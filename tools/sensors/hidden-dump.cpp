// Hidden-state dump for System-1 sensor research: one row {"id", "prompt", "marks": [byte offsets]} per line in.
// For every mark, the hidden state of the last token of prompt[:mark] is written (fp16) for each selected layer:
// layer inputs (llama_set_embeddings_layer_inp) and "final" (the final-norm row, as the MTP / JEV paths read it).
// The end of the prompt is always captured as the last mark. Each row is decoded from an empty context.
//
//   llama-hidden-dump -m model.gguf -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -c 8192 -b 2048 -ub 2048
//       --hd-in rows.jsonl --hd-out feats.f16 --hd-index index.jsonl --hd-layers 16,32,48,final [--hd-watch 248058,...]
//       [--hd-prefix prefix.txt]
// With --hd-prefix, every prompt must start with the prefix text: it is decoded once, its sequence state saved, and
// restored before each row (deep-context test without re-prefilling the prefix).
// feats.f16 layout: per row, per mark, per layer: n_embd fp16 values. index.jsonl: one line per row with
// {"id", "n_tokens", "pos": [...], "watch": [[logprob per watched token] per mark], "ms"}.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "../../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string in_path, out_path, index_path, layers_str = "final", watch_str, prefix_path;

    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        auto take = [&](const char * flag, std::string & dst) {
            if (strcmp(argv[i], flag) == 0 && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (take("--hd-in", in_path) || take("--hd-out", out_path) || take("--hd-index", index_path) ||
            take("--hd-layers", layers_str) || take("--hd-watch", watch_str) || take("--hd-prefix", prefix_path)) {
            continue;
        }
        rest.push_back(argv[i]);
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    if (in_path.empty() || out_path.empty() || index_path.empty()) {
        fprintf(stderr, "usage: %s <common args> --hd-in rows.jsonl --hd-out feats.f16 --hd-index index.jsonl "
                        "[--hd-layers 16,32,final] [--hd-watch tok,tok]\n", argv[0]);
        return 1;
    }
    common_init();
    params.embedding  = false;
    params.n_parallel = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_batch = llama_n_batch(ctx);
    const int n_ctx   = llama_n_ctx(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // -1 = final-norm row
    std::vector<int> layers;
    for (const auto & s : split(layers_str, ',')) {
        layers.push_back(s == "final" ? -1 : std::stoi(s));
    }
    bool need_final = false;
    for (int il : layers) {
        if (il < 0) {
            need_final = true;
        } else if (il >= n_layer) {
            LOG_ERR("layer %d >= n_layer %d (use 'final')\n", il, n_layer);
            return 1;
        } else {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
        }
    }
    // the final-norm row comes with the logits of output tokens; ask for it even when only the logits are watched
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    (void) need_final;

    std::vector<llama_token> watch;
    for (const auto & s : split(watch_str, ',')) {
        watch.push_back(std::stoi(s));
    }
    LOG_INF("hidden-dump: n_embd %d, n_layer %d, %zu layers, %zu watched tokens, n_ctx %d, n_batch %d\n",
            n_embd, n_layer, layers.size(), watch.size(), n_ctx, n_batch);

    std::ifstream fin(in_path);
    std::ofstream findex(index_path);
    FILE * fout = fopen(out_path.c_str(), "wb");
    if (!fin || !findex || !fout) {
        LOG_ERR("cannot open input/output files\n");
        return 1;
    }
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<ggml_fp16_t> h16(n_embd);

    std::string prefix_text;
    std::vector<llama_token> prefix_toks;
    std::vector<uint8_t> prefix_state;
    if (!prefix_path.empty()) {
        std::ifstream fp(prefix_path);
        prefix_text.assign(std::istreambuf_iterator<char>(fp), std::istreambuf_iterator<char>());
        prefix_toks = common_tokenize(vocab, prefix_text, false, true);
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i0 = 0; i0 < prefix_toks.size(); i0 += n_batch) {
            common_batch_clear(batch);
            const size_t n = std::min(prefix_toks.size() - i0, (size_t) n_batch);
            for (size_t k = 0; k < n; k++) {
                common_batch_add(batch, prefix_toks[i0 + k], (llama_pos) (i0 + k), { 0 }, false);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("prefix decode failed\n");
                return 1;
            }
        }
        prefix_state.resize(llama_state_seq_get_size(ctx, 0));
        llama_state_seq_get_data(ctx, prefix_state.data(), prefix_state.size(), 0);
        LOG_INF("hidden-dump: prefix %zu tokens, state %.1f MiB\n", prefix_toks.size(), prefix_state.size() / 1048576.0);
    }

    std::string line;
    long n_rows = 0, n_tok_total = 0;
    double ms_total = 0.0;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        const json row = json::parse(line);
        const std::string prompt = row.at("prompt").get<std::string>();
        std::vector<size_t> marks = row.value("marks", std::vector<size_t>());
        marks.push_back(prompt.size());

        const std::vector<llama_token> toks = common_tokenize(vocab, prompt, /*add_special*/ false, /*parse_special*/ true);
        if ((int) toks.size() > n_ctx) {
            LOG_ERR("row %ld: %zu tokens > n_ctx %d\n", n_rows, toks.size(), n_ctx);
            return 1;
        }
        // token position of each mark = last token of the prefix; the prefix must tokenize as a prefix of the whole
        std::vector<int> pos;
        for (size_t m : marks) {
            const auto pt = common_tokenize(vocab, prompt.substr(0, m), false, true);
            if (pt.empty() || pt.size() > toks.size() || !std::equal(pt.begin(), pt.end(), toks.begin())) {
                LOG_ERR("row %ld: mark %zu does not fall on a token boundary\n", n_rows, m);
                return 1;
            }
            pos.push_back((int) pt.size() - 1);
        }

        // feats[mark][layer] -> fp32 rows
        std::vector<std::vector<std::vector<float>>> feats(pos.size(), std::vector<std::vector<float>>(layers.size()));
        std::vector<std::vector<float>> wlp(pos.size(), std::vector<float>(watch.size(), 0.0f));

        size_t start = 0;
        if (!prefix_toks.empty()) {
            if (prompt.compare(0, prefix_text.size(), prefix_text) != 0 || toks.size() <= prefix_toks.size() ||
                !std::equal(prefix_toks.begin(), prefix_toks.end(), toks.begin())) {
                LOG_ERR("row %ld: prompt does not start with the prefix tokens\n", n_rows);
                return 1;
            }
            start = prefix_toks.size();
        }

        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        if (start > 0 && llama_state_seq_set_data(ctx, prefix_state.data(), prefix_state.size(), 0) == 0) {
            LOG_ERR("row %ld: prefix state restore failed\n", n_rows);
            return 1;
        }
        for (size_t i0 = start; i0 < toks.size(); i0 += n_batch) {
            common_batch_clear(batch);
            const size_t n = std::min(toks.size() - i0, (size_t) n_batch);
            for (size_t k = 0; k < n; k++) {
                const int p = (int) (i0 + k);
                const bool out = std::find(pos.begin(), pos.end(), p) != pos.end();
                common_batch_add(batch, toks[i0 + k], p, { 0 }, out);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("row %ld: llama_decode failed\n", n_rows);
                return 1;
            }
            for (size_t m = 0; m < pos.size(); m++) {
                const int p = pos[m];
                if (p < (int) i0 || p >= (int) (i0 + n)) {
                    continue;
                }
                const int bi = p - (int) i0;
                for (size_t l = 0; l < layers.size(); l++) {
                    const float * src = layers[l] < 0 ? llama_get_embeddings_nextn_ith(ctx, bi)
                                                      : llama_get_embeddings_layer_inp(ctx, (uint32_t) layers[l]) + (size_t) bi * n_embd;
                    if (!src) {
                        LOG_ERR("row %ld: no hidden row for layer %d\n", n_rows, layers[l]);
                        return 1;
                    }
                    feats[m][l].assign(src, src + n_embd);
                }
                if (!watch.empty()) {
                    const float * lg = llama_get_logits_ith(ctx, bi);
                    float mx = -INFINITY;
                    for (int v = 0; v < n_vocab; v++) {
                        mx = std::max(mx, lg[v]);
                    }
                    double se = 0.0;
                    for (int v = 0; v < n_vocab; v++) {
                        se += std::exp((double) lg[v] - mx);
                    }
                    for (size_t w = 0; w < watch.size(); w++) {
                        wlp[m][w] = (float) ((double) lg[watch[w]] - mx - std::log(se));
                    }
                }
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        for (size_t m = 0; m < pos.size(); m++) {
            for (size_t l = 0; l < layers.size(); l++) {
                ggml_fp32_to_fp16_row(feats[m][l].data(), h16.data(), n_embd);
                fwrite(h16.data(), sizeof(ggml_fp16_t), n_embd, fout);
            }
        }
        json out = { { "id", row.value("id", std::to_string(n_rows)) }, { "n_tokens", toks.size() },
                     { "pos", pos }, { "watch", wlp }, { "ms", ms } };
        findex << out.dump() << "\n";
        findex.flush();

        n_rows++;
        n_tok_total += toks.size();
        ms_total += ms;
        if (n_rows % 50 == 0) {
            LOG_INF("hidden-dump: %ld rows, %.1f ms/row, %.1f tok/s\n", n_rows, ms_total / n_rows, n_tok_total / (ms_total / 1000.0));
        }
    }
    LOG_INF("hidden-dump: done, %ld rows, %ld tokens, %.1f ms/row\n", n_rows, n_tok_total, n_rows ? ms_total / n_rows : 0.0);

    fclose(fout);
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
