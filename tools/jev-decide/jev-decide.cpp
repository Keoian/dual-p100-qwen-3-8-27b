// JEV System 1 decisions over a JSONL file: one row {"id", "kind", "state", "question", "options"} per line in,
// one row {"id", "n_tokens", "logits", "probs", "ms"} per line out. The backbone LoRA comes from --lora; the head and
// the calibration from --jev-head / --jev-calib. Each row is decoded from an empty context (no reuse between rows).
//
//   llama-jev-decide -m Qwen3.8-27B-UD-Q6_K.gguf --lora jev-27b-lora-f16.gguf -ngl 99 -sm tensor -fa 1
//       --jev-head head.safetensors --jev-calib calibration.json --jev-in rows.jsonl --jev-out out.jsonl

#include "arg.h"
#include "common.h"
#include "jev.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h" // llama_set_embeddings_nextn: the final-norm row of output tokens only

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    std::string head_path, calib_path, in_path, out_path, hidden_path;
    long limit = -1;

    // strip the tool's own flags, the rest goes to the common parser
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        auto take = [&](const char * flag, std::string & dst) {
            if (strcmp(argv[i], flag) == 0 && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        std::string lim;
        if (take("--jev-head", head_path) || take("--jev-calib", calib_path) || take("--jev-in", in_path) ||
            take("--jev-out", out_path) || take("--jev-hidden", hidden_path)) {
            continue;
        }
        if (take("--jev-limit", lim)) {
            limit = std::stol(lim);
            continue;
        }
        rest.push_back(argv[i]);
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    if (head_path.empty() || in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s <common args> --jev-head head.safetensors [--jev-calib calibration.json] "
                        "--jev-in rows.jsonl --jev-out out.jsonl [--jev-hidden h.f32] [--jev-limit N]\n", argv[0]);
        return 1;
    }
    common_init();

    // no embeddings mode (it makes every token an output): read the final-norm hidden row of the last token only
    params.embedding    = false;
    params.n_parallel   = 1;

    jev_head head;
    std::string err;
    if (!jev_head_load(head_path, calib_path, head, err)) {
        LOG_ERR("%s\n", err.c_str());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load the model\n");
        return 1;
    }
    if (llama_model_n_embd(model) != head.n_embd) {
        LOG_ERR("head n_embd %d != model n_embd %d\n", head.n_embd, llama_model_n_embd(model));
        return 1;
    }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_batch = llama_n_batch(ctx);
    const int n_ctx   = llama_n_ctx(ctx);
    LOG_INF("jev: head n_embd %d, temperatures noul %.4f score %.4f choice %.4f, n_ctx %d, n_batch %d, loras %zu\n",
            head.n_embd, head.temp[0], head.temp[1], head.temp[2], n_ctx, n_batch, params.lora_adapters.size());

    std::ifstream fin(in_path);
    std::ofstream fout(out_path);
    FILE * fh = hidden_path.empty() ? nullptr : fopen(hidden_path.c_str(), "wb");
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::string line;
    long n_rows = 0, n_tok_total = 0;
    double ms_total = 0.0;
    while (std::getline(fin, line) && (limit < 0 || n_rows < limit)) {
        if (line.empty()) {
            continue;
        }
        const json row = json::parse(line);
        jev_kind kind;
        if (!jev_kind_from_str(row.at("kind").get<std::string>(), kind)) {
            LOG_ERR("bad kind in row %ld\n", n_rows);
            return 1;
        }
        std::vector<std::string> options = row.value("options", std::vector<std::string>());
        if (!jev_check_options(kind, options, err)) {
            LOG_ERR("row %ld: %s\n", n_rows, err.c_str());
            return 1;
        }
        const std::string prompt = jev_build_prompt(kind, row.value("state", std::string()), row.at("question").get<std::string>(), options);
        const std::vector<llama_token> toks = common_tokenize(vocab, prompt, /*add_special*/ false, /*parse_special*/ true);
        if ((int) toks.size() > n_ctx) {
            LOG_ERR("row %ld: %zu tokens > n_ctx %d\n", n_rows, toks.size(), n_ctx);
            return 1;
        }

        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            common_batch_clear(batch);
            const size_t n = std::min(toks.size() - i, (size_t) n_batch);
            for (size_t k = 0; k < n; k++) {
                common_batch_add(batch, toks[i + k], (llama_pos) (i + k), { 0 }, i + k == toks.size() - 1);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("row %ld: llama_decode failed\n", n_rows);
                return 1;
            }
        }
        const float * h = llama_get_embeddings_nextn_ith(ctx, -1);
        if (!h) {
            LOG_ERR("row %ld: no embeddings\n", n_rows);
            return 1;
        }
        std::vector<float> raw;
        const std::vector<float> probs = jev_decide(head, kind, h, options.size(), &raw);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (fh) {
            fwrite(h, sizeof(float), head.n_embd, fh);
        }
        json out = { { "id", row.value("id", std::to_string(n_rows)) }, { "n_tokens", toks.size() },
                     { "logits", raw }, { "probs", probs }, { "ms", ms } };
        fout << out.dump() << "\n";
        fout.flush();

        n_rows++;
        n_tok_total += toks.size();
        ms_total += ms;
        if (n_rows % 100 == 0) {
            LOG_INF("jev: %ld rows, %.1f ms/row, %.1f tok/s\n", n_rows, ms_total / n_rows, n_tok_total / (ms_total / 1000.0));
        }
    }
    LOG_INF("jev: done, %ld rows, %ld tokens, %.1f ms/row, %.1f tok/s\n", n_rows, n_tok_total,
            n_rows ? ms_total / n_rows : 0.0, ms_total > 0 ? n_tok_total / (ms_total / 1000.0) : 0.0);

    if (fh) {
        fclose(fh);
    }
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
