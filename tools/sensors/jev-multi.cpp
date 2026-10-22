// JEV System 1 noul decisions for many questions over one state: one row {"id", "state", "questions": [...]} per line
// in, one row {"id", "n_tokens", "p_true": [...], "ms"} per line out. The common token prefix of the row's prompts
// (bare-v1: "[kind] noul\n[state] ...") is decoded once and its sequence state restored before each question.
//
//   llama-jev-multi -m Qwen3.8-27B-UD-Q6_K.gguf --lora jev-27b-lora-f16.gguf -ngl 99 -sm tensor -fa 1 -c 8192
//       -ctk q8_0 -ctv q8_0 --jev-head head.safetensors --jev-calib calibration.json --jm-in rows.jsonl --jm-out out.jsonl

#include "arg.h"
#include "common.h"
#include "jev.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

static bool decode_range(llama_context * ctx, llama_batch & batch, const std::vector<llama_token> & toks, size_t from,
                         size_t to, int n_batch, bool last_out) {
    for (size_t i = from; i < to; i += n_batch) {
        common_batch_clear(batch);
        const size_t n = std::min(to - i, (size_t) n_batch);
        for (size_t k = 0; k < n; k++) {
            common_batch_add(batch, toks[i + k], (llama_pos) (i + k), { 0 }, last_out && i + k == to - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string head_path, calib_path, in_path, out_path;
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        auto take = [&](const char * flag, std::string & dst) {
            if (strcmp(argv[i], flag) == 0 && i + 1 < argc) {
                dst = argv[++i];
                return true;
            }
            return false;
        };
        if (take("--jev-head", head_path) || take("--jev-calib", calib_path) || take("--jm-in", in_path) ||
            take("--jm-out", out_path)) {
            continue;
        }
        rest.push_back(argv[i]);
    }
    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    if (head_path.empty() || in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s <common args> --jev-head h.safetensors [--jev-calib c.json] --jm-in rows.jsonl --jm-out out.jsonl\n", argv[0]);
        return 1;
    }
    common_init();
    params.embedding  = false;
    params.n_parallel = 1;

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
    llama_set_embeddings_nextn(ctx, true, /*masked*/ true);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_batch = llama_n_batch(ctx);
    const int n_ctx   = llama_n_ctx(ctx);

    std::ifstream fin(in_path);
    std::ofstream fout(out_path);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<uint8_t> snap;
    const std::vector<std::string> noul_opts = { "false", "true" };

    std::string line;
    long n_rows = 0, n_q = 0;
    double ms_total = 0.0;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        const json row = json::parse(line);
        const std::string state = row.at("state").get<std::string>();
        const auto questions = row.at("questions").get<std::vector<std::string>>();

        std::vector<std::vector<llama_token>> toks;
        for (const auto & q : questions) {
            toks.push_back(common_tokenize(vocab, jev_build_prompt(JEV_KIND_NOUL, state, q, noul_opts), false, true));
            if ((int) toks.back().size() > n_ctx) {
                LOG_ERR("row %ld: prompt too long\n", n_rows);
                return 1;
            }
        }
        // common prefix, leaving at least one token per question to decode
        size_t lcp = toks[0].size();
        for (const auto & t : toks) {
            size_t k = 0;
            while (k < lcp && k < t.size() && t[k] == toks[0][k]) {
                k++;
            }
            lcp = std::min(lcp, k);
            lcp = std::min(lcp, t.size() - 1);
        }

        const auto t0 = std::chrono::steady_clock::now();
        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode_range(ctx, batch, toks[0], 0, lcp, n_batch, false)) {
            LOG_ERR("row %ld: prefix decode failed\n", n_rows);
            return 1;
        }
        snap.resize(llama_state_seq_get_size(ctx, 0));
        llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0);

        std::vector<float> p_true;
        for (size_t qi = 0; qi < questions.size(); qi++) {
            if (qi > 0) {
                llama_memory_clear(llama_get_memory(ctx), true);
                if (llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0) == 0) {
                    LOG_ERR("row %ld: state restore failed\n", n_rows);
                    return 1;
                }
            }
            if (!decode_range(ctx, batch, toks[qi], lcp, toks[qi].size(), n_batch, true)) {
                LOG_ERR("row %ld: question decode failed\n", n_rows);
                return 1;
            }
            const float * h = llama_get_embeddings_nextn_ith(ctx, -1);
            if (!h) {
                LOG_ERR("row %ld: no hidden row\n", n_rows);
                return 1;
            }
            p_true.push_back(jev_decide(head, JEV_KIND_NOUL, h, 2)[1]);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fout << json({ { "id", row.value("id", std::to_string(n_rows)) }, { "n_tokens", toks[0].size() },
                       { "prefix", lcp }, { "p_true", p_true }, { "ms", ms } }).dump() << "\n";
        fout.flush();
        n_rows++;
        n_q += questions.size();
        ms_total += ms;
        if (n_rows % 10 == 0) {
            LOG_INF("jev-multi: %ld rows, %.0f ms/row, %.0f ms/question\n", n_rows, ms_total / n_rows, ms_total / n_q);
        }
    }
    LOG_INF("jev-multi: done, %ld rows, %ld questions, %.0f ms/question\n", n_rows, n_q, n_q ? ms_total / n_q : 0.0);
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
