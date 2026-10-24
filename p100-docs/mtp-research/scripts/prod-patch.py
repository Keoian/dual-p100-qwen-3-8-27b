# prod-patch.py <repo>: put the lookup stop rule (LLAMA_SPEC_STOP_ISO) and its temperature gate on the default branch
# (p100-optimizations-keoian), without the research trace hooks. Tables and lookup are copied from the research
# branch's common/speculative.cpp (git show mtp-adaptive-research:common/speculative.cpp, or the working tree there).
import sys, subprocess
repo = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else None  # file with the research speculative.cpp (default: git show)
res = open(src).read() if src else subprocess.run(["git", "-C", repo, "show", "mtp-adaptive-research:common/speculative.cpp"],
                                                  capture_output=True, text=True, check=True).stdout
tabs = res[res.index("static const float spec_iso_x"):res.index("struct common_speculative_impl_draft_mtp")]

def sub(path, old, new, count=1):
    p = f"{repo}/{path}"; s = open(p).read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(p, "w").write(s.replace(old, new))

rule = '''// LLAMA_SPEC_STOP_ISO=<theta> (default off; start-qwen.sh sets 0.125): the MTP draft stops when the estimated
// survival of the drafted prefix through this step, prod_{i<d} iso_q(q_i) * iso_p1(p1_d), falls below theta, instead
// of the p-min test. q_i is the draft's probability of the token it drafted at step i, p1_d the top-1 probability of
// step d's distribution; the two maps are isotonic fits of the target's p/q acceptance (min(1, p/q)) on traced verify
// cycles (Qwen3.8-27B UD-Q6_K, sampled drafts at T 1.0; fit and measurements: branch mtp-adaptive-research,
// p100-docs/mtp-research). The decision for step d uses only step d's distribution and the tokens already drafted,
// never the token drawn at step d, so the speculative-sampling rule (per-token or block) keeps the output
// distribution exact. The depth-based p_cum guard still applies.
static const float spec_stop_iso = [] {
    const char * s = getenv("LLAMA_SPEC_STOP_ISO");
    return s ? (float) atof(s) : 0.0f;
}();
// LLAMA_SPEC_STOP_ISO_TEMP=<lo>,<hi> (default 0.5,0.9): the rule only for requests whose target temperature lies in
// [lo, hi]; others keep the p-min test. The tables were fit at the thinking-off sampling (target T 0.7, top-p 0.8,
// presence 1.5), where the rule measured +4.7% (chat) and +4.8% (long context, 32k-200k); at the thinking sampling
// (T 1.0, top-p 0.95) the target accepts more of the draft than the tables predict, the rule stops too early and
// measured -0.8% / +1.9% (budget / block verify), so it is off there.
static const std::pair<float, float> spec_stop_iso_temp = [] {
    const char * s = getenv("LLAMA_SPEC_STOP_ISO_TEMP");
    float lo = 0.5f, hi = 0.9f;
    if (s) {
        sscanf(s, "%f,%f", &lo, &hi);
    }
    return std::make_pair(lo, hi);
}();
'''
sub("common/speculative.cpp", "struct common_speculative_impl_draft_mtp : public common_speculative_impl {",
    rule + tabs + "struct common_speculative_impl_draft_mtp : public common_speculative_impl {")
sub("common/speculative.cpp", "    std::vector<float> p_cum;\n",
    "    std::vector<float> p_cum;\n    std::vector<float> s_iso; // LLAMA_SPEC_STOP_ISO: estimated survival of the drafted prefix\n")
sub("common/speculative.cpp", "            p_cum.resize(n_seq);\n            p_cum[seq_id] = 1.0f;\n",
    "            p_cum.resize(n_seq);\n            p_cum[seq_id] = 1.0f;\n            s_iso.resize(n_seq);\n            s_iso[seq_id] = 1.0f;\n")
sub("common/speculative.cpp",
'''                const bool  stop    = cur_p->data[0].p < params.p_min || pc_next < p_cum_min(dparams.at(seq_id).pos0);
''',
'''                const float t_tgt   = dparams.at(seq_id).target_temp;
                const bool  iso_on  = spec_stop_iso > 0.0f &&
                                      (t_tgt < 0.0f || (t_tgt >= spec_stop_iso_temp.first && t_tgt <= spec_stop_iso_temp.second));
                const bool  stop    = (iso_on ? s_iso[seq_id]*spec_iso_lookup(spec_iso_p1, cur_p->data[0].p) < spec_stop_iso
                                              : cur_p->data[0].p < params.p_min) ||
                                      pc_next < p_cum_min(dparams.at(seq_id).pos0);
''')
sub("common/speculative.cpp",
'''                auto & dp = dparams.at(seq_id);

                if (sample_temp > 0.0f) {''',
'''                auto & dp = dparams.at(seq_id);

                float q_id = cur_p->data[0].p; // the draft's probability of the token it drafts
                if (sample_temp > 0.0f) {''')
sub("common/speculative.cpp",
'''                    id = q[k].id;
                    dp.dists.push_back(std::move(q));
                }
''',
'''                    id = q[k].id;
                    q_id = q[k].p;
                    dp.dists.push_back(std::move(q));
                }

                if (iso_on) {
                    s_iso[seq_id] *= spec_iso_lookup(spec_iso_q, q_id);
                }
''')
sub("common/speculative.h",
'''    std::vector<std::vector<llama_token_data>> dists = {};
};''',
'''    std::vector<std::vector<llama_token_data>> dists = {};

    // the target sampler's temperature for this sequence's request (< 0: unknown). LLAMA_SPEC_STOP_ISO applies only
    // when it lies in LLAMA_SPEC_STOP_ISO_TEMP (or is unknown)
    float target_temp = -1.0f;
};''')
sub("tools/server/server-context.cpp",
'''                            /* .result   = */ &slot.spec_draft,
                        };
''',
'''                            /* .result   = */ &slot.spec_draft,
                        };
                        common_speculative_get_draft_params(spec.get(), slot.id).target_temp = slot.task->params.sampling.temp;
''')
print("patched")
