// test-spec-ratio: the ratio accept rule (common/spec-ratio.h), the same code the server runs.
//  1. one row, many draws: the emitted frequencies match the analytic r, r <= c p for every token, r sums to 1, and
//     forbidden tokens (p = 0) are never emitted; drafts concentrated on the target's least likely tokens.
//  2. a stream of verify cycles (random draft lengths, rejections end a cycle, bonus rows, single target-sampled
//     tokens, rollbacks replayed from a restored budget): every emitted token's spend equals max(0, log r(y)/p(y))
//     under the cap its row used, and every run of <= N consecutive tokens, whatever its alignment, has
//     prod r/p <= C (sum of clipped log-ratios <= log C).
// Run: ctest -R test-spec-ratio (LLAMA_BUILD_TESTS), or standalone:
//   g++ -O2 -std=c++17 -Iinclude -Iggml/include tests/test-spec-ratio.cpp -o test-spec-ratio
#include "../common/spec-ratio.h"

#include <cassert>
#include <cstdio>
#include <random>

static std::mt19937_64 rng(12345);
static double U() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

struct row_case {
    std::vector<llama_token_data> p; // target's sampler output (sums to 1, some ids with p = 0 left out)
    std::vector<llama_token_data> q; // draft distribution (sums to 1)
};

static std::vector<double> dirichlet(int n, double a) {
    std::gamma_distribution<double> g(a, 1.0);
    std::vector<double> v(n);
    double s = 0.0;
    for (auto & x : v) { x = g(rng) + 1e-12; s += x; }
    for (auto & x : v) { x /= s; }
    return v;
}

static row_case make_case(int V) {
    row_case rc;
    const double alphas[] = {0.05, 0.3, 1.0, 5.0};
    auto pv = dirichlet(V, alphas[rng() % 4]);
    // the sampler forbids some tokens: they are absent from p (cur_p holds only the survivors)
    std::vector<int> allowed;
    int top = 0;
    for (int i = 1; i < V; ++i) { if (pv[i] > pv[top]) top = i; }
    double s = 0.0;
    for (int i = 0; i < V; ++i) {
        if (i == top || U() > 0.25) { allowed.push_back(i); s += pv[i]; }
    }
    for (int i : allowed) { rc.p.push_back({(llama_token) i, 0.0f, (float) (pv[i] / s)}); }
    std::vector<double> qv(V, 0.0);
    if (rng() % 2) {
        // adversarial: the draft piles onto the target's least likely tokens (allowed or not)
        std::vector<int> ord(V);
        for (int i = 0; i < V; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return pv[a] < pv[b]; });
        const int k = 1 + (int) (rng() % 3);
        auto w = dirichlet(k, 1.0);
        for (int j = 0; j < k; ++j) qv[ord[j]] += 0.97 * w[j];
        auto e = dirichlet(V, 1.0);
        for (int i = 0; i < V; ++i) qv[i] += 0.03 * e[i];
    } else {
        qv = dirichlet(V, alphas[rng() % 4]);
    }
    // the draft keeps a truncated candidate list (top 20, as the MTP draft): renormalize over it
    std::vector<int> ord(V);
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return qv[a] > qv[b]; });
    const int nk = std::min(V, 20);
    double qs = 0.0;
    for (int j = 0; j < nk; ++j) qs += qv[ord[j]];
    for (int j = 0; j < nk; ++j) rc.q.push_back({(llama_token) ord[j], 0.0f, (float) (qv[ord[j]] / qs)});
    return rc;
}

static llama_token draw(const std::vector<llama_token_data> & d) {
    double u = U(), a = 0.0;
    for (size_t k = 0; k + 1 < d.size(); ++k) { a += d[k].p; if (u < a) return d[k].id; }
    return d.back().id;
}

static double p_of(const std::vector<llama_token_data> & d, llama_token t) {
    for (const auto & e : d) if (e.id == t) return e.p;
    return 0.0;
}

int main() {
    int fails = 0;
    // ---- 1. per-row distribution
    double worst_cap = 0.0, worst_dev = 0.0, worst_sum = 0.0;
    for (int t = 0; t < 3000; ++t) {
        const row_case rc = make_case(2 + (int) (rng() % 40));
        const double cs[] = {1.0, 1.1, 1.25, 1.5, 2.0, 3.0};
        const double c = cs[rng() % 6];
        std::vector<double> r;
        common_spec_ratio_dist(rc.p.data(), rc.p.size(), rc.q, c, r);
        double sr = 0.0;
        for (size_t j = 0; j < r.size(); ++j) {
            sr += r[j];
            worst_cap = std::max(worst_cap, r[j] / (c * rc.p[j].p) - 1.0);
        }
        worst_sum = std::max(worst_sum, std::fabs(sr - 1.0));
        if (t < 300) {   // Monte Carlo of the row decision itself
            const int n = 200000;
            std::vector<double> cnt(rc.p.size(), 0.0);
            for (int k = 0; k < n; ++k) {
                const llama_token y = draw(rc.q);
                llama_token out; double spend;
                common_spec_ratio_row(rc.p.data(), rc.p.size(), rc.q, y, c, U(), U(), rc.p[0].id, out, spend);
                bool found = false;
                for (size_t j = 0; j < rc.p.size(); ++j) {
                    if (rc.p[j].id == out) {
                        cnt[j] += 1.0; found = true;
                        const double want = std::max(0.0, std::log(r[j] / rc.p[j].p));
                        if (std::fabs(spend - want) > 1e-6) { if (fails++ < 5) printf("spend %g != %g\n", spend, want); }
                    }
                }
                if (!found) { if (fails++ < 5) printf("emitted a token the sampler forbids\n"); }
            }
            for (size_t j = 0; j < rc.p.size(); ++j) {
                // small expected counts: the exact Poisson tail of the observed count, as a z score
                const double f = cnt[j] / n, lam = n * r[j], k = cnt[j];
                double dev;
                if (lam < 30.0) {
                    double term = std::exp(-lam), cdf = 0.0;
                    for (int x = 0; x < (int) k; ++x) { cdf += term; term *= lam / (x + 1); }
                    const double tail = k > lam ? 1.0 - cdf : cdf + term;      // P(X >= k) or P(X <= k)
                    dev = std::sqrt(-2.0 * std::log(std::max(tail, 1e-300)));
                } else {
                    dev = std::fabs(f - r[j]) / std::sqrt(r[j] * (1 - r[j]) / n);
                }
                if (dev > 5.5 && getenv("DBG")) {
                    printf("row %d c %g tok %d: p %.6g q %.6g r %.6g emp %.6g (%.1f sd); |p|=%zu |q|=%zu\n", t, c, rc.p[j].id,
                           rc.p[j].p, p_of(rc.q, rc.p[j].id), r[j], f, dev, rc.p.size(), rc.q.size());
                }
                worst_dev = std::max(worst_dev, dev);
            }
        }
    }
    printf("rows: max r/(c p) - 1 = %.2e, max |sum r - 1| = %.2e, worst Monte Carlo deviation %.2f sd\n", worst_cap, worst_sum, worst_dev);
    // ~5,000 checks: a deviation score of 6.5 is a tail probability of ~1e-9 each
    if (worst_cap > 1e-6 || worst_sum > 1e-6 || worst_dev > 6.5) fails++;

    // ---- 2. stream of cycles with the phrase budget
    struct cfg { double c, C; int N; };
    const cfg cfgs[] = {{2.0, 1.5, 16}, {1.5, 1.25, 8}, {3.0, 2.0, 4}, {2.0, 1.1, 32}, {1.5, 1.5, 2}};
    for (const auto & cf : cfgs) {
        common_spec_budget B;
        std::vector<double> ell;    // log r/p of each emitted token (unclipped), in order
        std::vector<double> spent;  // the spend the budget recorded
        int n_replay = 0, n_low = 0, n_rows = 0;
        while (ell.size() < 200000) {
            if (rng() % 10 == 0) {  // a token the target samples itself (no draft): r = p
                B.push(0.0); ell.push_back(0.0); spent.push_back(0.0);
                continue;
            }
            const common_spec_budget saved = B;
            const int G = 1 + (int) (rng() % 4);
            std::vector<double> cyc_ell, cyc_sp;
            bool rejected = false;
            for (int i = 0; i < G && !rejected; ++i) {
                const row_case rc = make_case(2 + (int) (rng() % 30));
                const double c = B.cap(cf.c, cf.C, cf.N);
                n_rows++; n_low += c < std::min(cf.c, cf.C);
                const llama_token y = draw(rc.q);
                llama_token out; double spend;
                const bool keep = common_spec_ratio_row(rc.p.data(), rc.p.size(), rc.q, y, c, U(), U(), rc.p[0].id, out, spend);
                std::vector<double> r;
                common_spec_ratio_dist(rc.p.data(), rc.p.size(), rc.q, c, r);
                double rout = 0.0, pout = 0.0;
                for (size_t j = 0; j < rc.p.size(); ++j) if (rc.p[j].id == out) { rout = r[j]; pout = rc.p[j].p; }
                assert(pout > 0.0);
                B.push(spend);
                cyc_ell.push_back(std::log(rout / pout)); cyc_sp.push_back(spend);
                rejected = !keep;
            }
            if (!rejected) { B.push(0.0); cyc_ell.push_back(0.0); cyc_sp.push_back(0.0); } // bonus row from p
            if (rng() % 3 == 0) {
                // rollback: the server restores the pre-cycle sampler and replays the decided tokens with their spends
                const common_spec_budget after = B;
                B = saved;
                for (double s : cyc_sp) B.push(s);
                if (B.ring != after.ring) { if (fails++ < 5) printf("replay changed the budget state\n"); }
                n_replay++;
            }
            ell.insert(ell.end(), cyc_ell.begin(), cyc_ell.end());
            spent.insert(spent.end(), cyc_sp.begin(), cyc_sp.end());
        }
        // every run of <= N consecutive tokens: sum of clipped log ratios (>= the sum of the actual ones) <= log C
        double worst = -1e9, worst_raw = -1e9;
        int bad_spend = 0;
        for (size_t t = 0; t < ell.size(); ++t) {
            if (std::fabs(spent[t] - std::max(0.0, ell[t])) > 1e-5) bad_spend++;
            double s = 0.0, sr = 0.0;
            for (int k = 0; k < cf.N && k <= (int) t; ++k) {
                s += std::max(0.0, ell[t - k]); sr += ell[t - k];
                worst = std::max(worst, s); worst_raw = std::max(worst_raw, sr);
            }
        }
        printf("ratio:%g:%g:%g  %zu tokens, %d drafted rows (%.1f%% with the cap lowered by the budget), %d replays: max over runs <= N of "
               "prod r/p = %.6f (bound %g), spends wrong %d\n", cf.c, cf.C, (double) cf.N, ell.size(), n_rows,
               100.0 * n_low / n_rows, n_replay, std::exp(worst_raw), cf.C, bad_spend);
        if (worst > std::log(cf.C) + 1e-5 || bad_spend) fails++;
    }
    printf(fails ? "FAILED (%d)\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
