#pragma once

// ratio:<c>[:<C>:<N>] accept rule for sampled MTP drafts (CHANGES §24). Header-only so a test can drive the exact
// code the server runs.
//
// Row rule. A drafted token y, drawn from the draft distribution q, is kept with probability min(1, c p(y) / q(y));
// on rejection the token is drawn from the exact residual (p - q)+. p is the target's own sampler output at the row
// (after its penalties, top-k, top-p, min-p and temperature). The emitted distribution is
//     r(y) = min(q(y), c p(y)) + R (p(y) - q(y))+ / Z,   R = 1 - sum min(q, c p),   Z = sum (p - q)+
// and r(y) <= c p(y) for every y: where q >= p the residual term is 0; where q < p, R <= 1 - sum min(q, p) = Z for
// c >= 1, so r(y) <= p(y). Tokens with p = 0 are never emitted. c = 1 is the exact rule.
//
// Phrase budget. With C > 1 and N >= 2 each emitted token spends s = max(0, log(r(y) / p(y))), the log of the factor
// by which the rule made it likelier than the target alone, and the next row's cap is
//     c_t = min(c, max(1, C exp(-S_t))),   S_t = the spends of the previous N - 1 emitted tokens.
// By induction S_t <= log C, so every run of at most N consecutive emitted tokens, aligned with verify cycles or not,
// is at most C times likelier under the rule than under the target from the same prefix (chain rule; spends are
// clipped at 0 so tokens made less likely cannot refund the budget). Tokens the target samples itself (rejection
// fills, bonus rows, forced or grammar tokens) spend 0 and still occupy their place in the window.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

// spends of the last emitted tokens, newest last
struct common_spec_budget {
    std::deque<double> ring;

    static constexpr size_t MAX_N = 256;

    void push(double s) {
        ring.push_back(s);
        if (ring.size() > MAX_N) {
            ring.pop_front();
        }
    }

    // the cap for the next emitted token: c, lowered so that it and the previous n_win - 1 tokens spend <= log C
    double cap(double c, double C, int n_win) const {
        if (!(C > 1.0) || n_win < 2) {
            return c;
        }
        double S = 0.0;
        const size_t n = std::min(ring.size(), (size_t) n_win - 1);
        for (size_t k = 0; k < n; ++k) {
            S += ring[ring.size() - 1 - k];
        }
        return std::min(c, std::max(1.0, C * std::exp(-S)));
    }
};

// one drafted row. p: the target's sampler output (probabilities sum to 1); q: the distribution the draft token was
// drawn from (sums to 1); u1, u2 uniform in [0, 1). Sets out to the emitted token and spend to its log boost
// max(0, log r(out)/p(out)); returns whether the draft was kept. fallback: emitted when rounding leaves no residual.
template <typename P, typename Q>
static bool common_spec_ratio_row(const P * p, size_t n_p, const Q & q, llama_token draft, double c, double u1, double u2,
                                  llama_token fallback, llama_token & out, double & spend) {
    const auto q_of = [&](llama_token t) {
        for (const auto & e : q) {
            if (e.id == t) {
                return (double) e.p;
            }
        }
        return 0.0;
    };
    double p_x = 0.0;
    for (size_t j = 0; j < n_p; ++j) {
        if (p[j].id == draft) {
            p_x = p[j].p;
            break;
        }
    }
    const double q_x = q_of(draft);
    spend = 0.0;
    if (q_x > 0.0 && u1 * q_x < c * p_x) {
        out = draft;
        // r(draft) = min(q, c p) when q > p (no residual there); <= p otherwise
        if (q_x > p_x && p_x > 0.0) {
            spend = std::log(std::min(q_x, c * p_x) / p_x);
        }
        return true;
    }
    // rejected: the exact residual (p - q)+; every token it can emit has q < p, so r <= p and the spend is 0
    double sum_r = 0.0;
    for (size_t j = 0; j < n_p; ++j) {
        sum_r += std::max(0.0, (double) p[j].p - q_of(p[j].id));
    }
    out = fallback;
    if (sum_r > 0.0) {
        double t = u2 * sum_r;
        for (size_t j = 0; j < n_p; ++j) {
            const double r = std::max(0.0, (double) p[j].p - q_of(p[j].id));
            if (r > 0.0) {
                out = p[j].id;
                t -= r;
                if (t < 0.0) {
                    break;
                }
            }
        }
    }
    return false;
}

// the row's emitted distribution r over p's tokens (for the stats log and tests), and KL(r || p) in nats
template <typename P, typename Q>
static double common_spec_ratio_dist(const P * p, size_t n_p, const Q & q, double c, std::vector<double> & r) {
    const auto q_of = [&](llama_token t) {
        for (const auto & e : q) {
            if (e.id == t) {
                return (double) e.p;
            }
        }
        return 0.0;
    };
    double kept = 0.0, Z = 0.0;
    for (const auto & e : q) {
        double pe = 0.0;
        for (size_t j = 0; j < n_p; ++j) {
            if (p[j].id == e.id) {
                pe = p[j].p;
                break;
            }
        }
        kept += std::min((double) e.p, c * pe);
    }
    for (size_t j = 0; j < n_p; ++j) {
        Z += std::max(0.0, (double) p[j].p - q_of(p[j].id));
    }
    const double R = std::max(0.0, 1.0 - kept);
    r.assign(n_p, 0.0);
    double kl = 0.0;
    for (size_t j = 0; j < n_p; ++j) {
        const double pj = p[j].p, qj = q_of(p[j].id);
        r[j] = std::min(qj, c * pj) + (Z > 0.0 ? R * std::max(0.0, pj - qj) / Z : 0.0);
        if (r[j] > 0.0 && pj > 0.0) {
            kl += r[j] * std::log(r[j] / pj);
        }
    }
    return kl;
}
