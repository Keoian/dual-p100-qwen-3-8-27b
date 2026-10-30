# parity_sim.py: SIMULATION (CPU, recorded traces). Task 1c: how much exact-acceptance overlap sum(min(p, q)) the MTP
# draft gains when it samples the way the target samples, and from other valid draft distributions q.
#
# Data: trace-cfchat / trace-cfthink (72 suite prompts each, n-max 4 drafted to the end, no stop rule; the target's raw
# top-128 logits + full-vocab lse at every drafted row, teacher-forced past the first rejection; pen_hist = presence
# window). Rows = drafted rows of cf_ok cycles (no grammar sampler, no forced budget token).
#   p  = the target's served distribution (penalties -> top-k -> top-p -> min-p -> temperature), quality.Rows.dist
#   q0 = production's exact-mode draft: the draft's top-20 at T 1.0 (LLAMA_SPEC_SAMPLE_TEMP=1, DRAFT_TOPK=20)
#   The traces keep the draft's top-5 (T 1 probs over its top-20) and the sampled token's q0. Overlap is therefore
#   computed on the known top-5 exactly, plus a tail term in [0, min(q_tail, p_tail)]; for q0 the sampled estimator
#   mean(min(1, p(x)/q0(x))), x ~ q0 (recorded), is unbiased and includes the tail, which calibrates the tail term.
# Variants: draft temperature Td, draft top-p (on T 1 probs, the target chain's order), presence parity (the draft's
# own top-5 penalized by the target's window + drafted prefix), per-depth temperatures.
# Throughput: cfcore's cost model and production's stop tables/thetas (chat 0.125, thinking 0.15), stop score fed with
# E_{x~q}[tq(q(x))] per row (the drafted token is random), paired bootstrap over test prompts.
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
sys.path.insert(0, "/work/bench/mtp4"); sys.path.insert(0, "/work/bench/mtp3/analysis")
import cfcore, quality, qload
from quality import lookup

D = 4
OWN = {"cfchat": (0.7, 20, 0.8, 0.0, 1.5), "cfthink": (1.0, 20, 0.95, 0.0, 0.0)}  # temp, top_k, top_p, min_p, pres


class Data:
    def __init__(self, name):
        self.name = name
        self.S = cfcore.Set([name])
        p = self.S.parts[0]
        c = p["c"]
        self.c = c
        cx, _ = qload.load(name)
        assert (cx["t_us"] == c["t_us"]).all()
        R = quality.Rows(cx, cx["cf_ok"])
        # same row order as cfvec.Rows (cfcore.Set): (rc, rd) over cf_ok cycles
        assert (R.rc == np.where(c["cf_ok"])[0][p["rr"]]).all() and (R.rd == p["rd"]).all()
        self.R = R
        self.rr, self.rd = p["rr"], p["rd"]
        tid, pS, _ = R.dist(*OWN[name])
        self.tid, self.pS = tid, pS
        self.dtid = R.dtid
        self.dtp = np.where(R.dtid >= 0, np.nan_to_num(R.dtp), 0.0)
        self.dtwin = R.dtwin
        self.rest = np.clip(1.0 - self.dtp.sum(1), 0.0, 1.0)
        self.pS5 = np.stack([lookup(tid, pS, self.dtid[:, j], 0.0) for j in range(5)], 1)
        self.pS_tail = np.clip(1.0 - self.pS5.sum(1), 0.0, 1.0)
        # recorded draft token: q0 and served p
        self.x, self.q0x = R.x, R.q
        self.pSx = lookup(tid, pS, self.x, 0.0)
        self.a_T1 = np.where(self.q0x > 0, np.minimum(1.0, self.pSx / np.where(self.q0x > 0, self.q0x, 1)), 0.0)
        self.split = self.S.split[self.rr]
        self.cat = self.S.cat_[self.rr]

    def variant(self, Td=1.0, top_p=1.0, pres=0.0, top_k=20, Td_depth=None):
        """q over the draft's top-5 (n, 5) and the tail mass (n,), for draft sampling chain
        presence(pres) -> top-k -> top-p (T 1 probs) -> temperature Td (per depth if Td_depth)."""
        tp, rest = self.dtp, self.rest
        lq = np.where(tp > 0, np.log(np.maximum(tp, 1e-300)), -np.inf)
        if pres:
            lq = lq - pres * self.dtwin
        o = np.argsort(-lq, 1, kind="stable")
        lq = np.take_along_axis(lq, o, 1)
        ids = np.take_along_axis(self.dtid, o, 1)
        e = np.exp(lq)
        kk = min(top_k, 5)
        e[:, kk:] = 0.0
        rest_k = rest if top_k > 5 else np.zeros_like(rest)
        Z1 = e.sum(1) + rest_k           # T 1 normalizer over the top-k (penalties on tail ignored)
        p1 = e / Z1[:, None]
        cum = np.cumsum(p1, 1)
        if top_p < 1:
            hit = cum >= top_p
            n5 = np.where(hit.any(1), hit.argmax(1) + 1, 5)
            in_tail = ~hit.any(1)
            m_t = np.where(in_tail, np.minimum(rest_k / Z1, top_p - cum[:, -1]), 0.0)  # kept tail mass at T 1
        else:
            n5 = np.full(len(e), 5)
            m_t = rest_k / Z1
        keep = np.arange(5)[None, :] < n5[:, None]
        T = np.full(len(e), Td, float)
        if Td_depth is not None:
            T = np.asarray(Td_depth, float)[self.rd]
        w5 = np.where(keep & (p1 > 0), p1 ** (1.0 / T[:, None]), 0.0)
        # tail: m_t spread over tokens no larger than the 5th listed one
        cap = np.maximum(np.where(tp[:, 4] > 0, tp[:, 4] / Z1, 1e-12), 1e-12)
        ntail = np.maximum(m_t / cap, 1.0)
        wt = np.where(m_t > 0, ntail * (m_t / ntail) ** (1.0 / T), 0.0)
        Z = w5.sum(1) + wt
        return ids, w5 / Z[:, None], wt / Z

    def overlap(self, ids, q5, qt):
        """known-part overlap (lower bound) and the tail term's upper bound."""
        p5 = np.stack([lookup(self.tid, self.pS, ids[:, j], 0.0) for j in range(5)], 1)
        lo = np.minimum(p5, q5).sum(1)
        tail_hi = np.minimum(qt, np.clip(1.0 - p5.sum(1), 0, 1))
        return lo, tail_hi

    def iso_q_expect(self, ids, q5, qt, tq):
        """E_{x~q}[tq(q(x))]: the stop table's post-acceptance factor in expectation over the drafted token."""
        f = (q5 * np.interp(q5, cfcore.X24, tq)).sum(1)
        # tail tokens: q per token = qt/ntail, approximated with the smallest listed q
        qmin = np.where(q5 > 0, q5, np.inf).min(1)
        qmin = np.where(np.isfinite(qmin), qmin, 0.0)
        f += qt * np.interp(np.minimum(qmin, qt), cfcore.X24, tq)
        return f


def to_mat(Dt, v):
    out = np.zeros((Dt.S.n, D))
    out[Dt.rr, Dt.rd] = v
    return out


def throughput(Dt, a_rows, isoq_rows=None, theta=None, tables=None, split=cfcore.TEST):
    """(E, T) per cycle under the production stop rule (tables + theta) fed with a's and the expected iso_q factor;
    n-max 4 fixed if theta is None."""
    S = Dt.S
    A = to_mat(Dt, a_rows)
    if theta is None:
        k = S.n_real.copy(); st = np.full(S.n, D)
        return S.ET(A, k, st)
    tq, tp = tables
    post = to_mat(Dt, isoq_rows)
    prior = np.where(S.valid, np.interp(np.nan_to_num(S.p1), cfcore.X24, tp), 0.0)
    cp = np.c_[np.ones(S.n), np.cumprod(post, axis=1)[:, :-1]]
    k, st = S.k_of(cp * prior < theta)
    return S.ET(A, k, st)


def prod_stop(name):
    return (cfcore.PROD_CHAT, 0.125) if name == "cfchat" else (cfcore.PROD_THINK, 0.15)


def main():
    out = []
    pr = lambda s="": (print(s, flush=True), out.append(s))
    for name in ("cfchat", "cfthink"):
        Dt = Data(name)
        tables, theta = prod_stop(name)
        S0 = OWN[name]
        pr(f"## {name}: target {S0[0]}/{S0[2]}/{S0[1]}/min-p {S0[3]}, presence {S0[4]}; rows {len(Dt.x)}, "
           f"cycles {Dt.S.n}")
        # baseline q0 (production): known part + calibrated tail
        ids0, q50, qt0 = Dt.variant(1.0, 1.0, 0.0)
        lo0, th0 = Dt.overlap(ids0, q50, qt0)
        a0 = Dt.a_T1
        tail_share = (a0.mean() - lo0.mean()) / max(th0.mean(), 1e-9)
        pr(f"q0 check: listed q0 vs recorded top-5 max|diff| {np.abs(q50 - Dt.dtp[np.arange(len(q50))[:, None], np.argsort(-Dt.dtp, 1)]).max():.2e}")
        pr(f"q0 overlap: sampled estimator {a0.mean():.4f}; top-5 part {lo0.mean():.4f}; tail bound "
           f"{th0.mean():.4f} -> tail realizes {tail_share * 100:.0f}% of its bound")
        base_iso = Dt.iso_q_expect(ids0, q50, qt0, tables[0])
        # production's stop rule on recorded q (the deployed decision), for reference
        Eb, Tb = throughput(Dt, a0, base_iso, theta, tables)
        En, Tn = throughput(Dt, a0)
        rb = Dt.S.rate(Eb, Tb, cfcore.TEST)
        pr(f"baseline sim (test split): stop tables {rb:.2f} t/s, tok/cycle {Eb[Dt.S.split == cfcore.TEST].mean():.3f}; "
           f"n-max 4 fixed {Dt.S.rate(En, Tn, cfcore.TEST):.2f} t/s")

        def est(lo, th):
            return lo + tail_share * th   # tail term at the baseline's realized share of its bound

        rows = []

        def run(label, **kw):
            ids, q5, qt = Dt.variant(**kw)
            lo, th = Dt.overlap(ids, q5, qt)
            a = est(lo, th)
            iso = Dt.iso_q_expect(ids, q5, qt, tables[0])
            E, T = throughput(Dt, a, iso, theta, tables)
            # same draft lengths as baseline's policy would choose with the variant's own stop score
            g, ci = Dt.S.boot_gain((E, T), (Eb, Tb))
            byd = [a[Dt.rd == d].mean() for d in range(D)]
            m = Dt.S.split == cfcore.TEST
            rows.append((label, a.mean(), lo.mean(), (lo + th).mean(), byd, E[m].mean(), g, ci))
            pr(f"| {label} | {a.mean():.4f} | [{lo.mean():.4f}, {(lo + th).mean():.4f}] | "
               + " / ".join(f"{v:.3f}" for v in byd) + f" | {E[m].mean():.3f} | {g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] |")
            return a

        pr("")
        pr("| draft q | overlap (est.) | bounds | by depth 1/2/3/4 | tok/cycle | sim t/s vs prod (95% CI) |")
        pr("|---|---|---|---|---|---|")
        a_prod_est = run("prod: T1.0 top-20", Td=1.0)
        T0, k0, tp0, mp0, pres0 = S0
        run("+ top-p (target's)", Td=1.0, top_p=tp0)
        run("+ presence parity", Td=1.0, pres=pres0)
        run(f"+ T {T0}", Td=T0)
        run("full parity (pres, top-p, T)", Td=T0, top_p=tp0, pres=pres0)
        best = None
        for Td in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            for tpd in sorted({tp0, 0.8, 0.9, 0.95, 1.0}):
                ids, q5, qt = Dt.variant(Td=Td, top_p=tpd, pres=pres0)
                lo, th = Dt.overlap(ids, q5, qt)
                a = est(lo, th)
                dev = (Dt.split == cfcore.DEV) | (Dt.split == cfcore.VAL)
                sc = a[dev].mean()
                if best is None or sc > best[0]:
                    best = (sc, Td, tpd)
        pr(f"best (Td, top-p) on dev+val by overlap: Td {best[1]}, top-p {best[2]} (dev+val overlap {best[0]:.4f})")
        run(f"tuned: T {best[1]}, top-p {best[2]}, pres", Td=best[1], top_p=best[2], pres=pres0)
        # per-depth temperature, tuned greedily on dev+val
        Tdd = [best[1]] * D
        for d in range(D):
            bestd = None
            for Td in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
                Tx = list(Tdd); Tx[d] = Td
                ids, q5, qt = Dt.variant(top_p=best[2], pres=pres0, Td_depth=Tx)
                lo, th = Dt.overlap(ids, q5, qt)
                a = est(lo, th)
                dev = ((Dt.split == cfcore.DEV) | (Dt.split == cfcore.VAL)) & (Dt.rd == d)
                if bestd is None or a[dev].mean() > bestd[0]:
                    bestd = (a[dev].mean(), Td)
            Tdd[d] = bestd[1]
        run(f"per-depth T {Tdd}, top-p {best[2]}, pres", top_p=best[2], pres=pres0, Td_depth=Tdd)
        # greedy draft (q = delta at the draft's argmax): exact acceptance p(argmax)
        ag = Dt.pS5[np.arange(len(Dt.x)), np.argmax(Dt.dtp, 1)]
        Eg, Tg = throughput(Dt, ag, np.interp(np.ones_like(ag), cfcore.X24, tables[0]), theta, tables)
        g, ci = Dt.S.boot_gain((Eg, Tg), (Eb, Tb))
        pr(f"| greedy draft | {ag.mean():.4f} | exact | " + " / ".join(f"{ag[Dt.rd == d].mean():.3f}" for d in range(D))
           + f" | {Eg[Dt.S.split == cfcore.TEST].mean():.3f} | {g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] |")
        # by category (overlap gain of full parity)
        ids, q5, qt = Dt.variant(Td=T0, top_p=tp0, pres=pres0)
        lo, th = Dt.overlap(ids, q5, qt)
        ap = est(lo, th)
        pr("")
        pr("overlap by category, prod -> full parity: " + ", ".join(
            f"{cfcore.cfload.CATS[k]} {a_prod_est[Dt.cat == k].mean():.3f}->{ap[Dt.cat == k].mean():.3f}"
            for k in range(len(cfcore.cfload.CATS)) if (Dt.cat == k).any()))
        pr("")
    open("/work/bench/mtp5/out-parity.md", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
