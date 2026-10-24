# cfapprox2.py [set ...]: CF_ANALYSIS step 4 d (parametric families) and e (analytic estimate from the draft's own
# distribution reshaped by S). Same scoring as cfapprox.py; appends to out/cf/approx-<set>.json.
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.isotonic import IsotonicRegression
import cfcore as C
from cfapprox import Ev, stats, TABLE_BYTES

OUT = C.OUT
G = C.GRID
nG = len(G)


def feats(g):
    t, tp, tk, mp, pr = g
    return np.array([1.0, t, t * t, tp, pr, t * pr, tp * t, mp, (tk - 20) / 20.0])


def iso(x, y, w=None):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y, sample_weight=w)


# ---------- e: analytic reshaping of the draft's distribution ----------

def draft_cands(S, rest="uniform"):
    """per drafted row: the draft's top-20 (T = 1) as 5 recorded candidates + 15 reconstructed ones, the drawn token's
    column, and window membership. rest = 'uniform' (the unrecorded mass spread evenly over 15 tokens) or 'packed'
    (as many tokens at the 5th probability as the mass allows: the most concentrated tail)."""
    rc, rd = S.rows_c, S.rows_d
    tp = np.nan_to_num(S.tp[rc, rd]); tid = S.tid[rc, rd]; tw = S.twin[rc, rd]
    q = S.q[rc, rd]; x = S.did[rc, rd]
    n = len(q)
    P = np.zeros((n, 20)); W = np.zeros((n, 20), bool)
    P[:, :5] = tp; W[:, :5] = tw
    inx = tid == x[:, None]
    xin = inx.any(1)
    col = np.where(xin, inx.argmax(1), 5)
    Rm = np.clip(1.0 - tp.sum(1), 0.0, None)
    # x outside the top-5: it is one of the 15 others with its own q
    P[~xin, 5] = q[~xin]
    W[~xin, 5] = S.xwin[~xin]
    restm = np.where(xin, Rm, np.clip(Rm - q, 0.0, None))
    nfree = np.where(xin, 15, 14)
    start = np.where(xin, 5, 6)
    cap = tp[:, 4]
    for j in range(5, 20):
        m = j >= start
        idx = j - start
        if rest == "uniform":
            v = restm / nfree
        else:  # packed: tokens at the 5th probability until the mass runs out
            v = np.clip(restm - idx * cap, 0.0, cap)
        P[m, j] = v[m]
    return P, W, col, q, Rm


def reshape(P, W, g):
    """apply S's chain (penalties, top_k (no-op on 20), top_p, min_p, temperature) to the draft's T = 1 candidates."""
    t, tpp, tk, mp, pr = g
    w = P * np.where(W, np.exp(-pr), 1.0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    o = np.argsort(-w, axis=1, kind="stable")
    ws = np.take_along_axis(w, o, 1)
    keep = np.ones_like(ws, bool)
    if tpp < 1:
        cum = np.cumsum(ws, 1)
        hit = cum >= tpp
        nk = np.where(hit.any(1), hit.argmax(1) + 1, 20)
        keep &= np.arange(20)[None, :] < nk[:, None]
    if mp > 0:
        keep &= ws >= mp * ws[:, :1]
    keep[:, 0] = True
    ws = np.where(keep & (ws > 0), ws, 0.0) ** (1.0 / t)
    ws = ws / ws.sum(1, keepdims=True)
    out = np.empty_like(ws)
    np.put_along_axis(out, o, ws, 1)
    return out


def analytic_feats(S, g, cand):
    P, W, col, q, Rm = cand
    ph = reshape(P, W, g)
    n = len(q)
    r = np.minimum(1.0, ph[np.arange(n), col] / np.maximum(q, 1e-12))
    s = np.minimum(P, ph).sum(1)
    return r, s


def run(setname):
    E = Ev(setname)
    S = E.S
    # window membership of the drawn token (from the target's listed ids; outside them p_S = 0 anyway)
    S.xwin = np.zeros(len(S.rows_c), bool)
    off = 0
    for p in S.parts:
        import cfvec
        R = cfvec.Rows(p["c"])
        S.xwin[off:off + len(R.x)] = (R.win & (R.ids == R.x[:, None])).any(1)
        off += len(R.x)
    res = json.load(open(f"{OUT}/approx-{setname}.json"))

    def record(lab, rates_test, storage, extra=None):
        l = E.loss(rates_test)
        st = stats(l)
        st.update(storage=storage, loss=l.tolist())
        if extra:
            st.update(extra)
        res[lab] = st
        print(f"| {lab} | {st['mean'] * 100:+.2f}% | {st['worst'] * 100:+.2f}% | {st['worst10'] * 100:+.2f}% | {storage} |", flush=True)

    print(f"\n## {setname}: parametric and analytic approximations (loss vs own tables, test)")
    print("| approximation | mean | worst | worst 10% | storage (bytes) |\n|---|---|---|---|---|")
    rq = S.q[S.rows_c, S.rows_d]; rp = S.p1[S.rows_c, S.rows_d]
    dev = S.fit_w > 0

    # ---- d1: temperature-only lookup iso(q, T): per temperature, tables fit on dev pooled over its 48 settings,
    # theta per temperature chosen on val (mean val ratio over its settings); linear interpolation in T
    Ttab = {}
    for t in C.TEMPS:
        gis = [gi for gi, g in enumerate(G) if g[0] == t]
        ys = np.concatenate([S.a_rows(gi)[dev] for gi in gis])
        ws = np.tile(S.fit_w[dev], len(gis))
        tq = iso(np.tile(rq[dev], len(gis)), ys, ws).predict(C.X24)
        tp = iso(np.tile(rp[dev], len(gis)), ys, ws).predict(C.X24)
        vr = np.zeros(len(C.THETAS))
        for gi in gis:
            vr += S.theta_curve(S.a(gi), tq, tp) / E.r_own[C.VAL][gi]
        Ttab[t] = (tq, tp, float(C.THETAS[int(np.argmax(vr))]))

    def t_interp(t, tabs, temps):
        t = min(max(t, temps[0]), temps[-1])
        j = min(int(np.searchsorted(temps, t, side="right") - 1), len(temps) - 2)
        f = (t - temps[j]) / (temps[j + 1] - temps[j])
        a, b = tabs[temps[j]], tabs[temps[j + 1]]
        return (1 - f) * a[0] + f * b[0], (1 - f) * a[1] + f * b[1], (1 - f) * a[2] + f * b[2]

    for lab, temps in [("10 temperatures", C.TEMPS), ("T{0.3,0.7,1.2}", [0.3, 0.7, 1.2]), ("T{0.3,0.5,0.7,1.0,1.2}", [0.3, 0.5, 0.7, 1.0, 1.2])]:
        rt = [E.rate(gi, *E.kT(*t_interp(g[0], Ttab, temps)))[C.TEST] for gi, g in enumerate(G)]
        record(f"d. iso(q, T): temperature-only tables (marginal over the other fields), {lab}", rt, len(temps) * TABLE_BYTES,
               dict(thetas={str(t): Ttab[t][2] for t in temps}))
    # one marginal table set for all settings (fit pooled over the whole grid)
    gis = list(range(0, nG, 4))
    ys = np.concatenate([S.a_rows(gi)[dev] for gi in gis]); ws = np.tile(S.fit_w[dev], len(gis))
    tq1 = iso(np.tile(rq[dev], len(gis)), ys, ws).predict(C.X24); tp1 = iso(np.tile(rp[dev], len(gis)), ys, ws).predict(C.X24)
    vr = np.zeros(len(C.THETAS))
    for gi in range(nG):
        vr += S.theta_curve(S.a(gi), tq1, tp1) / E.r_own[C.VAL][gi]
    th1 = float(C.THETAS[int(np.argmax(vr))])
    rt = [E.rate(gi, *E.kT(tq1, tp1, th1))[C.TEST] for gi in range(nG)]
    record(f"d. one marginal table set (fit pooled over the grid), theta {th1}", rt, TABLE_BYTES,
           dict(tq=tq1.tolist(), tp=tp1.tolist(), theta=th1))

    # ---- d2: PCA family: tables = mean + sum_j c_j(S) v_j, c_j linear in setting features; theta linear too
    M = np.array([np.r_[E.own[gi][0], E.own[gi][1]] for gi in range(nG)])
    mu = M.mean(0)
    U, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
    Fm = np.array([feats(g) for g in G])
    th_own = np.array([E.own[gi][2] for gi in range(nG)])
    cth = np.linalg.lstsq(Fm, th_own, rcond=None)[0]
    evr = s ** 2 / (s ** 2).sum()
    for npc in (1, 2):
        coef = np.linalg.lstsq(Fm, (M - mu) @ Vt[:npc].T, rcond=None)[0]
        rt = []
        for gi, g in enumerate(G):
            tab = mu + (feats(g) @ coef) @ Vt[:npc]
            tq = np.clip(np.maximum.accumulate(tab[:24]), 0, 1); tp = np.clip(np.maximum.accumulate(tab[24:]), 0, 1)
            rt.append(E.rate(gi, *E.kT(tq, tp, float(feats(g) @ cth)))[C.TEST])
        record(f"d. PCA family: mean + {npc} component(s) linear in (T, T^2, top_p, pres, T*pres, T*top_p, min_p, top_k); theta linear",
               rt, (1 + npc) * 48 * 4 + (npc + 1) * 9 * 4, dict(var_explained=evr[:3].tolist()))

    # ---- e: analytic: reference tables x (reshaped-draft ratio at S / at the reference), + correction table
    ref = S.own_gi()[0]
    tq_r, tp_r, th_r = E.own[ref]
    out_e = {}
    for rest in ("uniform", "packed"):
        cand = draft_cands(S, rest)
        r_ref, s_ref = analytic_feats(S, G[ref], cand)
        cache = {}
        for gi, g in enumerate(G):
            r, sv = analytic_feats(S, g, cand)
            post = np.interp(rq, C.X24, tq_r) * r / np.maximum(r_ref, 1e-9)
            prior = np.interp(rp, C.X24, tp_r) * sv / np.maximum(s_ref, 1e-9)
            cache[gi] = (np.clip(post, 0, 1), np.clip(prior, 0, 1))
        # correction tables: isotonic of a on each estimate, dev rows pooled over a quarter of the grid
        sub = list(range(0, nG, 4))
        xs_post = np.concatenate([cache[gi][0][dev] for gi in sub]); xs_pr = np.concatenate([cache[gi][1][dev] for gi in sub])
        ys = np.concatenate([S.a_rows(gi)[dev] for gi in sub]); ws = np.tile(S.fit_w[dev], len(sub))
        cq = iso(xs_post, ys, ws); cp = iso(xs_pr, ys, ws)

        def score_of(post, prior):
            P_ = np.zeros((S.n, C.D)); Q_ = np.zeros((S.n, C.D))
            Q_[S.rows_c, S.rows_d] = post; P_[S.rows_c, S.rows_d] = prior
            cpq = np.c_[np.ones(S.n), np.cumprod(Q_, axis=1)[:, :-1]]
            return cpq * P_
        for corr in (False, True):
            rt_fix, rt_val = [], []
            for gi, g in enumerate(G):
                post, prior = cache[gi]
                if corr:
                    post, prior = cq.predict(post), cp.predict(prior)
                sc = score_of(post, prior)
                rt_fix.append(E.rate(gi, *E.kT_score(sc, th_r))[C.TEST])
                # theta per setting on val (an upper bound for this estimator's theta rule)
                M_ = np.minimum.accumulate(np.where(S.valid, sc, np.inf), axis=1)
                best = max(((E.rate(gi, *E.kT_score(M_, th), splits=(C.VAL,))[C.VAL], th) for th in C.THETAS))
                rt_val.append(E.rate(gi, *E.kT_score(sc, best[1]))[C.TEST])
            lab = f"e. analytic ({rest} tail{', + correction tables' if corr else ''}): ref tables x reshaped-draft ratio"
            record(lab + f", theta {th_r} fixed", rt_fix, TABLE_BYTES * (2 if corr else 1))
            record(lab + ", theta per setting (val)", rt_val, TABLE_BYTES * (2 if corr else 1))
        P, W, col, q, Rm = cand
        out_e[rest] = dict(mean_unrecorded_mass=float(Rm.mean()), p90=float(np.percentile(Rm, 90)),
                           x_outside_top5=float((col == 5).mean()))
    print(f"draft mass outside the recorded top-5: mean {out_e['uniform']['mean_unrecorded_mass']:.3f}, "
          f"p90 {out_e['uniform']['p90']:.3f}; drawn token outside the top-5: {out_e['uniform']['x_outside_top5']:.3%} of rows")
    res["_e_info"] = out_e
    json.dump(res, open(f"{OUT}/approx-{setname}.json", "w"))


if __name__ == "__main__":
    for s in sys.argv[1:] or C.available():
        run(s)
