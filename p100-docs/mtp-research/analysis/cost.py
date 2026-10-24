# cost.py (brief Q2): per-cycle cost model from the traces, calibrated against BASELINES' end-to-end runs.
# Production cycle time (no trace) = draft(steps) + pad + verify(W, n_active, ctx) + catch-up + accept + gap, i.e. the
# trace's cycle_us - trace_us + gap_us (trace_us excluded), plus ~10 ms per request (first token, first gap).
#   draft(steps) = d0 + d1*steps + dctx*ctx          fit on trace-I (p-min 0.2, 1..4 steps), n4p0 (4), n8p0 (8)
#   verify(W=5, n_act, ctx) = V5 - s*(5-n_act) + k5*ctx   fit on trace-I (n_act 1..5) and n4p0
#   verify(W=9, ctx)        = V9 + k9*ctx                 fit on n8p0 (W=9 > 8: n_active is not honoured, full width)
#   verify(W in 2,3,4,7)    calibrated from the B3 end-to-end runs (no trace at those widths); W=6,8 interpolated
#   other(W) = pad+catch-up+accept+gap, linear in W (n4p0, n8p0)
# Writes cache/cost.json and prints the calibration tables.
import json, os, numpy as np
from load import load, CACHE, CATS

RUNS = "/work/bench/mtp3/runs/"


def fit():
    rows = []  # (steps, W, n_act, ctx, draft_us, verify_us, other_us, run)
    for run, nmax in [("n4p0", 4), ("n8p0", 8), ("I", 4)]:
        c, _, _ = load(run)
        full = c["n_pad"] == nmax  # excludes the end-of-request cut cycles, which verify a narrower batch + rebuild
        st = c["n_draft_entries"]
        oth = c["tm_pad_us"] + c["tm_catchup_us"] + c["tm_accept_us"] + np.nan_to_num(c["tm_gap_us"], nan=np.nanmean(c["tm_gap_us"]))
        for i in np.where(full)[0]:
            rows.append((st[i], nmax + 1, c["n_real"][i] + 1, c["pos0"][i] / 1000.0, c["tm_draft_us"][i], c["tm_verify_us"][i], oth[i], run))
    st, W, na, ctx, dr, vr, ot = [np.array([r[j] for r in rows], float) for j in range(7)]
    run = np.array([r[7] for r in rows])
    # draft
    X = np.c_[np.ones_like(st), st, ctx * st]
    d0, d1, dctx = np.linalg.lstsq(X, dr, rcond=None)[0]
    # verify W=5
    m5 = W == 5
    X = np.c_[np.ones(m5.sum()), na[m5] - 5, ctx[m5]]
    V5, s5, k5 = np.linalg.lstsq(X, vr[m5], rcond=None)[0]
    m9 = W == 9
    X = np.c_[np.ones(m9.sum()), ctx[m9]]
    V9, k9 = np.linalg.lstsq(X, vr[m9], rcond=None)[0]
    # other
    o5 = ot[m5].mean(); o9 = ot[m9].mean()
    p = dict(d0=d0, d1=d1, dctx=dctx, V5=V5, s=s5, k5=k5, V9=V9, k9=k9, o5=o5, o9=o9)
    # per n_act means at W=5 (trace-I) for the table
    p["V5_by_nact"] = {int(k): float(vr[m5 & (na == k) & (run == "I")].mean()) for k in range(1, 6) if (m5 & (na == k) & (run == "I")).any()}
    p["n_by_nact"] = {int(k): int((m5 & (na == k) & (run == "I")).sum()) for k in range(1, 6)}
    # per-request overhead: predicted_ms - sum(cycle_us + gap_us) per request
    ov = []
    for r_ in ["n4p0", "n8p0", "I"]:
        c, _, r = load(r_)
        tot = (c["tm_cycle_us"] + np.nan_to_num(c["tm_gap_us"])).sum() / 1000
        ov.append((r["predicted_ms"].sum() - tot) / len(r["task"]))
    p["t_req"] = float(np.mean(ov)) * 1000  # us
    return p


def other(p, W):
    return p["o5"] + (p["o9"] - p["o5"]) * (W - 5) / 4.0


def kctx(p, W):
    """verify context slope (us per 1k ctx): q4p kernel for W<=5 (scaled by W), vec kernel for W 6..16 (scaled by W)."""
    return p["k5"] * W / 5.0 if W <= 5 else p["k9"] * W / 9.0


def verify(p, W, n_act, ctx):
    """ms-level model in us. W = verify width (n_max+1), n_act = real drafted + 1, ctx in k tokens."""
    V = p["V"][str(W)] if isinstance(p.get("V"), dict) else None
    s = p["s"] if 2 <= W <= 8 else 0.0  # n_active honoured only for 2..8-token ubatches
    return V - s * (W - n_act) + kctx(p, W) * ctx


def cycle_cost(p, steps, W, n_act, ctx):
    return p["d0"] + p["d1"] * steps + p["dctx"] * ctx * steps + verify(p, W, n_act, ctx) + other(p, W)


def b_runs(lab, cats_excl=("longctx",), splits=("dev", "val")):
    rows = [json.loads(l) for l in open(RUNS + lab + ".jsonl")]
    rows = [x for x in rows if x["category"] not in cats_excl and x["split"] in splits]
    n = sum(x["predicted_n"] for x in rows); ms = sum(x["predicted_ms"] for x in rows)
    dn = sum(x["draft_n"] for x in rows); da = sum(x["draft_acc"] for x in rows)
    cyc = sum(x["predicted_n"] - 1 - x["draft_acc"] for x in rows)
    ctx = np.mean([x["prompt_n"] + x["predicted_n"] / 2 for x in rows]) / 1000  # rough mean context (k tokens)
    return dict(n=n, ms=ms, dn=dn, da=da, cyc=cyc, nreq=len(rows), tps=n / ms * 1000, tpc=n / cyc, dpc=dn / cyc,
                mspc=ms / cyc, ctx=ctx)


B3 = {1: "B3-n1", 2: "B3-n2", 3: "B3-n3", 6: "B3-n6", 8: "B3-n8"}


def calibrate(p, pstop):
    """V(W) for W = 2,3,4,7 from the B3 runs: measured ms/cycle minus the modelled non-verify parts.
    pstop[n] = simulated fraction of cycles that decoded one step beyond their last added token (p-min stop)."""
    V = {"5": p["V5"], "9": p["V9"]}
    for n, lab in B3.items():
        if n + 1 in (5, 9):
            continue
        b = b_runs(lab)
        steps = b["dpc"] + pstop[n]
        nact = b["dpc"] + 1
        W = n + 1
        req = p["t_req"] * b["nreq"] / b["cyc"]
        s = p["s"] if 2 <= W <= 8 else 0.0
        rest = p["d0"] + p["d1"] * steps + p["dctx"] * b["ctx"] * steps + other(p, W) + req + kctx(p, W) * b["ctx"]
        V[str(W)] = b["mspc"] * 1000 - rest + s * (W - nact)
    V["6"] = V["5"] + (V["7"] - V["5"]) / 2
    V["8"] = V["7"] + (V["9"] - V["7"]) / 2
    # W=1 for reference: B0 plain decode per token
    b0 = b_runs("B0")
    V["1_plain_decode_per_token"] = b0["mspc"] * 1000
    p["V"] = V
    return p


if __name__ == "__main__":
    import sim
    p = fit()
    # p-min stop fractions at n-max n, p-min 0.2, from the n8p0 states (dev+val, no longctx)
    c, _, _ = load("n8p0")
    m = (c["cat"] != CATS.index("longctx")) & (c["split"] < 2)
    pstop = {}
    for n in range(1, 9):
        k, steps = sim.rule_pmin(c, n, 0.2)
        pstop[n] = float((steps - k)[m].mean())
    p = calibrate(p, pstop)
    json.dump(p, open(os.path.join(CACHE, "cost.json"), "w"), indent=1)
    print("## cost model parameters (us)")
    for k in ["d0", "d1", "dctx", "V5", "s", "k5", "V9", "k9", "o5", "o9", "t_req"]:
        print(f"  {k:6s} {p[k]:9.1f}")
    print("  verify by n_act at W=5 (trace-I means):", {k: round(v) for k, v in p["V5_by_nact"].items()}, "n:", p["n_by_nact"])
    print("  V(W) full-width verify at short context:", {k: round(v) for k, v in p["V"].items()})
    print("  p-min 0.2 extra-step fraction by n-max:", {k: round(v, 3) for k, v in pstop.items()})
