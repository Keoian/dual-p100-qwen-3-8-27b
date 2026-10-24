# cfvalidate.py <name>: (1) cfvec at the trace's own setting vs the recorded p_draft on every counterfactual row;
# (2) cfvec vs counterfactual.row_probs (the reference library, record by record) on random rows x random grid settings.
import json, sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import cfload, cfvec, counterfactual as CF

GRID = dict(temp=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], top_p=[0.8, 0.9, 0.95, 1.0], top_k=[20, 40],
            min_p=[0.0, 0.05], pres=[0.0, 0.75, 1.5])


def main(name, nrec=400, seed=1):
    c, meta = cfload.load(name)
    R = cfvec.Rows(c)
    s0 = meta["sampling"][0]
    own = dict(temp=s0[0], top_k=s0[1], top_p=s0[2], min_p=s0[3], pres=s0[4])
    a, px, ex = R.accept(**own)
    rec = c["t_p"][R.rc, R.rd]
    err = np.abs(px - rec)
    print(f"{name}: own setting {own}: {len(px)} rows, |p - p_draft| max {err.max():.3g}, p99 {np.percentile(err, 99):.3g}, "
          f"> 1e-4: {(err > 1e-4).mean():.4%}; inexact rows {int((~ex).sum())}")
    # reference library on random records
    rng = np.random.default_rng(seed)
    key = {(int(t), int(p)): i for i, (t, p) in enumerate(zip(c["task"], c["pos0"]))}
    row_of = {(int(cc), int(dd)): j for j, (cc, dd) in enumerate(zip(R.rc, R.rd))}
    lines = open(f"{cfload.RUNS}/trace-{name}.jsonl").readlines()
    pick = rng.choice(len(lines), min(nrec, len(lines)), replace=False)
    errs, n, n_inex_ref, n_inex_mine, mism = [], 0, 0, 0, 0
    settings = []
    for li in pick:
        r = json.loads(lines[li])
        ci = key.get((r["task"], r["pos0"]))
        if ci is None or not c["cf_ok"][ci]:
            continue
        for d in range(min(r["n_real"], cfload.D)):
            j = row_of.get((ci, d))
            if j is None:
                continue
            S = {k: float(rng.choice(v)) for k, v in GRID.items()}
            S["top_k"] = int(S["top_k"])
            setting = dict(temp=S["temp"], top_k=S["top_k"], top_p=S["top_p"], min_p=S["min_p"],
                           penalty_present=S["pres"], penalty_repeat=1.0, penalty_freq=0.0, penalty_last_n=64)
            info = {}
            pref = CF.row_probs(r, d, setting, info).get(r["draft"][d]["id"], 0.0)
            sub = cfvec.Rows.__new__(cfvec.Rows)
            for f in ("ids", "lg", "win", "x", "q", "tail_max"):
                setattr(sub, f, getattr(R, f)[j:j + 1])
            sub._sorted = {}
            pm, exm = sub.probs(**S)
            errs.append(abs(pm[0] - pref)); n += 1
            n_inex_ref += not info["exact"]; n_inex_mine += not exm[0]
            mism += (info["exact"] != bool(exm[0]))
    errs = np.array(errs)
    print(f"{name}: vs counterfactual.row_probs on {n} random (row, grid setting) pairs from {len(pick)} records: "
          f"max |dp| {errs.max():.3g}, mean {errs.mean():.3g}; inexact ref {n_inex_ref}, mine {n_inex_mine}, flag mismatches {mism}")
    # exactness over the whole grid (top_k filled inside the list)
    worst = 0
    for pres in GRID["pres"]:
        _, _, nab = R.sorted_for(pres)
        for k in GRID["top_k"]:
            worst = max(worst, int((nab < k).sum()))
    print(f"{name}: rows with top_k not filled inside the top-{cfload.K} list, worst grid setting: {worst} of {len(R.x)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cfchat")
