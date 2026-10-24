# features.py: per (cycle, depth) features available at draft-decision time, plus per-cycle context features.
# Draft side (step d, after its decode; free on the host): q of the sampled token, top-1/top-2 p, margin, top-5 mass,
# entropy H over the top-K, sampled==top-1, depth, cumulative log top-1 p and log q over steps 1..d, min top-1 p so far.
# Context: position in reply, context length, cycle index.
# Rolling (previous cycles of the same request, realised): first-token acceptance of the previous cycle, its EMA,
# previous accepted fraction. Difficulty (teacher side of the previous cycle, known before the next draft): teacher
# entropy / top-1 p at row 0 and at the row the last emitted token was drawn from.
# Proxy task type (from text already emitted, the first token included): inside a ``` fence, reply opened with { or [,
# tool-call markup seen, last emitted token ends a line.
# Oracle: category one-hot.
import json, numpy as np
from load import load, detok, CATS, DMAX, R

DRAFT = ["q", "p1", "p2", "margin", "p5", "H", "istop", "logq", "logp1", "depth", "cum_logp1", "cum_logq", "min_p1_prev", "meanH"]
# EXACT-SAFE set for the decision "add drafted token d?": only the step-d DISTRIBUTION (top-K p's, H) plus the earlier
# drafted tokens' q. A rule that looks at the sampled token's own q (or whether it is the top-1) changes the output
# distribution of the p/q rule (see ANALYSIS.md), so q_d / istop_d / cum_logq through d are excluded here.
PRIOR = ["p1", "p2", "margin", "p5", "H", "logp1", "depth", "cum_logp1", "min_p1_prev", "meanH",
         "q_prev", "istop_prev", "logq_prev", "cum_logq_prev"]
CTX = ["log_posrel", "log_pos0", "first_cycle"]
ROLL = ["prev_acc1", "ema_acc1", "prev_frac"]
DIFF = ["prev_tH0", "prev_tp1_0", "prev_rH", "prev_rp1"]
PROXY = ["in_fence", "json_mode", "tool_mode", "after_newline"]
CAT = [f"cat_{c}" for c in CATS]


def first_token_texts(run, c, emitted):
    fn = "I-on.jsonl" if run == "I" else f"T-{run}.jsonl"
    T = {json.loads(l)["task"]: json.loads(l) for l in open(f"{R}/runs/{fn}")}
    out = {}
    for task, t in T.items():
        idx = np.where(c["task"] == task)[0]
        if len(idx) == 0:
            out[task] = ""; continue
        s = detok(emitted[idx[0]])
        if "<function=" in s[:40] or "<tool_call>" in s[:40]:
            out[task] = "<tool_call>"; continue
        probe = s[:24].replace("<|im_end|>", "")
        j = t["text"].find(probe) if probe else -1
        out[task] = t["text"][:j] if j > 0 else ""
    return out


def build(run):
    c, emitted, req = load(run)
    n = len(c["task"])
    F = {}
    q = c["d_q"]; p1 = c["d_p1"]; p2 = c["d_p2"]
    eps = 1e-6
    F["q"] = q; F["p1"] = p1; F["p2"] = p2; F["margin"] = p1 - p2; F["p5"] = c["d_p5"]; F["H"] = c["d_H"]
    F["istop"] = c["d_istop"]; F["logq"] = np.log(q + eps); F["logp1"] = np.log(p1 + eps)
    F["depth"] = np.tile(np.arange(1, DMAX + 1, dtype=float), (n, 1))
    F["cum_logp1"] = np.nancumsum(np.log(p1 + eps), axis=1)
    F["cum_logq"] = np.nancumsum(np.log(q + eps), axis=1)
    mp = np.minimum.accumulate(np.nan_to_num(p1, nan=1.0), axis=1)
    F["min_p1_prev"] = np.c_[np.ones(n), mp[:, :-1]]
    F["meanH"] = np.nancumsum(c["d_H"], axis=1) / F["depth"]
    sh = lambda x, fill: np.c_[np.full(n, fill), x[:, :-1]]
    F["q_prev"] = sh(np.nan_to_num(q, nan=1.0), 1.0); F["istop_prev"] = sh(np.nan_to_num(c["d_istop"], nan=1.0), 1.0)
    F["logq_prev"] = np.log(F["q_prev"] + eps); F["cum_logq_prev"] = sh(F["cum_logq"], 0.0)
    # per-cycle features
    G = {k: np.zeros(n) for k in CTX + ROLL + DIFF + PROXY + CAT}
    G["log_posrel"] = np.log1p(c["posrel"]); G["log_pos0"] = np.log1p(c["pos0"])
    G["first_cycle"] = (c["cyc"] == 0).astype(float)
    first = first_token_texts(run, c, emitted)
    acc1_mean = (c["n_acc"] >= 1).mean()
    for task in np.unique(c["task"]):
        idx = np.where(c["task"] == task)[0]
        text = first.get(task, "")
        lead = text.lstrip()
        ema = acc1_mean
        prev = None
        for i in idx:
            stripped = text.lstrip()
            G["in_fence"][i] = float(text.count("```") % 2 == 1)
            G["json_mode"][i] = float(stripped[:1] in ("{", "[") or stripped.startswith("```json"))
            G["tool_mode"][i] = float("<tool_call>" in text or "<function=" in text)
            G["after_newline"][i] = float(text.endswith("\n"))
            if prev is None:
                G["prev_acc1"][i] = acc1_mean; G["ema_acc1"][i] = acc1_mean; G["prev_frac"][i] = 0.5
                G["prev_tH0"][i] = np.nanmean(c["t_H"][:, 0]); G["prev_tp1_0"][i] = np.nanmean(c["t_p1"][:, 0])
                G["prev_rH"][i] = np.nanmean(c["r_H"]); G["prev_rp1"][i] = np.nanmean(c["r_p1"])
            else:
                acc1 = float(c["n_acc"][prev] >= 1)
                ema = 0.7 * ema + 0.3 * acc1
                G["prev_acc1"][i] = acc1; G["ema_acc1"][i] = ema
                G["prev_frac"][i] = c["n_acc"][prev] / max(c["n_real"][prev], 1)
                G["prev_tH0"][i] = np.nan_to_num(c["t_H"][prev, 0], nan=0.0)
                G["prev_tp1_0"][i] = np.nan_to_num(c["t_p1"][prev, 0], nan=1.0)
                G["prev_rH"][i] = np.nan_to_num(c["r_H"][prev], nan=0.0)
                G["prev_rp1"][i] = np.nan_to_num(c["r_p1"][prev], nan=1.0)
            text = text + detok(emitted[i])
            prev = i
    for k, cat in enumerate(CATS):
        G[f"cat_{cat}"] = (c["cat"] == k).astype(float)
    return c, F, G


def matrix(c, F, G, names, rows_c, rows_d):
    """Design matrix for the (cycle, depth) pairs given."""
    cols = []
    for nm in names:
        if nm in F:
            cols.append(F[nm][rows_c, rows_d])
        else:
            cols.append(G[nm][rows_c])
    return np.nan_to_num(np.column_stack(cols).astype(float))


def all_pairs(c, Dmax):
    nr = np.minimum(c["n_real"], Dmax)
    rc = np.repeat(np.arange(len(nr)), nr)
    rd = np.concatenate([np.arange(k) for k in nr])
    return rc, rd
