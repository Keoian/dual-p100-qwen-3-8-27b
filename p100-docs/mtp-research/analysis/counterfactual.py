# counterfactual.py: the target's sampling distribution at a verify row under ANY target sampling setting, recomputed
# offline from a LLAMA_SPEC_TRACE_TOPK trace (per teacher row "raw" top-K logits and "lse_all", per cycle "pen_hist"),
# so the stop rule's acceptance tables can be built for settings the trace was not run at. The draft samples its tokens
# the same way whatever the request's sampling (LLAMA_SPEC_SAMPLE_TEMP over its top-K, recorded as q), so the p/q
# acceptance probability of a recorded drafted token x under setting S is min(1, p_S(x)/q(x)).
#
# The chain is this fork's, step by step, in float32 where the C++ is float32 (each step names the function it mirrors):
#   common/sampling.cpp common_sampler_init builds [logit_bias] -> params.samplers in order -> dist, and
#   common_sampler_sample applies the reasoning budget first (forcing: a point mass on the forced token, row_probs).
#   Default order, common/common.h common_params_sampling::samplers: penalties, dry, top_n_sigma, top_k, typ_p, top_p,
#   min_p, xtc, temperature. dry, top_n_sigma, typ_p, xtc, dynatemp, mirostat and logit_bias must be off (their
#   defaults, and the server's for these requests; the trace records none of them; ignore_eos is a logit bias).
#   Kmic's top-k prefilter (common_sampler::set_logits, prefilter_k) hands the chain the top_k + penalty_last_n largest
#   raw logits, sorted, instead of the whole vocabulary. It is exact (the penalties lower at most penalty_last_n
#   logits, so the top_k after them are among those), so it is not modelled: the steps run as on the whole vocabulary.
# Only the K largest raw logits are known. A result that depends on a token outside them sets info["exact"] = False
# (with the reasons in info["why"]) and is then an estimate: top_k not filled inside the list after penalties, top_p's
# nucleus or min_p's band reaching past it, a penalty window longer than the recorded one, a grammar. With top_k <= 0
# the softmaxes run over the whole vocabulary: the tokens outside the list enter through lse_all, which is exact (up to
# float rounding) at T = 1. In the default order top_p and min_p run before temperature, so at T = 1; dist normalizing
# the whole vocabulary at T != 1 (top_k <= 0, top_p >= 1, min_p <= 0) is an estimate (the tail is left out).
# Ties: C++'s partial sorts leave equal logits in unspecified order; here equal logits sort by ascending id.
# Usage: python3 counterfactual.py selfcheck <trace.jsonl>   |   python3 counterfactual.py test
import json, os, sys, tempfile
from collections import Counter
import numpy as np

F32 = np.float32
N_VOCAB = 248320  # Qwen3.8; a raw list this long has no tokens outside it
ORDER = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]
KEYS = ["temp", "top_k", "top_p", "min_p", "penalty_present", "penalty_repeat", "penalty_freq", "penalty_last_n"]


def _expf(x):
    """expf on a float32 array: exp in float64, rounded to float32 (glibc's expf is correctly rounded but for rare
    cases)."""
    return np.exp(np.asarray(x, np.float64)).astype(F32)


def _logf(p):
    """logf on a float32 scalar, the same way."""
    return F32(np.log(np.float64(p)))


def _check_off(s):
    """The samplers the trace cannot replay must be off (llama_sampler_init_* return an empty sampler then)."""
    bad = []
    if s.get("dry_multiplier", 0.0) != 0.0 and s.get("dry_base", 1.75) >= 1.0 and s.get("dry_penalty_last_n", 64) != 0:
        bad.append("dry")
    if s.get("top_n_sigma", -1.0) > 0.0:
        bad.append("top_n_sigma")
    if s.get("typ_p", s.get("typical_p", 1.0)) < 1.0:
        bad.append("typ_p")
    if s.get("xtc_probability", 0.0) > 0.0 and s.get("xtc_threshold", 0.1) <= 0.5:
        bad.append("xtc")
    if s.get("dynatemp_range", 0.0) > 0.0:
        bad.append("dynatemp")
    if s.get("mirostat", 0) != 0:
        bad.append("mirostat")
    if s.get("logit_bias"):
        bad.append("logit_bias")
    for name in s.get("samplers", ORDER):
        if name not in ORDER:
            bad.append(name)
    if bad:
        raise ValueError(f"not replayable: {', '.join(bad)}")


def penalties_active(s):
    """llama_sampler_penalties::is_disabled, after llama_sampler_init_penalties clamps penalty_last_n at 0."""
    return max(int(s["penalty_last_n"]), 0) > 0 and not (
        F32(s["penalty_repeat"]) == 1 and F32(s["penalty_freq"]) == 0 and F32(s["penalty_present"]) == 0)


def target_probs(raw, lse_all, hist, setting, info=None, n_vocab=N_VOCAB):
    """{id: p} after the sampler chain of `setting` (the trace's "sampling" keys; optional "min_keep", default 0, and
    "samplers", default ORDER) on the raw top-K logits `raw` ([[id, logit], ...]) of one verify row. lse_all: the
    row's log-sum-exp over the whole vocabulary (None: unknown). hist: the tokens the penalty sampler has seen, oldest
    first (the last penalty_last_n are used). info (optional dict) receives "exact" and "why". While the chain keeps
    the tokens outside the list (top_k <= 0 and no top_p or min_p cut inside it), only the listed ones come back,
    normalized over the whole vocabulary."""
    s = setting
    for k in KEYS:
        if k not in s:
            raise KeyError(k)
    _check_off(s)
    min_keep = int(s.get("min_keep", 0))
    if info is None:
        info = {}
    info["exact"], info["why"] = True, []

    def inexact(why):
        info["exact"] = False
        info["why"].append(why)

    ids = np.array([r[0] for r in raw], np.int64)
    lg = np.array([r[1] for r in raw], F32)
    m0 = float(lg.max())
    st = dict(ids=ids, lg=lg,
              tail=len(raw) < n_vocab,  # tokens outside the list are still candidates
              tail_max=F32(lg.min()),   # every one of them has a logit <= this
              tail_w=np.nan,            # their total weight sum e^(l - m0) at T = 1
              tail_w_ok=True,           # no penalty touched them
              scale=1.0)                # product of the temperatures applied so far
    if st["tail"] and lse_all is not None:
        st["tail_w"] = max(0.0, float(np.exp(lse_all - m0) - np.exp(lg.astype(np.float64) - m0).sum()))

    def resort():
        o = np.lexsort((st["ids"], -st["lg"]))
        st["ids"], st["lg"] = st["ids"][o], st["lg"][o]

    def keep(n):
        st["ids"], st["lg"] = st["ids"][:n], st["lg"][:n]

    def tail_weight(mx):
        """sum over the tail of e^(l - mx) at the current scale, if known."""
        if st["scale"] != 1.0 or not np.isfinite(st["tail_w"]):
            return None
        if not st["tail_w_ok"]:
            inexact("penalized tokens outside the list")
        return st["tail_w"]*np.exp(m0 - float(mx))

    resort()

    # llama_sampler_penalties_apply (src/llama-sampler.cpp): for each distinct token of the window, count c times:
    # logit = logit*repeat if logit <= 0 else logit/repeat; logit -= c*freq + (c > 0)*present (float32)
    def penalties():
        if not penalties_active(s):
            return
        rep, freq, pres = F32(s["penalty_repeat"]), F32(s["penalty_freq"]), F32(s["penalty_present"])
        n = int(s["penalty_last_n"])
        pos = {int(t): j for j, t in enumerate(st["ids"])}
        lg = st["lg"]
        for t, c in Counter(list(hist)[-n:]).items():
            j = pos.get(int(t))
            if j is None:
                st["tail_w_ok"] = False
                continue
            l = lg[j]
            l = l*rep if l <= 0 else l/rep
            lg[j] = l - (F32(c)*freq + F32(1.0)*pres)
        if not (rep >= 1 and freq >= 0 and pres >= 0) and st["tail"]:
            inexact("penalties that can raise a logit")  # the tail's bound no longer holds
        resort()

    # llama_sampler_top_k_impl: the k largest (k <= 0: llama_sampler_init_top_k returns an empty sampler)
    def top_k():
        k = int(s["top_k"])
        if k <= 0:
            return
        if st["tail"]:
            if int((st["lg"] > st["tail_max"]).sum()) < k:
                inexact("top_k not filled inside the list")
            st["tail"] = False
        keep(min(k, len(st["lg"])))

    # llama_sampler_top_p_apply: llama_sampler_softmax_impl (float32: e = expf(l - max), sum in order, p = e/sum),
    # then keep up to the first i with cumsum(p)[i] >= top_p and i + 1 >= min_keep (all if never)
    def top_p():
        p = F32(s["top_p"])
        if p >= 1:
            return
        lg = st["lg"]
        if st["tail"]:
            mx = lg[0]
            if st["tail_max"] > mx:
                inexact("the max may be outside the list")
            w = tail_weight(mx)
            if w is None:
                inexact("top_p over the whole vocabulary after temperature")
                w = 0.0
            e = np.exp(lg.astype(np.float64) - float(mx))
            cum = np.cumsum(e/(e.sum() + w))
            hit = (cum >= p) & (np.arange(1, len(lg) + 1) >= min_keep)
            if not hit.any():
                inexact("top_p nucleus past the list")
                return
            keep(int(np.argmax(hit)) + 1)
            st["tail"] = False
            return
        e = _expf(lg - lg[0])
        pp = e/np.cumsum(e, dtype=F32)[-1]
        cum = np.cumsum(pp, dtype=F32)
        hit = (cum >= p) & (np.arange(1, len(lg) + 1) >= min_keep)
        keep(int(np.argmax(hit)) + 1 if hit.any() else len(lg))

    # llama_sampler_min_p_apply: keep logit >= max + logf(min_p) (float32), at least min_keep, the first always
    def min_p():
        p = F32(s["min_p"])
        if p <= 0:
            return
        lg = st["lg"]
        thr = F32(lg[0] + _logf(p))
        if st["tail"]:
            if st["tail_max"] >= thr or st["tail_max"] > lg[0]:
                inexact("min_p band past the list")
            else:
                st["tail"] = False
        i = 1
        while i < len(lg) and not (lg[i] < thr and i >= min_keep):
            i += 1
        keep(i)

    # llama_sampler_temp_ext_apply with dynatemp_range 0 -> llama_sampler_temp_impl: logit /= temp (float32); temp <= 0
    # keeps the first max and sets the rest to -inf. temp == 1: llama_sampler_init_temp_ext returns an empty sampler
    def temperature():
        t = F32(s["temp"])
        if t == 1:
            return
        lg = st["lg"]
        if t <= 0:
            if st["tail"] and st["tail_max"] >= lg[0]:
                inexact("greedy max past the list")
            lg[1:] = -np.inf
            st["tail"] = False
            return
        st["lg"] = lg/t
        st["tail_max"] = F32(st["tail_max"]/t)
        st["scale"] *= float(t)

    steps = dict(penalties=penalties, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
    for name in s.get("samplers", ORDER):
        if name in steps:
            steps[name]()

    # llama_sampler_dist_apply: e = expf(l - max) (float32), sum in double in order, p = float(e/sum); one candidate: 1
    ids, lg = st["ids"], st["lg"]
    if st["tail"]:
        w = tail_weight(lg[0])
        if w is None:
            inexact("dist over the whole vocabulary at T != 1")
            w = 0.0
        e = np.exp(lg.astype(np.float64) - float(lg[0]))
        pp = e/(e.sum() + w)
    elif len(lg) == 1:
        pp = np.ones(1)
    else:
        e = _expf(lg - lg[0]).astype(np.float64)
        pp = (e/np.cumsum(e)[-1]).astype(F32)
    return dict(zip(ids.tolist(), [float(x) for x in pp]))


def grammar_cycle(rec):
    """the cycle had a grammar sampler (grmr, LLAMA_SPEC_TRACE_TOPK traces); older traces only say grammar or budget."""
    return rec["grmr"] if "grmr" in rec else rec["grammar"]


def row_probs(rec, i, setting=None, info=None, n_vocab=N_VOCAB):
    """target_probs at teacher row i of a trace record, under `setting` (None: the record's own "sampling"). A row
    where the reasoning budget was forcing is a point mass on the forced token, its top1, whatever the setting. The
    penalty window is pen_hist (the request's last penalty_last_n context tokens before the drafted ones) followed by
    the drafted tokens 0..i-1, which the teacher-forced clone accepted before row i."""
    if info is None:
        info = {}
    t = rec["teacher"][i]
    s = rec["sampling"] if setting is None else setting
    if t.get("forced"):
        info["exact"], info["why"] = True, []
        return {t["top1"][0]: 1.0}
    if "raw" not in t:
        raise ValueError("no raw logits (a trace without LLAMA_SPEC_TRACE_TOPK)")
    pen = rec.get("pen_hist")
    hist = list(pen or []) + [d["id"] for d in rec["draft"][:i]]
    p = target_probs(t["raw"], t.get("lse_all"), hist, s, info, n_vocab)
    if penalties_active(s):
        n_rec = int(rec["sampling"]["penalty_last_n"])
        if pen is None:
            info["exact"] = False
            info["why"].append("no penalty window recorded")
        elif int(s["penalty_last_n"]) > n_rec + i and len(pen) >= n_rec:
            info["exact"] = False
            info["why"].append("penalty window longer than the recorded one")
    if grammar_cycle(rec):
        info["exact"] = False
        info["why"].append("grammar")
    return p


def selfcheck(path, verbose=True):
    """recompute p_draft at each row's own sampling for every non-grammar teacher row with raw logits, and compare."""
    err, where, n_forced, n_inexact = [], [], 0, 0
    skip = Counter()
    for line in open(path):
        rec = json.loads(line)
        if grammar_cycle(rec):
            skip["grammar cycle"] += sum("p_draft" in t for t in rec["teacher"])
            continue
        for i, t in enumerate(rec["teacher"]):
            if "p_draft" not in t:
                continue  # the bonus row
            if i >= len(rec["draft"]):
                skip["no draft entry"] += 1
                continue
            if "raw" not in t and not t.get("forced"):
                skip["no raw logits"] += 1
                continue
            info = {}
            p = row_probs(rec, i, None, info).get(rec["draft"][i]["id"], 0.0)
            n_forced += bool(t.get("forced"))
            n_inexact += not info["exact"]
            err.append(abs(p - t["p_draft"]))
            where.append((rec["task"], rec["pos0"], i, p, t["p_draft"], ",".join(info["why"])))
    err = np.array(err)
    out = dict(rows=len(err), forced=n_forced, inexact=n_inexact, skipped=dict(skip))
    if len(err):
        out.update(max=float(err.max()), p99=float(np.percentile(err, 99)), frac_1e4=float((err > 1e-4).mean()))
    if verbose:
        print(f"{path}: {out['rows']} rows checked ({n_forced} forced, {n_inexact} not exact from the list), "
              f"skipped {dict(skip) or 'none'}")
        if len(err):
            print(f"|p - p_draft|: max {out['max']:.3g}, p99 {out['p99']:.3g}, rows above 1e-4 {out['frac_1e4']:.4%}")
            for j in np.argsort(-err)[:5]:
                if err[j] > 1e-4:
                    print("  task %d pos0 %d row %d: recomputed %.6g, recorded %.6g %s" % where[j])
    return out


# ---- tests (synthetic) ----

def _setting(**kw):
    s = dict(temp=1.0, top_k=0, top_p=1.0, min_p=0.0, penalty_present=0.0, penalty_repeat=1.0, penalty_freq=0.0,
             penalty_last_n=64)
    s.update(kw)
    return s


def _raw(logits, ids=None):
    ids = range(len(logits)) if ids is None else ids
    r = sorted(zip(ids, logits), key=lambda x: (-x[1], x[0]))
    return [[int(i), float(F32(l))] for i, l in r]


def _lse(logits):
    l = np.asarray(logits, np.float64)
    return float(l.max() + np.log(np.exp(l - l.max()).sum()))


def _close(a, b, tol=1e-6):
    assert set(a) == set(b), (a, b)
    for k in a:
        assert abs(a[k] - b[k]) <= tol, (k, a[k], b[k])


def _softmax(d):
    m = max(d.values())
    z = sum(np.exp(v - m) for v in d.values())
    return {k: float(np.exp(v - m)/z) for k, v in d.items()}


def test():
    # penalties: repeat divides a positive logit and multiplies a negative one, then frequency*count + presence
    lg = [2.0, 1.0, -1.0]
    s = _setting(penalty_repeat=2.0, penalty_freq=0.5, penalty_present=1.0)
    p = target_probs(_raw(lg), _lse(lg), [0, 0, 2], s, n_vocab=3)
    _close(p, _softmax({0: 2.0/2 - (2*0.5 + 1), 1: 1.0, 2: -1.0*2 - (0.5 + 1)}))
    # only the last penalty_last_n tokens count
    p = target_probs(_raw(lg), _lse(lg), [0, 0, 2], _setting(penalty_present=1.0, penalty_last_n=1), n_vocab=3)
    _close(p, _softmax({0: 2.0, 1: 1.0, 2: -2.0}))
    # penalty_last_n 0 disables them (llama_sampler_init_penalties)
    p = target_probs(_raw(lg), _lse(lg), [0], _setting(penalty_present=5.0, penalty_last_n=0), n_vocab=3)
    _close(p, _softmax({0: 2.0, 1: 1.0, 2: -1.0}))

    # penalties lowering a logit enough to reorder the top-k: 10 drops from first to third, out of the top 2
    ids, lg = [10, 11, 12, 13], [5.0, 4.9, 4.0, 1.0]
    s = _setting(top_k=2, penalty_present=1.5)
    info = {}
    p = target_probs(_raw(lg, ids), None, [10], s, info)
    assert info["exact"] and set(p) == {11, 12}, (p, info)
    _close(p, _softmax({11: 4.9, 12: 4.0}))
    assert target_probs(_raw(lg, ids), None, [], s).keys() == {10, 11}
    # the same with only the 3 largest recorded: 12 sits at the list's floor (4.0), so a token outside the list could
    # tie it for the second place
    info = {}
    target_probs(_raw(lg[:3], ids[:3]), None, [10], s, info)
    assert not info["exact"], info

    # top_k
    lg = [3.0, 2.0, 1.0, 0.0]
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=2), n_vocab=4)
    _close(p, _softmax({0: 3.0, 1: 2.0}))

    # top_p: p = .5/.3/.2 -> 0.75 keeps 2, 0.85 keeps 3, min_keep 3 keeps 3 at 0.4
    lg = np.log([0.5, 0.3, 0.2]).tolist()
    for tp, mk, n in [(0.75, 0, 2), (0.85, 0, 3), (0.4, 0, 1), (0.4, 3, 3)]:
        p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=3, top_p=tp, min_keep=mk), n_vocab=3)
        assert len(p) == n, (tp, mk, p)
        _close(p, _softmax({i: lg[i] for i in range(n)}))

    # min_p: keep logit >= max + log(min_p); min_keep wins over it
    lg = [0.0, -1.0, -2.0, -3.0]
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=4, min_p=0.1), n_vocab=4)
    _close(p, _softmax({0: 0.0, 1: -1.0, 2: -2.0}))
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=4, min_p=0.5, min_keep=3), n_vocab=4)
    assert set(p) == {0, 1, 2}, p
    # min_p runs before temperature (default order): T does not move its band
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=4, min_p=0.1, temp=0.25), n_vocab=4)
    _close(p, _softmax({0: 0.0, 1: -4.0, 2: -8.0}))

    # temperature: divides the logits; temp <= 0 is a point mass on the max
    lg = [1.0, 0.5, -0.5]
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=3, temp=0.5), n_vocab=3)
    _close(p, _softmax({0: 2.0, 1: 1.0, 2: -1.0}))
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=3, temp=0.0), n_vocab=3)
    assert p[0] == 1.0 and p[1] == 0.0 and p[2] == 0.0, p
    # a custom order: temperature before top_p changes the nucleus
    lg = np.log([0.5, 0.3, 0.2]).tolist()
    s = _setting(top_k=3, top_p=0.75, temp=0.5, samplers=["top_k", "temperature", "top_p"])
    assert len(target_probs(_raw(lg), _lse(lg), [], s, n_vocab=3)) == 2
    assert len(target_probs(_raw(lg), _lse(lg), [], dict(s, temp=3.0), n_vocab=3)) == 3

    # dist: the result sums to 1 and is the softmax of the survivors
    rng = np.random.default_rng(0)
    lg = rng.normal(0, 3, 50).tolist()
    p = target_probs(_raw(lg), _lse(lg), [], _setting(top_k=20, top_p=0.9, temp=0.7), n_vocab=50)
    assert abs(sum(p.values()) - 1) < 1e-6

    # top_k <= 0: the whole-vocabulary softmax takes the tokens outside the list from lse_all (exact at T = 1)
    lg = rng.normal(0, 2, 5000)
    full = target_probs(_raw(lg), _lse(lg), [], _setting(), n_vocab=5000)
    info = {}
    part = target_probs(_raw(lg)[:100], _lse(lg), [], _setting(), info, n_vocab=5000)
    assert info["exact"], info
    for k, v in part.items():
        assert abs(v - full[k]) < 1e-6, (k, v, full[k])
    info = {}
    target_probs(_raw(lg)[:100], _lse(lg), [], _setting(temp=0.5), info, n_vocab=5000)
    assert not info["exact"], info
    # top_p over the whole vocabulary with a nucleus inside the list: exact, and equal to the full computation
    lg = rng.normal(0, 1, 5000)
    lg[:30] += 12
    for tp in (0.5, 0.9):
        full = target_probs(_raw(lg), _lse(lg), [], _setting(top_p=tp, temp=0.7), n_vocab=5000)
        info = {}
        part = target_probs(_raw(lg)[:200], _lse(lg), [], _setting(top_p=tp, temp=0.7), info, n_vocab=5000)
        assert info["exact"] and part.keys() == full.keys(), (info, len(part), len(full))
        _close(part, full, 1e-5)

    # random settings: from the top 128, whenever the result is flagged exact it equals the whole-vocabulary one
    n_exact = 0
    for it in range(300):
        V = 3000
        lg = rng.normal(0, 3, V)
        lg[rng.choice(V, 40, replace=False)] += rng.uniform(5, 15, 40)
        top = np.argsort(-lg)[:60]
        hist = rng.choice(top, int(rng.integers(0, 100))).tolist() + rng.choice(V, 10).tolist()
        s = _setting(temp=float(rng.choice([0.0, 0.6, 0.7, 1.0, 1.3])), top_k=int(rng.choice([0, 1, 5, 20, 40])),
                     top_p=float(rng.choice([0.8, 0.95, 1.0])), min_p=float(rng.choice([0.0, 0.05, 0.2])),
                     penalty_present=float(rng.choice([0.0, 1.5])), penalty_repeat=float(rng.choice([1.0, 1.1])),
                     penalty_freq=float(rng.choice([0.0, 0.3])), penalty_last_n=int(rng.choice([0, 16, 64])),
                     min_keep=int(rng.choice([0, 0, 3])))
        full = target_probs(_raw(lg), _lse(lg), hist, s, n_vocab=V)
        info = {}
        part = target_probs(_raw(lg)[:128], _lse(lg), hist, s, info, n_vocab=V)
        if info["exact"]:
            n_exact += 1
            fz = {k: v for k, v in full.items() if v > 0}
            pz = {k: v for k, v in part.items() if v > 0}
            if len(pz) < len(fz):  # no cut inside the list: only the listed tokens come back
                assert len(part) == 128, (s, len(part))
                fz = {k: v for k, v in fz.items() if k in pz}
            _close(pz, fz, 2e-6)
    assert n_exact > 150, n_exact

    # the parser and selfcheck on synthetic trace lines in the TRACE_SCHEMA format with the new fields: a cycle with
    # penalties (row 1 sees drafted token 0 in its window), a forced row, a grammar cycle (skipped), a bonus row
    V = 4000
    S = dict(temp=0.7, top_k=20, top_p=0.8, min_p=0, penalty_present=1.5, penalty_repeat=1, penalty_freq=0,
             penalty_last_n=64)
    recs = []
    for c in range(3):
        rows = [rng.normal(0, 3, V) for _ in range(3)]
        for r in rows:
            r[:12] += 9
        pen = rng.integers(0, 12, 64).tolist()
        draft = [{"id": int(np.argmax(rows[0])), "q": 0.9, "top": [], "H": 0.3},
                 {"id": int(np.argsort(-rows[1])[1]), "q": 0.4, "top": [], "H": 1.1}]
        teacher = []
        for i, r in enumerate(rows):
            full = target_probs(_raw(r), _lse(r), pen + [d["id"] for d in draft[:i]], S, n_vocab=V)
            t = {"top1": list(max(full.items(), key=lambda x: x[1])), "H": 0.5, "logit_max": float(r.max()),
                 "argmax": int(np.argmax(r)), "raw": _raw(r)[:100], "lse_all": _lse(r)}
            if i < 2:
                t["p_draft"] = float(F32(full.get(draft[i]["id"], 0.0)))
                t["logit_draft"] = float(r[draft[i]["id"]])
            teacher.append(t)
        if c == 1:  # the reasoning budget forcing at row 1: a point mass on the forced token, the row's top1
            teacher[1] = {"p_draft": 0.0, "top1": [77, 1.0], "H": 0.0, "logit_max": 1.0, "argmax": 5,
                          "raw": [[5, 1.0]], "lse_all": 1.0, "forced": True}
        recs.append({"task": 7, "slot": 0, "t_us": 1000 + c, "pos0": 100 + 3*c, "n_ctx_used": 100 + 3*c,
                     "n_draft_padded": 2, "n_real": 2, "sampling": S, "draft_temp": 1, "rule": "dist-pertoken",
                     "grammar": c == 2, "grmr": c == 2, "pen_hist": pen, "draft": draft, "teacher": teacher,
                     "n_accepted": 1, "emitted": [draft[0]["id"], 3],
                     "timing": {"draft_us": 1, "pad_us": 1, "verify_us": 1, "verify_enqueue_us": 1, "catchup_us": 1,
                                "accept_us": 1, "trace_us": 1, "cycle_us": 1, "gap_us": 1}})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    try:
        out = selfcheck(f.name, verbose=False)
    finally:
        os.unlink(f.name)
    assert out["rows"] == 4 and out["forced"] == 1 and out["skipped"] == {"grammar cycle": 2}, out
    assert out["max"] < 1e-6 and out["inexact"] == 0, out
    # a counterfactual setting on the same record: no penalty, thinking sampling
    info = {}
    p = row_probs(recs[0], 1, dict(S, temp=1.0, top_p=0.95, penalty_present=0.0), info)
    assert info["exact"] and abs(sum(p.values()) - 1) < 1e-6, info
    # a longer penalty window than the recorded one is short of context
    info = {}
    row_probs(recs[0], 0, dict(S, penalty_last_n=128), info)
    assert not info["exact"], info
    print("counterfactual.py: all tests passed")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        test()
    elif len(sys.argv) == 3 and sys.argv[1] == "selfcheck":
        selfcheck(sys.argv[2])
    else:
        sys.exit("usage: counterfactual.py selfcheck <trace.jsonl> | counterfactual.py test")
