# cfvec.py: counterfactual.target_probs vectorized over many verify rows, for the drafted token of each row only, in
# the default sampler order (penalties -> top_k -> top_p -> min_p -> temperature -> dist) with repeat 1, freq 0,
# min_keep 0 and top_k > 0 (the grid's settings). Same float32 steps as counterfactual.py. Checked against it in
# cfgrid.py (validate): same p for every sampled (row, setting) pair.
import numpy as np

F32 = np.float32


class Rows:
    """Drafted rows of the counterfactual set: raw top-K logits (n, K) sorted descending (ties by id), ids, window
    membership, the drafted token x and its q."""

    def __init__(self, c, mask_rows=None):
        nr = np.minimum(c["n_real"], c["d_id"].shape[1])
        ok = c["cf_ok"]
        rc = np.repeat(np.arange(len(nr)), nr)
        rd = np.concatenate([np.arange(k) for k in nr])
        keep = ok[rc]
        if mask_rows is not None:
            keep &= mask_rows
        self.rc, self.rd = rc[keep], rd[keep]
        self.ids = c["r_id"][self.rc, self.rd]
        self.lg = c["r_lg"][self.rc, self.rd].astype(F32)
        self.win = c["r_win"][self.rc, self.rd]
        self.x = c["d_id"][self.rc, self.rd]
        self.q = c["d_q"][self.rc, self.rd]
        self.tail_max = self.lg[:, -1].copy()
        self._sorted = {}

    def sorted_for(self, pres):
        """logits after the presence penalty, re-sorted (descending, ties ascending id); cached per presence."""
        key = float(pres)
        if key not in self._sorted:
            lg = self.lg.copy()
            if pres != 0:
                lg = np.where(self.win, lg - F32(1.0) * F32(pres), lg).astype(F32)
            o = np.lexsort((self.ids, -lg), axis=-1)
            lgs = np.take_along_axis(lg, o, 1)
            ids = np.take_along_axis(self.ids, o, 1)
            n_above = (lg > self.tail_max[:, None]).sum(1)
            self._sorted[key] = (lgs, ids, n_above)
        return self._sorted[key]

    def probs(self, temp, top_k, top_p, min_p, pres):
        """p_S(x) per row and an exactness flag (top_k filled inside the list after penalties)."""
        lgs, ids, n_above = self.sorted_for(pres)
        k = int(top_k)
        assert k > 0
        lg = lgs[:, :k].copy()
        ids = ids[:, :k]
        n = len(lg)
        exact = n_above >= k
        nkeep = np.full(n, k)
        cols = np.arange(k)[None, :]
        # top_p (before temperature): float32 softmax over the top-k, cumsum, first index with cum >= p
        p = F32(top_p)
        if p < 1:
            e = np.exp((lg - lg[:, :1]).astype(np.float64)).astype(F32)
            ssum = np.cumsum(e, axis=1, dtype=F32)[:, -1:]
            pp = (e / ssum).astype(F32)
            cum = np.cumsum(pp, axis=1, dtype=F32)
            hit = cum >= p
            nkeep = np.where(hit.any(1), hit.argmax(1) + 1, k)
        # min_p: keep logit >= max + logf(min_p) (float32), within the kept prefix
        mp = F32(min_p)
        if mp > 0:
            thr = (lg[:, 0] + F32(np.log(np.float64(mp)))).astype(F32)
            nmp = (lg >= thr[:, None]).sum(1)
            nkeep = np.minimum(nkeep, np.maximum(nmp, 1))
        # temperature
        t = F32(temp)
        if t != 1:
            lg = (lg / t).astype(F32)
        # dist: e = expf(l - max) float32, summed in double, p = float(e / sum)
        e = np.exp((lg - lg[:, :1]).astype(np.float64)).astype(F32).astype(np.float64)
        e = np.where(cols < nkeep[:, None], e, 0.0)
        ssum = np.cumsum(e, axis=1)[:, -1:]
        pp = (e / ssum).astype(F32)
        pp = np.where(nkeep[:, None] == 1, np.where(cols == 0, F32(1), F32(0)), pp)
        hitx = (ids == self.x[:, None]) & (cols < nkeep[:, None])
        px = np.where(hitx.any(1), pp[np.arange(n), hitx.argmax(1)], 0.0).astype(np.float64)
        return px, exact

    def accept(self, temp, top_k, top_p, min_p, pres):
        px, exact = self.probs(temp, top_k, top_p, min_p, pres)
        a = np.where(self.q > 0, np.minimum(1.0, px / np.where(self.q > 0, self.q, 1.0)), 0.0)
        return a, px, exact
