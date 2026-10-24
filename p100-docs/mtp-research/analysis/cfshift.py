# cfshift.py: CF_ANALYSIS step 3, the content-shift check. For a setting S, tables fit on the OTHER trace's content
# (dev, at S) vs the trace's own (dev, at S): max table difference, and the simulated test throughput loss of using the
# cross-content tables on each trace's test split (theta from the other trace's val, or re-chosen on this trace's val).
# Named settings in detail (thinking setting from cfchat vs cfthink; chat setting from cfthink vs cfchat), then the
# whole grid. Writes out/cf/shift.json.
import json, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import cfcore as C

OUT = C.OUT


def main():
    rows = {s: json.load(open(f"{OUT}/grid-{s}.json")) for s in ("chat", "think")}
    sets = {s: C.Set(C.SETS[s]) for s in ("chat", "think")}
    other = {"chat": "think", "think": "chat"}
    res = {"named": {}, "grid": {}}
    print("## named settings: cross-content tables on each trace's test split")
    print("| evaluated on | setting | tables from | max abs diff iso_q / iso_p1 (mean) | theta own / cross | "
          "loss, cross theta [95%] | loss, theta re-chosen on own val [95%] |")
    print("|---|---|---|---|---|---|---|")
    for tgt, g in [("think", C.THINK_S), ("chat", C.CHAT_S), ("chat", C.THINK_S), ("think", C.CHAT_S)]:
        src = other[tgt]
        gi = C.GIDX[g]
        S = sets[tgt]
        A = S.a(gi)
        ro, rx = rows[tgt][gi], rows[src][gi]
        tq_o, tp_o, th_o = np.array(ro["tq"]), np.array(ro["tp"]), ro["theta"]
        tq_x, tp_x, th_x = np.array(rx["tq"]), np.array(rx["tp"]), rx["theta"]
        pol_o = S.pol_tables(A, tq_o, tp_o, th_o)
        pol_x = S.pol_tables(A, tq_x, tp_x, th_x)
        th_x2, _ = S.choose_theta(A, tq_x, tp_x)
        pol_x2 = S.pol_tables(A, tq_x, tp_x, th_x2)
        g1, ci1 = S.boot_gain(pol_x, pol_o)
        g2, ci2 = S.boot_gain(pol_x2, pol_o)
        dq, dp = np.abs(tq_o - tq_x), np.abs(tp_o - tp_x)
        print(f"| {tgt} | {C.sname(g)} | {src} | {dq.max():.3f} / {dp.max():.3f} ({dq.mean():.3f} / {dp.mean():.3f}) | "
              f"{th_o} / {th_x} (re-chosen {th_x2}) | {g1 * 100:+.2f}% [{ci1[0] * 100:+.2f}, {ci1[1] * 100:+.2f}] | "
              f"{g2 * 100:+.2f}% [{ci2[0] * 100:+.2f}, {ci2[1] * 100:+.2f}] |", flush=True)
        res["named"][f"{tgt}@{C.sname(g)}"] = dict(src=src, dq_max=float(dq.max()), dp_max=float(dp.max()),
                                                   loss_cross_theta=g1, ci1=ci1.tolist(), loss_own_theta=g2, ci2=ci2.tolist(),
                                                   tq_own=tq_o.tolist(), tq_cross=tq_x.tolist(), tp_own=tp_o.tolist(), tp_cross=tp_x.tolist())
        if tgt == "think" and g == C.THINK_S:
            print("\n  x     think-own q  chat-content q | think-own p1  chat-content p1   (production thinking q / p1)")
            for i, x in enumerate(C.X24):
                print(f"  {x:5.3f}  {tq_o[i]:.3f}       {tq_x[i]:.3f}          | {tp_o[i]:.3f}         {tp_x[i]:.3f}"
                      f"            ({C.PROD_THINK[0][i]:.3f} / {C.PROD_THINK[1][i]:.3f})")
            print()
    # whole grid: cross-content loss per setting, both directions
    print("\n## whole grid (480 settings): cross-content tables (theta from the other trace) vs own, test")
    print("| evaluated on | tables from | mean loss | worst | worst 10% | max abs table diff, median over grid (q / p1) |")
    print("|---|---|---|---|---|---|")
    for tgt in ("chat", "think"):
        src = other[tgt]
        S = sets[tgt]
        L, dqs, dps = [], [], []
        for gi in range(len(C.GRID)):
            A = S.a(gi)
            rx = rows[src][gi]
            r = S.rate(*S.pol_tables(A, np.array(rx["tq"]), np.array(rx["tp"]), rx["theta"]), C.TEST)
            L.append(r / rows[tgt][gi]["r_own"] - 1)
            dqs.append(np.abs(np.array(rows[tgt][gi]["tq"]) - rx["tq"]).max()); dps.append(np.abs(np.array(rows[tgt][gi]["tp"]) - rx["tp"]).max())
        L = np.array(L); l = np.sort(L)
        print(f"| {tgt} | {src} | {L.mean() * 100:+.2f}% | {l[0] * 100:+.2f}% | {l[:48].mean() * 100:+.2f}% | "
              f"{np.median(dqs):.3f} / {np.median(dps):.3f} |")
        res["grid"][tgt] = dict(src=src, loss=L.tolist(), mean=float(L.mean()), worst=float(l[0]), worst10=float(l[:48].mean()))
    # mean acceptance by content at the two named settings
    for s in ("chat", "think"):
        print(f"mean a ({s} content): at chat setting {rows[s][C.GIDX[C.CHAT_S]]['mean_a']:.3f}, "
              f"at thinking setting {rows[s][C.GIDX[C.THINK_S]]['mean_a']:.3f}; over the grid "
              f"{min(r['mean_a'] for r in rows[s]):.3f}..{max(r['mean_a'] for r in rows[s]):.3f}")
    json.dump(res, open(f"{OUT}/shift.json", "w"))


if __name__ == "__main__":
    main()
