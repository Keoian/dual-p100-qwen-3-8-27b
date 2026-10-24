# cfreport.py: the approximation comparison table of CF_ANALYSIS.md from out/cf/approx-*.json and recommend.json.
import json
import cfcore as C

A = {s: json.load(open(f"{C.OUT}/approx-{s}.json")) for s in ("chat", "think", "pooled")}
R = json.load(open(f"{C.OUT}/recommend.json"))["scores"]
pick = [
    ("a. production today (gate by temperature, else p-min 0.2)", "a. production (chat/think tables by temperature, else p-min 0.2)"),
    ("(ref) p-min 0.2 everywhere", "(ref) p-min 0.2 everywhere"),
    ("a. production's chat set at every setting (no gate)", "a2. production's chat tables/theta at every setting (no gate)"),
    ("a. production's thinking set at every setting", "a2. production's thinking tables/theta at every setting (no gate)"),
    ("a. production's two sets, chat T < 0.95, thinking T >= 0.95", "a'. production's two sets, gate widened (chat T < 0.95, thinking T >= 0.95)"),
    ("a. production's two sets keyed by thinking on/off", "a3. production's two sets keyed by thinking on/off (not temperature), no gate"),
    ("b. one set, best grid setting on val", "b. one table set for all"),
    ("b. nearest anchor, K=2", "b. nearest anchor, K=2"),
    ("b. nearest anchor, K=4", "b. nearest anchor, K=4"),
    ("b. nearest anchor, K=8", "b. nearest anchor, K=8"),
    ("c. multilinear T{0.3,1.2}", "c. multilinear T{0.3,1.2}"),
    ("c. multilinear T{0.3,0.7,1.2}", "c. multilinear T{0.3,0.7,1.2}"),
    ("c. multilinear T{0.3,0.7,1.2} x pres{0,1.5}", "c. multilinear T{0.3,0.7,1.2} x pr{0,1.5}"),
    ("c. multilinear T{0.3,0.7,1.2} x top_p{0.8,1} x pres{0,1.5}", "c. multilinear T{0.3,0.7,1.2} x p{0.8,1} x pr{0,1.5}"),
    ("c. multilinear T(5) x top_p(3) x pres(3)", "c. multilinear T{0.3,0.5,0.7,1.0,1.2} x p{0.8,0.9,1} x pr{0,0.75,1.5}"),
    ("c. multilinear full T(10) x top_p(4) x pres(3)", "c. multilinear full T(10) x p(4) x pr(3)"),
    ("d. one marginal set (fit over the whole grid)", "d. one marginal table set"),
    ("d. iso(q, T), marginal sets at T 0.3/0.7/1.2", "d. iso(q, T): temperature-only tables (marginal over the other fields), T{0.3,0.7,1.2}"),
    ("d. iso(q, T), marginal sets at 10 temperatures", "d. iso(q, T): temperature-only tables (marginal over the other fields), 10 temperatures"),
    ("d. PCA family, 1 component", "d. PCA family: mean + 1 component"),
    ("d. PCA family, 2 components", "d. PCA family: mean + 2 component"),
    ("e. analytic, uniform tail, fixed theta", "e. analytic (uniform tail): ref tables x reshaped-draft ratio, theta"),
    ("e. analytic, uniform tail + correction tables, fixed theta", "e. analytic (uniform tail, + correction tables): ref tables x reshaped-draft ratio, theta 0"),
    ("e. analytic, packed tail + correction tables, fixed theta", "e. analytic (packed tail, + correction tables): ref tables x reshaped-draft ratio, theta 0"),
    ("f. own tables, one theta (0.175) for all", "f. own tables, one theta for all"),
]
print("| approximation | storage (B) | chat: mean / worst / worst 10% | think: mean / worst / worst 10% | pooled: mean / worst / worst 10% |")
print("|---|---|---|---|---|")
for lab, key in pick:
    cells, stor = [], None
    for s in ("chat", "think", "pooled"):
        ks = [k for k in A[s] if k.startswith(key) and not k.startswith("_")]
        if not ks:
            cells.append("n/a"); continue
        v = A[s][ks[0]]
        stor = v["storage"]
        cells.append(f"{v['mean'] * 100:+.2f} / {v['worst'] * 100:+.2f} / {v['worst10'] * 100:+.2f}")
    print(f"| {lab} | {stor} | " + " | ".join(cells) + " |")
print()
print("| candidate | chat: mean [95%] / worst | think: mean [95%] / worst | pooled: mean [95%] / worst |")
print("|---|---|---|---|")
for lab in dict.fromkeys(k.split(" | ")[0] for k in R):
    cells = []
    for s in ("chat", "think", "pooled"):
        v = R[f"{lab} | {s}"]
        cells.append(f"{v['mean'] * 100:+.2f} [{v['ci'][0] * 100:+.2f}, {v['ci'][1] * 100:+.2f}] / {v['worst'] * 100:+.2f}")
    print(f"| {lab} | " + " | ".join(cells) + " |")
