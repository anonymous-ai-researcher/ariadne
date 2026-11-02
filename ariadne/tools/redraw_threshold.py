"""Keep the timing of a run that used the threshold decider (ADD only if the nearest memory is farther than tau),
replace the geometry by workloads of other seeds, re-take every decision exactly on the new geometry, and redo the
extraction of the unrelated concurrent in-ball insertions.  Tests whether timing plus threshold selection alone make
background ranks non-uniform."""
import sys, numpy as np, faiss
from scipy.stats import binomtest
from ariadne import audit as AU
from ariadne.workloads import make_workload
run = AU.load_run(sys.argv[1]); cfg = run["cfg"]; k = int(cfg.get("k", 10)); tau = AU._tau(run); recs = run["recs"]
n = len(recs); order_j = np.array([r["j"] for r in recs])
s = np.array([r["first"][2] for r in recs], dtype=float); sr = np.minimum(np.array([r["first"][3] for r in recs], dtype=float), s)
nbw = None; pos = {}
for p, op, item, j in run["clog"]:
    if op == "i": pos[j] = p
# virtual commit coordinate of every task: its log position if it inserted, else just after its validation position
vpos = np.array([pos.get(int(r["j"]), r["first"][2] + 0.5) for r in recs], dtype=float)
print("tau", round(tau, 3), "tasks", n, "inserted in the run", len(pos), "| mean window", float((s - sr).mean()))
tot = np.zeros(k, int); tot_exp = 0.0; add_share = []
for seed in [int(x) for x in sys.argv[2:]]:
    wl = make_workload(dict(cfg["workload"]), seed, k); X = wl.X[order_j]; par_full = wl.truth if wl.truth is not None else np.full(wl.n, -1)
    ix = faiss.IndexFlatIP(wl.base.shape[1]); ix.add(wl.base)
    rb = np.sqrt(np.maximum(2 - 2 * ix.search(np.ascontiguousarray(X), k + 1)[0], 0))          # base neighbours of every query
    T = np.sqrt(np.maximum(2 - 2 * X @ X.T, 0)); np.fill_diagonal(T, np.inf)
    add = np.zeros(n, bool)
    for i in np.argsort(vpos, kind="stable"):                                                   # decisions in commit order
        vis = add & (vpos < sr[i]); r1 = min(rb[i, 0], T[i, vis].min() if vis.any() else np.inf); add[i] = r1 > tau
    add_share.append(add.mean()); ranks = []; exp = 0.0
    for i in range(n):
        vis = add & (vpos < sr[i]); conc = add & (vpos >= sr[i]) & (vpos < s[i]); conc[i] = False
        src = np.where(conc)[0]; src = src[(par_full[order_j[src]] != order_j[i]) & (par_full[order_j[i]] != order_j[src])]
        if len(src) == 0: continue
        ro = np.sort(np.concatenate([rb[i, :k], np.sort(T[i, vis])[:k]]))[:k]
        t = T[i, src]; inb = t < ro[k - 1]; ranks.extend((1 + np.searchsorted(ro, t[inb])).tolist()); exp += len(src) * k / (wl.nb + vis.sum() + 1.0)
    h = np.bincount(np.array(ranks, dtype=int), minlength=k + 1)[1:]; tot += h; tot_exp += exp
    print(f"seed {seed:3d}: ADD share {add.mean():.3f}, unrelated in-ball {h.sum():4d} (expected {exp:.1f}) ranks {h.tolist()} rank-1 {100 * h[0] / max(h.sum(), 1):.1f}%", flush=True)
m = tot.sum(); print(f"pooled: in-ball {m} against {tot_exp:.1f} expected (ratio {m / tot_exp:.3f}, z = {(m - tot_exp) / np.sqrt(tot_exp):+.2f}); ranks {tot.tolist()}; rank-1 share {100 * tot[0] / m:.2f}% (binomial p = {binomtest(int(tot[0]), int(m), 1 / k).pvalue:.3g})")
