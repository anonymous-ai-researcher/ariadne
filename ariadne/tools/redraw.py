"""Keep the timing (commit log, read and validation positions) of one run, replace the geometry by workloads of other
seeds, and redo the law extraction of the concurrent in-ball insertions.  If the timing is independent of the geometry,
this is an exact replication of the run with new data."""
import sys, numpy as np
from scipy.stats import binomtest
from ariadne import audit as AU
from ariadne.workloads import make_workload
run = AU.load_run(sys.argv[1]); cfg = run["cfg"]; k = int(cfg.get("k", 10))
print("decider", cfg.get("decider"), "| protocol", cfg.get("protocol"), "| store", cfg.get("store"), "| tau_cos", cfg.get("tau_cos"), "| workload", cfg["workload"])
use = [r for r in run["recs"] if r["first"] is not None]; js = np.array([r["j"] for r in use])
s = np.array([r["first"][2] for r in use]); sr = np.minimum(np.array([r["first"][3] for r in use]), s)
tot = np.zeros(k, int); tot_exp = 0.0
for seed in [int(x) for x in sys.argv[2:]]:
    wl = make_workload(dict(cfg["workload"]), seed, k); ex = AU.Exact(run, wl)
    Q = wl.X[js]; _, ro = ex.topk(Q, sr, k + 1); T = ex.stream_dist(Q); par = wl.truth if wl.truth is not None else np.full(wl.n, -1)
    lo = np.searchsorted(ex.ins_pos, sr, side="left"); hi = np.searchsorted(ex.ins_pos, s, side="left"); ranks = []; exp = 0.0
    for x in range(len(js)):
        src = ex.ins_task[lo[x]:hi[x]]; src = src[(src != js[x]) & (par[src] != js[x]) & (par[js[x]] != src)]
        t = T[x, src]; inb = t < ro[x, k - 1]; ranks.extend((1 + np.searchsorted(ro[x, :k], t[inb])).tolist()); exp += len(src) * k / (wl.nb + lo[x] + 1.0)
    h = np.bincount(np.array(ranks, dtype=int), minlength=k + 1)[1:]; tot += h; tot_exp += exp
    print(f"seed {seed:3d}: in-ball {h.sum():4d} (expected {exp:.1f}) ranks {h.tolist()} rank-1 {100 * h[0] / max(h.sum(), 1):.1f}%")
n = tot.sum(); print(f"pooled: in-ball {n} against {tot_exp:.1f} expected (ratio {n / tot_exp:.3f}, z = {(n - tot_exp) / np.sqrt(tot_exp):+.2f}); ranks {tot.tolist()}; rank-1 share {100 * tot[0] / n:.2f}% (binomial p = {binomtest(int(tot[0]), int(n), 1 / k).pvalue:.3g})")
