"""Lag kernel from a static stream: predicting conflicts before running anything.

The lag kernel says how likely an item that arrives l positions later in the stream is to fall inside the
top-k ball of an earlier item's read (h_l), and with which effective dimension (D_l).  The engine measures
it from concurrent executions (python -m ariadne.law lag).  This module estimates the same quantities from
the data alone: for every stream item j it takes the exact top-k ball of j in the state "initial memory plus
the j earlier stream items" and tests the items j+1 .. j+L against it.  No execution, no concurrency.

With --predict it then predicts a concurrent run it has never seen: the run's law file supplies only how
many commits overlapped a read at each lag (a property of the scheduling, not of the data); the kernel
supplies the rest.  Output: predicted in-ball insertions per read and the conflict probability at each c,
next to what the run observed.

    python -m ariadne.static_kernel --law results/D3/L4_s1.law.pkl --out kernel_L4.json --predict results/D3/L8_s1.law.pkl
    python -m ariadne.static_kernel --workload '{"kind": "stream", "data": "data/agnews__wordllama.npz", "base": 60000, "stream": 6000}' --k 10 --out k.json

--law takes the workload (and k) from a law file; --workload takes a JSON workload section directly.
Output JSON: {"k", "n", "lags", "buckets": {bucket: {"overlap", "inball", "h", "D", ...}}, "per_lag": {"h": [...], "n": [...]},
              "exchangeable_h", "predictions": {name: {"pred_m", "obs_m", "cs", "obs", "pred", "err", "notes"}}}

Limits: the static state stores every earlier stream item (a run with deduplication stores fewer), and the
"other team" bucket is estimated from random pairs of items of different teams.
"""
import argparse, json, math, pickle
import numpy as np

from .audit import Exact
from .workloads import make_workload
from . import law as LW


def kernel(wl, k, L=64, sample=20000, seed=0, metric="cosine"):
    n, nb = wl.n, wl.nb
    run = {"cfg": {"workload": {}, "k": k, "store": {"metric": metric}}, "clog": [(j, "i", nb + j, j) for j in range(n)]}
    ex = Exact(run, wl)
    rng = np.random.default_rng(seed)
    js = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    _, ro = ex.topk(wl.X[js], js, k)
    rk = ro[:, k - 1]
    same_ov = np.zeros(LW.LAG_MAX + 1); same_in = np.zeros(LW.LAG_MAX + 1); ys = [[] for _ in range(L + 1)]; r1 = np.zeros(L + 1)
    for l in range(1, L + 1):
        ok = js + l < n
        a = js[ok]; b = a + l
        st = wl.team[a] == wl.team[b]
        a, b, rka, r1a = a[st], b[st], rk[ok][st], ro[ok][st][:, 0]
        if len(a) == 0:
            continue
        v = wl.X[a] - wl.X[b]
        t = np.sqrt(np.maximum(2.0 - 2.0 * (wl.X[a] * wl.X[b]).sum(1), 0.0)) if metric == "cosine" else np.sqrt((v * v).sum(1))
        inb = t < rka
        same_ov[l] = len(a); same_in[l] = int(inb.sum()); r1[l] = int((t[inb] < r1a[inb]).sum())
        ys[l] = np.log(rka[inb] / np.maximum(t[inb], LW.TMIN)).tolist()
    # other teams: random pairs of items from different teams
    oth = {"overlap": 0, "inball": 0, "y": [], "r1": 0}
    if len(set(wl.team.tolist())) > 1:
        b = rng.integers(0, n, size=len(js)); df = wl.team[js] != wl.team[b]
        a, b2, rka, r1a = js[df], b[df], rk[df], ro[df][:, 0]
        t = np.sqrt(np.maximum(2.0 - 2.0 * (wl.X[a] * wl.X[b2]).sum(1), 0.0))
        inb = t < rka
        oth = {"overlap": int(len(a)), "inball": int(inb.sum()), "y": np.log(rka[inb] / np.maximum(t[inb], LW.TMIN)).tolist(), "r1": int((t[inb] < r1a[inb]).sum())}
    tot_in = same_in.sum() + oth["inball"]
    buckets = {}
    for name, rg in LW.BUCKETS:
        if rg is None:
            ov, nin, y, n1 = oth["overlap"], oth["inball"], oth["y"], oth["r1"]
        else:
            lo, hi = rg[0], min(rg[1], L)
            if lo > L:
                buckets[name] = {"overlap": 0, "inball": 0, "h": None, "D": None, "share": None, "rank1": None}; continue
            ov = int(same_ov[lo:hi + 1].sum()); nin = int(same_in[lo:hi + 1].sum()); n1 = int(r1[lo:hi + 1].sum())
            y = [x for l in range(lo, hi + 1) for x in ys[l]]
        ym = float(np.mean(y)) if y else 0.0
        buckets[name] = {"overlap": ov, "inball": nin, "h": nin / ov if ov else None, "D": 1.0 / ym if (nin >= LW.MIN_BUCKET and ym > 0) else None,
                         "share": nin / tot_in if tot_in else None, "rank1": n1 / nin if nin else None}
    return {"k": k, "n": int(len(js)), "lags": L, "buckets": buckets,
            "per_lag": {"h": [float(same_in[l] / same_ov[l]) if same_ov[l] else None for l in range(1, L + 1)], "n": same_ov[1:L + 1].tolist()},
            "exchangeable_h": float(k / (nb + n / 2.0))}


class _Static:
    def __init__(self, K, n):
        self.K, self.n = K, n

    def lag_table(self):
        return self.K["buckets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law"); ap.add_argument("--workload"); ap.add_argument("--k", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lags", type=int, default=64); ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--predict", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.law:
        L = pickle.load(open(a.law, "rb")); wcfg, k, seed = L["workload"], int(L["k"]), int(L.get("workload_seed", a.seed))
    else:
        wcfg, k, seed = json.loads(a.workload), a.k, a.seed
    wl = make_workload(wcfg, seed, k)
    K = kernel(wl, k, a.lags, a.sample)
    print(f"static lag kernel: k={k}, {K['n']} stream items, lags 1..{a.lags}; exchangeable reference h = {100 * K['exchangeable_h']:.3f}%")
    for b, v in K["buckets"].items():
        if v["overlap"]:
            print(f"   lag {b:6s}: pairs {v['overlap']:8d}, in-ball {v['inball']:6d}, h = {100 * v['h']:7.3f}%, D = {('%.2f' % v['D']) if v['D'] else '  -'}, becomes new nearest {100 * (v['rank1'] or 0):5.1f}%")
    K["predictions"] = {}
    for path in a.predict:
        T = LW.LawData(path); notes = []
        model, tot = LW.lag_model(_Static(K, K["n"]), T, notes)
        ev = LW.evaluate(T, model) if model["pi"] else {"obs_m": float(T.m.mean()), "cs": list(T.cs), "obs": None, "pred": None, "err": None}
        K["predictions"][T.name] = {"pred_m": tot, **ev, "model": model, "notes": notes}
        print(f"prediction for {T.name}: in-ball insertions per read {tot:.3f} (observed {ev['obs_m']:.3f})")
        if ev.get("pred"):
            print(f"   conflict % at c = {list(T.cs)}\n   observed  {LW.fmt(ev['obs'])}\n   predicted {LW.fmt(ev['pred'])}   max |err| {ev['err']:.1f}")
        for x in notes:
            print("   note:", x)
    LW.save(a.out, K)


if __name__ == "__main__":
    main()
