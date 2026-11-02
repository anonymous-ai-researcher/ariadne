"""Microbenchmarks that need no store.

    python -m ariadne.micro --out micro.json                      cost of each validation predicate against k
    python -m ariadne.micro --audit RUN.pkl ... --out cost.json   cost of the offline audit against log size
"""
import argparse, json, time
import numpy as np
from .policies import Policy


def predicates(ks=(1, 5, 10, 50, 100, 1000), reps=20000, seed=0):
    rng = np.random.default_rng(seed); out = {}
    for k in ks:
        d = np.sort(rng.uniform(0.5, 1.0, size=k)); ids = rng.permutation(10 * k)[:k]
        A = [(int(i), float(x)) for i, x in zip(ids, d)]
        F = [(int(i), float(x * 0.999)) for i, x in zip(ids, d)]              # valid for every policy: the worst case (no early exit)
        row = {}
        for spec in ("exact", "set", "overlap:0.9", "top1", "radius:1.1", "rank1:1.1", "sumdist:1.1", "cadm:1.1"):
            p = Policy(spec, k); n = max(200, reps // max(k // 10, 1))
            t0 = time.perf_counter()
            for _ in range(n):
                p.valid(A, F)
            row[spec] = round(1e6 * (time.perf_counter() - t0) / n, 3)
        out[str(k)] = row
        print(f"k={k:5d}  microseconds per test: " + "  ".join(f"{s} {v}" for s, v in row.items()), flush=True)
    return out


def audit_cost(paths):
    from . import audit as AU
    out = []
    for p in paths:
        run = AU.load_run(p); n = len(run["recs"]); row = {"run": p, "tasks": n, "log_entries": len(run["clog"])}
        for name, fn in (("run", AU.audit_run), ("first", AU.first_validation), ("sep", AU.separation), ("law", AU.law_extract)):
            t0 = time.time()
            try:
                fn(run); row[name + "_s"] = round(time.time() - t0, 2); row[name + "_ms_per_task"] = round(1000 * (time.time() - t0) / n, 3)
            except (Exception, SystemExit) as e:
                row[name + "_s"] = None; row[name + "_error"] = repr(e)[:100]
        print(json.dumps(row), flush=True); out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--audit", nargs="*"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"audit": audit_cost(a.audit)} if a.audit else {"predicate_us": predicates()}
    json.dump(res, open(a.out, "w"), indent=1)
