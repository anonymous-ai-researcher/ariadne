"""Offline audit of run logs against exact k-NN.

Everything here is computed after the fact from a run file (config, task records, commit log): the
exact state of the store at any commit-log position is reconstructed from the workload and the log, so
the audit needs neither the store nor the index that the run used.

Commands (each takes one or more run files written by experiments/run.py):

  python -m ariadne.audit run    R.pkl ...   committed vs fresh vs exact: decision divergence for every proxy,
                                             approximation ratios, recall, contract violations, duplicate writes,
                                             and the safety-theorem check  rho(R | S) <= c * rho(F | S)
  python -m ariadne.audit first  R.pkl ...   counterfactual: what every validation predicate would have done at
                                             the first validation of each task (needs an instrumented run:
                                             protocol validate with policy none)
  python -m ariadne.audit sep    R.pkl ...   final store: pairwise separation, items within tau and tau/c of
                                             another item, redundant items (within tau of an earlier stored item)
  python -m ariadne.audit serial R.pkl ...   replay the tasks serially in commit order with exact search and the
                                             threshold rule; count tasks whose action differs
  python -m ariadne.audit law    R.pkl ...   extract the conflict-law data set (written next to the run as
                                             <run>.law.pkl; format documented in law_extract)
  python -m ariadne.audit replay R.pkl ...   re-ask the LLM decider on the fresh and on the exact answer and
                                             count changed decisions (uses the decider section of the run)

Options: --out FILE.jsonl (append one JSON line per run), --cs 1.02,1.05,...  --sample N  --tau-cos X
"""
import argparse, json, math, os, pickle, sys
import numpy as np

from .metrics import chord_from_cos, rho as rho_fn, insertion_conflicts, EPS
from .policies import Policy, PROXIES, make_proxy
from .workloads import make_workload

BIG = 10 ** 15
CS = (1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0)


def save_run(path, cfg, summary, recs, clog):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"                                       # atomic: an interrupted write never leaves a truncated log
    with open(tmp, "wb") as f:
        pickle.dump({"cfg": cfg, "summary": summary, "recs": recs, "clog": clog}, f, protocol=4)
    os.replace(tmp, path)


def load_run(path):
    with open(path, "rb") as f:
        return pickle.load(f)


_wl_cache = {}


def workload_of(cfg):
    key = json.dumps([cfg["workload"], cfg.get("workload_seed", cfg.get("seed", 1)), cfg.get("k", 10)], sort_keys=True, default=str)
    if key not in _wl_cache:
        _wl_cache.clear()
        _wl_cache[key] = make_workload(cfg["workload"], cfg.get("workload_seed", cfg.get("seed", 1)), int(cfg.get("k", 10)))
    return _wl_cache[key]


class Exact:
    """Exact top-k in the store state after the first s commit-log entries, for arbitrary queries."""

    def __init__(self, run, wl=None):
        self.cfg = run["cfg"]; self.wl = wl or workload_of(self.cfg)
        self.metric = (self.cfg.get("store") or {}).get("metric", "cosine")
        wl = self.wl; nb, n = wl.nb, wl.n
        self.ins = np.full(n, BIG, dtype=np.int64)          # commit-log position of the insertion of stream item j
        self.dele = np.full(nb + n, BIG, dtype=np.int64)    # position of the (first) deletion of an item
        self.ins_log = []                                   # (pos, task whose item was inserted), in order
        for pos, op, item, j in run["clog"]:
            if op == "i":
                self.ins[item - nb] = pos; self.ins_log.append((pos, item - nb))
            else:
                self.dele[item] = min(self.dele[item], pos)
        self.ins_pos = np.array([p for p, _ in self.ins_log], dtype=np.int64)
        self.ins_task = np.array([t for _, t in self.ins_log], dtype=np.int64)
        self.base_deleted = bool((self.dele[:nb] < BIG).any())
        import faiss
        self.faiss = faiss
        self.ix = (faiss.IndexFlatIP if self.metric == "cosine" else faiss.IndexFlatL2)(wl.base.shape[1])
        self.ix.add(np.ascontiguousarray(wl.base, dtype="float32"))
        self.xn = (wl.X ** 2).sum(1) if self.metric != "cosine" else None

    def _dist(self, q, V):
        """metric distances between rows of q and rows of V"""
        if self.metric == "cosine":
            return np.sqrt(np.maximum(2.0 - 2.0 * (q @ V.T), 0.0))
        return np.sqrt(np.maximum((q ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * (q @ V.T), 0.0))

    def stream_dist(self, Q):
        return self._dist(np.asarray(Q, dtype="float32"), self.wl.X)

    def vec(self, item):
        return self.wl.base[item] if item < self.wl.nb else self.wl.X[item - self.wl.nb]

    def recompute(self, q, A, s):
        """Distances of the items of an answer recomputed from the vectors, and how many of them are not in the state s."""
        if not A:
            return [], 0
        nb = self.wl.nb; ids = [i for i, _ in A]
        V = np.stack([self.vec(i) for i in ids])
        d = self._dist(np.asarray(q, dtype="float32")[None, :], V)[0]
        ghost = sum(1 for i in ids if (self.dele[i] < s) or (i >= nb and self.ins[i - nb] >= s))
        return [float(x) for x in d], ghost

    def topk(self, Q, s, kk):
        """Q [m, d], s [m] -> (ids [m, kk], dist [m, kk]); missing entries have id -1 and distance inf."""
        Q = np.ascontiguousarray(Q, dtype="float32"); s = np.asarray(s, dtype=np.int64)
        wl = self.wl; nb, n = wl.nb, wl.n; m = len(Q)
        ids = np.full((m, kk), -1, dtype=np.int64); dist = np.full((m, kk), np.inf)
        KB = min(nb, kk + (16 if self.base_deleted else 0))
        step = max(32, int(2e7 // max(n, 1)))
        for a in range(0, m, step):
            q = Q[a:a + step]; sa = s[a:a + step]
            Sb, Ib = self.ix.search(q, KB)
            db = np.sqrt(np.maximum(2.0 - 2.0 * Sb, 0.0)) if self.metric == "cosine" else np.sqrt(np.maximum(Sb, 0.0))
            if self.base_deleted:
                gone = self.dele[Ib] < sa[:, None]
                db = np.where(gone, np.inf, db)
                for r in np.where((~gone).sum(1) < min(kk, nb))[0]:          # too many of the candidates were deleted: brute force
                    d_all = self._dist(q[r:r + 1], wl.base)[0]
                    d_all[self.dele[:nb] < sa[r]] = np.inf
                    o = np.argsort(d_all)[:KB]; Ib[r] = o; db[r] = d_all[o]
            ds = self.stream_dist(q)
            live = (self.ins[None, :] < sa[:, None]) & (self.dele[None, nb:] >= sa[:, None])
            ds = np.where(live, ds, np.inf)
            ks = min(kk, n)
            part = np.argpartition(ds, ks - 1, axis=1)[:, :ks] if ks < n else np.tile(np.arange(n), (len(q), 1))
            dcat = np.concatenate([db, np.take_along_axis(ds, part, axis=1)], axis=1)
            icat = np.concatenate([Ib, nb + part], axis=1)
            o = np.argsort(dcat, axis=1, kind="stable")[:, :kk]
            dd = np.take_along_axis(dcat, o, axis=1); ii = np.take_along_axis(icat, o, axis=1)
            ii = np.where(np.isfinite(dd), ii, -1)
            ids[a:a + len(q), :dd.shape[1]] = ii; dist[a:a + len(q), :dd.shape[1]] = dd
        return ids, dist


def _ans(ids, dist, k):
    return [(int(i), float(d)) for i, d in zip(ids[:k], dist[:k]) if i >= 0]


def _pct(x, q):
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def _tau(run, args=None):
    cfg = run["cfg"]
    if args is not None and getattr(args, "tau_cos", None):
        return chord_from_cos(args.tau_cos)
    return chord_from_cos(float(cfg.get("tau_cos", 0.10))) if (cfg.get("store") or {}).get("metric", "cosine") == "cosine" else float(cfg.get("tau", 0.45))


# ----------------------------------------------------------------------------- committed vs fresh vs exact
def audit_run(run, cs=CS, tau=None):
    cfg, recs = run["cfg"], run["recs"]; k = int(cfg.get("k", 10))
    ex = Exact(run); wl = ex.wl; tau = tau or _tau(run)
    proxy = make_proxy(wl.labels, tau, k)
    done = [r for r in recs if r["R"] is not None and not r["gave_up"]]
    if not done:
        return {"error": "the run kept no answers (log_answers: none)"}
    # one row per read
    rows = [(r["j"], ri) for r in done for ri in range(len(r["R"]))]
    Q = np.stack([wl.queries(j)[ri] for j, ri in rows]); s = np.array([recs[j]["s"] for j, _ in rows])
    Ei, Ed = ex.topk(Q, s, k)
    pol = Policy((cfg.get("protocol") or {}).get("policy", "none"), k)
    c_run = getattr(pol, "c", None) if pol.name in ("cadm", "cert") else None
    div = {x: {p: 0 for p in PROXIES} for x in ("committed_vs_exact", "fresh_vs_exact", "committed_vs_fresh")}
    rR, rF, recR, recF = [], [], [], []
    dupw = dupw_floor = lostw = 0; safety_bad = 0; safety_n = 0; nF = 0
    row = 0; ghosts = mismatch = 0
    for r in done:
        j = r["j"]; rho_r = rho_f = 1.0
        for ri in range(len(r["R"])):
            A = r["R"][ri]; F = r["F"][ri] if r["F"] else None; E = _ans(Ei[row], Ed[row], k)
            ed = [d for _, d in E]
            # the auditor does not trust reported distances: it recomputes them from the vectors, and checks A within S
            da, g = ex.recompute(Q[row], A, s[row]); row += 1
            ghosts += g > 0; mismatch += any(abs(x - y) > 2e-3 for x, (_, y) in zip(da, A))
            rho_r = max(rho_r, rho_fn(da, ed) if (len(A) == len(E) and g == 0) else float("inf"))
            if F is not None:
                df, _ = ex.recompute(Q[row - 1], F, s[row - 1])
                rho_f = max(rho_f, rho_fn(df, ed) if len(F) == len(E) else float("inf"))
            if ri == 0:
                es = {i for i, _ in E}
                dA, dE = proxy(A), proxy(E)
                recR.append(len({i for i, _ in A} & es) / max(len(es), 1))
                for p in PROXIES:
                    div["committed_vs_exact"][p] += dA[p] != dE[p]
                if r["op"] == "add":
                    dupw += (not dA["dup"]) and dE["dup"]; lostw += dA["dup"] and not dE["dup"]
                if F is not None:
                    dF = proxy(F); nF += 1
                    recF.append(len({i for i, _ in F} & es) / max(len(es), 1))
                    for p in PROXIES:
                        div["fresh_vs_exact"][p] += dF[p] != dE[p]; div["committed_vs_fresh"][p] += dA[p] != dF[p]
                    if r["op"] == "add":
                        dupw_floor += (not dF["dup"]) and dE["dup"]
        rR.append(rho_r)
        if r["F"]:
            rF.append(rho_f)
            if c_run is not None and not r["capped"] and r["op"] != "read":
                safety_n += 1; safety_bad += not (rho_r <= c_run * rho_f * (1 + 1e-4) + 1e-5)
    n = len(done); rR, rF = np.array(rR), np.array(rF)
    fin = lambda x: x[np.isfinite(x)]
    out = {"tasks_audited": n, "answers_shorter_than_exact_or_with_missing_items": int(np.sum(~np.isfinite(rR))),
           "answers_with_items_not_in_state": int(ghosts), "answers_with_misreported_distances": int(mismatch),
           "div_committed_vs_exact_%": {p: round(100 * div["committed_vs_exact"][p] / n, 2) for p in PROXIES},
           "dup_written_per_100": round(100 * dupw / n, 2), "wrongly_skipped_per_100": round(100 * lostw / n, 2),
           "recall_committed": round(float(np.mean(recR)), 4),
           "rho_committed": {"p50": _pct(fin(rR), 50), "p90": _pct(fin(rR), 90), "p99": _pct(fin(rR), 99), "max": round(float(fin(rR).max()), 4),
                             **{f">{c}": round(100 * float(np.mean(rR > c + 1e-6)), 2) for c in cs if c > 1}}}
    if nF:
        out.update({"div_fresh_vs_exact_% (index floor)": {p: round(100 * div["fresh_vs_exact"][p] / nF, 2) for p in PROXIES},
                    "div_committed_vs_fresh_%": {p: round(100 * div["committed_vs_fresh"][p] / nF, 2) for p in PROXIES},
                    "dup_written_floor_per_100": round(100 * dupw_floor / nF, 2), "recall_fresh": round(float(np.mean(recF)), 4),
                    "rho_fresh": {"p50": _pct(fin(rF), 50), "p90": _pct(fin(rF), 90), "p99": _pct(fin(rF), 99), "max": round(float(fin(rF).max()), 4),
                                  **{f">{c}": round(100 * float(np.mean(rF > c + 1e-6)), 2) for c in cs if c > 1}}})
    out["_arrays"] = {"task": np.array([r["j"] for r in done]), "rho_committed": rR, "rho_fresh": rF, "recall_committed": np.array(recR),
                      "rounds": np.array([r["rounds"] for r in done]), "lat": np.array([r["lat"] for r in done])}
    if c_run is not None:
        out["safety_theorem"] = {"c": c_run, "checked": safety_n, "violations": int(safety_bad),
                                 "statement": "rho(committed | state at commit) <= c * rho(fresh | state at commit)"}
    return out


# ----------------------------------------------------------------------------- counterfactual at the first validation
def default_policies(k, cs=CS):
    P = ["none", "exact", "set", "overlap:0.9", "overlap:0.8", "top1", "topj:3"]
    for c in cs:
        if c > 1:
            P += [f"cadm:{c}", f"radius:{c}", f"rank1:{c}", f"sumdist:{c}"]
    return P + [f"oracle:{p}" for p in PROXIES]


def first_validation(run, cs=CS, policies=None, tau=None):
    cfg, recs = run["cfg"], run["recs"]; k = int(cfg.get("k", 10))
    ex = Exact(run); wl = ex.wl; tau = tau or _tau(run)
    proxy = make_proxy(wl.labels, tau, k)
    use = [r for r in recs if r["first"] is not None and r["first"][1] is not None]
    if not use:
        return {"error": "no first-validation data (run protocol validate, e.g. with policy none)"}
    n = len(use)
    Q = np.stack([wl.X[r["j"]] for r in use])
    s = np.array([r["first"][2] for r in use]); sr = np.array([r["first"][3] for r in use])
    Ei, Ed = ex.topk(Q, s, k); Oi, Od = ex.topk(Q, sr, k)
    pols = [Policy(p, k) for p in (policies or default_policies(k, cs))]
    res = {p.spec: {"invalid": 0, "dx": {q: 0 for q in PROXIES}, "df": {q: 0 for q in PROXIES}, "dup": 0, "rho": []} for p in pols}
    floor = {q: 0 for q in PROXIES}; floor_dup = 0; rho_f = []
    changed = logical = phys = 0
    for x, r in enumerate(use):
        A, F = r["first"][0][0], r["first"][1][0]
        E = _ans(Ei[x], Ed[x], k); O = _ans(Oi[x], Od[x], k)
        dA, dF, dE = proxy(A), proxy(F), proxy(E); ed = [d for _, d in E]
        for q in PROXIES:
            floor[q] += dF[q] != dE[q]
        add = r["op"] == "add"
        floor_dup += add and (not dF["dup"]) and dE["dup"]
        rho_f.append(rho_fn([d for _, d in F], ed)); ra = rho_fn([d for _, d in A], ed)
        ch = {i for i, _ in A} != {i for i, _ in F}; lg = {i for i, _ in E} != {i for i, _ in O}
        changed += ch; logical += lg; phys += ch and not lg
        for p in pols:
            v = res[p.spec]
            if not p.valid(A, F, proxy):
                v["invalid"] += 1
            else:
                for q in PROXIES:
                    v["dx"][q] += dA[q] != dE[q]; v["df"][q] += dA[q] != dF[q]
                v["dup"] += add and (not dA["dup"]) and dE["dup"]; v["rho"].append(ra)
    out = {"tasks": n, "changed_set_%": round(100 * changed / n, 2), "exact_topk_changed_%": round(100 * logical / n, 2),
           "physical_only_change_%": round(100 * phys / n, 2),
           "floor_fresh_vs_exact_%": {q: round(100 * floor[q] / n, 2) for q in PROXIES}, "floor_dup_written_per_100": round(100 * floor_dup / n, 2),
           "rho_fresh": {"p50": _pct(rho_f, 50), "p90": _pct(rho_f, 90), "p99": _pct(rho_f, 99)}, "policies": {}}
    for p in pols:
        v = res[p.spec]
        out["policies"][p.spec] = {"redo_%": round(100 * v["invalid"] / n, 2),
                                   "accepted_div_vs_exact_%": {q: round(100 * v["dx"][q] / n, 2) for q in PROXIES},
                                   "accepted_div_vs_fresh_%": {q: round(100 * v["df"][q] / n, 2) for q in PROXIES},
                                   "dup_written_per_100": round(100 * v["dup"] / n, 2), "rho_accepted_p99": _pct(v["rho"], 99),
                                   "rho_accepted_max": round(float(max(v["rho"])), 4) if v["rho"] else None}
    return out


# ----------------------------------------------------------------------------- final-state separation
def separation(run, tau=None):
    cfg = run["cfg"]; ex = Exact(run); wl = ex.wl; nb = wl.nb; tau = tau or _tau(run)
    end = len(run["clog"])
    live = np.where((ex.ins < end + 1) & (ex.dele[nb:] > end))[0]
    live = live[np.argsort(ex.ins[live])]                   # commit order
    if len(live) == 0:
        return {"stored_stream_items": 0}
    V = wl.X[live]
    KN = 8
    ids, dist = ex.topk(V, np.full(len(live), end, dtype=np.int64), KN)     # live neighbours only (deleted items are excluded)
    own = ids == (nb + live)[:, None]
    dist = np.where(own, np.inf, dist)
    nn = dist.min(axis=1)                                                    # nearest OTHER live item
    pol = Policy((cfg.get("protocol") or {}).get("policy", "none"), int(cfg.get("k", 10)))
    c = getattr(pol, "c", 1.0) if pol.name in ("cadm", "cert") else 1.0
    # redundancy: walk the stored stream items in commit order and keep one iff no base item and no kept item is within tau
    rank_of = {int(nb + j): a for a, j in enumerate(live)}
    kept = np.zeros(len(live), dtype=bool); redundant = 0
    for a in range(len(live)):
        near = False
        for i, d in zip(ids[a], dist[a]):
            if d >= tau:
                continue
            if i < nb or kept[rank_of[int(i)]]:
                near = True; break
        if near:
            redundant += 1
        else:
            kept[a] = True
    return {"stored_stream_items": int(len(live)), "tau": round(tau, 4), "c": c,
            "min_pairwise_distance": round(float(nn.min()), 4), "nn_distance_p1": _pct(nn, 1),
            "items_within_tau_of_another": int((nn < tau - 1e-9).sum()), "items_within_tau_over_c": int((nn < tau / c - 1e-9).sum()),
            "separation_proposition": {"bound": round(tau / c, 4), "violations": int((nn < tau / c - 1e-6).sum()),
                                       "note": "holds for exact validation reads; with approximate reads the bound is tau / (c * rho)"},
            "redundant_items": int(redundant), "redundant_per_100_tasks": round(100 * redundant / wl.n, 2)}


# ----------------------------------------------------------------------------- serial reference execution
def serial_reference(run, tau=None, limit=20000):
    """Replays the tasks one at a time, in commit order, with exact search and the threshold rule."""
    from .deciders import threshold_rule
    cfg, recs = run["cfg"], run["recs"]; k = int(cfg.get("k", 10))
    ex = Exact(run); wl = ex.wl; nb = wl.nb; tau = tau or _tau(run)
    if (cfg.get("decider") or {}).get("kind", "threshold") != "threshold":
        return {"error": "serial reference is defined for the threshold decider"}
    order = sorted((r for r in recs if not r["gave_up"]), key=lambda r: (r["s"], len(r["ops"] or []) > 0, r["j"]))[:limit]
    Q = np.stack([wl.X[r["j"]] for r in order])
    Sb, Ib = ex.ix.search(np.ascontiguousarray(Q), 1)
    bd = np.sqrt(np.maximum(2 - 2 * Sb[:, 0], 0)) if ex.metric == "cosine" else np.sqrt(np.maximum(Sb[:, 0], 0))
    base_dead = set(); live = []                             # live: indices into `order` of inserted items
    alive = np.zeros(len(order), dtype=bool); diff = 0; acts = {}; same_store = 0
    for a, r in enumerate(order):
        cand = []
        if int(Ib[a, 0]) not in base_dead:
            cand.append((float(bd[a]), int(Ib[a, 0])))
        elif base_dead:                                      # nearest base item was deleted: exact scan of the base
            d_all = ex._dist(Q[a:a + 1], wl.base)[0]; d_all[list(base_dead)] = np.inf
            i = int(np.argmin(d_all)); cand.append((float(d_all[i]), i))
        idx = np.where(alive[:a])[0]
        if len(idx):
            d = ex._dist(Q[a:a + 1], Q[idx])[0]; i = int(np.argmin(d)); cand.append((float(d[i]), nb + order[idx[i]]["j"]))
        cand.sort()
        dec = threshold_rule(r["op"], [(cand[0][1], cand[0][0])] if cand else [], tau)
        acts[dec.action] = acts.get(dec.action, 0) + 1
        if dec.action in ("UPDATE", "DELETE"):
            t = dec.target
            if t < nb:
                base_dead.add(t)
            else:
                for b in idx:
                    if order[b]["j"] == t - nb:
                        alive[b] = False
        if dec.action in ("ADD", "UPDATE"):
            alive[a] = True
        diff += dec.action != (r["dec"][0] if r["dec"] else None)
    stored_run = {i for _, op, i, _ in run["clog"] if op == "i"} - {i for _, op, i, _ in run["clog"] if op == "d"}
    stored_ref = {nb + order[a]["j"] for a in np.where(alive)[0]}
    out = {"tasks": len(order), "actions_serial_exact": acts, "tasks_with_different_action": int(diff),
           "different_action_per_100": round(100 * diff / len(order), 2),
           "stored_run": len(stored_run), "stored_serial_exact": len(stored_ref),
           "stored_only_in_run": len(stored_run - stored_ref), "stored_only_in_serial": len(stored_ref - stored_run)}
    ne = (run.get("summary") or {}).get("events")
    if ne:
        # Rows the run stores beyond the serial execution of the same tasks, per 100 facts.  For planted paraphrases this
        # equals dup_per_100_events; for real duplicate clusters (members can be farther apart than tau, so a serial
        # execution also stores several) it is the part of the duplicates that the race causes.
        out["excess_stored_per_100_events"] = round(100.0 * (len(stored_run) - len(stored_ref)) / ne, 2)
    return out


# ----------------------------------------------------------------------------- conflict-law data set
LAG_MAX = 4096


def law_extract(run, cs=CS, name=None):
    """Data set for the conflict law, from the first read of every task of an instrumented run.

    For task j let sr / s be the commit-log sizes at its read and at its first validation.  Everything is
    exact-to-exact (index error plays no role).  Returned dict (also the content of <run>.law.pkl):

      name, k, n, nb, agents, tau, cs           scalars / grid
      team   [n]            team of every task;  workload: the workload section;  meta: workload meta (planted parameters)
      ro     [n, k+1]       exact neighbour distances at the read (old state), metric distance
      rn     [n, k+1]       exact neighbour distances at the first validation (new state)
      rho    [n]            max_i ro_i / rn_i : the stale exact answer is c-admissible  <=>  rho <= c
      nc     [n]            insertions committed in [sr, s)                (overlapping commits)
      N      [n]            live items at the read
      m      [n]            insertions among them that fall inside the old top-k ball
      ev     [E, 10]        one row per in-ball concurrent insertion:
                            task, m of the task, old rank p (1-based), t, r_p, r_1, r_k, lag, same_team, source task
                            (lag = |source - task| in stream order)
      lid    [n]            maximum-likelihood local intrinsic dimension from ro[:, :k]
      loo_suf [n, k]        suffix maxima of r_{i+1} / r_i : an insertion exchangeable with the neighbours that lands
                            just before old rank i conflicts at c  <=>  loo_suf[:, i-1] > c
      conc_same [LAG_MAX+1] number of overlapping commits from the same team at each lag (index LAG_MAX = that or more)
      conc_other            number of overlapping commits from other teams
      char_ok, char_tot     characterization theorem on single-insertion reads: predicted vs observed, over the c grid
      task   [n]            stream task id of every row (tasks without first-validation data are left out);
                            ev[:, 0] is a ROW index into these arrays, ev[:, 9] is a stream task id
      seed, workload_seed   seeds of the run (the workload can be rebuilt with make_workload(workload, workload_seed, k))
    """
    cfg, recs = run["cfg"], run["recs"]; k = int(cfg.get("k", 10))
    ex = Exact(run); wl = ex.wl; nb = wl.nb
    use = [r for r in recs if r["first"] is not None]
    if not use:
        raise SystemExit("law: the run has no first-validation data")
    if any(op == "d" for _, op, _, _ in run["clog"]):
        print("law: warning: the run contains deletions; only insertions are analysed", file=sys.stderr)
    js = np.array([r["j"] for r in use]); n = len(use)
    s = np.array([r["first"][2] for r in use]); sr = np.minimum(np.array([r["first"][3] for r in use]), s)
    Q = wl.X[js]
    _, ro = ex.topk(Q, sr, k + 1); _, rn = ex.topk(Q, s, k + 1)
    rho = np.max(np.maximum(ro[:, :k], 1e-6) / np.maximum(rn[:, :k], 1e-6), axis=1)
    lo = np.searchsorted(ex.ins_pos, sr, side="left"); hi = np.searchsorted(ex.ins_pos, s, side="left")
    nc = hi - lo
    live_at = np.array([nb - int((ex.dele[:nb] < x).sum()) for x in sr]) if ex.base_deleted else np.full(n, nb)
    N = live_at + lo
    m = np.zeros(n, dtype=int); ev = []
    conc_same = np.zeros(LAG_MAX + 1, dtype=np.int64); conc_other = 0
    team = wl.team
    for x in range(n):
        if nc[x] == 0:
            continue
        src = ex.ins_task[lo[x]:hi[x]]; src = src[src != js[x]]
        if len(src) == 0:
            continue
        same = team[src] == team[js[x]]; lag = np.abs(src - js[x])
        np.add.at(conc_same, np.minimum(lag[same], LAG_MAX), 1); conc_other += int((~same).sum())
        t = ex._dist(Q[x:x + 1], wl.X[src])[0]
        inb = t < ro[x, k - 1]
        m[x] = int(inb.sum())
        for a in np.where(inb)[0]:
            p = 1 + int(np.searchsorted(ro[x, :k], t[a]))
            ev.append((x, m[x], p, float(t[a]), float(ro[x, p - 1]), float(ro[x, 0]), float(ro[x, k - 1]), int(lag[a]), int(same[a]), int(src[a])))
    ev = np.array(ev, dtype=float).reshape(-1, 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        lid = -1.0 / np.mean(np.log(np.maximum(ro[:, :k - 1], 1e-9) / ro[:, k - 1:k]), axis=1)
    ratio = ro[:, 1:k + 1] / np.maximum(ro[:, :k], 1e-9)
    suf = np.maximum.accumulate(ratio[:, ::-1], axis=1)[:, ::-1]
    ok = tot = 0
    for row in ev[ev[:, 1] == 1] if len(ev) else []:
        x = int(row[0])
        for c in cs:
            if c <= 1:
                continue
            ok += insertion_conflicts(list(ro[x, :k]), float(row[3]), c) == bool(rho[x] > c + 1e-9); tot += 1
    return {"name": name or cfg.get("name", "run"), "k": k, "n": n, "nb": nb, "agents": wl.n_clients(), "tau": _tau(run), "cs": list(cs),
            "seed": int(cfg.get("seed", 1)), "workload_seed": int(cfg.get("workload_seed", cfg.get("seed", 1))),
            "team": team[js], "task": js, "workload": cfg["workload"], "meta": {a: b for a, b in wl.meta.items() if a != "t_plant"},
            "ro": ro, "rn": rn, "rho": rho, "nc": nc, "N": N, "m": m, "ev": ev, "lid": lid, "loo_suf": suf,
            "conc_same": conc_same, "conc_other": conc_other, "char_ok": int(ok), "char_tot": int(tot)}


def law_brief(L):
    k = L["k"]; ev = L["ev"]
    single = L["m"] == 1
    return {"name": L["name"], "k": k, "reads": L["n"], "reads_with_in_ball_insertion_%": round(100 * float(np.mean(L["m"] > 0)), 2),
            "in_ball_insertions_per_read": round(float(L["m"].mean()), 4), "single_insertion_reads": int(single.sum()), "events": int(len(ev)),
            "expected_background_per_read": round(float(np.mean(L["nc"] * k / (L["N"] + 1.0))), 4),
            "median_lid": round(float(np.median(L["lid"][np.isfinite(L["lid"])])), 2),
            "new_nearest_share_%": round(100 * float(np.mean(ev[:, 2] == 1)), 1) if len(ev) else None,
            "characterization": f'{L["char_ok"]}/{L["char_tot"]}',
            "conflict_%_single_insertion": {str(c): round(100 * float(np.mean(L["rho"][single] > c)), 2) for c in L["cs"] if c > 1} if single.any() else None,
            "inadmissible_%_all_reads": {str(c): round(100 * float(np.mean(L["rho"] > c)), 2) for c in L["cs"] if c > 1}}


# ----------------------------------------------------------------------------- LLM replay
def replay_llm(run, sample=None, seed=0):
    """Decision sensitivity with a real model: decide again on the fresh answer and on the exact answer.
    sample: tasks replayed per run (default 500, or the environment variable ARIADNE_REPLAY_SAMPLE)."""
    if sample is None:
        sample = int(os.environ.get("ARIADNE_REPLAY_SAMPLE", "500"))
    from .deciders import make_decider
    cfg, recs = run["cfg"], run["recs"]; k = int(cfg.get("k", 10))
    ex = Exact(run); wl = ex.wl; tau = _tau(run)
    dcfg = dict(cfg.get("decider") or {}); dcfg["think"] = "none"; dcfg["replay_latency"] = False
    if dcfg.get("kind") != "llm":
        return {"error": "replay needs a run with the llm decider"}
    dec = make_decider(dcfg, wl, tau)
    use = [r for r in recs if r["R"] is not None and not r["gave_up"] and r["op"] != "read"]
    rng = np.random.default_rng(seed)
    use = [use[i] for i in rng.permutation(len(use))[:sample]]
    Q = np.stack([wl.X[r["j"]] for r in use]); Ei, Ed = ex.topk(Q, np.array([r["s"] for r in use]), k)
    d_rf = d_re = d_fe = nF = 0; by = {}
    for x, r in enumerate(use):
        E = _ans(Ei[x], Ed[x], k)
        kR = dec.decide(r["j"], [r["R"][0]] + r["R"][1:])[0].key()
        kE = dec.decide(r["j"], [E] + r["R"][1:])[0].key()
        d_re += kR != kE
        by.setdefault(kR[0], [0, 0]); by[kR[0]][0] += 1; by[kR[0]][1] += kR != kE
        if r["F"]:
            kF = dec.decide(r["j"], [r["F"][0]] + r["F"][1:])[0].key(); nF += 1
            d_rf += kR != kF; d_fe += kF != kE
    n = len(use)
    return {"sampled": n, "decision_changed_committed_vs_exact_%": round(100 * d_re / n, 2),
            "decision_changed_committed_vs_fresh_%": round(100 * d_rf / nF, 2) if nF else None,
            "decision_changed_fresh_vs_exact_% (index floor)": round(100 * d_fe / nF, 2) if nF else None,
            "by_committed_action": {a: {"n": v[0], "changed_%": round(100 * v[1] / v[0], 2)} for a, v in by.items()}, "decider": dec.stats()}


CMDS = {"run": audit_run, "first": first_validation, "sep": separation, "serial": serial_reference, "replay": replay_llm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS) + ["law"]); ap.add_argument("runs", nargs="+")
    ap.add_argument("--out"); ap.add_argument("--cs"); ap.add_argument("--sample", type=int, default=500); ap.add_argument("--tau-cos", type=float)
    a = ap.parse_args()
    cs = tuple(float(x) for x in a.cs.split(",")) if a.cs else CS
    for path in a.runs:
        if path.endswith(".law.pkl"):                       # law files live next to the run files: not a run
            continue
        run = load_run(path); tau = _tau(run, a)
        if a.cmd == "law":
            L = law_extract(run, cs, name=os.path.basename(path)[:-4])
            with open(path[:-4] + ".law.pkl", "wb") as f:
                pickle.dump(L, f, protocol=4)
            res = law_brief(L)
        elif a.cmd == "replay":
            res = replay_llm(run, a.sample)
        elif a.cmd in ("run", "first"):
            res = CMDS[a.cmd](run, cs, tau=tau) if a.cmd == "run" else first_validation(run, cs, tau=tau)
        else:
            res = CMDS[a.cmd](run, tau=tau)
        arr = res.pop("_arrays", None)
        if arr is not None:
            np.savez_compressed(path[:-4] + ".audit.npz", **arr)
        line = {"run": path, "name": run["cfg"].get("name"), "cell": run["cfg"].get("cell"), "audit": a.cmd, **res}
        print(json.dumps(line, default=float), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(line, default=float) + "\n")


if __name__ == "__main__":
    main()
