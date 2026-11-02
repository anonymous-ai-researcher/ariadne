"""Does the auditor catch a broken protocol?  Fault injection into run logs.

Takes the log of a correct run (protocol validate with policy cadm:c or cert:c) and plants three kinds of
faults that a buggy implementation would produce, then audits the mutated log:

  skip      the commit used the answer of the first read although validation had rejected it
            (a protocol that forgets to validate)                        -> safety-theorem violations
  reorder   the commit-time position of a task is moved back in the commit log
            (validation against a state older than the commit state)     -> safety-theorem violations
  ghost     the committed answer contains an item that is not in the store at the commit state
            (a deleted or not yet inserted item)                         -> answers with items not in state
  distance  the store misreports distances (too small by 10%)           -> misreported distances

For each fault it reports how many planted faults there are and how many violations the audit finds.
A fault class that is planted but not detected makes the command exit with status 1.

For skip the row also has, per task: flagged (planted tasks that violate the safety bound in the mutated log),
violations_elsewhere (violations on tasks that were not planted, must be 0), and for the planted tasks that do not
violate it, min_slack = min over them of c x rho_fresh - rho(held | S) (>= 0 by construction of the count; reported so
that "the others still satisfy the bound" is visible per task), and fresh_approx = how many of them have rho_fresh > 1
(the fresh answer at commit is itself approximate).

A reordered commit is an error only if the move crosses a write that matters to the committed answer.  Moving the
commit position of a task from s back to s - 200 removes the writes at positions s - 200 .. s - 1 from the state the
answer is judged against.  Removing an insertion that is not in the answer can only make the exact distances larger,
so it can only lower the approximation ratio of the answer; the move can raise the ratio only if that window holds
(a) the insertion of an item of the committed answer (the answer then holds an item that is not in the state), or
(b) the deletion of an item closer to the query than the answer's k-th item (it is back in the state and can enter the
exact top-k).  For reorder the row also has, computed from the commit log independently of the audit:
  crossing              planted moves whose window holds such a write (a relevant write) for one of the task's reads
  flagged               planted moves whose task violates the safety bound in the mutated log (per task)
  flagged_not_crossing  flagged moves that cross no relevant write (a false alarm of the auditor: must be 0)
  violations_elsewhere  violations in the mutated log on tasks that were not moved (must be 0)

    python -m ariadne.selftest results/B1/<cell>.pkl [--share 0.05]
"""
import argparse, copy, json, sys
import numpy as np
from . import audit as AU


def mutate(run, kind, share, rng):
    run = {"cfg": run["cfg"], "clog": run["clog"], "recs": copy.deepcopy(run["recs"])}
    recs = run["recs"]; planted = 0
    if kind == "skip":
        run["_skipped"] = []
        for r in recs:
            if r["first"] is not None and r["first_ok"] is False and not r["gave_up"]:
                r["R"] = r["first"][0]; planted += 1; run["_skipped"].append(r["j"])
    elif kind == "reorder":
        cand = [r for r in recs if r["R"] is not None and r["s"] > 200 and not r["gave_up"]]
        run["_moved"] = []
        for r in (cand[i] for i in rng.permutation(len(cand))[:max(1, int(share * len(cand)))]):
            run["_moved"].append((r["j"], r["s"], r["s"] - 200))
            r["s"] -= 200; planted += 1
    elif kind == "ghost":
        nb = len(AU.workload_of(run["cfg"]).base); n = len(recs)
        cand = [r for r in recs if r["R"] and r["R"][0] and not r["gave_up"]]
        for r in (cand[i] for i in rng.permutation(len(cand))[:max(1, int(share * len(cand)))]):
            late = [x for x in recs if x["ops"] and x["s"] > r["s"] and any(op == "i" for op, _ in x["ops"])]
            if not late:
                continue
            item = next(it for op, it in late[0]["ops"] if op == "i")
            A = list(r["R"][0]); A[-1] = (int(item), A[-1][1]); r["R"] = [A] + list(r["R"][1:]); planted += 1
    elif kind == "distance":
        cand = [r for r in recs if r["R"] and r["R"][0] and not r["gave_up"]]
        for r in (cand[i] for i in rng.permutation(len(cand))[:max(1, int(share * len(cand)))]):
            r["R"] = [[(i, 0.9 * d) for i, d in r["R"][0]]] + list(r["R"][1:]); planted += 1
    return run, planted


def _violating(run, audit):
    """Tasks (ids) whose committed answer violates the safety bound in an audit (the per-task form of the count)."""
    st = audit.get("safety_theorem") or {}
    c = st.get("c"); arr = audit.get("_arrays") or {}
    if c is None or "task" not in arr:
        return set()
    recs = run["recs"]
    out = set(); f = 0                                      # rho_fresh has one entry per audited task that has a fresh answer
    for j, rr in zip(arr["task"], arr["rho_committed"]):
        r = recs[int(j)]
        if not r["F"]:
            continue
        rf = arr["rho_fresh"][f]; f += 1
        if r["capped"] or r["op"] == "read":
            continue
        if not (rr <= c * rf * (1 + 1e-4) + 1e-5):
            out.add(int(j))
    return out


def _per_task(run, audit):
    """task id -> (rho_committed, rho_fresh, c) for the audited non-capped writing tasks with a fresh answer."""
    st = audit.get("safety_theorem") or {}
    c = st.get("c"); arr = audit.get("_arrays") or {}
    out = {}
    if c is None or "task" not in arr:
        return out
    recs = run["recs"]; f = 0
    for j, rr in zip(arr["task"], arr["rho_committed"]):
        r = recs[int(j)]
        if not r["F"]:
            continue
        rf = arr["rho_fresh"][f]; f += 1
        if r["capped"] or r["op"] == "read":
            continue
        out[int(j)] = (float(rr), float(rf), float(c))
    return out


def skip_relevance(clean, mutated, audit_clean, audit_mut):
    """flagged / violations_elsewhere / min_slack / fresh_approx for the planted skip faults (see the doc)."""
    sk = set(mutated.get("_skipped") or [])
    viol = _violating(mutated, audit_mut) - _violating(clean, audit_clean)
    T = _per_task(mutated, audit_mut)
    ok = [T[j] for j in sk - viol if j in T]
    slack = [c * rf * (1 + 1e-4) + 1e-5 - rr for rr, rf, c in ok]
    return {"flagged": len(viol & sk), "violations_elsewhere": len(viol - sk), "not_flagged": len(sk - viol),
            "not_flagged_audited": len(ok), "min_slack": (min(slack) if slack else None),
            "fresh_approx": sum(1 for _, rf, _ in ok if rf > 1 + 1e-9)}


def reorder_relevance(clean, mutated, audit_clean, audit_mut):
    """crossing / flagged / flagged_not_crossing / violations_elsewhere for the planted reorder moves (see the doc)."""
    moved = mutated.get("_moved") or []
    if not moved:
        return {"crossing": 0, "flagged": 0, "flagged_not_crossing": 0, "violations_elsewhere": 0}
    ex = AU.Exact(clean); wl = ex.wl
    by_pos = sorted(clean["clog"], key=lambda e: e[0])
    crossing = set()
    for j, s_old, s_new in moved:
        r = clean["recs"][j]
        window = [(op, int(item)) for pos, op, item, _ in by_pos if s_new <= pos < s_old]
        if not window:
            continue
        for ri, A in enumerate(r["R"] or []):
            if not A:
                continue
            q = wl.queries(j)[ri]
            ids = {int(i) for i, _ in A}
            da, _ = ex.recompute(q, A, s_old)
            rk = max(da) if da else 0.0
            dels = [it for op, it in window if op != "i"]
            dd = ex._dist(np.asarray(q, dtype="float32")[None, :], np.stack([ex.vec(it) for it in dels]))[0] if dels else []
            if any(op == "i" and it in ids for op, it in window) or any(float(x) < rk for x in dd):
                crossing.add(j); break
    viol = _violating(mutated, audit_mut) - _violating(clean, audit_clean)
    mv = {j for j, _, _ in moved}
    flagged = viol & mv
    return {"crossing": len(crossing), "flagged": len(flagged), "flagged_not_crossing": len(flagged - crossing),
            "violations_elsewhere": len(viol - mv)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--share", type=float, default=0.05); ap.add_argument("--out")
    a = ap.parse_args(); bad = 0
    for path in a.runs:
        run = AU.load_run(path); rng = np.random.default_rng(0)
        base = AU.audit_run(run)
        if "safety_theorem" not in base:
            print(f"{path}: the run does not use cadm:c or cert:c, skipped"); continue
        row = {"run": path, "clean": {"safety_violations": base["safety_theorem"]["violations"], "ghost": base["answers_with_items_not_in_state"],
                                      "misreported": base["answers_with_misreported_distances"]}}
        for kind, key in (("skip", "safety"), ("reorder", "safety"), ("ghost", "ghost"), ("distance", "misreported")):
            m, planted = mutate(run, kind, a.share, rng)
            r = AU.audit_run(m)
            found = {"safety": r["safety_theorem"]["violations"], "ghost": r["answers_with_items_not_in_state"],
                     "misreported": r["answers_with_misreported_distances"]}[key]
            row[kind] = {"planted": planted, "detected": int(found)}
            if kind == "reorder":
                row[kind].update(reorder_relevance(run, m, base, r))
            if kind == "skip":
                row[kind].update(skip_relevance(run, m, base, r))
            if planted > 0 and found == 0:
                bad += 1
        print(json.dumps(row), flush=True)
        if a.out:
            open(a.out, "a").write(json.dumps(row) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
