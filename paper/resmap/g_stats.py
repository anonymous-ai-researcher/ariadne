"""Intervals and tests that the text prints next to its 'no difference' statements.  Every key is computed from the
same rows as the statement it qualifies; the run (seed) is the unit of replication.

  STAT.cert.lo.min / .hi.max      C1 audit_run, cells A64 cadm:1.05, A64 cadm:1.1, L4 cadm:1.1: per run
                                  d = rho_committed['>c'] - rho_fresh['>c'] (paired within the run); 90 % Student t
                                  interval of the mean of d; smallest lower and largest upper end over the three cells
  STAT.cert.margin.min            smallest equivalence margin of the three cells, max(0.25, 0.25 x mean rho_fresh['>c'])
  STAT.cert.seeds                 seeds per cell (smallest over the three cells)
  STAT.D4.rel / .lo / .hi         D4a audit_law [A64, exchangeable]: per run 100 (in_ball / expected_background - 1);
                                  mean and 90 % Student t interval
  STAT.D4.seeds                   runs
  STAT.E3.lo / .hi / .p           E3 [A64, cadm:1.1, max_rounds 30]: extra calls of reread minus rethink; 95 % Welch
                                  interval of the difference of the means and the two-sided Welch p value
  STAT.B4.excess.max / .hi.max    B4 audit_serial: per (case, operation mix), mean different_action_per_100 of cadm1.1
                                  minus that of exact; largest difference and largest one-sided 95 % Welch upper bound
"""
import math
import statistics as st

from scipy import stats

C1, D4A, E3, B4 = "C1_calls_vs_policy", "D4a_exchangeable_controls", "E3_retry_strategy", "B4_deletes_updates"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _tci(v, conf):
    n = len(v)
    m = st.mean(v)
    if n < 2:
        return m, m, m
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * st.stdev(v) / math.sqrt(n)
    return m, m - h, m + h


def _welch(a, b, conf=0.95, one_sided=False):
    """difference of means a - b, its interval (two-sided, or the one-sided upper bound) and the two-sided p value"""
    d = st.mean(a) - st.mean(b)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, d, 1.0 if d == 0 else 0.0
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va else 0) + (vb ** 2 / (len(b) - 1) if vb else 0))
    q = stats.t.ppf(conf if one_sided else 0.5 + conf / 2, df)
    p = 2 * stats.t.sf(abs(d) / se, df)
    return d, d - q * se, d + q * se, p


def resolve(R):
    # ---- certified contract: committed against fresh shares above c, paired within each run
    lo, hi, mg, ns, cells = [], [], [], [], []
    for case, c, pol in (("A64", "1.05", "cadm:1.05"), ("A64", "1.1", "cadm:1.1"), ("L4", "1.1", "cadm:1.1")):
        rows = R.sel(C1, {"case": case, "protocol.policy": pol}, "audit_run.jsonl")
        pr = [(R.get(r, "rho_committed.>" + c), R.get(r, "rho_fresh.>" + c)) for r in rows]
        pr = [(a, b) for a, b in pr if _num(a) and _num(b)]
        if len(pr) < 2:
            continue
        m, l, h = _tci([a - b for a, b in pr], 0.90)
        lo.append(l); hi.append(h); ns.append(len(pr)); cells.append("%s %s" % (case, pol))
        mg.append(max(0.25, 0.25 * st.mean(b for _, b in pr)))
    src = "C1 audit_run, %s: 90%% t interval of the paired difference of the shares above c" % ", ".join(cells)
    R.put("STAT.cert.lo.min", min(lo) if lo else None, src + "; smallest lower end (points)")
    R.put("STAT.cert.hi.max", max(hi) if hi else None, src + "; largest upper end (points)")
    R.put("STAT.cert.margin.min", min(mg) if mg else None, src + "; smallest margin max(0.25, 0.25 x mean fresh share)")
    R.put("STAT.cert.seeds", min(ns) if ns else None, src + "; runs per cell (smallest)")

    # ---- exchangeable split against the background rate (A64)
    rows = R.sel(D4A, {"case": "A64", "order": "exchangeable"}, "audit_law.jsonl")
    rel = [100.0 * (r["in_ball_insertions_per_read"] / r["expected_background_per_read"] - 1.0) for r in rows
           if _num(r.get("in_ball_insertions_per_read")) and _num(r.get("expected_background_per_read")) and r["expected_background_per_read"]]
    src = "D4a audit_law [A64, exchangeable]: per run 100 (in_ball / expected_background - 1), %d runs" % len(rel)
    if len(rel) >= 2:
        m, l, h = _tci(rel, 0.90)
        R.put("STAT.D4.rel", m, src + ": mean"); R.put("STAT.D4.lo", l, src + ": 90% t lower end"); R.put("STAT.D4.hi", h, src + ": 90% t upper end")
    else:
        for k in ("STAT.D4.rel", "STAT.D4.lo", "STAT.D4.hi"):
            R.put(k, None, src + " -> fewer than two runs")
    R.put("STAT.D4.seeds", len(rel) or None, src)

    # ---- retry strategy: reread against rethink
    cell = {"case": "A64", "protocol.policy": "cadm:1.1", "protocol.max_rounds": 30}
    a = R.vals(R.sel(E3, dict(cell, **{"protocol.on_invalid": "reread"})), "extra_calls_per_100")
    b = R.vals(R.sel(E3, dict(cell, **{"protocol.on_invalid": "rethink"})), "extra_calls_per_100")
    src = "E3 [A64, cadm:1.1, max_rounds 30]: extra calls reread (%d runs) minus rethink (%d runs), Welch" % (len(a), len(b))
    if len(a) >= 2 and len(b) >= 2:
        d, l, h, p = _welch(a, b)
        R.put("STAT.E3.lo", l, src + ": 95% lower end"); R.put("STAT.E3.hi", h, src + ": 95% upper end"); R.put("STAT.E3.p", p, src + ": two-sided p")
    else:
        for k in ("STAT.E3.lo", "STAT.E3.hi", "STAT.E3.p"):
            R.put(k, None, src + " -> too few runs")

    # ---- deletions and replacements: actions that differ from a serial execution, c = 1.1 against re-execution
    rows = R.sel(B4, None, "audit_serial.jsonl")
    grp = {}
    for r in rows:
        k = (r.get("case"), r.get("workload.ops"))
        if r.get("proto") in ("cadm1.1", "exact") and _num(r.get("different_action_per_100")):
            grp.setdefault(k, {}).setdefault(r["proto"], []).append(r["different_action_per_100"])
    ex, hb = [], []
    for k, g in grp.items():
        if len(g.get("cadm1.1", [])) >= 2 and len(g.get("exact", [])) >= 2:
            d, _, h, _ = _welch(g["cadm1.1"], g["exact"], one_sided=True)
            ex.append(d); hb.append(h)
    src = "B4 audit_serial, %d (case, operation mix) cells: different_action_per_100 of cadm1.1 minus exact" % len(ex)
    R.put("STAT.B4.excess.max", max(ex) if ex else None, src + "; largest difference of the means")
    R.put("STAT.B4.hi.max", max(hb) if hb else None, src + "; largest one-sided 95% Welch upper bound")
