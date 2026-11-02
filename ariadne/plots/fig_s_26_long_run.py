"""Figure S26: stability over a long run (time series in windows of tasks).

Reads     F5_long_run/summary.jsonl and the per-task logs F5_long_run/<cell>.pkl (ariadne.audit.load_run: for every
          task its number of rounds, whether its first validation passed, the time of its validation read, its latency
          and its commit time).
Shows     in consecutive windows of tasks (5000 tasks per window, or a tenth of the run for shorter runs), in commit
          order: (a) extra model calls per 100 tasks, (b) share of tasks whose first validation failed (the conflict
          probability), (c) mean time of the validation read, (d) 99th percentile of the task latency.  One line per
          policy, one row per workload; seeds are averaged per window and the band is their interval.
Paper     appendix figure (drift between the first and the last window).
"""
import numpy as np

from . import style as st, load as ld, panels as pn

EXP = "F5_long_run"
NAME = "fig_s_26_long_run"
KEY = "protocol.policy"
METRICS = (("extra", "extra calls\nper 100 tasks"), ("fail", "first validation\nfailed (%)"), ("vread", "validation\nread (ms)"), ("p99", "p99 latency (s)"))


def windows(path, width=None):
    """Per-window statistics of one run: dict metric -> array, plus the window centres (in tasks)."""
    from ariadne.audit import load_run
    recs = [r for r in load_run(path)["recs"] if r.get("t1") is not None and not r.get("gave_up")]
    recs.sort(key=lambda r: r["t1"])
    n = len(recs)
    if n < 20:
        return None
    w = width or (5000 if n >= 20000 else max(n // 10, 10))
    k = n // w
    out = {m: [] for m, _ in METRICS}
    for i in range(k):
        W = recs[i * w:(i + 1) * w]
        out["extra"].append(100.0 * (np.mean([r["rounds"] for r in W]) - 1.0))
        ok = [r.get("first_ok") for r in W if r.get("first_ok") is not None]
        out["fail"].append(100.0 * (1.0 - np.mean(ok)) if ok else np.nan)
        tv = [r["t_vread"] for r in W if r.get("t_vread") is not None]
        out["vread"].append(1000.0 * np.mean(tv) if tv else np.nan)
        out["p99"].append(float(np.percentile([r["lat"] for r in W], 99)))
    return {m: np.array(v) for m, v in out.items()}, (np.arange(k) + 0.5) * w


def main(results="results", suffix="", out="figures"):
    R = [r for r in ld.rows(results, EXP, suffix=suffix) if r.get(KEY) is not None]
    if not R:
        return ld.skip(NAME, "no rows")
    cases = st.case_order(ld.values(R, "case"))
    pols = st.policy_order(ld.values(R, KEY))
    data = {}
    for r in R:
        p = ld.cell_pkl(results, EXP, r, suffix)
        if not p:
            continue
        try:
            w = windows(p)
        except Exception as e:
            ld.note("%s: could not read %s (%r)" % (EXP, p, e)); continue
        if w:
            data.setdefault((r["case"], r[KEY]), []).append(w)
    if not data:
        return ld.skip(NAME, "no per-task logs (<cell>.pkl)")
    nr, nc = len(cases), len(METRICS)
    fig, axes = st.subplots(nr, nc, cols=2, aspect=0.20 * nr + 0.07, sharex="row", squeeze=False)
    for j, c in enumerate(cases):
        for i, (m, lab) in enumerate(METRICS):
            ax = axes[j][i]
            for p in pols:
                runs = data.get((c, p))
                if not runs:
                    continue
                k = min(len(w[0][m]) for w in runs)
                x = runs[0][1][:k]
                S = [ld.stats([w[0][m][t] for w in runs]) for t in range(k)]
                if all(s.n == 0 for s in S):
                    continue
                st.line_ci(ax, x / 1000.0, S, **st.policy_line(p, ms=2.4))
            st.floor0(ax); st.grid(ax, "y")
            if j == 0:
                st.panel_label(ax, "abcd"[i], lab.replace("\n", " "))
            if i == 0:
                ax.set_ylabel(st.dataset_label(c, short=True) if nr > 1 else "")
            if j == nr - 1:
                ax.set_xlabel("tasks committed (thousands)")
    pn.policy_legend(fig, pols)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
