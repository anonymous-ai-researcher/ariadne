"""Figure S34: anomalies by protocol (annotated heat table).

Reads     A4_anomaly_taxonomy/audit_serial.jsonl (replay of the tasks serially in commit order with exact search:
          different_action_per_100, stored_only_in_run, stored_only_in_serial), audit_run.jsonl (dup_written_per_100,
          answers_with_items_not_in_state, div_committed_vs_exact_%.link, rho_committed) and summary.jsonl
          (extra_calls_per_100).  Rows are protocols, columns are anomaly counts per 100 tasks (or %).
Shows     one table per workload: every cell is the mean over seeds, printed, and shaded by its value relative to the
          largest value of its column in that table (white = 0).  "n/a" marks a quantity the audit cannot compute for
          that protocol (the uninstrumented run logs no validation read).  The last column is the price in model calls.
Paper     table "anomalies by protocol" (main text or appendix); this figure is its visual form.
"""
from . import style as st, load as ld, panels as pn

EXP = "A4_anomaly_taxonomy"
NAME = "fig_s_34_anomalies"
ORDER = ["raw", "none", "objects", "cadm1.1-noexist", "cadm1.1", "exact", "lock_long", "lock-long"]


def per100(key):
    return lambda r: (100.0 * ld.num(r.get(key)) / max(ld.num(r.get("tasks", r.get("tasks_audited")), 1), 1)) if r.get(key) is not None else None


COLS = [("different_action_per_100", "action differs\nfrom serial"), ("dup_written_per_100", "duplicate\ninsertion"),
        (per100("stored_only_in_run"), "extra item\nstored"), (per100("stored_only_in_serial"), "item\nmissing"),
        (per100("answers_with_items_not_in_state"), "decision on a\ndeleted item"), ("div_committed_vs_exact_%.link", "nearest item\ndiffers (%)"),
        ("rho_committed.>1.1", "answer above\nratio 1.1 (%)"), ("extra_calls_per_100", "extra model\ncalls")]
NEUTRAL = {7}                                      # columns that are a cost, not an anomaly (grey shading)


def fmt(v):
    if v != v:
        return "n/a"
    if v == 0:
        return "0"
    return "%.2f" % v if abs(v) < 1 else "%.1f" % v if abs(v) < 100 else "%.0f" % v


def table(ax, R, protos, show_rows=True, show_cols=True):
    import numpy as np
    M = np.full((len(protos), len(COLS)), np.nan)
    for i, p in enumerate(protos):
        Rp = ld.where(R, proto=p)
        for j, (key, _) in enumerate(COLS):
            s = ld.stats([ld.num(ld.get(r, key)) for r in Rp])
            if s.n:
                M[i, j] = s.mean
    for j in range(len(COLS)):
        col = M[:, j]
        top = np.nanmax(col) if np.isfinite(col).any() else 0.0
        cmap = st.WARM_CMAP if j not in NEUTRAL else st.plt.get_cmap("Greys")
        for i in range(len(protos)):
            v = M[i, j]
            t = 0.0 if (v != v or top <= 0) else (v / top) * (1.0 if j not in NEUTRAL else 0.45)
            fc = cmap(t) if v == v else "#F4F4F4"
            ax.add_patch(st.plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white", linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, fmt(v), ha="center", va="center", fontsize=st.SMALL if v == v else st.TINY,
                    color=st.MUTED if (v != v or v == 0) else st.text_on(st.plt.matplotlib.colors.to_hex(fc)))
    ax.set_xlim(0, len(COLS)); ax.set_ylim(len(protos), 0)
    ax.set_xticks([j + 0.5 for j in range(len(COLS))]); ax.set_yticks([i + 0.5 for i in range(len(protos))])
    ax.set_xticklabels([c[1] for c in COLS] if show_cols else [], fontsize=st.TINY, rotation=38, ha="left", rotation_mode="anchor")
    ax.set_yticklabels([st.pretty_math(p) for p in protos] if show_rows else [])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    if len(COLS) - 1 in NEUTRAL:
        ax.plot([len(COLS) - 1, len(COLS) - 1], [0, len(protos)], color="#888888", lw=0.5)


def main(results="results", suffix="", out="figures"):
    S = ld.rows(results, EXP, suffix=suffix)
    if not S:
        return ld.skip(NAME, "no rows")
    R = ld.join(ld.join(S, ld.rows(results, EXP, "audit_serial.jsonl", suffix, quiet=True)), ld.rows(results, EXP, "audit_run.jsonl", suffix, quiet=True))
    cases = st.case_order(ld.values(R, "case"))
    protos = [p for p in ORDER if ld.where(R, proto=p)] + [p for p in ld.values(R, "proto") if p not in ORDER]
    n = len(cases)
    nr, nc = pn.grid_of(n, 2)
    h_in = (0.62 + 0.165 * len(protos)) * nr + 0.05
    w = st.COL1 if nc == 1 else st.COL2
    fig, axes = st.subplots(nr, nc, figsize=(w, h_in), squeeze=False)
    flat = [a for row in axes for a in row]
    for i, (ax, c) in enumerate(zip(flat, cases)):
        table(ax, ld.where(R, case=c), protos, show_rows=(i % nc == 0))
        if n > 1:
            ax.set_xlabel("(%s) %s" % ("abcdefgh"[i], st.dataset_label(c)), fontsize=st.FONT)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("every cell is per 100 tasks, except the two columns marked (%)", fontsize=st.SMALL, color=st.MUTED)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
