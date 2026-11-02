"""Figure S27: Algorithm AV on every store.

Reads     F1_stores/summary.jsonl joined with audit_run.jsonl  (one case per store; proto: none, exact, cadm1.1;
          extra_calls_per_100, cs_mean_ms; dup_written_per_100 and safety_theorem.violations from the audit).
Shows     per store, one bar per protocol: (a) extra model calls per 100 tasks, (b) duplicate writes per 100 tasks
          against exact k-NN (0 for the validated protocols), (c) mean time in the critical section (log axis, one mark per
          protocol, since a bar on a log axis has no meaningful base).  The number of safety-theorem violations over all
          audited commits is printed at the top of panel (b), above the bars.  Caps are intervals over seeds.
Paper     appendix, with the portability table.
"""
from . import style as st, load as ld, panels as pn

EXP = "F1_stores"
NAME = "fig_s_27_stores"
ORDER = ["pgvector-hnsw", "pgvector-ivfflat", "qdrant", "chroma", "milvus", "lancedb", "weaviate", "faiss-hnsw", "faiss-ivf", "faiss-flat"]


def hi_of(ax):
    """Top of the bars and of their caps in an axes."""
    import numpy as np
    tops = [p.get_y() + p.get_height() for p in ax.patches]
    for c in ax.collections:
        try:
            for seg in c.get_segments():
                tops.append(float(np.max(np.asarray(seg)[:, 1])))
        except (AttributeError, ValueError, IndexError):
            pass
    return max(tops + [0.0])


def marks(ax, R, stores, protos, key):
    """One mark with its interval caps per store and protocol, dodged within the store's slot."""
    A = ld.agg(R, ["case", "proto"], key)
    n = max(len(protos), 1)
    w = 0.66
    for j, p in enumerate(protos):
        for i, s in enumerate(stores):
            v = A.get((s, p))
            if v is None or not v.n or v.mean != v.mean:
                continue
            st.point_ci(ax, i - w / 2 + (j + 0.5) * w / n, v, **st.policy_marker(p))
    ax.set_xticks(range(len(stores)))
    rot = 28 if len(stores) > 3 else 0
    ax.set_xticklabels([st.store_label(s) for s in stores], rotation=rot, ha="right" if rot else "center", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(stores) - 0.4)


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    stores = [s for s in ORDER if ld.where(R, case=s)] + [s for s in ld.values(R, "case") if s not in ORDER]
    protos = st.policy_order(ld.values(R, "proto"))
    fig, axes = st.subplots(3, 1, cols=1, aspect=1.02, sharex=True)
    for ax, (key, lab) in zip(axes, (("extra_calls_per_100", "extra calls\nper 100 tasks"), ("dup_written_per_100", "duplicate writes\nper 100 tasks"),
                                     ("cs_mean_ms", "critical\nsection (ms)"))):
        log = key == "cs_mean_ms"
        if log:
            marks(ax, R, stores, protos, key)
        else:
            pn.bars(ax, R, "case", "proto", key, groups=stores, group_label=st.store_label, zero_label=True, rotate=28 if len(stores) > 3 else 0)
        ax.set_ylabel(lab); st.grid(ax, "y")
        if log:
            ax.set_yscale("log"); ax.autoscale(enable=True, axis="y")
            lo, hi = ax.get_ylim(); ax.set_ylim(lo / 1.15, hi * 1.15)
            st.log_ticks(ax, "y", steps=(1, 2, 3, 5))
    pn.letters(axes, dy=0.98)
    # both numbers come from the audit rows the figure reads (audit_run.jsonl, joined): commits checked against the
    # safety theorem and violations found, summed over the runs that carry the check (the c-admissible runs)
    Rs = [r for r in R if ld.get(r, "safety_theorem.checked") is not None]
    chk = sum(int(ld.num(ld.get(r, "safety_theorem.checked"), 0)) for r in Rs)
    bad = sum(int(ld.num(ld.get(r, "safety_theorem.violations"), 0)) for r in Rs)
    cs_chk = sorted({ld.num(ld.get(r, "safety_theorem.c")) for r in Rs})
    if max([ld.num(r.get("dup_written_per_100"), 0) for r in R] + [0]) == 0:
        axes[1].set_ylim(0, 1.0)
    if chk:
        hi = hi_of(axes[1])
        axes[1].set_ylim(0, hi * 1.5 if hi > 0 else 1.5)
        who = ("$c$=%g, " % cs_chk[0]) if len(cs_chk) == 1 and cs_chk[0] == cs_chk[0] else ""
        axes[1].text(0.99, 0.98, "safety theorem: %s violations in %s audited commits\n(%s%d runs)" % (format(bad, ","), format(chk, ","), who, len(Rs)),
                     transform=axes[1].transAxes,
                     ha="right", va="top", fontsize=st.SMALL, color=st.MUTED)
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    H = [(b, Line2D([], [], **st.policy_marker(p))) for b, p in zip(st.legend_handles(protos, "bar"), protos)]
    fig.legend(handles=H, labels=[st.pretty_math(p) for p in protos], loc="outside upper center", ncol=len(protos), handlelength=2.4,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
