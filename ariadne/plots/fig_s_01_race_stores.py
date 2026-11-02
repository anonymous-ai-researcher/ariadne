"""Figure S1: the check-then-insert race on every store.

Reads     A1_race_stores/summary.jsonl  (dup_per_100_events per store, protocol, target concurrency, seed).
Shows     grouped bars of duplicate rows per 100 facts: one group per store, one bar per protocol (no isolation,
          re-read before the write, re-read under a lock), one panel per concurrency level; caps are 95%
          intervals over seeds.  A correct execution stores one row per fact, so the right bar of every group is 0.
Paper     main text, next to the table "the race on real systems" (the per-concurrency panels can move to the
          appendix: with one concurrency level the figure is one column wide).
"""
from . import style as st, load as ld

EXP = "A1_race_stores"
NAME = "fig_s_01_race_stores"
STORE_ORDER = ["pgvector-rc", "pgvector-rr", "qdrant", "chroma", "milvus", "lancedb", "weaviate", "faiss"]


def small_value_labels(ax, groups, series, A, below=5.0, width=0.82):
    """Value of every bar lower than `below` (unreadable on the 0-100 axis), written vertically above its cap.
    Neighbouring bars are narrower than a label, so a label that would touch the label of the bar to its right
    starts above it.  Call it after the y limits are final."""
    fig = ax.figure
    fig.canvas.draw()
    px = 72.0 / fig.dpi
    bw = width / max(len(series), 1)
    fs = st.TINY
    for i, g in enumerate(groups):
        floor = None                                  # top (points) of the label placed to the right
        for j in range(len(series) - 1, -1, -1):
            s = A.get((g, series[j]))
            if s is None or s.n == 0 or s.mean != s.mean or s.mean >= below:
                floor = None
                continue
            txt = "0" if s.mean == 0 else ("%.1f" % s.mean if s.mean >= 0.05 else "%.2f" % s.mean)
            top = s.hi if (s.n > 1 and s.hi > s.lo) else s.mean
            x = i - width / 2 + (j + 0.5) * bw
            y_pt = ax.transData.transform((x, top))[1] * px + 1.0
            base = max(y_pt, floor + 0.8) if floor is not None else y_pt
            ax.annotate(txt, (x, top), xytext=(0, base - y_pt + 1.0), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=fs, color=st.MUTED, annotation_clip=False)
            floor = base + len(txt.replace(".", "")) * fs * 0.56 + fs * 0.28 * txt.count(".")


def draw(ax, R, protos, stores):
    A = ld.agg(R, ["case", "proto"], "dup_per_100_events")
    st.grouped_bars(ax, stores, protos, lambda g, s: A.get((g, s)), st.policy_bar, zero_label=False)
    ax.set_xticks(range(len(stores)))
    ax.set_xticklabels([st.store_label(s) for s in stores], rotation=28 if len(stores) > 4 else 0,
                       ha="right" if len(stores) > 4 else "center", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(stores) - 0.4)
    st.grid(ax, "y")
    return A


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    R = [r for r in R if r.get("dup_per_100_events") is not None]
    if not R:
        return ld.skip(NAME, "no rows with dup_per_100_events")
    protos = st.policy_order(ld.values(R, "proto"))
    stores = [s for s in STORE_ORDER if ld.where(R, case=s)] + [s for s in ld.values(R, "case") if s not in STORE_ORDER]
    concs = ld.values(R, "workload.conc") or [None]
    n = len(concs)
    fig, axes = st.subplots(1, n, cols=1 if n == 1 else 2, aspect=0.60 if n == 1 else 0.30, sharey=True, squeeze=False)
    aggs = []
    for i, (ax, c) in enumerate(zip(axes[0], concs)):
        aggs.append(draw(ax, R if c is None else ld.where(R, **{"workload.conc": c}), protos, stores))
        if n > 1:
            st.panel_label(ax, "abcdefgh"[i], "%s concurrent agents" % c)
    axes[0][0].set_ylabel("duplicate rows per 100 facts")
    st.floor0(axes[0][0])
    fig.legend(handles=st.legend_handles(protos, "bar"), loc="outside upper center", ncol=len(protos), handlelength=1.4)
    for ax, A in zip(axes[0], aggs):
        small_value_labels(ax, stores, protos, A)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
