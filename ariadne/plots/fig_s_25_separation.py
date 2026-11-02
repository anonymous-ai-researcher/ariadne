"""Figure S25: the separation proposition.

Reads     B2_separation/audit_sep.jsonl  (final store of every run: min_pairwise_distance among the stored stream items,
          the duplicate threshold tau in metric distance, the tolerance c of the run).
Shows     smallest pairwise distance in the final store divided by tau, against c, one line per workload: filled marks
          for certified validation (exact validation reads), hollow marks for approximate validation (c-admissible),
          the workload's mark in the right column for no isolation.  The black curve is the bound 1 / c of the proposition
          (tau-deduplicating insertions under c-serializability keep the store (tau / c)-separated): certified runs
          must stay on or above it, and the planted workload approaches it.  Caps are 95% Student t intervals of the mean over
          the seeds (3 seeds: t with 2 degrees of freedom; see load.Stats), cut at 0 by the axis.  Every series is moved
          sideways by a small step of its own, so that coinciding series stay visible.
Paper     appendix (one-line plot), or next to the proposition.
"""
import numpy as np

from . import style as st, load as ld

EXP = "B2_separation"
NAME = "fig_s_25_separation"
KEY = "protocol.policy"


def main(results="results", suffix="", out="figures"):
    S = [r for r in ld.rows(results, EXP, "audit_sep.jsonl", suffix) if r.get("min_pairwise_distance") is not None and r.get("tau")]
    if not S:
        return ld.skip(NAME, "no separation audit")
    for r in S:
        r["_rel"] = float(r["min_pairwise_distance"]) / float(r["tau"])
    val = [r for r in S if st.family(r[KEY]) in ("cert", "cadm")]
    cs = sorted({st.spec_c(r[KEY]) for r in val}) or [1.0, 2.0]
    has_none = bool(ld.where(S, **{KEY: "none"}))
    fig, ax = st.subplots(cols=1, aspect=0.62)
    pos = st.c_axis(ax, cs, right=["no\nisolation"] if has_none else [])
    grid = np.linspace(1.0, max(cs), 200)
    pos_cs = [c for c in cs if c > 1]
    if pos_cs:
        g = 1.0 + np.logspace(np.log10(min(pos_cs) - 1.0) - 0.2, np.log10(max(pos_cs) - 1.0), 200)
        ax.plot(g - 1.0, 1.0 / g, color=st.INK, lw=1.0, zorder=2)
        if 1.0 in cs:
            ax.plot([pos(1.0), g[0] - 1.0], [1.0, 1.0 / g[0]], color=st.INK, lw=1.0, zorder=2)
        # label under the curve at the smallest tolerance above 1, where no interval reaches up to it
        c_lab = pos_cs[0]
        ax.annotate(r"bound $\tau / c$", (c_lab - 1.0, 1.0 / c_lab), xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                    fontsize=st.SMALL, color=st.INK)
    cases = st.case_order(ld.values(S, "case"))
    lines = st.case_lines(cases)
    for c in cases:                     # a "+" reads as an error-bar cap: diamond instead
        if lines[c].get("marker") == "+":
            lines[c] = dict(lines[c], marker="D", ms=st.MS * 0.85)
    ends = []
    # every series (workload x certified / approximate) is moved sideways by its own small step (log axis), so that
    # series that coincide (all certified runs at 1 for small c; fan-out certified and approximate) stay visible
    fig.canvas.draw()
    bb = ax.get_position()
    x0, x1 = ax.get_xlim()
    dec_per_pt = np.log10(x1 / x0) / (bb.width * fig.get_size_inches()[0] * 72)
    slot = lambda j, fam: j - (len(cases) - 1) / 2.0 + (1.0 if fam == "cadm" else 0.0)   # approximate: one step beside its own
    step = 10 ** (2.8 * dec_per_pt)
    for j, c in enumerate(cases):
        kw = dict(lines[c])
        for fam, hollow in (("cert", False), ("cadm", True)):
            Rf = [r for r in ld.where(S, case=c) if st.family(r[KEY]) == fam]
            xs, St = ld.series(Rf, KEY, "_rel", xnum=st.spec_c)
            if not xs:
                continue
            px = [pos(x) * step ** slot(j, fam) for x in xs]
            k2 = dict(kw, mfc="white" if hollow else kw["color"], mec=kw["color"], lw=0.6 if hollow else 1.0, ls=(0, (1, 1.2)) if hollow else kw["ls"])
            ax.plot(px, [s.mean for s in St], zorder=3, clip_on=False, **k2)   # marks near 0 stay whole
            for x, s in zip(px, St):
                if s.n > 1 and s.hi > s.lo:
                    ax.errorbar([x], [s.mean], yerr=[[s.mean - s.lo], [s.hi - s.mean]], fmt="none", ecolor=kw["color"], elinewidth=0.6, capsize=1.5)
        s = ld.stats([r["_rel"] for r in ld.where(S, case=c, **{KEY: "none"})])
        if s.n and has_none:
            # the workload's own mark (an "x" would read as the fan-out workload), dodged within the column
            k = cases.index(c) - (len(cases) - 1) / 2.0
            st.point_ci(ax, pos("no\nisolation") * 1.13 ** k, s, color=kw["color"], marker=kw.get("marker", "o"),
                        ms=kw.get("ms", st.MS), mfc=kw["color"], mec=kw["color"], mew=1.0, ls="none", clip_on=False)
    st.floor0(ax)
    ax.set_ylabel(r"smallest stored distance / $\tau$")
    st.grid(ax, "y")
    from matplotlib.lines import Line2D
    H = [Line2D([], [], label=st.dataset_label(c, short=True), **lines[c]) for c in cases]
    if any(st.family(r[KEY]) == "cadm" for r in val):
        H.append(Line2D([], [], color=st.MUTED, marker="o", mfc="white", mec=st.MUTED, lw=0.6, ls=(0, (1, 1.2)), ms=st.MS, label="approximate validation"))
    fig.legend(handles=H, loc="outside upper center", ncol=3, handlelength=2.2)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
