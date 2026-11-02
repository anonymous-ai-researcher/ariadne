"""Figure S31: isolation level of the read phase and of the critical section (PostgreSQL).

Reads     E5_isolation_of_phases/summary.jsonl joined with audit_run.jsonl  (grid: protocol.read_isolation,
          protocol.cs_isolation in {READ COMMITTED, REPEATABLE READ}; extra_calls_per_100; dup_written_per_100 from the
          audit).
Shows     for the four combinations: (a) duplicate writes per 100 tasks, (b) extra model calls per 100 tasks, one bar
          per workload.  The shaded combinations run the critical section at REPEATABLE READ, whose snapshot is taken
          before the commit lock is granted: the validation read misses the writes it waited for, so validation
          passes (few extra calls) and duplicates are written.  Caps are intervals over seeds.  Bars take colour and
          hatch from the data set (fig_s_32_per_decision.case_bar, as in figures S32, S33 and S35).
Paper     implementation note in the prototype section (a pitfall), appendix figure.
"""
from . import style as st, load as ld, panels as pn
from .fig_s_32_per_decision import case_bar

EXP = "E5_isolation_of_phases"
NAME = "fig_s_31_isolation_phases"
RI, CI = "protocol.read_isolation", "protocol.cs_isolation"
AB = {"READ COMMITTED": "RC", "REPEATABLE READ": "RR"}


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    for r in R:
        r["_combo"] = (r.get(CI), r.get(RI))
    combos = sorted({r["_combo"] for r in R}, key=lambda t: (t[0] != "READ COMMITTED", t[1] != "READ COMMITTED"))
    cases = st.case_order(ld.values(R, "case"))
    fig, axes = st.subplots(1, 2, cols=1, aspect=0.58)
    kw_of = case_bar(cases)
    for ax, (key, lab) in zip(axes, (("dup_written_per_100", "duplicate writes per 100 tasks"), ("extra_calls_per_100", "extra calls per 100 tasks"))):
        A = ld.agg(R, ["_combo", "case"], key)
        st.grouped_bars(ax, combos, cases, lambda g, s: A.get((g, s)), kw_of)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(["%s\n%s" % (AB.get(c[0], c[0]), AB.get(c[1], c[1])) for c in combos])
        ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(combos) - 0.4)
        bad = [i for i, c in enumerate(combos) if c[0] != "READ COMMITTED"]
        if bad:
            ax.axvspan(min(bad) - 0.5, max(bad) + 0.5, color=st.lighten(st.ACCENT, 0.88), lw=0, zorder=0)
        ax.set_ylabel(lab); st.floor0(ax, headroom=0.05 if ax is axes[0] else 0.36); st.grid(ax, "y")
        ax.annotate("critical section:\nread phase:", (0, 0), xycoords="axes fraction", xytext=(-3, -3.5), textcoords="offset points", ha="right", va="top",
                    fontsize=st.TINY, color=st.MUTED, linespacing=1.25) if ax is axes[0] else None
    if any(c[0] != "READ COMMITTED" for c in combos):
        bad = [i for i, c in enumerate(combos) if c[0] != "READ COMMITTED"]
        axes[1].text(0.5 * (min(bad) + max(bad)), 0.985, "critical section\nat RR: unsound", transform=axes[1].get_xaxis_transform(),
                     ha="center", va="top", fontsize=st.TINY, color=st.ACCENT)
    pn.letters(axes)
    from matplotlib.patches import Patch
    H = [Patch(label=st.dataset_label(c, short=True), **{("facecolor" if k == "color" else k): v for k, v in kw_of(c).items()}) for c in cases]
    fig.legend(handles=H, loc="outside upper center", ncol=len(cases), handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
