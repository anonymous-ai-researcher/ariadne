"""Figure S17: how much search effort the validation read should get.

Reads     E4_validation_budget/summary.jsonl joined with audit_run.jsonl  (grid val = ef<n> | exact: search budget of
          the validation read, which runs inside the critical section).
Shows     against the validation budget, one line per workload: (a) mean time of the validation read (log axis), (b) share of
          committed answers whose approximation ratio exceeds c (solid) and of the validation reads themselves
          (thin dashed: the validation reads alone, which the guarantee c * rho_F inherits), (c) throughput.
          Bands are intervals over seeds.
Paper     ablation (appendix figure, or a row group of the ablation table), one column.
"""
from . import style as st, load as ld, panels as pn
from .fig_s_15_contract_vs_budget import viol

EXP = "E4_validation_budget"
NAME = "fig_s_17_validation_budget"


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    vals = ld.values(R, "val")
    vals = [v for v in vals if v != "exact"] + [v for v in vals if v == "exact"]
    cases = st.case_order(ld.values(R, "case"))
    lines = st.case_lines(cases)
    fig, axes = st.subplots(1, 3, cols=1, aspect=0.52)
    lab = lambda v: "exact" if v == "exact" else "%g" % ld.number_in(v)
    for ax, (key, ylabel) in zip(axes, (("cs_vread_mean_ms", "validation read (ms)"), (viol("rho_committed"), "answers above\nthe tolerance (%)"),
                                        ("tps", "tasks per second"))):
        pos = pn.ordinal(ax, vals, lab, rotate=40 if len(vals) > 3 else 0)
        if key == "cs_vread_mean_ms":       # the workloads moved a little apart sideways: their small reads are close
            for j, c in enumerate(cases):
                dx = (j - (len(cases) - 1) / 2) * 0.14
                pn.lines(ax, R, "val", key, by="case", xpos={k: v + dx for k, v in pos.items()}, kw_of=lambda c: lines[c],
                         order=st.case_order, only=[c])
        else:
            pn.lines(ax, R, "val", key, by="case", xpos=pos, kw_of=lambda c: lines[c], order=st.case_order)
        if not isinstance(key, str):
            pn.lines(ax, R, "val", viol("rho_fresh"), by="case", xpos=pos, band=False, order=st.case_order,
                     kw_of=lambda c: dict(lines[c], color=st.darken(lines[c]["color"], 0.45), lw=0.6, ls=(0, (2, 1.2)), marker="none", zorder=6))
            # drawn on top, darker: from ef = 40 on they lie on the committed curves
        ax.set_ylabel(ylabel)
        if key == "cs_vread_mean_ms":
            # log axis: the approximate reads (fractions of a ms) of the two workloads would lie on one another at the
            # foot of a linear axis that has to reach the exact read
            ax.set_yscale("log")
            lo_ = min([min([v for v in l.get_ydata() if v > 0] or [1.0]) for l in ax.lines] or [1.0])
            ax.set_ylim(lo_ / 1.6, ax.get_ylim()[1])
            st.log_ticks(ax, "y", steps=(1, 3))
        else:
            st.floor0(ax)
        st.grid(ax, "y")
    pn.letters(axes)
    fig.supxlabel("search budget of the validation read (ef)", fontsize=st.FONT)
    H = [st.plt.Line2D([], [], label=st.dataset_label(c, short=True), **lines[c]) for c in cases]
    H.append(st.plt.Line2D([], [], color=st.MUTED, lw=0.6, ls=(0, (2, 1.2)), label="validation reads alone"))
    fig.legend(handles=H, loc="outside upper center", ncol=3, handlelength=2.2)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
