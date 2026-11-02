"""Figure S33: which part of the c-admissibility predicate matters (ablation of the validation predicate).

Reads     E1_predicate_components/summary.jsonl joined with audit_run.jsonl  (policies: exact, set, overlap, top-j, the
          single-rank distance tests radius / rank1 / sumdist, c-admissible, certified; extra_calls_per_100,
          dup_written_per_100, rho_committed[">c"], div_committed_vs_exact_%.link).
Shows     per predicate (rows, grouped by family), one bar per workload: (a) extra model calls per 100 tasks, (b)
          duplicate writes per 100 tasks, (c) share of committed answers whose approximation ratio exceeds the tolerance
          (1.1 for the tests that have no c), (d) share of committed decisions whose link target differs from the one on
          the exact answer.  The per-rank test is the cheapest one that keeps (b) and (c) at the index floor.  Caps are
          intervals over seeds.  Bars take colour and hatch from the data set (case_bar, as in figures S31, S32, S35).
Paper     ablation: main text table, this figure in the appendix (two columns).
"""
from . import style as st, load as ld, panels as pn
from .fig_s_32_per_decision import case_bar

EXP = "E1_predicate_components"
NAME = "fig_s_33_predicates"
KEY = "protocol.policy"


def above(r):
    d = r.get("rho_committed") or {}
    c = st.spec_c(r.get(KEY)) or 1.1
    return d.get(">%s" % c, d.get(">1.1"))


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    R = [r for r in R if r.get(KEY) is not None]
    if not R:
        return ld.skip(NAME, "no rows")
    pols = st.policy_order(ld.values(R, KEY))
    cases = st.case_order(ld.values(R, "case"))
    metrics = (("extra_calls_per_100", "extra calls per 100 tasks"), ("dup_written_per_100", "duplicate writes per 100 tasks"),
               (above, "answers above the tolerance (%)"), ("div_committed_vs_exact_%.link", "link target differs (%)"))
    wide = len(pols) > 6
    fig, axes = st.subplots(1, 4, cols=2, aspect=0.10 + 0.021 * len(pols) * max(len(cases), 1.5), sharey=True)
    kw = case_bar(cases)
    for ax, (key, lab) in zip(axes, metrics):
        A = ld.agg(R, [KEY, "case"], key)
        st.grouped_bars(ax, pols, cases, lambda g, s: A.get((g, s)), kw, horizontal=True)
        ax.set_xlabel(lab); st.grid(ax, "x"); ax.tick_params(axis="y", length=0)
        ax.margins(x=0.10)
        fam = [st.family(p) for p in pols]
        for i in range(1, len(pols)):
            if fam[i] != fam[i - 1]:
                ax.axhline(i - 0.5, color=st.LIGHT, lw=0.5, zorder=0.5)
    axes[0].set_yticks(range(len(pols))); axes[0].set_yticklabels([st.pretty_math(p) for p in pols])
    for lab, p in zip(axes[0].get_yticklabels(), pols):
        if st.family(p) == "cadm":
            lab.set_fontweight("bold")
    axes[0].set_ylim(len(pols) - 0.45, -0.55)
    pn.zero_floor(axes, axis="x")
    pn.letters(axes)
    from matplotlib.patches import Patch
    H = [Patch(label=st.dataset_label(c, short=True), **{("facecolor" if k == "color" else k): v for k, v in kw(c).items()}) for c in cases]
    fig.legend(handles=H, loc="outside upper center", ncol=len(cases), handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
