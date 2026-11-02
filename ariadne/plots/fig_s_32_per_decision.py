"""Figure S32: a tolerance per operation.

Reads     E6_per_decision_tolerance/summary.jsonl joined with audit_run.jsonl and audit_serial.jsonl  (variants: one c for
          every operation, or a loose c for additions with a strict c, exact validation, or re-deciding by rule for the
          destructive operations (replace, delete); extra_calls_per_100, div_committed_vs_exact_%.link,
          different_action_per_100).
Shows     per variant (rows: "add" = additions, "replace, delete" = the destructive operations), one bar per workload:
          (a) extra model calls per 100 tasks, (b) share of committed decisions whose
          link target (which memory is replaced) differs from the one on the exact answer, (c) tasks per 100 whose
          action differs from a serial exact execution in commit order.  Caps are intervals over seeds.  Bars take colour
          and hatch from the data set (case_bar, shared with figures S31, S33 and S35).
Paper     discussion paragraph, appendix figure or table.
"""
import re

from . import style as st, load as ld, panels as pn

EXP = "E6_per_decision_tolerance"
NAME = "fig_s_32_per_decision"


def variant_label(v):
    v = str(v)
    m = re.match(r"^all-(\d+(?:\.\d+)?)$", v)
    if m:
        return "all: $c$=%g" % float(m.group(1))
    m = re.match(r"^add-?([a-z]+|\d+(?:\.\d+)?)-destr-?([a-z]+|\d+(?:\.\d+)?)$", v)
    if m:
        f = lambda x: {"redecide": "re-decide", "exact": "exact"}.get(x, "$c$=%g" % float(x) if re.match(r"^\d", x) else x)
        return "add: %s;\nreplace, delete: %s" % (f(m.group(1)), f(m.group(2)))
    return v


HATCH = {"locomo": "", "agnews": "///", "synthetic": "...", "longmemeval": "xxx", "qqp": "\\\\\\", "msmarco": "---"}


def case_bar(cases):
    """Bar style of a workload: colour and hatch of its data set (the same in every figure); further cases of the same
    data set get lighter shades of the colour."""
    def kw(c):
        key = st.dataset_key(c)
        same = [x for x in cases if st.dataset_key(x) == key]
        k = same.index(c) if c in same else 0
        col = st.dataset_style(c)["color"]
        return dict(color=st.lighten(col, 0.4 * k / max(len(same) - 1, 1)) if k else col, edgecolor="white", linewidth=0.5,
                    hatch=HATCH.get(key, "xxx"))
    return kw


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    R = ld.join(R, ld.rows(results, EXP, "audit_serial.jsonl", suffix, quiet=True))
    if not R:
        return ld.skip(NAME, "no rows")
    variants = ld.values(R, "proto")
    variants = sorted(variants, key=lambda v: (not str(v).startswith("all"), ld.natkey(v)))
    cases = st.case_order(ld.values(R, "case"))
    metrics = (("extra_calls_per_100", "extra calls\nper 100 tasks"), ("div_committed_vs_exact_%.link", "link target\ndiffers (%)"),
               ("different_action_per_100", "action differs\nper 100 tasks"))
    fig, axes = st.subplots(1, 3, cols=1, aspect=0.20 + 0.075 * len(variants) * max(len(cases), 1.4), sharey=True)
    kw = case_bar(cases)
    for ax, (key, lab) in zip(axes, metrics):
        A = ld.agg(R, ["proto", "case"], key)
        st.grouped_bars(ax, variants, cases, lambda g, s: A.get((g, s)), kw, horizontal=True)
        ax.set_xlabel(lab, fontsize=st.SMALL); st.grid(ax, "x"); ax.tick_params(axis="y", length=0)
        ax.margins(x=0.12)
    axes[0].set_yticks(range(len(variants))); axes[0].set_yticklabels([variant_label(v) for v in variants])
    axes[0].set_ylim(len(variants) - 0.45, -0.55)
    pn.zero_floor(axes, axis="x")
    pn.letters(axes)
    from matplotlib.patches import Patch
    H = [Patch(label=st.dataset_label(c, short=True), **{("facecolor" if k == "color" else k): v for k, v in kw(c).items()}) for c in cases]
    fig.legend(handles=H, loc="outside upper center", ncol=len(cases), handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
