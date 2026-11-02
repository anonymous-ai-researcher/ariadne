"""Figure S35: does the auditor catch protocols that look right but are not?

Reads     B5a_broken_variants/audit_run.jsonl joined with summary.jsonl  (variants run live: correct; no-lock = validation
          without mutual exclusion; group-lock = one commit lock per team on a shared store; rr-cs = critical section at
          REPEATABLE READ, PostgreSQL only).  dup_written_per_100: duplicate writes against exact k-NN;
          dup_written_floor_per_100: duplicate writes that the validation read itself would have produced;
          rho_fresh[">1.1"]: validation reads whose approximation ratio exceeds 1.1.
Shows     per variant, one bar per workload: (a) duplicate writes per 100 tasks (symlog axis, linear below 0.1, each bar
          labelled with its mean: two decimals below 1, one below 10), (b) fresh answers (validation reads) above ratio 1.1.  The duplicates computed from
          the validation reads (dup_written_floor_per_100: above 0 means the validation read missed writes committed
          before it, the tell-tale of a stale snapshot) get a panel of their own only when they differ from (a) in some
          row; in the current runs they equal (a) everywhere and the panel is left out.  Caps are intervals over seeds.
Paper     appendix paragraph "validating the auditor".
"""
from . import style as st, load as ld, panels as pn
from .fig_s_32_per_decision import case_bar

EXP = "B5a_broken_variants"
NAME = "fig_s_35_broken_variants"
ORDER = ["correct", "no-lock", "group-lock", "rr-cs"]
LABEL = {"correct": "correct", "no-lock": "no lock", "group-lock": "lock per team", "rr-cs": "crit. section at RR"}


def bar_label(v):
    """One rule for every bar label: 0, two decimals below 1, one decimal below 10, whole numbers above."""
    return "0" if v == 0 else "%.2f" % v if abs(v) < 1 else "%.1f" % v if abs(v) < 10 else "%.0f" % v


def symlog_values(ax, variants, cases, A, width=0.82, thresh=0.1):
    """Symlog value axis (linear below `thresh`) and the mean printed above every bar (rotated), so that values
    of a few hundredths stay readable next to values of several units."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    n = max(len(cases), 1)
    bw = width / n
    top = 0.0
    for i, g in enumerate(variants):
        for j, c in enumerate(cases):
            s = A.get((g, c))
            if s is None or not s.n or s.mean != s.mean:
                continue
            hi = s.hi if s.n > 1 and s.hi > s.lo else s.mean
            top = max(top, hi, s.mean)
            ax.annotate(bar_label(s.mean), (i - width / 2 + (j + 0.5) * bw, max(hi, s.mean)), xytext=(0, 1.2),
                        textcoords="offset points", ha="center", va="bottom", rotation=90, fontsize=st.TINY, color=st.MUTED)
    ax.set_yscale("symlog", linthresh=thresh, linscale=0.6)
    ticks = [0, 0.05, 0.1, 0.3, 1, 3, 10, 30, 100]
    ax.set_ylim(0, max(top, thresh) * 3.0)
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if t <= ax.get_ylim()[1]]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_locator(NullLocator())


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    R = [r for r in R if r.get("dup_written_per_100") is not None]
    if not R:
        return ld.skip(NAME, "no rows with the run audit")
    variants = [v for v in ORDER if ld.where(R, proto=v)] + [v for v in ld.values(R, "proto") if v not in ORDER]
    cases = st.case_order(ld.values(R, "case"))
    kw = case_bar(cases)
    same = all(ld.num(r.get("dup_written_floor_per_100")) == ld.num(r.get("dup_written_per_100")) for r in R
               if r.get("dup_written_floor_per_100") is not None)
    metrics = [("dup_written_per_100", "duplicate writes\nper 100 tasks")]
    if not same:
        metrics.append(("dup_written_floor_per_100", "duplicates from the\nvalidation read, per 100 tasks"))
    metrics.append(("rho_fresh.>1.1", "fresh answers above\nratio 1.1 (%)"))
    fig, axes = st.subplots(1, len(metrics), cols=1, aspect=0.64)
    for ax, (key, lab) in zip(axes, metrics):
        A = ld.agg(R, ["proto", "case"], key)
        small = key.startswith("dup_written")
        st.grouped_bars(ax, variants, cases, lambda g, s: A.get((g, s)), kw, zero_label=not small)
        if small:
            symlog_values(ax, variants, cases, A)
        ax.set_xticks(range(len(variants))); ax.set_xticklabels([LABEL.get(v, v) for v in variants], fontsize=st.TINY, rotation=40, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.set_ylabel(lab); st.grid(ax, "y")
    pn.zero_floor([a for a in axes if a.get_yscale() == "linear"])
    pn.letters(axes)
    from matplotlib.patches import Patch
    H = [Patch(label=st.dataset_label(c, short=True), **{("facecolor" if k == "color" else k): v for k, v in kw(c).items()}) for c in cases]
    fig.legend(handles=H, loc="outside upper center", ncol=len(cases), handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
