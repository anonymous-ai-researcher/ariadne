"""Figure S22: read-mostly traffic.

Reads     C13_read_mix/summary.jsonl joined with audit_run.jsonl  (grid mix = read<percent of read-only tasks>, proto:
          exact, cadm1.1, cadm1.1-ro (read-only tasks validate too), none; rho_committed from the audit).
Shows     against the share of read-only tasks: (a) extra model calls per 100 tasks, (b) throughput, (c) 99th
          percentile of the approximation ratio of the answers at their commit (or reply) position, which for
          read-only tasks is their staleness.  One line per protocol; bands are intervals over seeds.  With several
          workloads there is one row per workload.  The workload is named above the legend.
Paper     appendix figure.
"""
from . import style as st, load as ld, panels as pn

EXP = "C13_read_mix"
NAME = "fig_s_22_read_mix"
METRICS = (("extra_calls_per_100", "extra calls per 100 tasks"), ("tps", "tasks per second"), ("rho_committed.p99", "p99 approximation ratio"))


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    mixes = sorted(ld.values(R, "mix"), key=ld.number_in)
    cases = st.case_order(ld.values(R, "case"))
    protos = st.policy_order(ld.values(R, "proto"))
    kw_of = lambda p: st.policy_line(p, **(dict(mfc="white", ls=(0, (4, 1.5))) if str(p).endswith("-ro") else {}))
    fig, axes = st.subplots(len(cases), 3, cols=1, aspect=0.46 * len(cases) + 0.17, sharex=True, squeeze=False)
    for j, c in enumerate(cases):
        for i, (key, lab) in enumerate(METRICS):
            ax = axes[j][i]
            pos = pn.ordinal(ax, mixes, lambda v: "%g" % ld.number_in(v))
            pn.lines(ax, ld.where(R, case=c), "mix", key, by="proto", xpos=pos, kw_of=kw_of)
            if key != "rho_committed.p99":
                st.floor0(ax)
            ax.set_ylabel(lab + ("\n[%s]" % st.dataset_label(c, short=True) if len(cases) > 1 and i == 0 else "")); st.grid(ax, "y")
    pn.letters(axes[0])
    fig.supxlabel("read-only tasks (%)", fontsize=st.FONT)
    from matplotlib.lines import Line2D
    H = [Line2D([], [], label=st.pretty_math(p, short=False).replace(", validates reads)", ", reads validated)"), **kw_of(p)) for p in protos]
    title = "workload: " + ", ".join(st.dataset_label(c) for c in cases)
    fig.legend(handles=H, loc="outside upper center", ncol=2, handlelength=2.2, title=title, title_fontsize=st.SMALL)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
