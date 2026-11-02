"""Figure S20: transactions with several similarity reads.

Reads     B3_multi_read/summary.jsonl  (grid: workload.extra_reads, workload.extra_source = prev | random | expand,
          policy; extra_calls_per_100, cs_mean_ms per workload and seed).
Shows     against the number of extra reads per task: top row, extra model calls per 100 tasks; bottom row, mean time
          in the critical section (every read is validated there).  One column per workload and source of the extra
          queries, one line per policy; bands are intervals over seeds.
Paper     appendix figure.
"""
from . import style as st, load as ld, panels as pn

EXP = "B3_multi_read"
NAME = "fig_s_20_multi_read"
KEY = "protocol.policy"
SOURCE = {"prev": "team history", "random": "random items", "expand": "query variants"}


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    cases = st.case_order(ld.values(R, "case"))
    srcs = [s for s in SOURCE if ld.where(R, **{"workload.extra_source": s})] or ld.values(R, "workload.extra_source")
    cols = [(c, s) for c in cases for s in srcs][:6]
    reads = sorted(ld.values(R, "workload.extra_reads"))
    pols = st.policy_order(ld.values(R, KEY))
    nc = len(cols)
    fig, axes = st.subplots(2, nc, cols=1 if nc <= 2 else 2, aspect=1.05 if nc == 1 else 0.80 if nc == 2 else 0.42, sharex=True, sharey="row", squeeze=False)
    for i, (c, s) in enumerate(cols):
        Rp = ld.where(R, case=c, **{"workload.extra_source": s})
        for j, key in enumerate(("extra_calls_per_100", "cs_mean_ms")):
            ax = axes[j][i]
            pos = pn.ordinal(ax, reads, lambda v: "%g" % v)
            pn.lines(ax, Rp, "workload.extra_reads", key, by=KEY, xpos=pos, kw_of=st.policy_line_distinct)
            st.grid(ax, "y")
        title = SOURCE.get(s, str(s)) if len(cases) == 1 else "%s,\n%s" % (st.dataset_label(c, short=True), SOURCE.get(s, str(s)))
        st.panel_label(axes[0][i], "abcdefgh"[i], title if nc > 1 or len(srcs) > 1 else st.dataset_label(c))
    pn.zero_floor([a for row in axes for a in row])
    axes[0][0].set_ylabel("extra model calls\nper 100 tasks"); axes[1][0].set_ylabel("critical section (ms)")
    fig.supxlabel("extra similarity reads per task (queries from: team history = earlier items of the team, random stored items,\n"
                  "query variants = noisy copies of the task's own vector)" if nc > 1 else "extra similarity reads per task", fontsize=st.FONT if nc == 1 else st.SMALL)
    pn.policy_legend(fig, pols, kind="line_distinct")
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
