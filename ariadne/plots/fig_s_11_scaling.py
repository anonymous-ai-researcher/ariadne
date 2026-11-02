"""Figure S11: throughput and tail latency against the number of agents.

Reads     C5_latency_throughput/summary.jsonl  (tps, lat_p99_s per number of agents, protocol, think-time setting, seed).
Shows     (a) committed tasks per second and (b) 99th percentile of the task latency against the number of agents
          (closed loop), log-log, one line per protocol; bands are 95% intervals over seeds.  The thin dashed grey line in
          (a) is a slope-1 guide (linear scaling) three times above the first point of the unprotected run.
Writes    fig_s_11_scaling (thinking time 0.2 s, or the first setting if 0.2 s was not run) and, with several
          think-time settings, fig_s_11_scaling_all (one row of panels per setting).
Paper     main text, figure "throughput and tail latency vs agents" (one column); the _all variant in the appendix.
"""
from . import style as st, load as ld, panels as pn

EXP = "C5_latency_throughput"
NAME = "fig_s_11_scaling"


MAIN_THINK = "0.2s"            # the thinking time of the figure in the main text (its caption says so)


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    thinks = ld.values(R, "think") or [None]
    one = [MAIN_THINK] if MAIN_THINK in thinks else thinks[:1]
    written = draw(R, one, NAME, out)
    if len(thinks) > 1:                                       # every thinking time: appendix
        written += draw(R, thinks, NAME + "_all", out)
    return written


def draw(R, thinks, name, out):
    protos = st.policy_order(ld.values(R, "proto"))
    nrow = len(thinks)
    fig, axes = st.subplots(nrow, 2, cols=1, aspect=0.50 * nrow + 0.16, sharex=True, squeeze=False)
    for i, th in enumerate(thinks):
        Rt = ld.where(R, think=th) if th is not None else R
        ax_t, ax_l = axes[i]
        # no isolation is the reference: drawn last, on top of the lines that coincide with it (SSI short, c = 1.1)
        kw = lambda v: dict(st.policy_line(v), **({"zorder": 6} if st.family(v) == "none" else {}))
        got = pn.lines(ax_t, Rt, "workload.agents", "tps", by="proto", kw_of=kw)
        pn.lines(ax_l, Rt, "workload.agents", "lat_p99_s", by="proto", kw_of=kw)
        for ax in (ax_t, ax_l):
            ax.set_xscale("log"); ax.set_yscale("log")
            xs = sorted(ld.values(Rt, "workload.agents"))
            ax.set_xticks(xs[::2] if len(xs) > 6 else xs); ax.set_xticklabels(["%g" % x for x in (xs[::2] if len(xs) > 6 else xs)])
            ax.xaxis.set_minor_locator(st.plt.NullLocator())
            st.grid(ax, "y")
        ref = next((p for p in protos if st.family(p) == "none" and p in got), None) or (protos[0] if protos and protos[0] in got else None)
        if ref and len(got[ref][0]) > 1:
            xs, S = got[ref]
            # a slope-1 guide, lifted by a factor of 3 above the first point of the reference run into the empty upper
            # left of the panel (at the data's own level it would run under the lines that scale linearly); it stops
            # at the middle of the range, where its label sits
            lift = 3.0
            xg = [x for x in xs if x <= xs[len(xs) // 2]] or xs
            ax_t.plot(xg, [lift * S[0].mean * x / xs[0] for x in xg], color="#9A9A9A", lw=0.6, ls=(0, (3, 1.6)), zorder=0.5)
            ax_t.annotate("linear (slope 1)", (xg[-1], lift * S[0].mean * xg[-1] / xs[0]), xytext=(-2, 2), textcoords="offset points",
                          ha="right", va="bottom", fontsize=st.TINY, color=st.MUTED)
        st.log_ticks(ax_t, "y"); st.log_ticks(ax_l, "y")
        ax_t.set_ylabel("tasks per second"); ax_l.set_ylabel("p99 latency (s)")
        note = None if th is None or nrow == 1 else "think time %s" % str(th).replace("s", " s")
        st.panel_label(ax_t, "abcdefgh"[2 * i], note); st.panel_label(ax_l, "abcdefgh"[2 * i + 1], note)
    for ax in axes[-1]:
        ax.set_xlabel("agents")
    pn.policy_legend(fig, protos, label=pn.short_label)
    return st.save(fig, name, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
