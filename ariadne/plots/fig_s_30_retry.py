"""Figure S30: what to do after a rejected validation.

Reads     E3_retry_strategy/summary.jsonl  (grid: protocol.on_invalid = rethink (think again on the validation answer) |
          reread (read again outside the lock) | abort (give up), protocol.max_rounds, policy; extra_calls_per_100,
          capped, gave_up, lat_p99_s per workload and seed).
Shows     against the round limit: (a) extra model calls per 100 tasks, (b) tasks per 100 that were committed at the
          cap without a valid answer or given up (these are outside the contract and must be reported), (c) 99th
          percentile of the task latency.  Colour gives the policy, marker and line style the strategy.  One row per
          workload; bands are intervals over seeds.  Round limits are placed at equal steps, with an axis break before
          a limit that is far from its predecessor (30); the strategies are dodged slightly along x and "think again"
          is drawn on top.
Paper     ablation (appendix figure or a row group of the ablation table).
"""
from . import style as st, load as ld, panels as pn

EXP = "E3_retry_strategy"
NAME = "fig_s_30_retry"
KEY, ON, CAP = "protocol.policy", "protocol.on_invalid", "protocol.max_rounds"
STRAT = {"rethink": ("think again on F (ours)", "o", "-", True), "reread": ("read again, no lock", "s", (0, (4, 1.5)), False),
         "abort": ("give up", "x", (0, (1, 1.2)), False)}


def axis_break(ax, x):
    """Axis break mark (two slanted strokes on a white gap) on the bottom spine at data position x (linear axis)."""
    tr = ax.get_xaxis_transform()
    ax.plot([x - 0.09, x + 0.09], [0, 0], transform=tr, color="white", lw=2.6, clip_on=False, zorder=5, solid_capstyle="butt")
    for dx in (-0.06, 0.06):
        ax.annotate("", xy=(x + dx, 0), xycoords=tr, xytext=(1.3, 2.8), textcoords="offset points", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333", shrinkA=0, shrinkB=0), zorder=6)
        ax.annotate("", xy=(x + dx, 0), xycoords=tr, xytext=(-1.3, -2.8), textcoords="offset points", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333", shrinkA=0, shrinkB=0), zorder=6)


def outside(r):
    return 100.0 * (ld.num(r.get("capped"), 0) + ld.num(r.get("gave_up"), 0)) / max(ld.num(r.get("tasks"), 1), 1)


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    cases = st.case_order(ld.values(R, "case"))
    pols = st.policy_order(ld.values(R, KEY))
    strats = [s for s in STRAT if ld.where(R, **{ON: s})] + [s for s in ld.values(R, ON) if s not in STRAT]
    caps = sorted(ld.values(R, CAP))
    metrics = (("extra_calls_per_100", "extra calls\nper 100 tasks"), (outside, "capped or given up\nper 100 tasks"), ("lat_p99_s", "p99 latency (s)"))
    fig, axes = st.subplots(len(cases), 3, cols=1, aspect=0.46 * len(cases) + 0.20, sharex=True, squeeze=False)
    for j, c in enumerate(cases):
        for i, (key, lab) in enumerate(metrics):
            ax = axes[j][i]
            pos = pn.ordinal(ax, caps, lambda v: "%g" % v)
            ns = len(strats)
            for k in reversed(range(ns)):                  # the first strategy (ours, filled) last, i.e. on top
                s = strats[k]
                _, mk, ls, fill = STRAT.get(s, (s, "d", "-", False))
                kw_of = lambda p, mk=mk, ls=ls, fill=fill, k=k: st.policy_line(p, marker=mk, ls=ls, zorder=3.2 - 0.3 * k,
                                                                                **({} if fill else dict(mfc="white")))
                dx = (k - 0.5 * (ns - 1)) * 0.08
                pn.lines(ax, ld.where(R, case=c, **{ON: s}), CAP, key, by=KEY, xpos={v: x + dx for v, x in pos.items()}, kw_of=kw_of)
            for a, b in zip(caps, caps[1:]):
                if float(b) / max(float(a), 1e-9) > 3:
                    axis_break(ax, 0.5 * (pos[a] + pos[b]))
            ax.set_ylabel(lab + ("\n[%s]" % st.dataset_label(c, short=True) if len(cases) > 1 and i == 0 else ""))
            st.grid(ax, "y")
    pn.zero_floor([a for row in axes for a in row])
    pn.letters(axes[0])
    fig.supxlabel("round limit (model calls per task)", fontsize=st.FONT)
    from matplotlib.lines import Line2D
    H = [Line2D([], [], label=st.pretty_math(p), **st.policy_line(p, marker="none")) for p in pols]
    H += [Line2D([], [], color=st.MUTED, marker=STRAT.get(s, (s, "d", "-", False))[1], ls=STRAT.get(s, (s, "d", "-", False))[2], ms=st.MS, mec=st.MUTED,
                 mfc=st.MUTED if STRAT.get(s, (s, "d", "-", False))[3] else "white", label=STRAT.get(s, (s,))[0]) for s in strats]
    fig.legend(handles=H, loc="outside upper center", ncol=3, handlelength=2.2, columnspacing=0.9)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
