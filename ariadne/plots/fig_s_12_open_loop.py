"""Figure S12: open-loop arrivals.

Reads     C8b_open_loop/summary.jsonl  (cases arr = poisson<rate> | bursty<rate>, lat_p50_s, lat_p99_s, tps per policy, seed).
Shows     (a) task latency including queueing against the arrival rate: 99th percentile (lines with bands) and median
          (thin lines without markers) for Poisson arrivals, hollow marks for bursty arrivals at the same mean rate;
          (b) completed tasks per second against the offered rate (the diagonal is "keeps up").
Paper     appendix, with the think-time figure (C8).
"""
import re

from . import style as st, load as ld, panels as pn

EXP = "C8b_open_loop"
NAME = "fig_s_12_open_loop"
KEY = "protocol.policy"
BURST_DX = 1.2           # bursty marks are drawn at 1.2 x their rate so that they do not hide the Poisson marks


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    for r in R:
        m = re.match(r"^([a-z]+)(\d+(?:\.\d+)?)", str(r.get("arr")))
        r["_kind"], r["_rate"] = (m.group(1), float(m.group(2))) if m else ("poisson", float("nan"))
    P, B = ld.where(R, _kind="poisson"), ld.where(R, _kind=lambda k: k != "poisson")
    pols = st.policy_order(ld.values(R, KEY))
    fig, (ax_l, ax_t) = st.subplots(1, 2, cols=1, aspect=0.62)
    pn.lines(ax_l, P, "_rate", "lat_p99_s", by=KEY)
    pn.lines(ax_l, P, "_rate", "lat_p50_s", by=KEY, band=False, kw_of=lambda p: st.policy_line(p, marker="none", lw=0.6, alpha=0.8))
    pn.lines(ax_t, P, "_rate", "tps", by=KEY)
    for p in pols:
        for ax, val in ((ax_l, "lat_p99_s"), (ax_t, "tps")):
            for rate, s in ld.agg(ld.where(B, **{KEY: p}), "_rate", val).items():
                st.point_ci(ax, rate * BURST_DX, s, **st.policy_marker(p, mfc="white", ms=st.MS + 0.6))
    rates = sorted(ld.values(R, "_rate"))
    for ax in (ax_l, ax_t):
        ax.set_xscale("log")
        if len(rates) > 1:
            ax.set_xlim(rates[0] / 1.3, rates[-1] * 1.3 * (BURST_DX if B else 1.0))
        else:
            ax.set_xlim(rates[0] / 2, rates[0] * 2)
        st.log_ticks(ax, "x")
        ax.set_xlabel("arrival rate (tasks/s)")
        st.grid(ax, "y")
    ax_l.set_yscale("log"); st.log_ticks(ax_l, "y")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lo_l, hi_l = ax_l.get_ylim()
    yt = [t for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70) if lo_l <= t <= hi_l]
    if len(yt) >= 2:                     # the 1-2-5 ticks leave the upper half of (a) unlabelled
        ax_l.yaxis.set_major_locator(FixedLocator(yt)); ax_l.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
        ax_l.yaxis.set_minor_locator(NullLocator())
    ax_t.set_yscale("log")
    lo, hi = ax_t.get_xlim()
    ax_t.plot([lo, hi], [lo, hi], color="#B5B5B5", lw=0.6, zorder=0.5)
    ax_t.set_ylim(min(lo, ax_t.get_ylim()[0]), hi)
    st.log_ticks(ax_t, "y")
    ax_l.set_ylabel("task latency (s)"); ax_t.set_ylabel("completed tasks/s")
    st.panel_label(ax_l, "a"); st.panel_label(ax_t, "b")
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color=st.INK, lw=st.LW, label="Poisson, p99"),
             Line2D([], [], color=st.INK, lw=0.6, alpha=0.8, label="Poisson, median")]
    if B:
        extra.append(Line2D([], [], ls="none", marker="o", mfc="white", mec=st.INK, ms=st.MS, label="bursty, p99 (shifted right)"))
    pn.policy_legend(fig, pols, extra=extra)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
