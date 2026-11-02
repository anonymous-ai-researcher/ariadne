"""Figure S19: can a task be starved?  Distribution of the number of rounds per task.

Reads     C11_starvation/summary.jsonl and the per-task arrays C11_starvation/<cell>.audit.npz (rounds = model calls of
          each committed task), grid: policy and protocol.max_rounds (30 = cap, 3 = bounded retries then commit).
Shows     complementary CDF of rounds per task (share of tasks that needed at least x model calls), log y, tasks
          pooled over seeds; one panel per worst-case workload (hot topics, large fan-out, many agents, heavy-tailed
          think time).  Solid: cap of 30 rounds; dashed, hollow marks: bounded at the smaller cap (vertical line).
Paper     appendix figure (CCDF of rounds).
"""
import numpy as np

from . import style as st, load as ld, panels as pn

EXP = "C11_starvation"
NAME = "fig_s_19_rounds_ccdf"
KEY, CAP = "protocol.policy", "protocol.max_rounds"
CASE = {"fan8-hot": "fan-out 8, hot topics", "fan16-hot": "fan-out 16, hot topics", "A256": "AG News, 256 agents", "L8": "LoCoMo, 8 agents"}


def main(results="results", suffix="", out="figures"):
    R = [r for r in ld.rows(results, EXP, suffix=suffix) if r.get(KEY) is not None]
    if not R or not ld.npz(results, EXP, suffix):
        return ld.skip(NAME, "no summary or no <cell>.audit.npz files")
    cases = ld.values(R, "case")
    pols = st.policy_order(ld.values(R, KEY))
    caps = sorted(ld.values(R, CAP)) or [None]
    big = max(c for c in caps if c is not None) if caps[0] is not None else None
    nr, nc = pn.grid_of(len(cases), 4)
    fig, axes = st.subplots(nr, nc, cols=1 if nc <= 2 else 2, aspect=(0.62 if nc == 1 else 0.42 if nc == 2 else 0.27) * nr + 0.06,
                            sharey=True, sharex=True, squeeze=False)
    flat = [a for row in axes for a in row]
    xmax, ymin = 2, 100.0
    for i, (ax, c) in enumerate(zip(flat, cases)):
        for cap in caps:
            for p in pols:
                rows = ld.where(R, case=c, **({KEY: p, CAP: cap} if cap is not None else {KEY: p}))
                v = ld.pooled(results, EXP, rows, "rounds", suffix)
                if not v.size:
                    continue
                xs = np.arange(1, int(v.max()) + 1)
                y = np.array([100.0 * np.mean(v >= x) for x in xs])
                main_cap = cap is None or cap == big
                kw = st.policy_line(p, ms=2.6 if main_cap else 2.4, lw=1.0 if main_cap else 0.8)
                if not main_cap:
                    kw.update(ls=(0, (2, 1.2)), mfc="white")
                ax.plot(xs, y, zorder=3 if main_cap else 2.5, **kw)
                xmax = max(xmax, int(v.max())); ymin = min(ymin, 100.0 / v.size)
        for cap in caps:
            if cap is not None and cap != big:
                ax.axvline(cap, color=st.MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
        st.panel_label(ax, "abcdefgh"[i], CASE.get(c, st.dataset_label(c)))
        st.grid(ax, "y")
    for ax in flat[len(cases):]:
        ax.set_visible(False)
    ax0 = flat[0]
    ax0.set_yscale("log"); ax0.set_ylim(ymin / 1.6, 140); st.log_ticks(ax0, "y", steps=(1,))
    ax0.set_xlim(0.5, xmax + 0.5)
    step = 1 if xmax <= 10 else 2 if xmax <= 20 else 5
    ax0.set_xticks([1] + list(range(step if step > 1 else 2, xmax + 1, step)))
    for row in axes:
        row[0].set_ylabel("tasks with at least $x$ rounds (%)")
    fig.supxlabel("rounds per task (model calls)", fontsize=st.FONT)
    from matplotlib.lines import Line2D
    extra = []
    if len([c for c in caps if c is not None]) > 1:
        extra = [Line2D([], [], color=st.INK, lw=1.0, marker="o", ms=2.6, label="cap of %s rounds" % big),
                 Line2D([], [], color=st.INK, lw=0.8, ls=(0, (2, 1.2)), marker="o", ms=2.4, mfc="white", label="at most %s rounds" % min(caps))]
    pn.policy_legend(fig, pols, extra=extra)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
