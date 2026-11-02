"""Figure T8: the mechanism in one picture.  Removing the order of the stream removes the conflicts.

Reads     D4a_exchangeable_controls/audit_law.jsonl  (per cell: case, order, seed, in_ball_insertions_per_read,
          expected_background_per_read, new_nearest_share_%, conflict_%_single_insertion, inadmissible_%_all_reads)
          D4b_controls_analysis/fits.json  (optional: single.exch, the exchangeable-only prediction of the conflict
          probability of the same single-insertion reads)
Shows     the same items in four orders, from the stream as it is to no structure at all:
            natural       the stream as it is
            block         blocks of consecutive items shuffled (short-range order kept)
            shuffle       stream order shuffled (siblings exist but are far apart in time)
            exchangeable  initial memory and stream pooled and split at random (no sibling structure)
          (a) in-ball concurrent insertions per read (marks on a log axis, 95% interval over seeds); the black
              tick on each mark is the first-principles background n_c k / (N + 1), which needs no fit: the exchangeable control
              sits on it, every bit of order adds insertions on top of it.
          (b) probability that one in-ball insertion breaks c-admissibility at c = --c (single-insertion reads);
              the tick is the exchangeable-only prediction for the same reads.
          (c) the product: share of ALL reads whose held answer is no longer c-admissible at validation.
          One bar colour per data set (colours of the system-side figures).
Paper     main text, next to the law figure (single column).

Options   --c 1.1
"""
import numpy as np

from . import _theory as T
from . import load as ld

NAME = "fig_t_08_controls"
EXP = "D4a_exchangeable_controls"
ORDERS = ["natural", "block", "shuffle", "exchangeable"]
ORDER_LABEL = {"natural": "natural order", "block": "blocks shuffled", "shuffle": "shuffled", "exchangeable": "exchangeable"}


def order_key(o):
    s = str(o)
    return "block" if s.startswith("block") else s


def ckey(d, c):
    """Value of a {"1.1": x} dict at tolerance c."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if abs(float(k) - c) < 1e-9:
            return v
    return None


def main(results="results", suffix="", out="figures", c=1.1):
    plt = T.setup()
    R = ld.rows(results, EXP, "audit_law.jsonl", suffix)
    R = [r for r in R if r.get("order") is not None and r.get("in_ball_insertions_per_read") is not None]
    if not R:
        return T.skip(NAME, "no audit_law.jsonl rows with an order")
    F = T.jload(results, "D4b_controls_analysis", "fits.json", suffix, quiet=True) or {}
    for r in R:
        r["_order"] = order_key(r["order"])
        r["_conf1"] = ckey(r.get("conflict_%_single_insertion"), c)
        r["_all"] = ckey(r.get("inadmissible_%_all_reads"), c)
        fe = F.get(r.get("cell") or r.get("name")) or {}
        sg = fe.get("single") or {}
        r["_exch"] = None
        if sg.get("exch") and fe.get("cs"):
            j = [i for i, cc in enumerate(fe["cs"]) if abs(cc - c) < 1e-9]
            r["_exch"] = sg["exch"][j[0]] if j else None
    cases = T.case_sort(sorted({r["case"] for r in R}))
    orders = [o for o in ORDERS if any(r["_order"] == o for r in R)]
    panels = [("in_ball_insertions_per_read", "expected_background_per_read", "in-ball insertions\nper read"),
              ("_conf1", "_exch", "conflict after one\nin-ball insertion (%)"),
              ("_all", None, "reads with a stale\nanswer (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(T.COL1, 0.62 + 0.30 * len(orders) * max(len(cases), 1) ** 0.6), sharey=True)
    nc = len(cases); bh = 0.76 / nc
    for pi, (ax, (key, ref, lab)) in enumerate(zip(axes, panels)):
        A = ld.agg(R, ["case", "_order"], key)
        B = ld.agg(R, ["case", "_order"], ref) if ref else {}
        vmax, vmin = 0.0, float("inf")
        for i, o in enumerate(orders):
            for j, case in enumerate(cases):
                s = A.get((case, o))
                if s is None or s.n == 0:
                    continue
                y = i + (j - (nc - 1) / 2) * bh
                col = T.ds_style(case)["color"]
                if pi == 0:      # (a) spans two decades: marks on a log axis (a bar on a log axis has no base)
                    ax.plot([s.mean], [y], ls="none", marker=T.ds_style(case).get("marker", "o"), ms=4.0, mfc=col, mec="white", mew=0.4, zorder=6)
                    vmin = min(vmin, s.lo if s.n > 1 and s.lo > 0 else s.mean)
                else:
                    ax.barh(y, s.mean, height=bh * 0.9, color=col if o != "exchangeable" else T.lighten(col, 0.0), edgecolor="white", linewidth=0.4, zorder=2)
                if s.n > 1:
                    ax.errorbar([s.mean], [y], xerr=[[s.mean - s.lo], [s.hi - s.mean]], fmt="none", ecolor=T.INK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
                vmax = max(vmax, s.hi if s.n > 1 else s.mean)
                b = B.get((case, o))
                if b is not None and b.n > 0 and np.isfinite(b.mean):
                    ax.plot([b.mean, b.mean], [y - bh * 0.5, y + bh * 0.5], color=T.INK, lw=1.3, zorder=5, solid_capstyle="butt")
                    vmax = max(vmax, b.mean)
                    if b.mean > 0:
                        vmin = min(vmin, b.mean)
                val = ("%.2f" % s.mean) if key == "in_ball_insertions_per_read" else ("%.0f" % s.mean if s.mean >= 10 else "%.1f" % s.mean)
                ax.annotate(val, (max(s.hi if s.n > 1 else s.mean, b.mean if (b is not None and b.n > 0 and np.isfinite(b.mean)) else 0), y), xytext=(2.5, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=T.TINY, color=T.INK)
        if pi == 0 and vmax > 0 and np.isfinite(vmin) and vmin > 0:
            ax.set_xscale("log")
            ax.set_xlim(vmin / 1.8, vmax * 4.0)                      # room for the value labels on the right
            from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
            ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(-4, 4)]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10))); ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(0, vmax * 1.32 if vmax > 0 else 1)
        ax.set_xlabel(lab, fontsize=T.SMALL, linespacing=1.1)
        ax.tick_params(axis="x", labelsize=T.TINY)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(pi == 0)
        ax.grid(True, axis="x", color=T.LIGHT, lw=0.4)
        T.panel(ax, "abc"[pi], dy=1.02)
    axes[0].set_yticks(range(len(orders))); axes[0].set_yticklabels([ORDER_LABEL[o] for o in orders])
    axes[0].set_ylim(len(orders) - 0.5, -0.5)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Patch(fc=T.ds_style(cs)["color"], ec="none", label=T.case_label(cs, long=True)) for cs in cases]
    # the black tick means different things in (a) and (b): one entry each
    tk = lambda: Line2D([], [], color=T.INK, lw=1.3, marker="|", ms=6, mew=1.3, ls="none")
    tks = [(tk(), "tick (a): first-principles\nbackground $n_c\\,k\\,/\\,(N+1)$"), (tk(), "tick (b): exchangeable-\nonly prediction")]
    if len(hs) == 2:                                                   # rows: data sets, then the two ticks (legend fills by column)
        hh = [hs[0], tks[0][0], hs[1], tks[1][0]]; ll = [hs[0].get_label(), tks[0][1], hs[1].get_label(), tks[1][1]]
        nco = 2
    else:
        hh = hs + [t[0] for t in tks]; ll = [h.get_label() for h in hs] + [t[1] for t in tks]
        nco = 2 if len(hh) <= 4 else 3
    fig.legend(hh, ll, loc="outside upper center", ncol=nco, handlelength=1.0, columnspacing=1.2, fontsize=T.TINY, labelspacing=0.35)
    for case in cases:
        for o in orders:
            rr = [r for r in R if r["case"] == case and r["_order"] == o]
            if rr:
                print("  %s: %-5s %-12s seeds %d: in-ball per read %.4f (first principles %.4f), conflict at c=%g single %.1f%%, all reads %.2f%%" % (
                    NAME, case, o, len(rr), np.mean([r["in_ball_insertions_per_read"] for r in rr]), np.mean([r["expected_background_per_read"] for r in rr]),
                    c, np.mean([T.fnum(r["_conf1"]) for r in rr]), np.mean([T.fnum(r["_all"]) for r in rr])))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--c", dict(type=float, default=1.1))])
