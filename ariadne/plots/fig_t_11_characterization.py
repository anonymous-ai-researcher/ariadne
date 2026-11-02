"""Figure T11: the characterization theorem on one real neighbour list.

Reads     D3a_law_real/<cell>.law.pkl  (one read: its old exact neighbour distances r_1 <= ... <= r_k; default: a
          read of the first of L4, L8, L2, A64, A16 chosen for legibility, see pick(); nothing is invented)
          B7_characterization/audit_law.jsonl and D3a_law_real/audit_law.jsonl  (only printed: on how many real
          single-insertion cases the theorem was compared with the observed outcome, and how many agree)
Shows     (a) the theorem rank by rank, on a log distance axis.  Row p is the interval r_{p-1} <= t < r_p of the
              distances at which a concurrent insertion takes old rank p.  Vermillion: the insertion breaks
              c-admissibility of the held exact answer; blue: the held answer stays c-admissible.  An insertion
              at rank p conflicts iff r_p > c t (the vermillion part ends at r_p / c), or some later neighbours
              are more than a factor c apart (r_i > c r_{i-1} for an i > p: the whole row is vermillion; such
              gaps are marked on top).  Outside the ball (t >= r_k) nothing can break.
          (b) the same list for every tolerance: the conflict region in the (t, c) plane.  Its boundary is the
              conflict radius R_c = max{r_i / c : r_i > c r_{i-1}}, a staircase of 1 / c pieces that jumps where
              a gap of the list stops exceeding c.  The dashed line is the c of panel (a).  Averaging this
              picture over all reads gives the conflict landscape (fig_t_02).
Paper     main text, schematic next to the characterization theorem (single column).

Options   --cell L4_s1   --row N   --c 1.1
"""
import os

import numpy as np

from . import _theory as T
from . import load as ld

NAME = "fig_t_11_characterization"
EXP = "D3a_law_real"
PREFER = ("L4", "L8", "L2", "A64", "A16")


def pick(L, c):
    """Row of a read whose neighbour list shows every case of the theorem and is legible on a log axis: a gap
    larger than c at a middle rank, r_1 well inside the ball, neighbours not on top of each other."""
    k = int(L["k"]); ro = np.asarray(L["ro"], dtype=float)[:, :k]
    ok = np.all(np.isfinite(ro), axis=1) & (ro[:, 0] > 1e-3)
    best = None
    for x in np.where(ok)[0]:
        r = ro[x]; ratio = r[1:] / r[:-1]
        gaps = np.where(ratio > c)[0] + 2                                 # ranks i with r_i > c r_{i-1}
        if len(gaps) == 0 or len(gaps) > 2 or gaps.max() < 3 or gaps.max() > k - 2:
            continue
        spread = r[0] / r[-1]
        if not 0.35 <= spread <= 0.8:
            continue
        score = float(np.sort(np.log(ratio))[:3].mean()) * 40 - abs(spread - 0.55) * 2 - abs(float(ratio.max()) - 1.22) * 3
        if best is None or score > best[0]:
            best = (score, int(x))
    return None if best is None else best[1]


def main(results="results", suffix="", out="figures", cell=None, row=None, c=1.1):
    plt = T.setup()
    import matplotlib.patheffects as pe
    laws = T.law_paths(results, EXP, suffix)
    if not laws:
        return T.skip(NAME, "no law files in %s" % T.exp_dir(results, EXP, suffix))
    order = [cell] if cell else sorted(laws, key=lambda n: (PREFER.index(T.split_cell(n)[0]) if T.split_cell(n)[0] in PREFER else 99, n))
    chosen = None
    for name in order:
        if name not in laws:
            continue
        L = T.law_load(laws[name])
        x = row if row is not None else pick(L, c)
        if x is not None:
            chosen = (name, L, int(x)); break
    if chosen is None:
        return T.skip(NAME, "no read with a gap larger than c = %g at a middle rank" % c)
    name, L, x = chosen
    k = int(L["k"]); r = np.asarray(L["ro"], dtype=float)[x, :k]
    print("  %s: read = law row %d (task %d) of %s, k = %d, distances %s" % (NAME, x, int(L["task"][x]), name, k, " ".join("%.3f" % v for v in r)))
    tot = agree = 0
    for exp in ("B7_characterization", EXP):
        if not os.path.exists(os.path.join(T.exp_dir(results, exp, suffix), "audit_law.jsonl")):
            continue
        for rr in ld.rows(results, exp, "audit_law.jsonl", suffix):
            m = str(rr.get("characterization", "")).split("/")
            if len(m) == 2 and m[1].isdigit():
                agree += int(m[0]); tot += int(m[1])
    if tot:
        print("  %s: characterization theorem against observed outcomes on real single-insertion cases: %d of %d agree" % (NAME, agree, tot))

    x0, x1 = r[0] / 1.42, r[-1] * 1.13
    fig = plt.figure(figsize=(T.COL1, 3.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.08)
    ax = fig.add_subplot(gs[0]); ab = fig.add_subplot(gs[1], sharex=ax)
    col_c, col_a = T.lighten(T.C_CONF, 0.25), T.lighten(T.C_ADM, 0.62)

    # ---- (a) rank by rank
    gap = np.zeros(k, dtype=bool); gap[1:] = r[1:] > c * r[:-1]
    h = 0.36
    for p in range(1, k + 1):
        lo = r[p - 2] if p > 1 else x0; hi = r[p - 1]
        later = bool(gap[p:].any())
        cut = hi if later else min(hi, max(lo, hi / c))
        if cut > lo:
            ax.fill_between([lo, cut], p - h, p + h, color=col_c, lw=0, zorder=2)
        if hi > cut:
            ax.fill_between([cut, hi], p - h, p + h, color=col_a, lw=0, zorder=2)
        if not later and lo < hi / c < hi:
            ax.plot([hi / c, hi / c], [p - h - 0.12, p + h + 0.12], color=T.INK, lw=0.6, zorder=4)
    ax.fill_between([r[-1], x1], k + 1 - h, k + 1 + h, color="#E6E6E6", lw=0, zorder=2)
    for i in range(k):
        for a in (ax, ab):
            a.axvline(r[i], color="#BDBDBD", lw=0.4, zorder=1)
    ax.set_xscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(k + 1.62, -2.15)
    ax.set_yticks(list(range(1, k + 1)) + [k + 1]); ax.set_yticklabels(["%d" % p for p in range(1, k + 1)] + ["out"])
    ax.tick_params(axis="y", length=0, pad=2); ax.tick_params(axis="x", labelbottom=False, which="both")
    ax.set_ylabel("old rank $p$ of the insertion")
    ax.spines["left"].set_visible(False)
    # labels r_i on top, skipping the ones that would collide
    last = -1e9; tr = ax.transData
    for i in range(k):
        px = tr.transform((r[i], 0))[0]
        if px - last >= 9.5 * fig.dpi / 72.0 or i == k - 1:
            ax.text(r[i], -1.22, "$r_{%d}$" % (i + 1) if i + 1 < k else "$r_k$", ha="center", va="bottom", fontsize=T.TINY, color=T.INK, zorder=6,
                    clip_on=False)
            last = px
    for i in np.where(gap)[0]:                                            # gaps r_i > c r_{i-1}
        ax.annotate("", xy=(r[i], -0.10), xytext=(r[i - 1], -0.10), arrowprops=dict(arrowstyle="<->", lw=0.6, color=T.INK, shrinkA=0, shrinkB=0,
                                                                                    mutation_scale=5), zorder=7)
        wide = (tr.transform((r[i], 0))[0] - tr.transform((r[i - 1], 0))[0]) > 30 * fig.dpi / 72.0
        ax.text(np.sqrt(r[i] * r[i - 1]) if wide else r[i] * 1.008, -0.28 if wide else -0.10, "gap $r_{%d}/r_{%d}$ = %.2f > $c$" % (i + 1, i, r[i] / r[i - 1]) if wide
                else "%.2f > $c$" % (r[i] / r[i - 1]), ha="center" if wide else "left", va="bottom" if wide else "center", fontsize=T.TINY,
                color=T.INK, zorder=7, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    # the cut of the first row without a later gap, labelled once
    for p in range(1, k + 1):
        lo = r[p - 2] if p > 1 else x0; hi = r[p - 1]
        if not gap[p:].any() and lo < hi / c < hi:
            ax.annotate("$r_{%d}/c$" % p, (hi / c, p + h + 0.12), xytext=(0, -1.0), textcoords="offset points", ha="center", va="top", fontsize=T.TINY,
                        color=T.INK, zorder=7, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
            break
    xa = x0 * 1.015                                                      # key in the empty lower-left corner (rows k-2, k-1), not in the "out" row
    for yy, colr, txt in ((k - 1.9, col_c, "conflict"), (k - 0.9, col_a, "stays admissible")):
        ax.fill_between([xa, xa * 1.035], yy - 0.30, yy + 0.30, color=colr, lw=0, zorder=3)
        ax.text(xa * 1.05, yy, txt, ha="left", va="center", fontsize=T.SMALL, color=T.INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.text(np.sqrt(r[-1] * x1), k + 1, "outside", ha="center", va="center", fontsize=T.TINY, color=T.MUTED, zorder=5)
    T.panel(ax, "a", "one insertion at distance $t$, tolerance $c$ = %g" % c, dy=1.03)

    # ---- (b) every tolerance
    cs = 1.0 + np.logspace(np.log10(0.008), np.log10(1.3), 600)
    Rc = T.conflict_radius(r[None, :], k, cs)[0]
    y = cs - 1.0
    ab.fill_betweenx(y, x0, np.minimum(Rc, r[-1]), color=col_c, lw=0, zorder=2)
    ab.fill_betweenx(y, np.minimum(Rc, r[-1]), r[-1], color=col_a, lw=0, zorder=2)
    ab.fill_betweenx(y, r[-1], x1, color="#E6E6E6", lw=0, zorder=2)
    ab.plot(np.minimum(Rc, r[-1]), y, color=T.darken(T.C_CONF, 0.2), lw=0.9, zorder=4)
    ab.axhline(c - 1.0, color=T.INK, lw=0.7, ls=(0, (3, 1.5)), zorder=5)
    ab.text(x0 * 1.02, (c - 1.0) * 1.07, "$c$ = %g, panel (a)" % c, ha="left", va="bottom", fontsize=T.TINY, color=T.INK, zorder=6)
    T.c_axis(ab, "y", ticks=(1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0), lim=(cs[0], cs[-1]), label="tolerance $c$")
    ab.text(x0 * 1.02, y[0] * 1.6, "conflict: $t < R_c$", ha="left", va="bottom", fontsize=T.SMALL, color=T.INK, zorder=6)
    ab.text(r[-1] / 1.015, y[-1] / 1.25, "admissible: $R_c \\leq t < r_k$", ha="right", va="top", fontsize=T.SMALL, color=T.INK, zorder=6)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    cand = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
    ticks = [v for v in cand if x0 * 1.02 <= v <= x1 / 1.02]
    while len(ticks) > 6:
        ticks = ticks[::2]
    ab.xaxis.set_major_locator(FixedLocator(ticks)); ab.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ab.xaxis.set_minor_locator(NullLocator()); ab.xaxis.set_minor_formatter(NullFormatter())
    ab.set_xlabel("distance $t$ of the insertion to the query (log scale)")
    T.panel(ab, "b", "the same neighbor list, every tolerance", dy=1.02)
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--cell", dict(default=None)), ("--row", dict(type=int, default=None)), ("--c", dict(type=float, default=1.1))])
