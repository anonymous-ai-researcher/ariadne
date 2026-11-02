"""Figure T4: the closed-form conflict law over (c, D_s), and where the real workloads sit.

Reads     D1_closed_form/grid.json  (closed-form conflict probability on a dense grid: conflict[D][k][pi][Ds][c])
          D3b_law_analysis/fits.json  (fitted sibling dimension D_s of every configuration: f1.Ds, f1.Ds_ci,
          sibling share f1.pi, pooled local dimension lid_pooled)
Shows     two maps of the closed form for a fixed local dimension D and k (default: the D of the grid closest to
          the median pooled local dimension of the workloads, k = 10): probability that ONE in-ball insertion
          breaks c-admissibility, as filled contours with labelled iso-lines, over the tolerance c (log scale in
          c - 1) and the sibling dimension D_s (log scale).  Left: every in-ball insertion is a sibling
          (pi = 1); right: a diluted mixture (default pi = 0.5; the rest is background).  The dashed line
          D_s = D is the background law itself (alpha = 1): above it siblings are more spread out than the
          stored points, below it they concentrate around the query.  For a large c the iso-lines follow
          c^(-D_s) = const (straight lines in these coordinates would be D_s ln c = const).
          Strip on the right: the fitted D_s of every real configuration with its 95% interval (over seeds when
          there are several, else the bootstrap interval of the fit), in the colour and marker of its data set;
          the horizontal guide lines carry the median of each data set across the maps, so that a workload is a
          horizontal cut through the map; the data sets and the line D_s = D are named at the right end of the
          strip (not on the maps, where they would cover the iso-line labels).  Iso-line labels are placed by
          label_spots() away from the guide lines; the c axis has no tick at 1.01 (it would touch 1.02).
Paper     main text, heat-map figure of the law (both columns; with --single one map and the strip fit in one
          column).

Options   --D 16 --k 10 --pi 1.0,0.5 (values of the grid)   --exp D3b_law_analysis   --single
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_04_law_heatmap"
LEVELS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LINES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]


def nearest(vals, x):
    vals = list(vals)
    return int(np.argmin([abs(float(v) - float(x)) for v in vals]))


def workloads(F, k):
    """[(case, data set key, D_s, lo, hi, share, D)] of the configurations with the k of the map."""
    out = []
    for case, entries in T.by_case(F).items():
        es = [e for _, e in entries if e.get("f1") and e["f1"].get("Ds") and e["f1"]["Ds"][0] is not None and int(e.get("k", 0)) == k]
        if not es:
            continue
        m, lo, hi, n = T.mean_ci([e["f1"]["Ds"][0] for e in es])
        if n == 1:
            ci = (es[0]["f1"].get("Ds_ci") or [[None, None]])[0]
            lo, hi = T.fnum(ci[0]), T.fnum(ci[1])
        out.append((case, T.ds_key(case), m, lo, hi, float(np.mean([sum(p for p in e["f1"]["pi"] if p) for e in es])),
                    T.nmean([T.fnum(e.get("lid_pooled")) for e in es])))
    order = T.case_sort([w[0] for w in out])
    return sorted(out, key=lambda w: order.index(w[0]))


def label_spots(ax, x, Ds, Z, avoid=()):
    """Manual positions of the iso-line labels: for every level, the point on its line (searched along rows of
    D_s) that stays inside the map and farthest from the labels already placed, from the edges and from the
    horizontal guide lines at the D_s values in avoid (all distances in axes fractions)."""
    lx = np.log(x); ly = np.log(Ds)
    fx = lambda v: (np.log(v) - lx[0]) / (lx[-1] - lx[0]); fy = lambda v: (np.log(v) - ly[0]) / (ly[-1] - ly[0])
    rows = np.unique(np.linspace(0, len(Ds) - 1, 60).astype(int))
    spots, placed = [], []
    for lev in LINES:
        best, score = None, -1.0
        for r in rows:
            z = Z[r]; d = z - lev
            idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
            for i in idx:
                t = d[i] / (d[i] - d[i + 1]); xv = float(np.exp(lx[i] + t * (lx[i + 1] - lx[i])))
                ax_, ay = fx(xv), fy(Ds[r])
                if not (0.05 <= ax_ <= 0.95 and 0.08 <= ay <= 0.92):
                    continue
                if any(abs(ay - fy(a)) < 0.045 for a in avoid):
                    continue
                dmin = min([np.hypot((ax_ - px) * 0.8, ay - py) for px, py in placed] + [0.35])
                if dmin > score:
                    best, score = (xv, float(Ds[r]), ax_, ay), dmin
        if best is not None and score > 0.06:
            spots.append(best[:2]); placed.append(best[2:])
    return spots


def draw_map(ax, c, Ds, Z, D, first, avoid=()):
    import matplotlib.patheffects as pe
    from matplotlib.colors import BoundaryNorm
    x = c - 1.0
    norm = BoundaryNorm(LEVELS, T.CONF_CMAP.N)
    cf = ax.contourf(x, Ds, Z, levels=LEVELS, cmap=T.CONF_CMAP, norm=norm)
    cl = ax.contour(x, Ds, Z, levels=LINES, colors=[T.INK], linewidths=0.5)
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlim(x[0], x[-1]); ax.set_ylim(Ds[0], Ds[-1])
    ax.set_yscale("log")
    T.c_axis(ax, "x", lim=(float(c[0]), float(c[-1])), label="tolerance $c$  (log scale in $c-1$)")
    from matplotlib.ticker import FixedLocator
    ax.xaxis.set_major_locator(FixedLocator([t - 1.0 for t in T.C_TICKS if t != 1.01 and c[0] - 1e-9 <= t <= c[-1] + 1e-9]))
    labs = ax.clabel(cl, fmt=lambda v: "%g%%" % (100 * v), fontsize=T.TINY, inline_spacing=1.5,
                     manual=label_spots(ax, x, Ds, Z, avoid), zorder=7) or []
    for t in labs:
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white", alpha=0.6)])
    yt = [v for v in (0.5, 1, 2, 4, 8, 16, 32) if Ds[0] <= v <= Ds[-1]]
    ax.set_yticks(yt); ax.set_yticklabels(["%g" % v for v in yt]); ax.minorticks_off()
    ax.set_ylim(Ds[0], Ds[-1])
    if Ds[0] < D < Ds[-1]:
        ax.axhline(D, color=T.INK, lw=0.6, ls=(0, (4, 2)))
    if first:
        ax.set_ylabel("sibling dimension $D_s$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    return cf


def main(results="results", suffix="", out="figures", D=None, k=10, pi="1.0,0.5", exp="D3b_law_analysis", single=False):
    plt = T.setup()
    import matplotlib.patheffects as pe
    G = T.jload(results, "D1_closed_form", "grid.json", suffix)
    if not G:
        return T.skip(NAME, "no grid.json")
    F = T.jload(results, exp, "fits.json", suffix, quiet=True) or {}
    ki = nearest(G["k"], k); k = int(G["k"][ki])
    W = workloads(F, k)
    if D is None:
        D = float(np.median([w[6] for w in W])) if W else 16.0
    di = nearest(G["D"], D); Dg = float(G["D"][di])
    pis = [float(v) for v in str(pi).split(",")][:1 if single else 2]
    c = np.array(G["c"], dtype=float); Ds = np.array(G["Ds"], dtype=float)
    A = np.array(G["conflict"], dtype=float)[di, ki]                     # [pi][Ds][c]
    ns = max(len(W), 1); n = len(pis)
    strip = (0.10 + 0.060 * (ns + 2.9)) if n == 2 else (0.10 + 0.095 * (ns + 3.4))
    fig = plt.figure(figsize=(T.COL2, 2.45) if n == 2 else (T.COL1, 2.25))
    gs = fig.add_gridspec(1, n + 2, width_ratios=[1.0] * n + [strip * (1.0 if n == 2 else 0.62), 0.035])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    axs = fig.add_subplot(gs[0, n], sharey=axes[0]); cax = fig.add_subplot(gs[0, n + 1])
    cf = None
    meds = {}
    for case, dk, m, lo, hi, share, Dw in W:
        meds.setdefault(dk, []).append(m)
    guides = [float(np.median(v)) for v in meds.values()] + [Dg]
    for i, (ax, p) in enumerate(zip(axes, pis)):
        pj = nearest(G["pi"], p); pg = float(G["pi"][pj])
        cf = draw_map(ax, c, Ds, A[pj], Dg, i == 0, avoid=guides)
        if i > 0:
            ax.tick_params(labelleft=False)
        what = "every in-ball insertion is a sibling" if pg >= 1 else "background only" if pg <= 0 else \
            "%g%% siblings, %g%% background" % (100 * pg, 100 * (1 - pg))
        if n > 1:                                            # a single map has no panel title: the caption states D, k and the share
            n0 = len(ax.texts)
            T.panel(ax, "abc"[i], what)
            for t in ax.texts[n0:]:
                t.set_in_layout(False)                       # titles may run over the gap; they must not widen it

    # strip: fitted D_s of the workloads, and guide lines across the maps at the median of each data set
    XL = 2.9 if n == 2 else 3.4                            # room right of the markers for the labels of the guide lines
    axs.set_xlim(-0.7, ns - 0.3 + XL)
    seen = {}
    for j, (case, dk, m, lo, hi, share, Dw) in enumerate(W):
        st = T.ds_style(case)
        T.caps(axs, [j], [m], [lo], [hi], color=st["color"], marker=st["marker"], ms=3.6, capsize=1.6)
        seen.setdefault(dk, []).append(m)
    for dk, vals in seen.items():
        med = float(np.median(vals)); st = T.ds_style(dk)
        for ax in axes:
            ax.axhline(med, color="white", lw=1.6, alpha=0.75, zorder=4)
            ax.axhline(med, color=T.INK, lw=0.7, ls=(0, (1, 1.3)), zorder=5)
        axs.plot([-0.7, ns - 0.3 + 0.15], [med, med], color=T.INK, lw=0.7, ls=(0, (1, 1.3)), zorder=1)
        axs.text(ns - 0.3 + 0.3, med, st["label"], ha="left", va="center", fontsize=T.TINY, color=T.INK, zorder=6)
    if Ds[0] < Dg < Ds[-1]:
        axs.plot([-0.7, ns - 0.3 + 0.15], [Dg, Dg], color=T.INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
        axs.text(ns - 0.3 + 0.3, Dg, "$D_s = D$\n(background)", ha="left", va="center", fontsize=T.TINY, color=T.INK, zorder=6,
                 linespacing=1.0)
    axs.set_xticks(range(len(W))); axs.set_xticklabels([w[0] for w in W], rotation=90, fontsize=T.TINY)
    axs.tick_params(axis="x", length=0, pad=1.5); axs.tick_params(axis="y", labelleft=False, which="both", length=0)
    for s in ("left", "top", "right"):
        axs.spines[s].set_visible(False)
    axs.text(0.0, 1.03, "fitted $D_s$" + ("    (maps: $D$ = %g, $k$ = %d)" % (Dg, k) if n > 1 else ""), transform=axs.transAxes,
             fontsize=T.SMALL, ha="left", va="bottom").set_in_layout(False)
    if n > 1:
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))     # room for the panel titles (kept out of the layout)
    if not W:
        axs.text(0.5, 0.5, "no fit", transform=axs.transAxes, ha="center", va="center", fontsize=T.TINY, color=T.MUTED)
    cb = fig.colorbar(cf, cax=cax, ticks=LEVELS, spacing="uniform")
    cb.ax.set_yticklabels(["%g" % (100 * v) for v in LEVELS]); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5); cb.set_label("conflict probability (%)", labelpad=2)
    for case, dk, m, lo, hi, share, Dw in W:
        print("  %s: %-8s D_s = %.2f [%.2f, %.2f], sibling share %.2f, pooled local dimension %.1f" % (NAME, case, m, lo, hi, share, Dw))
    print("  %s: maps for D = %g, k = %d, pi = %s" % (NAME, Dg, k, ", ".join("%g" % float(G["pi"][nearest(G["pi"], p)]) for p in pis)))
    return T.save(fig, NAME + ("_single" if single else ""), out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--D", dict(type=float, default=None, help="local dimension of the maps (a value of the grid; default: closest to the workloads)")),
                          ("--k", dict(type=int, default=10)), ("--pi", dict(default="1.0,0.5", help="sibling shares of the two maps")),
                          ("--exp", dict(default="D3b_law_analysis")), ("--single", dict(action="store_true", help="one map, single column"))])
