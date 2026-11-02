"""Figure T7: planted and recovered.  Does the estimation pipeline find a law that was put there?

Reads     D2b_planted_analysis/planted.json  (one entry per synthetic run "<Ds>_<pi>_<D>_<k>_s<seed>": "planted" {Ds,
          pi, D}, "fit" {Ds, Ds_ci}, "bg" {n, obs, lam_bg}, "k", "n")
          D4b_controls_analysis/controls.json  (exchangeable controls on real data, same format; optional)
Shows     (a) fitted sibling dimension against the planted one, both on log scales.  Small dots: the cells of
              the grid (one per manifold dimension and k, mean over workload seeds); large markers: mean over all
              runs with that planted D_s and sibling share, with the 95% interval over runs (with fewer than
              three runs: the bootstrap intervals of the fits, combined).  One shade per sibling share; shares are dodged horizontally.
              The diagonal is perfect recovery.  Below: the same as the ratio fitted / planted.
          (b) background check with no parameter: observed in-ball insertions per read that are not siblings
              against the first-principles value n_c k / (N + 1), one point per synthetic cell (seeds pooled,
              95% Poisson interval), plus the exchangeable controls on real data (rings in the colour of the
              data set).  The band is +/- 10% around the diagonal.
Paper     appendix, figure "planted vs recovered"; one sentence in the main text.

Options   --exp D2b_planted_analysis   --controls-exp D4b_controls_analysis
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_07_planted_recovery"
BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none")       # text over grid lines: a white backing


def pi_color(pi, pis):
    """Ordered shade for a sibling share (light = small share); pi = 0 (no siblings) is grey."""
    from matplotlib.colors import to_hex
    if pi <= 0:
        return T.GREY
    pos = [p for p in pis if p > 0]
    i = pos.index(pi) if pi in pos else 0
    return to_hex(T.COOL_CMAP(0.42 + 0.55 * (i / max(len(pos) - 1, 1)) if len(pos) > 1 else 0.75))


def main(results="results", suffix="", out="figures", exp="D2b_planted_analysis", controls_exp="D4b_controls_analysis"):
    plt = T.setup()
    P = T.jload(results, exp, "planted.json", suffix)
    if not P:
        return T.skip(NAME, "no planted.json")
    C = T.jload(results, controls_exp, "controls.json", suffix, quiet=True) or {}
    runs = [(nm, e) for nm, e in P.items() if isinstance(e, dict) and e.get("planted")]
    pis = sorted({float(e["planted"]["pi"]) for _, e in runs if e["planted"].get("pi") is not None})
    fig = plt.figure(figsize=(T.COL2, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.0, 0.36])
    ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[1, 0], sharex=ax); axb = fig.add_subplot(gs[:, 1])

    # ---- (a) fitted against planted
    fitted = [(float(e["planted"]["Ds"]), float(e["planted"]["pi"]), T.split_cell(nm)[0], T.fnum((e.get("fit") or {}).get("Ds")),
               (e.get("fit") or {}).get("Ds_ci") or [None, None]) for nm, e in runs
              if e["planted"].get("Ds") is not None and float(e["planted"].get("pi") or 0) > 0]
    fitted = [f for f in fitted if np.isfinite(f[3])]
    dss = sorted({f[0] for f in fitted}); pp = [p for p in pis if p > 0]
    inside = tot = 0
    for j, pi in enumerate(pp):
        col = pi_color(pi, pis); dodge = 1.055 ** (j - (len(pp) - 1) / 2)
        for ds in dss:
            sel = [f for f in fitted if f[0] == ds and f[1] == pi]
            if not sel:
                continue
            cells = {}
            for f in sel:
                cells.setdefault(f[2], []).append(f[3])
            if len(cells) > 1:
                cm = np.array([np.mean(v) for v in cells.values()])
                ax.plot(np.full(len(cm), ds * dodge), cm, ls="none", marker="o", ms=1.9, mfc=T.lighten(col, 0.35), mec="none", zorder=2)
                axr.plot(np.full(len(cm), ds * dodge), cm / ds, ls="none", marker="o", ms=1.9, mfc=T.lighten(col, 0.35), mec="none", zorder=2)
            m, lo, hi, n = T.pooled_ci([f[3] for f in sel], [f[4][0] for f in sel], [f[4][1] for f in sel])
            T.caps(ax, [ds * dodge], [m], [lo], [hi], color=col, ms=3.6, capsize=1.6)
            T.caps(axr, [ds * dodge], [m / ds], [lo / ds], [hi / ds], color=col, ms=3.2, capsize=1.6)
            if np.isfinite(lo) and np.isfinite(hi):
                tot += 1; inside += int(lo <= ds <= hi)
    if dss:
        lo, hi = min(dss) / 1.35, max(dss) * 1.35
        ax.plot([lo, hi], [lo, hi], color=T.INK, lw=0.6, zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(plt.FixedLocator(dss)); a.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v)); a.set_minor_locator(plt.NullLocator())
        axr.axhline(1.0, color=T.INK, lw=0.6, zorder=1)
        axr.fill_between([lo, hi], 0.9, 1.1, color="#EDEDED", lw=0, zorder=0)
        axr.set_ylim(0.62, 1.38); axr.set_yticks([0.8, 1.0, 1.2])
    else:
        ax.text(0.5, 0.5, "no run with planted siblings", transform=ax.transAxes, ha="center", va="center", fontsize=T.SMALL, color=T.MUTED)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("fitted $D_s$"); axr.set_ylabel("fitted / planted"); axr.set_xlabel("planted sibling dimension $D_s$")
    ax.grid(True, color=T.LIGHT, lw=0.4); axr.grid(True, axis="x", color=T.LIGHT, lw=0.4)
    T.panel(ax, "a", "sibling dimension: planted and recovered")
    for j, pi in enumerate(pp):
        ax.text(0.085, 0.93 - 0.095 * (j + 1), "%g%%" % (100 * pi), transform=ax.transAxes, ha="left", va="center", fontsize=T.SMALL, color=T.INK,
                bbox=BOX, zorder=6)
        ax.plot([0.055], [0.93 - 0.095 * (j + 1)], transform=ax.transAxes, ls="none", marker="o", ms=3.6, mfc=pi_color(pi, pis), mec=pi_color(pi, pis), zorder=7)
    if pp:
        ax.text(0.04, 0.955, "sibling share", transform=ax.transAxes, ha="left", va="center", fontsize=T.SMALL, color=T.MUTED, bbox=BOX, zorder=6)
    if fitted:
        per = sorted({sum(1 for f in fitted if f[1] == p) for p in pp})
        ax.text(0.98, 0.05, "%s runs%s" % (format(len(fitted), ","), (", %d per share" % per[0]) if len(per) == 1 else ""),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=T.SMALL, color=T.INK, bbox=BOX, zorder=6)

    # ---- (b) background rate against first principles
    cells = {}
    for nm, e in runs:
        bg = e.get("bg") or {}
        if bg.get("lam_bg") is None or bg.get("n") is None or not e.get("n"):
            continue
        c = cells.setdefault(T.split_cell(nm)[0], dict(n=0.0, lam=0.0, reads=0.0, pi=float(e["planted"].get("pi") or 0), real=False))
        c["n"] += float(bg["n"]); c["lam"] += float(bg["lam_bg"]) * float(e["n"]); c["reads"] += float(e["n"])
    for nm, e in C.items():
        bg = (e.get("bg") or {}) if isinstance(e, dict) else {}
        if bg.get("lam_bg") is None or bg.get("n") is None or not e.get("n"):
            continue
        c = cells.setdefault("ctl:" + T.split_cell(nm)[0], dict(n=0.0, lam=0.0, reads=0.0, pi=0.0, real=True, case=T.split_cell(nm)[0]))
        c["n"] += float(bg["n"]); c["lam"] += float(bg["lam_bg"]) * float(e["n"]); c["reads"] += float(e["n"])
    xs, ys = [], []
    drawn = []                                                         # the cells that are drawn (the legend lists only these)
    for key, c in sorted(cells.items(), key=lambda kv: (kv[1]["real"], kv[1]["pi"])):
        if c["n"] <= 0 or c["lam"] <= 0:
            continue
        x = c["lam"] / c["reads"]; y = c["n"] / c["reads"]; h = 1.96 / np.sqrt(c["n"])
        xs.append(x); ys.append(y); drawn.append(c)
        if c["real"]:
            st = T.ds_style(c["case"])
            T.caps(axb, [x], [y], [y * (1 - h)], [y * (1 + h)], color=st["color"], marker=st["marker"], mfc="white", ms=4.4, capsize=1.6, zorder=6)
        else:
            T.caps(axb, [x], [y], [y * (1 - h)], [y * (1 + h)], color=pi_color(c["pi"], pis), ms=3.2, capsize=1.2)
    if xs:
        lo, hi = min(xs + ys) / 1.5, max(xs + ys) * 1.5
        axb.plot([lo, hi], [lo, hi], color=T.INK, lw=0.6, zorder=1)
        axb.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color="#EDEDED", lw=0, zorder=0)
        axb.set_xscale("log"); axb.set_yscale("log"); axb.set_xlim(lo, hi); axb.set_ylim(lo, hi); axb.set_aspect("equal", adjustable="box")
        tn = sum(c["n"] for c in drawn); tl = sum(c["lam"] for c in drawn)
        T.log_plain(axb)
        axb.text(0.97, 0.04, "pooled, %s insertions:\nobserved / predicted\n= %.3f $\\pm$ %.3f" % (format(int(tn), ","), tn / tl, 1.96 * np.sqrt(tn) / tl),
                 transform=axb.transAxes, ha="right", va="bottom", fontsize=T.TINY, color=T.INK, linespacing=1.25, bbox=BOX, zorder=6)
        print("  %s: background rate pooled over %d cells: observed / first principles = %.3f (+/- %.3f)" % (NAME, len(xs), tn / tl, 1.96 * np.sqrt(tn) / tl))
    axb.set_xlabel("first principles: $n_c\\,k\\,/\\,(N+1)$ per read"); axb.set_ylabel("observed background insertions per read")
    axb.grid(True, color=T.LIGHT, lw=0.4)
    T.panel(axb, "b", "background rate, no parameter")
    yy = 0.93
    syn = sorted({c["pi"] for c in drawn if not c["real"]})
    labs = [("synthetic, no siblings" if p <= 0 else "synthetic, %g%% siblings" % (100 * p), pi_color(p, pis), "o", pi_color(p, pis)) for p in syn]
    if drawn and not syn:
        axb.text(0.04, 0.975, "real exchangeable controls only", transform=axb.transAxes, ha="left", va="top", fontsize=T.SMALL, color=T.MUTED,
                 bbox=BOX, zorder=6)
        yy = 0.88
    seen = []
    for c in drawn:
        if c["real"] and T.ds_key(c["case"]) not in seen:
            seen.append(T.ds_key(c["case"])); st = T.ds_style(c["case"])
            labs.append(("%s, exchangeable control" % st["label"], st["color"], st["marker"], "white"))
    for txt, col, mk, mfc in labs:
        axb.plot([0.06], [yy - 0.026], transform=axb.transAxes, ls="none", marker=mk, ms=3.4, mfc=mfc, mec=col, mew=0.7, clip_on=False, zorder=7)
        axb.text(0.11, yy, txt, transform=axb.transAxes, ha="left", va="top", fontsize=T.TINY, color=T.INK, bbox=BOX, zorder=6)
        yy -= 0.075
    if tot:
        print("  %s: planted D_s inside the 95%% interval of the summary marker in %d of %d (D_s, share) groups" % (NAME, inside, tot))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D2b_planted_analysis")), ("--controls-exp", dict(default="D4b_controls_analysis"))])
