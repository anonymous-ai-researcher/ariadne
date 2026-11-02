"""Figure T2: the conflict landscape.  Where does an insertion have to land to break c-admissibility, and where
do real insertions land?

Reads     D3a_law_real/*.law.pkl (all cells of a data set with k = --k are pooled: concurrency levels and seeds).
          Fields used: ro (old neighbour distances of every read), ev (in-ball concurrent insertions: t, r_k),
          nc and N (overlapping commits and store size, for the first-principles background count).  No fit and
          no model output is used.
Shows     one block per data set (default LoCoMo and AG News):
          centre  heat map of P(conflict | u, c): the share of the reads for which ONE insertion at normalized
                  distance u = t / r_k breaks c-admissibility of the held exact answer.  For every read this is a
                  deterministic statement of the characterization theorem about its neighbour list (the insertion
                  conflicts iff t < R_c with R_c = max{r_i / c : r_i > c r_{i-1}}), evaluated exactly on a grid of
                  240 x 140 values and averaged over the reads; iso-lines at 10%, 50%, 90%.  Both axes are
                  logarithmic in the distance to the trivial case: c - 1 and 1 - u (the query is at the left
                  edge, the k-th neighbour at the right edge), because everything happens near the rim.
          top     where real insertions land: histogram of u over all in-ball concurrent insertions (outline),
                  split into the BACKGROUND part that first principles predict (grey: sum of n_c k / (N + 1)
                  insertions, placed like the stored neighbours, r_i / r_{k+1} of the static lists; no parameter)
                  and the excess over it, the SIBLINGS (purple).  Medians of the observed
                  insertions (purple, dotted) and of the background law (grey, dashed), keyed in the map of the first block.  The
                  hatched last bin next to the rim also holds the insertions with 1 - u < DMIN (clipped into it).
          right   the landscape integrated over those positions: probability that one insertion breaks
                  c-admissibility as a function of c, for the observed insertions (theorem applied to each of
                  them, solid) and for the background law (dashed).
Paper     main text, data-driven view of the characterization theorem (both columns), directly before the law:
          the excess mass sits closer to the query than the background, where the landscape is red at every
          practical c.

Options   --k 10   --datasets locomo,agnews (or "all")
"""
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from . import _theory as T

NAME = "fig_t_02_conflict_landscape"
MED_LS_OBS, MED_LS_BG = (0, (1, 1.4)), (0, (3.2, 1.6))      # the two median lines: dotted (observed), dashed (background)
EXP = "D3a_law_real"
NU, NC = 240, 140
DMIN = 0.004               # smallest depth 1 - u shown: insertions closer to the rim than this are drawn in the last bin


def pooled(paths, k):
    """Arrays of one data set: neighbour lists of all reads, positions of all in-ball insertions with the row of
    their read, static reference positions, and the first-principles background count."""
    ro, u, xr, stat = [], [], [], []
    off = 0; lam = 0.0
    for p in paths:
        L = T.law_load(p)
        if int(L["k"]) != k:
            continue
        r = np.asarray(L["ro"], dtype=float)
        ok = np.all(np.isfinite(r[:, :k]), axis=1) & (r[:, k - 1] > 0)
        idx = np.full(len(r), -1); idx[ok] = off + np.arange(int(ok.sum()))
        ro.append(r[ok, :k]); off += int(ok.sum())
        lam += float((np.asarray(L["nc"], dtype=float) * k / (np.asarray(L["N"], dtype=float) + 1.0))[ok].sum())
        if r.shape[1] > k:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (r[ok, :k] / r[ok, k:k + 1]).ravel()
            stat.append(s[np.isfinite(s) & (s > 0) & (s <= 1)])
        ev = np.asarray(L["ev"], dtype=float).reshape(-1, 10)
        if len(ev):
            x = ev[:, 0].astype(int); good = idx[x] >= 0
            u.append(np.clip(ev[good, 3] / ev[good, 6], 0, 1)); xr.append(idx[x[good]])
    if not ro:
        return None
    cat = lambda a, dt=float: np.concatenate(a).astype(dt) if a else np.zeros(0, dtype=dt)
    return {"ro": np.concatenate(ro), "u": cat(u), "x": cat(xr, int), "stat": cat(stat), "lam": lam}


def landscape(ro, k, us, cs):
    """P [len(us), len(cs)] and the normalized conflict radii [reads, len(cs)]."""
    R = T.conflict_radius(ro, k, cs) / ro[:, k - 1:k]
    P = np.empty((len(us), len(cs)))
    for j in range(len(cs)):
        srt = np.sort(R[:, j])
        P[:, j] = 1.0 - np.searchsorted(srt, us, side="right") / len(srt)
    return P, R


def steps(e, h):
    return np.repeat(e, 2)[1:-1], np.repeat(h, 2)


def block(fig, gs, P, us, cs, D, R, k, title, letter, first):
    plt = T.plt
    ax = fig.add_subplot(gs[1, 0]); at = fig.add_subplot(gs[0, 0], sharex=ax); ar = fig.add_subplot(gs[1, 1], sharey=ax)
    y = cs - 1.0
    im = ax.pcolormesh(1.0 - us, y, P.T, cmap=T.CONF_CMAP, vmin=0, vmax=1, shading="gouraud", rasterized=True)
    cl = ax.contour(1.0 - us, y, P.T, levels=[0.1, 0.5, 0.9], colors=[T.INK], linewidths=0.55)
    T.c_axis(ax, "y", lim=(cs[0], cs[-1]), label="tolerance $c$  (log scale in $c-1$)" if first else None)
    ax.set_xscale("log"); ax.set_xlim(1.0, DMIN)
    T.clabel_inside(ax, cl, fmt=lambda v: "%d%%" % round(100 * v), fontsize=T.TINY, inline_spacing=2)
    x = 1.0 - us                                                           # depth below the rim, log scale, reversed
    ax.set_xscale("log"); ax.set_xlim(1.0, DMIN)
    ut = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    ax.xaxis.set_major_locator(FixedLocator([1 - v for v in ut])); ax.xaxis.set_major_formatter(FixedFormatter(["%g" % v for v in ut]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("insertion at $u = t\\,/\\,r_k$  (log scale in $1-u$)")
    if not first:
        ax.tick_params(labelleft=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)

    # top: where insertions land = background predicted from first principles + excess (siblings)
    n_ev = len(D["u"]); bins = 50 if n_ev >= 4000 else 40 if n_ev >= 800 else 25
    bins = int(bins * 0.6)
    e = np.logspace(np.log10(DMIN), 0.0, bins + 1); wd = np.diff(np.log(e))            # bins of the depth 1 - u, density per log unit
    dep = lambda v: np.clip(1.0 - np.asarray(v, dtype=float), DMIN * 1.0001, 1.0)
    h_obs = np.histogram(dep(D["u"]), e)[0] / max(n_ev, 1) / wd
    w_bg = min(D["lam"] / n_ev, 1.0) if n_ev else 1.0
    h_bg = w_bg * (np.histogram(dep(D["stat"]), e)[0] / max(len(D["stat"]), 1) / wd) if len(D["stat"]) else np.zeros(bins)
    lo = np.minimum(h_bg, h_obs)
    xs, yb = steps(e, lo); _, yo = steps(e, h_obs)
    at.fill_between(xs, 0, yb, color="#C9C9C9", lw=0, zorder=2)
    at.fill_between(xs, yb, yo, color=T.lighten(T.C_SIB, 0.35), lw=0, zorder=2)
    at.plot(xs, yo, color=T.darken(T.C_SIB, 0.35), lw=0.7, zorder=3, solid_joinstyle="miter")
    # the last bin (next to the rim) also holds the insertions with 1 - u < DMIN, clipped into it: hatched
    at.fill_between([e[0], e[1]], 0, h_obs[0], facecolor="none", edgecolor="none", hatch="//////",
                    hatchcolor=T.darken(T.C_SIB, 0.35), lw=0, zorder=3.5)
    m = float(h_obs.max()) if n_ev else 1.0
    at.set_ylim(0, m * 1.08); at.set_yticks([]); at.tick_params(labelbottom=False, length=0)
    for s in ("left", "top", "right"):
        at.spines[s].set_visible(False)
    # observed median dotted, background median dashed (as the background law in the right panel)
    for u, col, mls in ((D["u"], T.darken(T.C_SIB, 0.35), MED_LS_OBS), (D["stat"], T.MUTED, MED_LS_BG)):
        if len(u) >= 5:
            med = max(1.0 - float(np.median(u)), DMIN)
            ax.axvline(med, color=col, lw=0.7, ls=mls, zorder=4)
            at.plot([med, med], [0, m * 1.08], color=col, lw=0.7, ls=mls, zorder=1)
    T.panel(at, letter, title, dy=1.04)

    # right: the landscape integrated over the positions (theorem applied to every observed insertion)
    if n_ev >= 5:
        ar.plot(100 * (D["u"][:, None] < R[D["x"], :]).mean(axis=0), y, color=T.darken(T.C_SIB, 0.2), lw=1.1)
    if len(D["stat"]):
        sub = D["stat"] if len(D["stat"]) <= 20000 else np.random.default_rng(0).choice(D["stat"], 20000, replace=False)
        rr = R if len(R) <= 4000 else R[np.random.default_rng(1).choice(len(R), 4000, replace=False)]
        pc = np.array([1.0 - (np.searchsorted(np.sort(rr[:, j]), sub, side="right") / len(rr)).mean() for j in range(R.shape[1])])   # E_s P(R_c > s)
        ar.plot(100 * pc, y, color=T.C_EXCH, lw=0.9, ls=(0, (3, 1.4)))
    ar.set_xlim(0, 100); ar.set_xticks([0, 50, 100]); ar.tick_params(labelleft=False)
    ar.set_xlabel("conflict (%)")
    ar.grid(True, axis="x", color=T.LIGHT, lw=0.4)
    return ax, at, ar, im, w_bg


def main(results="results", suffix="", out="figures", k=10, datasets="locomo,agnews"):
    plt = T.setup()
    laws = T.law_paths(results, EXP, suffix)
    if not laws:
        return T.skip(NAME, "no law files in %s" % T.exp_dir(results, EXP, suffix))
    groups = {}
    for name, p in laws.items():
        L = T.law_load(p)
        if int(L["k"]) == k:
            groups.setdefault(T.ds_key(L), []).append(p)
    want = [d.strip() for d in datasets.split(",")] if datasets != "all" else sorted(groups, key=lambda d: T.ds_style(d).get("order", 9))
    keys = [d for d in want if d in groups]
    if not keys:
        return T.skip(NAME, "no law file with k = %d for the data sets %s" % (k, datasets))
    us = 1.0 - np.logspace(0.0, np.log10(DMIN), NU)                       # u from 0 to 1 - DMIN, log-spaced in the depth 1 - u
    cs = 1.0 + np.logspace(np.log10(0.005), 0.0, NC)
    n = len(keys)
    W = T.COL2 if n > 1 else T.COL1 * 1.3
    fig = plt.figure(figsize=(W, 2.60))
    fig.set_layout_engine("none")
    left = 0.42 / W; right = 1 - 0.50 / W
    outer = fig.add_gridspec(1, n + 1, width_ratios=[1.0] * n + [0.03], wspace=0.12, left=left, right=right, top=0.925, bottom=0.135)
    im = None
    for i, d in enumerate(keys):
        D = pooled(groups[d], k)
        P, R = landscape(D["ro"], k, us, cs)
        gs = outer[0, i].subgridspec(2, 2, height_ratios=[0.30, 1.0], width_ratios=[1.0, 0.24], hspace=0.05, wspace=0.05)
        title = "%s: %s reads, %s in-ball concurrent insertions" % (T.ds_style(d)["label"], format(len(D["ro"]), ","), format(len(D["u"]), ","))
        ax, at, ar, im, w_bg = block(fig, gs, P, us, cs, D, R, k, title, "abcdef"[i], i == 0)
        ak = fig.add_subplot(gs[0, 1]); ak.axis("off")                     # key of the histogram, in the free corner
        for yy, col, txt in ((0.78, T.lighten(T.C_SIB, 0.35), "siblings\n(excess) %.0f%%" % (100 * (1 - w_bg))),
                             (0.26, "#C9C9C9", "background\n(predicted) %.0f%%" % (100 * w_bg))):
            ak.add_patch(plt.Rectangle((0.02, yy - 0.09), 0.10, 0.18, fc=col, ec="none", transform=ak.transAxes, clip_on=False))
            ak.text(0.17, yy, txt, transform=ak.transAxes, fontsize=T.TINY, color=T.INK, ha="left", va="center", linespacing=1.05)
        if i == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            hs = [Line2D([], [], color=T.darken(T.C_SIB, 0.35), lw=0.8, ls=MED_LS_OBS),
                  Line2D([], [], color=T.MUTED, lw=0.8, ls=MED_LS_BG),
                  Patch(facecolor=T.lighten(T.C_SIB, 0.35), edgecolor=T.darken(T.C_SIB, 0.35), hatch="//////",
                        hatchcolor=T.darken(T.C_SIB, 0.35), lw=0.5)]
            ax.legend(hs, ["median,\nobserved", "median,\nbackground", "clipped:\n$1-u<%g$" % DMIN],
                      loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=T.TINY, handlelength=1.5, handletextpad=0.3, labelspacing=0.3,
                      borderpad=0.25, borderaxespad=0.15,
                      frameon=True, framealpha=1.0, edgecolor="none", facecolor="white").set_zorder(6)
            ar.text(0.94, 0.97, "observed\ninsertions", transform=ar.transAxes, color=T.darken(T.C_SIB, 0.35), fontsize=T.TINY, ha="right", va="top",
                    linespacing=1.1)
            ar.text(0.94, 0.80, "background\nlaw", transform=ar.transAxes, color=T.MUTED, fontsize=T.TINY, ha="right", va="top", linespacing=1.1)
        print("  %s: %s: %d reads, %d in-ball insertions, first-principles background share %.3f; median u observed %.3f, background law %.3f" % (
            NAME, d, len(D["ro"]), len(D["u"]), w_bg, float(np.median(D["u"])) if len(D["u"]) else float("nan"),
            float(np.median(D["stat"])) if len(D["stat"]) else float("nan")))
    cax = fig.add_subplot(outer[0, n].subgridspec(2, 1, height_ratios=[0.30, 1.0], hspace=0.05)[1, 0])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100"]); cb.outline.set_linewidth(0.4)
    cb.set_label("reads in conflict (%)", labelpad=2)
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--k", dict(type=int, default=10)), ("--datasets", dict(default="locomo,agnews"))])
