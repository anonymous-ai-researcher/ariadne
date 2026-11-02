"""Figure T3: the conflict law against what happened.  Conflict probability of single-insertion reads.

Reads     D3b_law_analysis/fits.json  ("single": observed conflict probability with its bootstrap interval, and the
          predictions exch / closed / semi1 / semi2 with their max absolute errors; "f1", "f2", "lid_pooled").
          When the law files D3a_law_real/<cell>.law.pkl are there as well, the same quantities are evaluated on a
          dense grid of c with the library functions of ariadne.law and the fitted parameters of fits.json
          (nothing is refitted): the observed survival curve of the admissibility ratio, the exchangeable-only
          prediction, the semi-parametric forms; the closed form is always drawn densely from its formula.
Shows     one panel per configuration (shared axes): P(the held answer is no longer c-admissible | exactly one
          concurrent insertion fell inside the ball) against c on a log scale in c - 1.
            black dots, caps     observed at the c grid of the audit, 95% interval (over seeds when there are
                                 several, else the bootstrap interval of the file); thin black line: observed, dense
            grey dashed          exchangeable only: the insertion is placed like a stored neighbour (no parameter)
            orange               closed form (Poisson neighbours, D = pooled local dimension, fitted D_s)
            blue                 semi-parametric form, one sibling scale
            green dash-dotted    semi-parametric form, two sibling scales
          The table in each panel gives the max absolute error over the c grid, in percentage points (fits.json
          err_*; mean over seeds).  Conflict outcomes are never used in a fit: every curve is a prediction.
          Without a fits.json the law files of --law-exp are fitted in process (_theory.quick_fit: the same
          library calls, no bootstrap; the intervals of the observed points are then Wilson intervals).
Paper     main text, figure "law vs observed" (both columns), next to the fit table.

Options   --exp D3b_law_analysis --law-exp D3a_law_real  (the same figure for other fit files, e.g. the controls:
          --exp D4b_controls_analysis --law-exp D4a_exchangeable_controls --name fig_t_03b_law_controls)
          --cases L4,A64 (default: all)   --cols N   --logy   --no-dense
"""
import os

import numpy as np

from . import _theory as T

NAME = "fig_t_03_law_vs_observed"
KEYS = ("exch", "closed", "semi1", "semi2")
DENSE = 1.0 + np.logspace(np.log10(0.012), np.log10(1.15), 70)


def closed_curve(e, cs):
    """Closed form on a dense grid from the fitted parameters of a fits.json entry (same formula as law.fit_one)."""
    from .. import law as LW
    f1 = e.get("f1") or {}
    D = T.fnum(e.get("lid_pooled")); k = int(e["k"])
    comps = [(p, d) for p, d in zip(f1.get("pi") or [], f1.get("Ds") or []) if p is not None and d is not None and p > 0]
    if not np.isfinite(D) or D <= 0:
        return None
    ps = sum(p for p, _ in comps); ds = comps[0][1] if comps else D
    return np.array([100 * LW.mix_conflict(c, ps, ds / D, D, k) for c in cs])


def dense_curves(path, e, cs):
    """Observed and predicted curves of one cell on a dense grid, from its law file and its fitted parameters."""
    from .. import law as LW
    C = LW.LawData(path)
    single = np.where(C.m == 1)[0]
    if len(single) == 0:
        return None
    out = {"n": len(single), "obs": np.array([100 * np.mean(C.rho[single] > c) for c in cs]),
           "exch": np.array([100 * float((C.loo_suf[single] > c).mean()) for c in cs])}
    for key, f in (("semi1", e.get("f1")), ("semi2", e.get("f2"))):
        if f and f.get("pi") and all(v is not None for v in f["pi"]):
            out[key] = np.array(C.pred_conflict({"pi": f["pi"], "Ds": f["Ds"]}, single, list(cs)), dtype=float)
    return out


def gather(entries, law_paths, cells, dense):
    """Everything of one configuration, averaged over its seeds."""
    es = [e for _, e in entries if e.get("single")]
    if not es:
        return None
    cs = np.array(es[0]["cs"], dtype=float)
    es = [e for e in es if list(e["cs"]) == list(es[0]["cs"])]
    G = {"cs": cs, "k": int(es[0]["k"]), "seeds": len(es), "n": int(sum(e["single"]["n"] for e in es))}
    m, lo, hi, _ = T.col_mean_ci([e["single"]["obs"] for e in es])
    if len(es) == 1:
        ci = es[0]["single"].get("ci") or []
        lo = T.arr([c[0] for c in ci]) if ci else np.full(len(cs), np.nan)
        hi = T.arr([c[1] for c in ci]) if ci else np.full(len(cs), np.nan)
    G["obs"], G["lo"], G["hi"] = m, lo, hi
    for key in KEYS:
        rows = [e["single"].get(key) for e in es if e["single"].get(key) is not None]
        G[key] = T.col_mean_ci(rows)[0] if rows else None
        errs = [T.fnum(e["single"].get("err_" + key)) for e in es]
        G["err_" + key] = float(np.nanmean(errs)) if np.isfinite(errs).any() else float("nan")
    G["Ds"] = T.nmean([T.fnum(((e.get("f1") or {}).get("Ds") or [None])[0]) for e in es]) if es else float("nan")
    cl = [closed_curve(e, DENSE) for e in es]
    cl = [c for c in cl if c is not None]
    G["dense"] = {"closed": np.mean(cl, axis=0)} if cl else {}
    if dense:
        ds = []
        for cell, e in zip(cells, [e for _, e in entries]):
            if e.get("single") and cell in law_paths:
                try:
                    d = dense_curves(law_paths[cell], e, DENSE)
                except (SystemExit, KeyError, ValueError) as err:
                    T.note("%s: dense curves of %s not available (%s)" % (NAME, cell, err)); d = None
                if d:
                    ds.append(d)
        if ds and len(ds) == len(es):
            w = np.array([d["n"] for d in ds], dtype=float); w = w / w.sum()
            for key in ("obs", "exch", "semi1", "semi2"):
                if all(key in d for d in ds):
                    G["dense"][key] = sum(wi * d[key] for wi, d in zip(w, ds))
            p = G["dense"]["obs"] / 100.0; ntot = sum(d["n"] for d in ds)
            _, blo, bhi = T.wilson(p * ntot, ntot)
            G["dense"]["obs_lo"], G["dense"]["obs_hi"] = 100 * blo, 100 * bhi
    return G


def top_of(G):
    """Largest value drawn in a panel (observed points and intervals, dense curves, predictions)."""
    vals = [G["obs"], G["hi"]] + [G[k] for k in KEYS if G.get(k) is not None] + [v for v in G["dense"].values()]
    v = np.concatenate([np.asarray(x, dtype=float).ravel() for x in vals])
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else 100.0


def draw(ax, G, logy=False):
    x = G["cs"] - 1.0; xd = DENSE - 1.0; dn = G["dense"]
    for key in KEYS:
        st = T.PRED[key]
        if key in dn:
            ax.plot(xd, dn[key], color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3 if key != "exch" else 2)
        elif G.get(key) is not None:
            ax.plot(x, G[key], color=st["color"], ls=st["ls"], lw=st["lw"], zorder=3 if key != "exch" else 2)
    if "obs" in dn:
        T.band(ax, xd, dn["obs_lo"], dn["obs_hi"], T.INK, alpha=0.10)
        ax.plot(xd, dn["obs"], color=T.INK, lw=0.6, zorder=4)
    T.caps(ax, x, G["obs"], G["lo"], G["hi"], color=T.INK, ms=2.9, zorder=6)
    T.c_axis(ax, "x", lim=(1.012, 2.15), label=None)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", color=T.LIGHT, lw=0.4)


def error_table(ax, G, x=0.97, y=0.95):
    """Max absolute errors of the predictors (percentage points), with a swatch of each line."""
    rows = [(k, G["err_" + k]) for k in KEYS if np.isfinite(G["err_" + k])]
    if not rows:
        return
    dy = 0.092
    ax.text(x, y, "max |error|, points", transform=ax.transAxes, ha="right", va="top", fontsize=T.TINY, color=T.MUTED)
    for i, (k, v) in enumerate(rows):
        yy = y - dy * (i + 1) - 0.035
        st = T.PRED[k]
        ax.plot([x - 0.275, x - 0.135], [yy, yy], transform=ax.transAxes, color=st["color"], ls=st["ls"], lw=st["lw"], clip_on=False, zorder=7)
        ax.text(x, yy, "%.1f" % v, transform=ax.transAxes, ha="right", va="center", fontsize=T.TINY, color=T.INK, zorder=7)


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis", law_exp="D3a_law_real", cases=None, cols=None, logy=False,
         no_dense=False, name=NAME, width=None):
    plt = T.setup()
    F, quick = T.fits_or_quick(results, exp, suffix, law_exp, name)
    if not F:
        return T.skip(name, "no fits.json and no law files")
    law_paths = T.law_paths(results, law_exp, suffix)
    B = T.by_case(F)
    cellnames = {}
    for nm in F:
        if not str(nm).startswith("_") and isinstance(F[nm], dict):
            cellnames.setdefault(T.split_cell(nm)[0], []).append(nm)
    order = [c for c in T.case_sort(B) if (cases is None or c in cases.split(","))]
    Gs = []
    for c in order:
        G = gather(B[c], law_paths, cellnames[c], not no_dense)
        if G is not None:
            Gs.append((c, G))
        else:
            T.note("%s: %s has no single-insertion block (too few in-ball insertions)" % (name, c))
    if not Gs:
        return T.skip(name, "no configuration with a fitted law")
    n = len(Gs); nr, nc = T.grid_layout(n, cols)
    w = float(width) if width else (T.COL2 if nc >= 3 else T.COL1 * (1.0 if nc == 1 else 1.55))
    lcol = T.legend_cols(w, 5); lrow = int(np.ceil(5 / lcol))
    fig, axes = plt.subplots(nr, nc, figsize=(w, 0.22 + 0.14 * lrow + nr * (1.45 if nc >= 4 else 1.62)), sharex=True, sharey=True, squeeze=False)
    top = max(top_of(G) for _, G in Gs)
    for i, (c, G) in enumerate(Gs):
        ax = axes.flat[i]
        draw(ax, G, logy)
        lab = T.case_label(c, long=nc <= 3)
        # the sample size goes on its own line under the panel title (left-aligned), so that title and sample
        # size never share a line in narrow panels
        ax.set_title("%s" % lab, fontsize=T.FONT, pad=3 + 1.35 * T.TINY, loc="left")
        ax.text(0.0, 1.02, "%s reads%s" % (format(G["n"], ","), "" if G["seeds"] == 1 else ", %d seeds" % G["seeds"]), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=T.TINY, color=T.MUTED)
        error_table(ax, G)
        if not logy:
            ax.set_ylim(0, min(100, top * 1.06))
    T.hide_unused(axes, n)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("tolerance $c$")
    for j in range(nc):                                                   # panels above an empty slot keep their tick labels
        col = [axes[r, j] for r in range(nr) if axes[r, j].get_visible()]
        if col and col[-1] is not axes[-1, j]:
            col[-1].tick_params(labelbottom=True); col[-1].set_xlabel("tolerance $c$")
    fig.supylabel("conflict probability after one in-ball insertion (%)" if nr > 1 else "conflict probability (%)", fontsize=T.FONT)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=T.INK, marker="o", ms=2.9, lw=0.6, label="observed")] + \
         [Line2D([], [], color=T.PRED[k]["color"], ls=T.PRED[k]["ls"], lw=T.PRED[k]["lw"], label=T.PRED[k]["label"]) for k in KEYS]
    fig.legend(handles=hs, loc="outside upper center", ncol=lcol, handlelength=2.4, columnspacing=1.6 if lcol >= 5 else 1.0)
    for c, G in Gs:
        print("  %s: %-10s single-insertion reads %6d, seeds %d, D_s %.2f | max |err| exch %.1f closed %.1f semi1 %.1f semi2 %.1f" % (
            name, c, G["n"], G["seeds"], G["Ds"], G["err_exch"], G["err_closed"], G["err_semi1"], G["err_semi2"]))
    return T.save(fig, name, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis")), ("--law-exp", dict(default="D3a_law_real")),
                          ("--cases", dict(default=None, help="comma-separated case names (default: all)")),
                          ("--cols", dict(type=int, default=None)), ("--logy", dict(action="store_true")),
                          ("--no-dense", dict(action="store_true", help="use only the c grid of fits.json")),
                          ("--name", dict(default=NAME, help="name of the output files")),
                          ("--width", dict(type=float, default=None, help="figure width in inches (3.33 forces a single column)"))])
