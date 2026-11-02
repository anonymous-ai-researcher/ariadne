"""Figure T10: the fitted object.  Density of y = ln(r_k / t) of the in-ball insertions against the model.

Reads     D3b_law_analysis/fits.json  ("y_hist": {edges, centres, density}, "y_model1", "y_model2", "y_bg" evaluated at
          the centres; f1 / f2: shares, sibling dimensions, KS distance and its parametric-bootstrap p-value;
          "w_bg": the first-principles background share)
Shows     one panel per configuration, density on a log scale (an exponential law is a straight line whose slope
          is the dimension): histogram of y over all in-ball concurrent insertions (grey steps), the fitted
          mixture with one sibling scale (blue) and two (green, dash-dotted), and the background shape alone
          (grey dashed; what the density would be without siblings).  Only the sibling part is fitted: the
          background share is fixed from first principles.  Printed: fitted D_s (one scale, with its 95%
          interval), the background share, and the KS distance (mean over seeds) with its parametric-bootstrap
          p-value: the range over the seeds, or "<= 1/(B+1)" when every seed is at the floor of the bootstrap (with
          many insertions a formal rejection is expected; the distance is the effect size).  With several seeds the densities
          are interpolated to the grid of the first seed and averaged.
Paper     appendix (goodness of fit of the sibling spectrum).

Options   --exp D3b_law_analysis   --cases L4,A64   --cols N
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_10_y_density"


def gather(entries):
    es = [e for _, e in entries if e.get("y_hist") and e.get("y_model1")]
    if not es:
        return None
    e0 = es[0]; ed = T.arr(e0["y_hist"]["edges"]); mid = 0.5 * (ed[1:] + ed[:-1])
    G = {"edges": ed, "mid": mid, "seeds": len(es), "events": int(sum(e.get("events") or 0 for e in es))}
    for key, get in (("hist", lambda e: e["y_hist"]["density"]), ("m1", lambda e: e["y_model1"]), ("m2", lambda e: e["y_model2"]), ("bg", lambda e: e["y_bg"])):
        rows = []
        for e in es:
            ee = T.arr(e["y_hist"]["edges"]); mm = 0.5 * (ee[1:] + ee[:-1])
            v = get(e)
            if v is not None:
                rows.append(np.interp(mid, mm, T.arr(v), right=np.nan))
        if rows:
            A = np.array(rows); cnt = np.isfinite(A).sum(axis=0)
            G[key] = np.where(cnt > 0, np.nansum(A, axis=0) / np.maximum(cnt, 1), np.nan)
        else:
            G[key] = None
    ds = [T.fnum(((e.get("f1") or {}).get("Ds") or [None])[0]) for e in es]
    ci = [((e.get("f1") or {}).get("Ds_ci") or [[None, None]])[0] for e in es]
    G["Ds"], G["lo"], G["hi"], _ = T.pooled_ci(ds, [c[0] for c in ci], [c[1] for c in ci])
    G["w_bg"] = T.nmean([T.fnum(e.get("w_bg")) for e in es])
    G["ks"] = T.nmean([T.fnum((e.get("f1") or {}).get("ks")) for e in es])
    G["ks_p"] = T.nmean([T.fnum((e.get("f1") or {}).get("ks_p")) for e in es])
    G["ks_ps"] = [v for v in (T.fnum((e.get("f1") or {}).get("ks_p")) for e in es) if np.isfinite(v)]
    G["Ds2"] = [T.nmean([T.fnum(((e.get("f2") or {}).get("Ds") or [None, None])[j]) for e in es]) for j in range(2)]
    return G


def p_floor(F):
    """Smallest p-value the parametric bootstrap can return, 1 / (B + 1): the smallest ks_p in the file when its
    reciprocal is an integer (B bootstrap draws), else 0 (no floor detected)."""
    ps = [T.fnum((e.get("f1") or {}).get("ks_p")) for e in F.values() if isinstance(e, dict)]
    ps = [v for v in ps if np.isfinite(v) and v > 0]
    if not ps:
        return 0.0
    m = min(ps)
    return m if abs(1.0 / m - round(1.0 / m)) < 1e-6 else 0.0


def p_text(ps, floor):
    """Bootstrap p-values of the seeds as text: at the floor "<= floor", else the range over the seeds."""
    if not ps:
        return ""
    fmt = lambda v: ("$\\leq$ %.2g" % v) if floor > 0 and v <= floor * (1 + 1e-6) else ("%.2g" % v)
    lo, hi = min(ps), max(ps)
    if fmt(lo) == fmt(hi):
        return fmt(lo) if lo > floor * (1 + 1e-6) or floor <= 0 else "$\\leq$ %.3f" % floor
    return "%s to %s" % (fmt(lo), fmt(hi))


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis", cases=None, cols=None, name=NAME):
    plt = T.setup()
    F = T.jload(results, exp, "fits.json", suffix)
    if not F:
        return T.skip(name, "no fits.json")
    B = T.by_case(F)
    pfloor = p_floor(F)
    Gs = [(c, gather(B[c])) for c in T.case_sort(B) if cases is None or c in cases.split(",")]
    Gs = [(c, G) for c, G in Gs if G is not None]
    if not Gs:
        return T.skip(name, "no configuration with a fitted density")
    n = len(Gs); nr, nc = T.grid_layout(n, cols)
    w = T.COL2 if nc >= 3 else T.COL1 * (1.0 if nc == 1 else 1.55)
    lcol = T.legend_cols(w, 4); lrow = int(np.ceil(4 / lcol))
    fig, axes = plt.subplots(nr, nc, figsize=(w, 0.20 + 0.14 * lrow + nr * 1.48), squeeze=False)
    for i, (c, G) in enumerate(Gs):
        ax = axes.flat[i]; ed = G["edges"]; mid = G["mid"]; h = G["hist"]
        pos = h[h > 0]
        floor = float(pos.min()) / 2.2 if len(pos) else 1e-3
        hh = np.where(h > 0, h, floor)
        ax.fill_between(np.repeat(ed, 2)[1:-1], floor, np.repeat(hh, 2), color="#D4D4D4", lw=0, zorder=1)
        if G["bg"] is not None:
            ax.plot(mid, G["w_bg"] * G["bg"], color=T.C_EXCH, ls=T.PRED["exch"]["ls"], lw=0.9, zorder=3)
        if G["m2"] is not None:
            ax.plot(mid, G["m2"], color=T.C_SEMI2, ls=T.PRED["semi2"]["ls"], lw=1.0, zorder=4)
        ax.plot(mid, G["m1"], color=T.C_SEMI1, lw=1.1, zorder=5)
        ax.set_yscale("log"); ax.set_xlim(0, ed[-1])
        top = float(max(np.nanmax(h), np.nanmax(G["m1"]))) * 2.2
        ax.set_ylim(floor, top)
        ax.set_title(T.case_label(c, long=nc <= 3), fontsize=T.FONT, pad=3, loc="left")
        txt = "$D_s$ = %.2f" % G["Ds"]
        if np.isfinite(G["lo"]) and np.isfinite(G["hi"]):
            txt += " [%.2f, %.2f]" % (G["lo"], G["hi"])
        txt += "\nbackground share %.0f%%" % (100 * G["w_bg"])
        if np.isfinite(G["ks"]):
            pt = p_text(G["ks_ps"], pfloor)
            txt += "\nKS %.3f" % G["ks"] + ("" if not pt else (", $p$ " + pt) if (pt.startswith("$") or " to " in pt) else ", $p$ = " + pt)
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=T.TINY, color=T.INK, linespacing=1.25)
        print("  %s: %-8s %d insertions, D_s %.2f [%.2f, %.2f], two scales %.2f / %.2f, background share %.3f, KS %.3f (p %.3g)" % (
            name, c, G["events"], G["Ds"], G["lo"], G["hi"], G["Ds2"][0], G["Ds2"][1], G["w_bg"], G["ks"], G["ks_p"]))
    T.hide_unused(axes, n)
    for ax in axes.flat[max(n - nc, 0):n]:
        ax.set_xlabel("$y = \\ln(r_k\\,/\\,t)$")
    fig.supylabel("density of the in-ball insertions", fontsize=T.FONT)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Patch(fc="#D4D4D4", ec="none", label="observed"),
          Line2D([], [], color=T.C_SEMI1, lw=1.1, label="background + 1 sibling scale"),
          Line2D([], [], color=T.C_SEMI2, lw=1.0, ls=T.PRED["semi2"]["ls"], label="background + 2 scales"),
          Line2D([], [], color=T.C_EXCH, lw=0.9, ls=T.PRED["exch"]["ls"], label="background part alone")]
    fig.legend(handles=hs, loc="outside upper center", ncol=lcol, handlelength=2.0, columnspacing=1.4 if lcol >= 4 else 1.0)
    return T.save(fig, name, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis")), ("--cases", dict(default=None)), ("--cols", dict(type=int, default=None)),
                          ("--name", dict(default=NAME))])
