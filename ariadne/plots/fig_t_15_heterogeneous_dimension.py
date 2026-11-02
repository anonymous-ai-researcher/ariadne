"""Figure T15: one intrinsic dimension D in the closed form, data with several.

Reads     D5_heterogeneous_dimension/fits.json  (python -m ariadne.law fit on the law files of D5; when the file is
          missing the law files D5_heterogeneous_dimension/*.law.pkl are fitted in process with _theory.quick_fit,
          the same library calls without bootstrap)
          D5_heterogeneous_dimension/*.law.pkl  (dense curves, optional; planted D and D_s from the workload section)
          D5_heterogeneous_dimension/strata_lid.json and D3b_law_analysis/strata_lid.json  (optional; written by
          python -m ariadne.law strata --by lid: per stratum of the reads by local intrinsic dimension, the
          error of the closed form with the pooled D of the file, with the D of the stratum, and of the
          semi-parametric form)
Shows     (a) one panel per synthetic store (homogeneous manifolds and mixtures of two manifold dimensions, as
              named in the panel): conflict probability of single-insertion reads against c, observed and
              predicted, exactly as in fig_t_03 (same code).  The closed form uses ONE pooled D; the
              semi-parametric form uses the neighbour distances of every read and no D.
          (b) reads stratified by their local intrinsic dimension (only when a strata_lid.json exists): for every
              configuration the max absolute error of each predictor, averaged over the strata (weighted by
              reads; summary.mean_err_common of the strata file): closed form with the pooled D (ring), closed
              form with the D of the stratum (filled), semi-parametric (square), exchangeable only (cross).
Paper     appendix paragraph on the limits of the closed form.

Options   --exp D5_heterogeneous_dimension   --real-exp D3b_law_analysis   --cols N
"""
import numpy as np

from . import _theory as T
from . import fig_t_03_law_vs_observed as F3

NAME = "fig_t_15_heterogeneous_dimension"
STRATA_KEYS = (("closed_pooled", "o", "white", T.C_CLOSED, "closed form, pooled $D$"), ("closed_own", "o", T.C_CLOSED, T.C_CLOSED, "closed form, $D$ of the stratum"),
               ("semi_global", "s", T.C_SEMI1, T.C_SEMI1, "semi-parametric"), ("exch", "x", T.GREY, T.GREY, "exchangeable only"))


LEG = {"exch": "exchangeable only", "closed": "closed form", "semi1": "semi-param., 1 scale", "semi2": "semi-param., 2 scales"}


def parse_case(case):
    """(D, D_s) from a synthetic case id "<D>_<D_s>" or "<D1>+<D2>_<D_s>" (the naming of D5), else (None, None)."""
    import re
    m = re.match(r"^(\d+(?:\+\d+)*)_(\d+(?:\.\d+)?)$", str(case))
    if not m:
        return None, None
    D = [float(v) for v in m.group(1).split("+")]
    return (D if len(D) > 1 else D[0]), float(m.group(2))


def synth_label(case, law):
    wl = (law or {}).get("workload") or {}
    D, Ds = wl.get("D"), wl.get("Ds")
    if D is None:                                                       # no law file at hand: the case id carries D and D_s
        D, Ds = parse_case(case)
    if D is None:
        return case
    dl = " + ".join("%g" % d for d in D) if isinstance(D, (list, tuple)) else "%g" % D
    kind = "mixed" if isinstance(D, (list, tuple)) and len(D) > 1 else "homogeneous"
    return "%s: $D$ = %s%s" % (kind, dl, (", $D_s$ = %g" % Ds) if Ds is not None else "")


def strata_rows(S, synth, laws):
    out = []
    B = T.by_case({k: v for k, v in S.items() if isinstance(v, dict) and v.get("by") == "lid"})
    for case in T.case_sort(B):
        ms = [(e.get("summary") or {}).get("mean_err_common") or (e.get("summary") or {}).get("mean_err") or {} for _, e in B[case]]
        ms = [m for m in ms if m]
        if not ms:
            continue
        r = {"case": case, "synth": synth}
        for k, *_ in STRATA_KEYS:
            v = [T.fnum(m.get(k)) for m in ms]
            r[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        cell = [n for n in laws if T.split_cell(n)[0] == case]
        r["label"] = synth_label(case, T.law_load(laws[cell[0]]) if cell else None).replace("homogeneous: ", "").replace("mixed: ", "") if synth else case
        out.append(r)
    return out


def main(results="results", suffix="", out="figures", exp="D5_heterogeneous_dimension", real_exp="D3b_law_analysis", cols=None):
    plt = T.setup()
    F, quick = T.fits_or_quick(results, exp, suffix, exp, NAME)
    laws = T.law_paths(results, exp, suffix)
    B = T.by_case(F)
    cellnames = {}
    for nm in F:
        cellnames.setdefault(T.split_cell(nm)[0], []).append(nm)
    Gs = []
    for c in sorted(B, key=lambda s: (("+" in s), s)):
        G = F3.gather(B[c], laws, cellnames[c], True)
        if G is not None:
            Gs.append((c, G))
    S5 = T.jload(results, exp, "strata_lid.json", suffix, quiet=True) or {}
    S3 = T.jload(results, real_exp, "strata_lid.json", suffix, quiet=True) or {}
    rows = strata_rows(S5, True, laws) + strata_rows(S3, False, {})
    if not Gs and not rows:
        return T.skip(NAME, "no fitted D5 configuration and no strata_lid.json")
    n = len(Gs); nr, nc = T.grid_layout(max(n, 1), cols if cols else (2 if (rows and n > 1) else None))
    has_b = bool(rows)
    wa = min(nc * 1.75, T.COL2 - (2.3 if has_b else 0.0)) if n else 0.0
    W = min(T.COL2, max(wa + (2.3 if has_b else 0.0), T.COL1))
    H = max(0.5 + nr * 1.55 if n else 0.0, 0.9 + 0.17 * len(rows) if has_b else 0.0, 1.9)
    fig = plt.figure(figsize=(W, H))
    subs = fig.subfigures(1, 2 if (n and has_b) else 1, width_ratios=[wa, W - wa] if (n and has_b) else None)
    subs = list(np.atleast_1d(subs))
    if n:
        sf = subs[0]
        axes = sf.subplots(nr, nc, sharex=True, sharey=True, squeeze=False)
        top = max(F3.top_of(G) for _, G in Gs)
        for i, (c, G) in enumerate(Gs):
            ax = axes.flat[i]
            F3.draw(ax, G)
            cell = cellnames[c][0]
            ax.set_title(synth_label(c, T.law_load(laws[cell]) if cell in laws else None), fontsize=T.SMALL, pad=3, loc="left")
            F3.error_table(ax, G)
            ax.set_ylim(0, min(100, top * 1.06))
            if i == 0:
                T.panel(ax, "a", dy=1.14 if nr > 1 or True else 1.03)
        T.hide_unused(axes, n)
        for ax in axes.flat[max(n - nc, 0):n]:
            ax.set_xlabel("tolerance $c$")
        for ax in axes[:, 0]:
            ax.set_ylabel("conflict probability (%)")
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=T.INK, marker="o", ms=2.9, lw=0.6, label="observed")] + \
             [Line2D([], [], color=T.PRED[k]["color"], ls=T.PRED[k]["ls"], lw=T.PRED[k]["lw"], label=LEG.get(k, T.PRED[k]["label"])) for k in F3.KEYS]
        sf.legend(handles=hs, loc="outside upper center", ncol=3 if nc < 3 else 5, handlelength=2.0, columnspacing=0.8, fontsize=T.TINY)
        for c, G in Gs:
            print("  %s: %-10s single-insertion reads %6d | max |err| exch %.1f closed (one pooled D) %.1f semi1 %.1f semi2 %.1f%s" % (
                NAME, c, G["n"], G["err_exch"], G["err_closed"], G["err_semi1"], G["err_semi2"], "  (fitted in process)" if quick else ""))
    if has_b:
        sf = subs[-1]
        ax = sf.subplots(1, 1)
        # synthetic rows first, then real ones: a gap with a rule between the groups and a label above each group
        ys, y, heads, prev = [], 0.0, [], None
        for r in rows:
            if r["synth"] != prev:
                if prev is not None:
                    y += 0.5
                    ax.axhline(y - 0.5 - 0.25, color=T.MUTED, lw=0.5, zorder=0)
                heads.append((y - 0.05, "synthetic stores" if r["synth"] else "real data")); y += 0.75; prev = r["synth"]
            ys.append(y); y += 1.0
        for yy, txt in heads:
            ax.text(0.0, yy, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=T.TINY, fontweight="bold", color=T.INK,
                    zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        for i, r in zip(ys, rows):
            vals = [r[k] for k, *_ in STRATA_KEYS[:3] if np.isfinite(r[k])]
            if vals:
                ax.plot([min(vals), max(vals)], [i, i], color="#D8D8D8", lw=1.6, zorder=1, solid_capstyle="round")
            # the closed forms a little above the row line and the semi-parametric square a little below it: where
            # the errors are equal the filled circle would hide the square
            dy = {"closed_pooled": -0.15, "closed_own": -0.15, "semi_global": 0.15}
            for k, mk, mfc, mec, _ in STRATA_KEYS:
                if np.isfinite(r[k]):
                    ax.plot([r[k]], [i + dy.get(k, 0.0)], ls="none", marker=mk, ms=3.9 if mk != "x" else 3.6, mfc=mfc, mec=mec, mew=0.8, zorder=3 if k != "closed_own" else 4)
        ax.set_yticks(ys); ax.set_yticklabels([r["label"] for r in rows], fontsize=T.TINY)
        ax.set_ylim(ys[-1] + 0.5, -0.5); ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, None); ax.grid(True, axis="x", color=T.LIGHT, lw=0.4); ax.spines["left"].set_visible(False)
        ax.set_xlabel("max |error| inside LID strata (points)")
        T.panel(ax, "b" if n else "a", dy=1.03)
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], ls="none", marker=mk, ms=3.9 if mk != "x" else 3.6, mfc=mfc, mec=mec, mew=0.8, label=lab) for _, mk, mfc, mec, lab in STRATA_KEYS]
        sf.legend(handles=hs, loc="outside upper center", ncol=2, handletextpad=0.2, columnspacing=0.8, fontsize=T.TINY)
        for r in rows:
            print("  %s: strata by LID, %-10s mean max |err|: closed pooled D %.1f, closed own D %.1f, semi-parametric %.1f, exchangeable %.1f" % (
                NAME, r["case"], r["closed_pooled"], r["closed_own"], r["semi_global"], r["exch"]))
    else:
        T.note("%s: no strata_lid.json (python -m ariadne.law strata --by lid): panel (b) is left out" % NAME)
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D5_heterogeneous_dimension")), ("--real-exp", dict(default="D3b_law_analysis")),
                          ("--cols", dict(type=int, default=None))])
