"""Figure T9: where in the ranking does a concurrent insertion land?  Predicted, never fitted.

Reads     D3b_law_analysis/fits.json  ("rank": observed shares of the old rank p = 1..k of all in-ball insertions,
          "pred1" / "pred2" predicted by the law with one / two sibling scales, "exch" the uniform distribution
          of an exchangeable insertion; "events"; f1.rank_chi2_p)
Shows     one panel per configuration: observed share of every old rank (bars, 95% Wilson interval from the
          number of in-ball insertions, seeds pooled), the prediction of the law with one sibling scale (blue
          line and dots) and with two (green, dash-dotted), and the uniform exchangeable distribution 1 / k (grey
          dashed).  The sibling scales were fitted on the insertion DISTANCES only; ranks were never used, so
          the agreement is a test.  Printed in each panel: share of insertions that become the new nearest
          neighbour, observed and predicted with one and with two sibling scales, and the number of insertions.  No per-seed test statistic is printed (the paper's rank test is pooled and
          burst-corrected, computed elsewhere).
Paper     appendix (rank test of the law); the rank-1 shares can be quoted in the main text.

Options   --exp D3b_law_analysis   --cases L4,A64   --cols N
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_09_rank_distribution"


def gather(entries):
    es = [e for _, e in entries if e.get("rank") and e["rank"].get("obs")]
    es = [e for e in es if len(e["rank"]["obs"]) == len(es[0]["rank"]["obs"])] if es else []
    if not es:
        return None
    w = np.array([float(e.get("events") or 0) for e in es]); w = w / w.sum() if w.sum() > 0 else np.full(len(es), 1.0 / len(es))
    G = {"k": len(es[0]["rank"]["obs"]), "events": int(sum(e.get("events") or 0 for e in es)), "seeds": len(es)}
    for key in ("obs", "pred1", "pred2", "exch"):
        rows = [T.arr(e["rank"][key]) for e in es if e["rank"].get(key) is not None]
        G[key] = sum(wi * r for wi, r in zip(w, rows)) if len(rows) == len(es) else None
    ps = [T.fnum((e.get("f1") or {}).get("rank_chi2_p")) for e in es]
    G["chi2_p"] = float(np.nanmean(ps)) if np.isfinite(ps).any() else float("nan")
    return G


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis", cases=None, cols=None, name=NAME):
    plt = T.setup()
    F = T.jload(results, exp, "fits.json", suffix)
    if not F:
        return T.skip(name, "no fits.json")
    B = T.by_case(F)
    Gs = [(c, gather(B[c])) for c in T.case_sort(B) if cases is None or c in cases.split(",")]
    Gs = [(c, G) for c, G in Gs if G is not None]
    if not Gs:
        return T.skip(name, "no configuration with a rank block")
    n = len(Gs); nr, nc = T.grid_layout(n, cols)
    w = T.COL2 if nc >= 3 else T.COL1 * (1.0 if nc == 1 else 1.55)
    lcol = T.legend_cols(w, 4); lrow = int(np.ceil(4 / lcol))
    fig, axes = plt.subplots(nr, nc, figsize=(w, 0.20 + 0.14 * lrow + nr * 1.42), squeeze=False, sharey=True)
    top = max(float(max(np.nanmax(G["obs"]), np.nanmax(G["pred1"]) if G["pred1"] is not None else 0)) for _, G in Gs)
    for i, (c, G) in enumerate(Gs):
        ax = axes.flat[i]; k = G["k"]; p = np.arange(1, k + 1)
        _, lo, hi = T.wilson(G["obs"] * G["events"], G["events"])
        ax.bar(p, 100 * G["obs"], width=0.78, color="#C9C9C9", edgecolor="none", zorder=2)
        ax.errorbar(p, 100 * G["obs"], yerr=[100 * (G["obs"] - lo), 100 * (hi - G["obs"])], fmt="none", ecolor=T.MUTED, elinewidth=0.5,
                    capsize=1.0 if k <= 20 else 0, capthick=0.5, zorder=3)
        if G["exch"] is not None:
            ax.plot([0.4, k + 0.6], [100 * G["exch"][0]] * 2, color=T.C_EXCH, ls=T.PRED["exch"]["ls"], lw=0.9, zorder=4)
        if G["pred2"] is not None:
            ax.plot(p, 100 * G["pred2"], color=T.C_SEMI2, ls=T.PRED["semi2"]["ls"], lw=1.0, zorder=5)
        if G["pred1"] is not None:
            ax.plot(p, 100 * G["pred1"], color=T.C_SEMI1, lw=1.0, marker="o" if k <= 20 else None, ms=2.4, mfc=T.C_SEMI1, mec="white", mew=0.3, zorder=6)
        ax.set_xlim(0.3, k + 0.7)
        ax.set_xticks(list(range(1, k + 1)) if k <= 10 else [1] + list(range(10, k + 1, 10)) if k > 20 else [1] + list(range(5, k + 1, 5)))
        ax.set_ylim(0, 100 * top * 1.36)                               # headroom: the two text lines sit above every mark
        ax.set_title(T.case_label(c, long=nc <= 3), fontsize=T.FONT, pad=3, loc="left")
        # rank-1 share: observed, and predicted with one and (where fitted) two sibling scales
        two = T.ds_key(c) == "agnews" and G["pred2"] is not None          # the model of the paper's tables (for the log line)
        txt = "rank 1: %.0f%% observed, %s insertions" % (100 * G["obs"][0], format(G["events"], ","))
        pp = [(lab, v[0]) for lab, v in (("1 scale", G["pred1"]), ("2 scales", G["pred2"])) if v is not None]
        if len(pp) == 2 and round(100 * pp[0][1]) == round(100 * pp[1][1]):
            txt += "\npredicted %.0f%% (1 or 2 scales)" % (100 * pp[0][1])
        elif pp:
            txt += "\npredicted " + ", ".join("%.0f%% (%s)" % (100 * v, lab) for lab, v in pp)
        ax.text(0.97, 0.96, txt, transform=ax.transAxes, ha="right", va="top", fontsize=T.TINY, color=T.INK, linespacing=1.25, zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.grid(True, axis="y", color=T.LIGHT, lw=0.4)
        print("  %s: %-8s k=%d, %d insertions: rank-1 share observed %.3f, predicted %.3f (1 scale) %.3f (2 scales; shown: %s), uniform %.3f" % (
            name, c, k, G["events"], G["obs"][0], G["pred1"][0] if G["pred1"] is not None else float("nan"),
            G["pred2"][0] if G["pred2"] is not None else float("nan"), "2" if two else "1", 1.0 / k))
    T.hide_unused(axes, n)
    for ax in axes.flat[max(n - nc, 0):n]:
        ax.set_xlabel("old rank of the insertion")
    fig.supylabel("share of in-ball insertions (%)", fontsize=T.FONT)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Patch(fc="#C9C9C9", ec="none", label="observed"),
          Line2D([], [], color=T.C_SEMI1, lw=1.0, marker="o", ms=2.4, label="predicted, 1 sibling scale"),
          Line2D([], [], color=T.C_SEMI2, lw=1.0, ls=T.PRED["semi2"]["ls"], label="predicted, 2 scales"),
          Line2D([], [], color=T.C_EXCH, lw=0.9, ls=T.PRED["exch"]["ls"], label="exchangeable (uniform)")]
    fig.legend(handles=hs, loc="outside upper center", ncol=lcol, handlelength=2.0, columnspacing=1.4 if lcol >= 4 else 1.0)
    return T.save(fig, name, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis")), ("--cases", dict(default=None)), ("--cols", dict(type=int, default=None)),
                          ("--name", dict(default=NAME))])
