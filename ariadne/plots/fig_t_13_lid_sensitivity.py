"""Figure T13: how much does the closed form depend on the estimate of the local intrinsic dimension D?

Reads     D3b_law_analysis/fits.json  ("lid_sensitivity": {"median" | "pooled" | "twonn": {"D", "err_closed"}}: the
          closed form evaluated with three estimates of D, and its max absolute error against the observed
          conflict probability of the single-insertion reads; "single": err_semi1 and err_exch for reference)
Shows     one row per configuration.
          (a) the three estimates of D: median of the per-read maximum-likelihood estimates (circle), the pooled
              estimate 1 / mean ln(r_{k+1} / r_i) used everywhere else (square), and TwoNN from the two nearest
              neighbours (triangle).
          (b) max absolute error of the closed form (percentage points) with each of them, same markers; the
              black tick is the semi-parametric form, which uses the actual neighbour distances of every read
              and needs no D; the grey cross is the exchangeable-only prediction.
          Values are means over seeds; the thin line spans the three estimators.
Paper     appendix (limits of the closed form: it needs a neighbourhood-scale estimate of D; TwoNN looks at a
          smaller scale and is off).

Options   --exp D3b_law_analysis
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_13_lid_sensitivity"
EST = (("median", "o", "median per-read MLE"), ("pooled", "s", "pooled MLE (used)"), ("twonn", "^", "TwoNN"))


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis"):
    plt = T.setup()
    F, _ = T.fits_or_quick(results, exp, suffix, None, NAME)
    if not F:
        return T.skip(NAME, "no fits.json")
    B = T.by_case(F)
    rows = []
    for case in T.case_sort(B):
        es = [e for _, e in B[case] if e.get("lid_sensitivity")]
        if not es:
            continue
        r = {"case": case}
        for key, _, _ in EST:
            r["D_" + key] = T.nmean([T.fnum((e["lid_sensitivity"].get(key) or {}).get("D")) for e in es])
            r["e_" + key] = T.nmean([T.fnum((e["lid_sensitivity"].get(key) or {}).get("err_closed")) for e in es])
        # semi-parametric reference: the model of the paper's tables, two sibling scales on AG News, one elsewhere (as fig_t_06)
        sk = "err_semi2" if T.ds_key(case) == "agnews" else "err_semi1"
        r["semi"] = T.nmean([T.fnum((e.get("single") or {}).get(sk)) for e in es])
        if not np.isfinite(r["semi"]):
            r["semi"] = T.nmean([T.fnum((e.get("single") or {}).get("err_semi1")) for e in es])
        r["exch"] = T.nmean([T.fnum((e.get("single") or {}).get("err_exch")) for e in es])
        rows.append(r)
    if not rows:
        return T.skip(NAME, "no configuration with a lid_sensitivity block")
    n = len(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(T.COL1, 0.72 + 0.20 * n), sharey=True, gridspec_kw=dict(width_ratios=[0.8, 1.0]))
    for i, r in enumerate(rows):
        col = T.ds_style(r["case"])["color"]
        for ax, pre in ((a1, "D_"), (a2, "e_")):
            vals = [r[pre + k] for k, _, _ in EST if np.isfinite(r[pre + k])]
            if vals:
                ax.plot([min(vals), max(vals)], [i, i], color=T.lighten(col, 0.55), lw=1.6, zorder=2, solid_capstyle="round")
            for k, mk, _ in EST:
                if np.isfinite(r[pre + k]):
                    ax.plot([r[pre + k]], [i], ls="none", marker=mk, ms=3.9, mfc=col if k == "pooled" else "white", mec=col, mew=0.8, zorder=4)
        if np.isfinite(r["semi"]):
            a2.plot([r["semi"], r["semi"]], [i - 0.30, i + 0.30], color=T.INK, lw=1.3, zorder=5, solid_capstyle="butt")
        if np.isfinite(r["exch"]):
            a2.plot([r["exch"]], [i], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, zorder=3)
    a1.set_yticks(range(n)); a1.set_yticklabels([r["case"] for r in rows])
    a1.set_ylim(n - 0.5, -0.5); a1.tick_params(axis="y", length=0); a2.tick_params(axis="y", length=0)
    a1.set_xscale("log")
    lo, hi = a1.get_xlim()
    tk = [v for v in (2, 3, 5, 7, 10, 15, 20, 30, 50) if lo <= v <= hi]
    a1.set_xticks(tk); a1.set_xticklabels(["%g" % v for v in tk]); a1.minorticks_off()
    a1.set_xlabel("estimate of $D$"); a2.set_xlabel("max |error| (points)")
    a2.set_xlim(0, None); a2.spines["left"].set_visible(False)
    for ax in (a1, a2):
        ax.grid(True, axis="x", color=T.LIGHT, lw=0.4)
    T.panel(a1, "a"); T.panel(a2, "b")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="none", marker=mk, ms=3.9, mfc=T.MUTED if k == "pooled" else "white", mec=T.MUTED, mew=0.8, label=lab) for k, mk, lab in EST]
    hs += [Line2D([], [], color=T.INK, lw=1.3, label="semi-param., 1 scale\n(2 scales on AG News)"), Line2D([], [], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, label="exchangeable only")]
    fig.legend(handles=hs, loc="outside upper center", ncol=2, handletextpad=0.3, columnspacing=1.0, fontsize=T.TINY)
    for r in rows:
        print("  %s: %-8s D median %.1f pooled %.1f TwoNN %.1f | closed-form max |err| %.1f / %.1f / %.1f | semi-parametric %.1f" % (
            NAME, r["case"], r["D_median"], r["D_pooled"], r["D_twonn"], r["e_median"], r["e_pooled"], r["e_twonn"], r["semi"]))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis"))])
