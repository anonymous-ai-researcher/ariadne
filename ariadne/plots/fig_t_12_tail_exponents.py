"""Figure T12: the tail of the conflict curve is a power law whose exponent is the sibling dimension.

Reads     D3b_law_analysis/fits.json  ("tail": {"obs": {"1.1-1.2": [slope, lo, hi], "1.2-1.5": [...], "1.5-2": [...]},
          "exch": {...}, "model1": {...}, "model2": {...}, "Ds1", "Ds2"}; slopes are -d ln P(conflict) / d ln c of the
          single-insertion reads, with bootstrap intervals for the observed ones)
Shows     one panel per range of c.  Horizontal axis: the fitted sibling dimension D_s of a configuration (one
          scale; fitted on insertion distances only).  Vertical axis: the tail exponent in that range of c:
          observed (marker in the colour of the data set, 95% interval: over seeds when there are three or more,
          else the bootstrap interval of the file), predicted by the law with one sibling scale (black ring) and
          two (dark grey diamond; neutral colours, so that they are not read as data sets), and by the exchangeable-only model (grey cross: the exponent that the local dimension of
          the stored data would give).  One point per configuration (all concurrency levels and k).  On the diagonal the exponent equals D_s, the limit for large c; the
          closed form says P(conflict) ~ c^(-D_s) there.  Configurations whose observed exponent is undefined in
          a range (no conflict left at the larger c) are missing from that panel.
Paper     appendix (tail exponent); supports the sentence that D_s can be read off the tail.

Options   --exp D3b_law_analysis
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_12_tail_exponents"
RANGES = ("1.1-1.2", "1.2-1.5", "1.5-2")
# the law's markers are neutral (hollow black / dark grey), so that they cannot be read as observed points of a data set
C_LAW1, C_LAW2 = T.INK, T.MUTED


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis"):
    plt = T.setup()
    F = T.jload(results, exp, "fits.json", suffix)
    if not F:
        return T.skip(NAME, "no fits.json")
    B = T.by_case(F)
    rows = []
    for case in T.case_sort(B):
        es = [e for _, e in B[case] if e.get("tail") and e["tail"].get("obs")]
        if not es:
            continue
        ds = T.nmean([T.fnum(e["tail"].get("Ds1")) for e in es])
        r = {"case": case, "Ds": ds}
        for w in RANGES:
            ob = [(e["tail"]["obs"].get(w) or [None, None, None]) for e in es]
            m, lo, hi, n = T.pooled_ci([o[0] for o in ob], [o[1] for o in ob], [o[2] for o in ob])
            r[w] = (m, lo, hi)
            for key in ("exch", "model1", "model2"):
                vals = [T.fnum((e["tail"].get(key) or {}).get(w)) for e in es]
                r[key + w] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        rows.append(r)
    if not rows:
        return T.skip(NAME, "no configuration with a tail block")
    fig, axes = plt.subplots(1, len(RANGES), figsize=(T.COL2, 2.45), sharex=True, sharey=True)
    allv = [r["Ds"] for r in rows]
    for w in RANGES:
        for r in rows:
            allv += [v for v in (r[w][0], r["exch" + w], r["model1" + w]) if np.isfinite(v) and v > 0]
    lo, hi = max(min(allv) / 1.5, 0.2), max(allv) * 1.5
    for i, (ax, w) in enumerate(zip(axes, RANGES)):
        ax.plot([lo, hi], [lo, hi], color=T.INK, lw=0.6, zorder=1)
        for r in rows:
            st = T.ds_style(r["case"]); x = r["Ds"]
            if not np.isfinite(x):
                continue
            ys = [v for v in (r[w][0], r["exch" + w], r["model1" + w], r["model2" + w]) if np.isfinite(v) and v > 0]
            if len(ys) > 1:
                ax.plot([x, x], [min(ys), max(ys)], color="#D0D0D0", lw=0.6, zorder=1)
            if np.isfinite(r["exch" + w]) and r["exch" + w] > 0:
                ax.plot([x], [r["exch" + w]], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, zorder=2)
            if np.isfinite(r["model2" + w]) and r["model2" + w] > 0:
                ax.plot([x], [r["model2" + w]], ls="none", marker="D", ms=3.4, mfc="white", mec=C_LAW2, mew=0.8, zorder=3)
            if np.isfinite(r["model1" + w]) and r["model1" + w] > 0:
                ax.plot([x], [r["model1" + w]], ls="none", marker="o", ms=4.0, mfc="white", mec=C_LAW1, mew=0.8, zorder=4)
            m, l, h = r[w]
            if np.isfinite(m) and m > 0:
                T.caps(ax, [x], [m], [max(l, lo) if np.isfinite(l) else np.nan], [h], color=st["color"], marker=st["marker"], ms=3.8, capsize=1.6, zorder=6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        T.log_plain(ax)
        ax.grid(True, color=T.LIGHT, lw=0.4)
        a, b = w.split("-")
        T.panel(ax, "abc"[i], "$c$ from %s to %s" % (a, b))
        ax.set_xlabel("fitted sibling dimension $D_s$ (1 scale)")
    axes[0].set_ylabel("tail exponent  $-\\mathrm{d}\\ln P\\,/\\,\\mathrm{d}\\ln c$")
    from matplotlib.lines import Line2D
    dsn = []
    for r in rows:
        if T.ds_key(r["case"]) not in dsn:
            dsn.append(T.ds_key(r["case"]))
    hs = [Line2D([], [], ls="none", marker=T.ds_style(d)["marker"], ms=3.8, mfc=T.ds_style(d)["color"], mec=T.ds_style(d)["color"],
                 label="observed, %s" % T.ds_style(d)["label"]) for d in dsn] + [
          Line2D([], [], ls="none", marker="o", ms=4.0, mfc="white", mec=C_LAW1, mew=0.8, label="law, 1 scale"),
          Line2D([], [], ls="none", marker="D", ms=3.4, mfc="white", mec=C_LAW2, mew=0.8, label="law, 2 scales"),
          Line2D([], [], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, label="exchangeable only"),
          Line2D([], [], color=T.INK, lw=0.6, label="exponent = $D_s$")]
    fig.legend(handles=hs, loc="outside upper center", ncol=min(len(hs), 4 if len(hs) > 6 else 6), handletextpad=0.3, columnspacing=1.0)
    for r in rows:
        print("  %s: %-8s D_s %.2f | observed exponent %s | exchangeable only %s" % (
            NAME, r["case"], r["Ds"], " ".join("%s: %.2f" % (w, r[w][0]) for w in RANGES), " ".join("%.2f" % r["exch" + w] for w in RANGES)))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis"))])
