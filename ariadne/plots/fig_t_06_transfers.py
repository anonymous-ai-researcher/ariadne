"""Figure T6: prediction without refitting.  What transfers and what does not.

Reads     D3b_law_analysis/transfer_*.json  (mode window | k | lag | time | params: "cs", "obs", "se", "pred", "err",
          "ref": {"exch", "err_exch", "self", "err_self"}, "src", "dst", "scales")
          D3b_law_analysis/static_*.json    (predictions of the kernel estimated from the static stream: per target
          "cs", "obs", "se", "pred", "err")
          D3b_law_analysis/fits.json        (only for the reference values of the static predictions: the target's
          own fit single.err_semi1 / semi1 and the exchangeable-only prediction single.err_exch / exch)
Shows     (a) one row per transfer, grouped by what is transferred: the max absolute error of the predicted
              conflict probability over the c grid, in percentage points (filled marker, colour of the target
              data set, keyed above the panel; the marker shape is the kind of transfer).  The grey bar spans the two references of the same target: its own fit (ring, the best
              one can hope for) and the exchangeable-only prediction that knows nothing about siblings (cross).
              A transfer works when its marker sits at the ring; the parameters of another corpus land at or
              beyond the cross.
          (b) for every kind of transfer the observed conflict curve of one target (dots, +/- 1.96 standard
              errors) with the transferred prediction (solid), the target's own fit (dotted) and the
              exchangeable-only prediction (grey dashed).  The transfer shown is the WORST of its kind
              (--pick worst | best | median).
Paper     main text, next to (or instead of) the table "prediction without refitting" (both columns).

Options   --pick worst   --exp D3b_law_analysis
"""
import re

import numpy as np

from . import _theory as T

NAME = "fig_t_06_transfers"
# names of the transfer modes as the text uses them (group headers of (a), panel titles of (b))
MODE_LONG = {"static": "from the static stream (no execution)", "lag": "lag kernel (another concurrency level)",
             "window": "pooled parameters (window)", "k": "across $k$", "time": "across time (1st to 2nd half)",
             "params": "across collections"}
MODE_SHORT = {"static": "static stream", "lag": "lag kernel", "window": "pooled parameters", "k": "across $k$",
              "time": "across time", "params": "across collections"}


def collect(TR, ST, F):
    """List of dicts: mode, src, dst, label, err, err_self, err_exch, cs, obs, se, pred, self, exch, ds."""
    out = []
    for fn, t in TR.items():
        if t.get("err") is None or not t.get("cs"):
            continue
        mode = str(t.get("mode")); src, dst = T.split_cell(t.get("src"))[0], T.split_cell(t.get("dst") or t.get("src"))[0]
        ref = t.get("ref") or {}
        if mode == "time":
            lab = "%s, 1st $\\rightarrow$ 2nd half" % src
        else:
            lab = "%s $\\rightarrow$ %s" % (src, dst)
        if t.get("scales") and int(t["scales"]) > 1:
            lab += " (%d scales)" % int(t["scales"])
        out.append(dict(mode=mode, src=src, dst=dst, label=lab, err=float(t["err"]), err_self=T.fnum(ref.get("err_self")), err_exch=T.fnum(ref.get("err_exch")),
                        cs=T.arr(t["cs"]), obs=T.arr(t["obs"]), se=T.arr(t.get("se") or [None] * len(t["cs"])), pred=T.arr(t["pred"]),
                        self_=T.arr(ref["self"]) if ref.get("self") else None, exch=T.arr(ref["exch"]) if ref.get("exch") else None, ds=T.ds_key(dst)))
    for fn, S in ST.items():
        for nm, pr in (S.get("predictions") or {}).items():
            if pr.get("err") is None or not pr.get("cs"):
                continue
            dst = T.split_cell(nm)[0]; fe = (F.get(nm) or {}).get("single") or {}
            same_grid = fe and list((F.get(nm) or {}).get("cs") or []) == list(pr["cs"])
            sk = "semi2" if T.ds_key(dst) == "agnews" else "semi1"     # the row's own model: two sibling scales on AG News (Table 5)
            out.append(dict(mode="static", src="stream", dst=dst, label="stream $\\rightarrow$ %s" % dst, err=float(pr["err"]),
                            err_self=T.fnum(fe.get("err_" + sk)), err_exch=T.fnum(fe.get("err_exch")), cs=T.arr(pr["cs"]), obs=T.arr(pr["obs"]),
                            se=T.arr(pr.get("se") or [None] * len(pr["cs"])), pred=T.arr(pr["pred"]),
                            self_=T.arr(fe[sk]) if same_grid and fe.get(sk) else None,
                            exch=T.arr(fe["exch"]) if same_grid and fe.get("exch") else None, ds=T.ds_key(dst)))
    order = {m: i for i, m in enumerate(T.MODES)}
    out.sort(key=lambda r: (order.get(r["mode"], 99), T.ds_style(r["ds"]).get("order", 9), r["label"]))
    return out


def dots(ax, rows):
    y = 0.0; ticks, labels, heads = [], [], []
    last = None
    for r in rows:
        if r["mode"] != last:
            y += 0.75 if last is not None else 0.0
            heads.append((y - 0.62, r["mode"])); last = r["mode"]
        col = T.ds_style(r["ds"])["color"]
        a, b = r["err_self"], r["err_exch"]
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([a, b], [y, y], color="#D8D8D8", lw=2.6, solid_capstyle="round", zorder=1)
        if np.isfinite(b):
            ax.plot([b], [y], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, zorder=2)
        if np.isfinite(a):
            ax.plot([a], [y], ls="none", marker="o", ms=3.8, mfc="white", mec=T.MUTED, mew=0.7, zorder=3)
        ax.plot([r["err"]], [y], ls="none", marker=T.MODE_MARK.get(r["mode"], "o"), ms=6.6 if r["mode"] == "params" else 4.6, mfc=col, mec="white",
                mew=0.5, zorder=4)
        ticks.append(y); labels.append(r["label"]); y += 1.0
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=T.TINY)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(y - 0.4, -1.15)
    xmax = max([r["err"] for r in rows] + [r["err_exch"] for r in rows if np.isfinite(r["err_exch"])]) * 1.06
    ax.set_xlim(0, xmax)
    for yy, m in heads:
        ax.text(0.0, yy, MODE_LONG.get(m, T.MODE_LONG.get(m, m)), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=T.TINY, color=T.INK,
                fontweight="bold")
    ax.grid(True, axis="x", color=T.LIGHT, lw=0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("max |error| of the predicted conflict probability (points)")


def curve(ax, r, first_col, last_row):
    col = T.ds_style(r["ds"])["color"]; x = r["cs"] - 1.0
    if r["exch"] is not None:
        ax.plot(x, r["exch"], color=T.C_EXCH, ls=(0, (4, 1.6)), lw=0.9, zorder=1)
    if r["self_"] is not None:
        ax.plot(x, r["self_"], color=T.INK, ls=(0, (1, 1.2)), lw=0.8, zorder=2)
    ax.plot(x, r["pred"], color=col, lw=1.2, zorder=3)
    se = np.where(np.isfinite(r["se"]), r["se"], 0.0)
    T.caps(ax, x, r["obs"], r["obs"] - 1.96 * se, r["obs"] + 1.96 * se, color=T.INK, ms=2.6, zorder=5, capsize=1.2)
    T.c_axis(ax, "x", ticks=(1.02, 1.05, 1.1, 1.2, 1.5, 2.0), lim=(1.014, 2.25), label="tolerance $c$" if last_row else None)
    ax.set_xticklabels(["1.02", "1.05", "1.1", "1.2", "1.5", "2"])
    ax.tick_params(axis="x", labelsize=5.5)          # six labels, 1.05 included, on a narrow panel
    ax.set_ylim(0, None)
    ax.grid(True, axis="y", color=T.LIGHT, lw=0.4)
    ax.set_title(MODE_SHORT.get(r["mode"], T.MODE_LABEL.get(r["mode"], r["mode"])), fontsize=T.SMALL, pad=2.5, loc="left")
    ax.text(0.97, 0.95, "%s\nmax |err| %.1f" % (re.sub(r" \(\d+ scales\)$", "", r["label"]), r["err"]), transform=ax.transAxes, ha="right", va="top", fontsize=T.TINY, color=T.INK,
            linespacing=1.25)


def main(results="results", suffix="", out="figures", exp="D3b_law_analysis", pick="worst"):
    plt = T.setup()
    TR = T.jglob(results, exp, "transfer_*.json", suffix)
    ST = T.jglob(results, exp, "static_*.json", suffix)
    F = T.jload(results, exp, "fits.json", suffix, quiet=True) or {}
    rows = collect(TR, ST, F)
    if not rows:
        return T.skip(NAME, "no transfer_*.json or static_*.json in %s" % T.exp_dir(results, exp, suffix))
    modes = [m for m in T.MODES if any(r["mode"] == m for r in rows)]
    reps = []
    for m in modes:
        rs = sorted([r for r in rows if r["mode"] == m], key=lambda r: r["err"])
        reps.append(rs[-1] if pick == "worst" else rs[0] if pick == "best" else rs[len(rs) // 2])
    nm = len(reps); ncol = 3 if nm > 4 else 2 if nm > 1 else 1; nrow = int(np.ceil(nm / ncol))
    h = max(2.6, 0.70 + 0.125 * (len(rows) + 0.75 * len(modes)))
    W = T.COL2
    fig = plt.figure(figsize=(W, h))
    fig.set_layout_engine("none")
    top, bottom = 1 - 0.34 / h, 0.36 / h
    axd = fig.add_axes([1.10 / W, bottom, 2.02 / W, top - bottom])
    dots(axd, rows)
    fig.canvas.draw()                                                     # left margin from the widest row label
    r = fig.canvas.get_renderer()
    wl = max([t.get_window_extent(r).width for t in axd.get_yticklabels()] + [1.0]) / fig.dpi
    left = min(wl + 0.10, 1.75)
    axd.set_position([left / W, bottom, (3.14 - left) / W, top - bottom])
    fig.text(0.02 / W, 1 - 0.02 / h, "(a)", fontsize=T.FONT, fontweight="bold", ha="left", va="top")
    fig.text(0.02 / W + 0.026, 1 - 0.02 / h, "every transfer", fontsize=T.FONT, ha="left", va="top")
    from matplotlib.lines import Line2D
    # colour key: the filled (transferred) marks and the lines of (b) have the colour of the TARGET data set
    tgt = [d for d in ("locomo", "agnews") if any(r["ds"] == d for r in rows)] + sorted({r["ds"] for r in rows} - {"locomo", "agnews"})
    hs = [Line2D([], [], ls="none", marker="s", ms=4.2, mfc=T.ds_style(d)["color"], mec="white", mew=0.5,
                 label="to %s" % T.ds_style(d)["label"]) for d in tgt]
    hs += [Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec=T.MUTED, mew=0.7, label="own fit"),
           Line2D([], [], ls="none", marker="x", ms=3.6, mec=T.GREY, mew=0.9, label="exchangeable only")]
    axd.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=len(hs), fontsize=T.TINY, handletextpad=0.1, borderaxespad=0.2,
               columnspacing=0.7)
    sub = fig.add_gridspec(nrow, ncol, left=3.62 / W, right=1 - 0.05 / W, top=top - 0.20 / h, bottom=bottom, wspace=0.16, hspace=0.34)
    from matplotlib.legend_handler import HandlerTuple
    hb = [tuple(Line2D([], [], color=T.ds_style(d)["color"], lw=1.2) for d in tgt),
          Line2D([], [], color=T.INK, ls=(0, (1, 1.2)), lw=0.8, label="own fit"),
          Line2D([], [], color=T.C_EXCH, ls=(0, (4, 1.6)), lw=0.9, label="exchangeable only"),
          Line2D([], [], ls="none", marker="o", ms=2.6, mfc=T.INK, mec=T.INK, label="observed ($\\pm$1.96 s.e.)")]
    lb = ["transferred (target's color)", "own fit", "exchangeable only", "observed ($\\pm$1.96 s.e.)"]
    fig.legend(handles=hb, labels=lb, loc="upper left", bbox_to_anchor=(3.30 / W, 1 - 0.16 / h), ncol=4, fontsize=T.TINY, handlelength=2.0,
               handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    ax0 = None
    for i, r in enumerate(reps):
        ax = fig.add_subplot(sub[i // ncol, i % ncol], sharex=ax0, sharey=ax0)
        ax0 = ax0 or ax
        curve(ax, r, i % ncol == 0, i // ncol == nrow - 1 or i + ncol >= nm)
        if not (i // ncol == nrow - 1 or i + ncol >= nm):
            ax.tick_params(labelbottom=False)
        if i % ncol:
            ax.tick_params(labelleft=False)
    fig.text(3.30 / W, 1 - 0.02 / h, "(b)", fontsize=T.FONT, fontweight="bold", ha="left", va="top")
    fig.text(3.30 / W + 0.026, 1 - 0.02 / h, "the %s transfer of each kind: conflict probability (%%) against $c$" % pick, fontsize=T.FONT, ha="left", va="top")
    for m in modes:
        rs = [r for r in rows if r["mode"] == m]
        print("  %s: %-7s %2d transfers, max |err| %s points (own fit %s, exchangeable only %s)" % (
            NAME, m, len(rs), "/".join("%.1f" % r["err"] for r in rs), "/".join("%.1f" % r["err_self"] for r in rs),
            "/".join("%.1f" % r["err_exch"] for r in rs)))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D3b_law_analysis")), ("--pick", dict(default="worst", choices=["worst", "best", "median"]))])
