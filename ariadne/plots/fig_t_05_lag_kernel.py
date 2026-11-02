"""Figure T5: the lag kernel.  Siblings resolved by their distance in the stream.

Reads     D3a_law_real/*.law.pkl          in-ball insertions (ev: t, r_k, lag, same team) and static neighbour lists
          D3b_law_analysis/lag.json       per cell and lag bucket: overlapping commits, in-ball insertions, h, D
          D3b_law_analysis/static_*.json  the same kernel estimated from the static stream (no execution), and
                                          its predictions of in-ball insertions per read
          D3b_law_analysis/transfer_*.json  predicted and observed in-ball insertions per read of every transfer
Shows     (a, b) ridgelines, one data set each (cells of all concurrency levels and seeds pooled): density of
              y = ln(r_k / t) of the in-ball concurrent insertions in each stream-lag bucket (lag 1, 2 to 3, ...,
              32 or more, other team).  Every ridge is a histogram scaled so that its fitted exponential starts
              at 1: dark line = fitted Exp(D_b) with D_b = 1 / mean(y) (the effective dimension of the bucket,
              printed at the right with the number of insertions); grey dashed = the background shape, the
              density of ln(r_{k+1} / r_i) of the stored neighbours, scaled the same way.  The ridges narrow
              towards the background shape as the lag grows: the effective dimension grows with the lag.
          (c) in-ball probability h of an overlapping commit per lag bucket, one line per concurrency level
              (95% Wilson intervals, cells of the seeds pooled), one panel per data set.  The lines coincide: the
              kernel does not depend on the concurrency level (lines from light to dark with the number of
              agents).  Dashed: the exchangeable reference k / (N + 1).  Hollow diamonds: the estimate from the
              static stream alone (same symbol as in panel d).
          (d) in-ball insertions per read, predicted without refitting against observed, for every transfer
              (window, k, lag, time, params) and for the static kernel; the diagonal is a perfect prediction and
              the grey band is a factor of 1.25 around it.
Paper     main text or appendix, lag-kernel figure (both columns).

Options   --datasets locomo,agnews   --k 10   --min-events 8
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_05_lag_kernel"
MODE_MARK, MODE_LABEL = T.MODE_MARK, T.MODE_LABEL


# ------------------------------------------------------------------------------------------------ data
def pooled_events(paths, k):
    y, lag, same, stat = [], [], [], []
    for p in paths:
        L = T.law_load(p)
        if int(L["k"]) != k:
            continue
        ev = np.asarray(L["ev"], dtype=float).reshape(-1, 10)
        if len(ev):
            y.append(np.maximum(np.log(np.maximum(ev[:, 6], 1e-3) / np.maximum(ev[:, 3], 1e-3)), 0.0)); lag.append(ev[:, 7]); same.append(ev[:, 8] > 0.5)
        r = np.asarray(L["ro"], dtype=float)
        if r.shape[1] > k:
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.log(r[:, k:k + 1] / np.maximum(r[:, :k], 1e-6)).ravel()
            stat.append(s[np.isfinite(s) & (s >= 0)])
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return cat(y), cat(lag), cat(same).astype(bool), cat(stat)


def bucket_mask(b, lag, same):
    if b == "other":
        return ~same
    lo, hi = T.BUCKET_RANGE[b]
    return same & (lag >= lo) & (lag <= hi)


def lag_table(LJ):
    """{case: {bucket: (inball, overlap)}} pooled over seeds, {case: exchangeable h}, {case: k}."""
    tab, exch, ks = {}, {}, {}
    for case, entries in T.by_case(LJ).items():
        t = {}
        for _, e in entries:
            for b, v in (e.get("buckets") or {}).items():
                a = t.setdefault(b, [0, 0]); a[0] += int(v.get("inball") or 0); a[1] += int(v.get("overlap") or 0)
        tab[case] = t
        exch[case] = float(np.mean([T.fnum(e.get("exchangeable_h")) for _, e in entries])); ks[case] = int(entries[0][1].get("k", 0))
    return tab, exch, ks


# ------------------------------------------------------------------------------------------------ panels
def ridges(ax, y, lag, same, stat, min_events, first):
    rows = [(b, bucket_mask(b, lag, same)) for b in T.BUCKETS]
    rows = [(b, m) for b, m in rows if int(m.sum()) >= min_events]
    if not rows:
        ax.text(0.5, 0.5, "too few in-ball insertions", transform=ax.transAxes, ha="center", va="center", fontsize=T.SMALL, color=T.MUTED)
        ax.set_yticks([]); return
    xmax = float(np.clip(np.percentile(y, 98.5), 0.25, 3.0))
    nb = 26 if len(y) >= 1500 else 18
    e = np.linspace(0, xmax, nb + 1); xs = np.linspace(0, xmax, 200)
    bg = None
    if len(stat) >= 200:
        hb = np.histogram(stat, np.linspace(0, xmax, 61))[0].astype(float)
        mid = 0.5 * (np.linspace(0, xmax, 61)[1:] + np.linspace(0, xmax, 61)[:-1])
        hb = hb / max(hb[:2].mean(), 1e-12)                                # scaled to start at 1 like the ridges
        rate = 1.0 / max(float(stat.mean()), 1e-9)
        bg = (mid, hb, rate)
    import matplotlib.patheffects as pe
    H = 0.86; n = len(rows)
    for i, (b, m) in enumerate(rows):
        base = n - 1 - i; yy = y[m]; col = T.bucket_color(b)
        Db = 1.0 / max(float(yy.mean()), 1e-9)
        h = np.histogram(yy, e)[0] / (len(yy) * np.diff(e)) / Db           # the fitted exponential starts at 1
        sx, sy = np.repeat(e, 2)[1:-1], np.repeat(np.minimum(h, 1.16), 2)
        ax.fill_between(sx, base, base + H * sy, color=T.lighten(col, 0.45) if b != "other" else "#DADADA", lw=0, zorder=2 + i * 0.01)
        ax.plot(sx, base + H * sy, color=T.darken(col, 0.15), lw=0.5, zorder=3)
        if bg is not None:
            ax.plot(bg[0], base + H * np.minimum(bg[1], 1.16), color=T.MUTED, lw=0.7, ls=(0, (2.5, 1.3)), zorder=4)
        ax.plot(xs, base + H * np.exp(-Db * xs), color=T.INK, lw=0.9, zorder=5)
        ax.plot([0, xmax], [base, base], color=T.INK, lw=0.4, zorder=6)
        halo = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.85)]
        ax.text(1.0, base + 0.56, "$D$ = %.1f" % Db, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=T.SMALL, color=T.INK,
                path_effects=halo, zorder=8)
        ax.text(1.0, base + 0.30, "%s ins." % format(len(yy), ","), transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=T.TINY,
                color=T.MUTED, path_effects=halo, zorder=8)
    ax.set_yticks([n - 1 - i + 0.02 for i in range(n)])
    ax.set_yticklabels([T.BUCKET_LABEL[b] for b, _ in rows], va="bottom")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(-0.05, n - 1 + H * 1.2); ax.set_xlim(0, xmax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("$y = \\ln(r_k\\,/\\,t)$")
    if bg is not None:
        ax.text(1.0, n - 1 + H * 1.2, "background: $D$ = %.1f" % bg[2], transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=T.TINY, color=T.MUTED)


def h_panel(ax, cases, tab, exch, static, color):
    xs = np.arange(len(T.BUCKETS))
    n = len(cases)
    from matplotlib.lines import Line2D
    keys = []
    for j, case in enumerate(cases):
        t = tab[case]
        x, p, lo, hi = [], [], [], []
        for i, b in enumerate(T.BUCKETS):
            ib, ov = t.get(b, (0, 0))
            if ov >= 30 and ib >= 1:
                pp, l, h = T.wilson(ib, ov)
                x.append(i + (j - (n - 1) / 2) * 0.13); p.append(float(pp)); lo.append(float(l)); hi.append(float(h))
        if not x:
            continue
        shade = T.lighten(color, 0.55 * (1 - (j + 1) / n)) if n > 1 else color
        same = [i for i, xx in enumerate(x) if T.BUCKETS[int(round(xx - (j - (n - 1) / 2) * 0.13))] != "other"]
        ax.plot([x[i] for i in same], [p[i] for i in same], color=shade, lw=0.9, zorder=3)
        T.caps(ax, x, p, lo, hi, color=shade, ms=3.0, marker="o", capsize=1.2, zorder=4)
        keys.append((case, Line2D([], [], color=shade, lw=0.9, marker="o", ms=3.0, mfc=shade, mec=shade)))
    if static:
        sx, sy = [], []
        for i, b in enumerate(T.BUCKETS):
            v = (static.get("buckets") or {}).get(b) or {}
            if v.get("h") and (v.get("overlap") or 0) >= 30 and v["h"] > 0:
                sx.append(i); sy.append(v["h"])
        ax.plot(sx, sy, ls="none", marker="D", ms=4.6, mfc="none", mec=T.INK, mew=0.7, zorder=6)
    ex = [exch[c] for c in cases if np.isfinite(exch[c])]
    if ex:
        ax.axhline(float(np.mean(ex)), color=T.C_EXCH, lw=0.8, ls=(0, (4, 1.6)), zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(["1", "2-3", "4-7", "8-15", "16-31", "32+", "other"], fontsize=T.TINY, rotation=45, ha="right",
                                         rotation_mode="anchor")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if hi / lo < 60 else (1.0,)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10))); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.grid(True, axis="y", color=T.LIGHT, lw=0.4)
    return (float(np.mean(ex)) if ex else None), keys


def main(results="results", suffix="", out="figures", datasets="locomo,agnews", k=10, min_events=8):
    plt = T.setup()
    laws = T.law_paths(results, "D3a_law_real", suffix)
    LJ = T.jload(results, "D3b_law_analysis", "lag.json", suffix) or {}
    ST = T.jglob(results, "D3b_law_analysis", "static_*.json", suffix)
    TR = T.jglob(results, "D3b_law_analysis", "transfer_*.json", suffix)
    if not laws and not LJ:
        return T.skip(NAME, "no law files and no lag.json")
    groups = {}
    for name, p in laws.items():
        L = T.law_load(p)
        if int(L["k"]) == k:
            groups.setdefault(T.ds_key(L), []).append(p)
    tab, exch, ks = lag_table(LJ)
    keys = [d.strip() for d in datasets.split(",")]
    keys = [d for d in keys if d in groups or any(T.ds_key(c) == d and ks[c] == k for c in tab)]
    if not keys:
        return T.skip(NAME, "no data for the data sets %s at k = %d" % (datasets, k))
    static_of = {}
    for fn, S in ST.items():
        names = list((S.get("predictions") or {}).keys())
        d = T.ds_key(names[0]) if names else T.ds_key(fn.replace("static_", ""))
        if int(S.get("k", k)) == k:
            static_of[d] = S

    nd = len(keys)
    fig = plt.figure(figsize=(T.COL2, 2.75))
    gs = fig.add_gridspec(1, nd + 2, width_ratios=[0.95] * nd + [1.32, 1.0])
    # (a, b) ridgelines
    for i, d in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        y, lag, same, stat = pooled_events(groups.get(d, []), k)
        ridges(ax, y, lag, same, stat, min_events, i == 0)
        T.panel(ax, "abcdef"[i], "%s: where in the ball" % T.ds_style(d)["label"])
    # (c) in-ball probability per lag bucket
    sub = gs[0, nd].subgridspec(nd, 1, hspace=0.10)
    axh = None
    for i, d in enumerate(keys):
        ax = fig.add_subplot(sub[i, 0], sharex=axh)
        axh = axh or ax
        cases = [c for c in T.case_sort(tab) if T.ds_key(c) == d and ks[c] == k]
        ref, keys_h = h_panel(ax, cases, tab, exch, static_of.get(d), T.ds_style(d)["color"])
        # shade key (light to dark = fewer to more agents): one entry per concurrency level, in an empty corner: below
        # the short-lag points in the upper panel, above the exchangeable line in the lower one
        ag = [T.agents_of(c) for c in cases]
        by_agents = bool(ag) and all(ag) and len(set(ag)) == len(ag)
        names = [("%d" % T.agents_of(c)) if by_agents else c for c, _ in keys_h]
        title = T.ds_style(d)["label"] + (", agents:" if by_agents else ":")
        if keys_h:
            ncol = len(keys_h) if len(keys_h) <= 3 else 2
            leg = ax.legend([h for _, h in keys_h], names, title=title, loc="lower left", bbox_to_anchor=(0.01, 0.41 if i == 0 else 0.04),
                            ncol=ncol, fontsize=T.TINY, title_fontsize=T.TINY, handlelength=1.5, handletextpad=0.35, columnspacing=0.8,
                            labelspacing=0.15, borderaxespad=0.0, borderpad=0.2, frameon=False, alignment="left")
            leg.get_title().set_color(T.darken(T.ds_style(d)["color"], 0.15))
            leg.set_zorder(7)
        if i < nd - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("stream lag (same team)")
        if i == 0:
            T.panel(ax, "abcdef"[nd], "in-ball probability $h$")
            if ref:
                ax.annotate("exchangeable", (0.02, ref), xycoords=ax.get_yaxis_transform(), xytext=(0, 2), textcoords="offset points", fontsize=T.TINY,
                            color=T.MUTED, ha="left", va="bottom")
    # (d) predicted against observed in-ball insertions per read
    ax = fig.add_subplot(gs[0, nd + 1])
    pts = []
    for fn, t in TR.items():
        if t.get("pred_m") is not None and t.get("obs_m") is not None:
            mode = str(t.get("mode", "?"))
            pts.append((mode, T.ds_key(str(t.get("dst") or t.get("src"))), float(t["obs_m"]), float(t["pred_m"])))
    for fn, S in ST.items():
        for nm, pr in (S.get("predictions") or {}).items():
            if pr.get("pred_m") is not None and pr.get("obs_m") is not None:
                pts.append(("static", T.ds_key(nm), float(pr["obs_m"]), float(pr["pred_m"])))
    if pts:
        v = np.array([[p[2], p[3]] for p in pts]); v = v[v > 0]
        lo, hi = float(v.min()) / 1.5, float(v.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], color=T.INK, lw=0.6, zorder=1)
        ax.fill_between([lo, hi], [lo / 1.25, hi / 1.25], [lo * 1.25, hi * 1.25], color="#EDEDED", lw=0, zorder=0)
        # marks are small, half transparent and edged in the colour of their corpus; marks of different transfers
        # that predict the same cell are moved sideways by a fixed step per kind of transfer (log axis)
        dodge = {"window": -1, "lag": 1, "k": -1, "time": 2, "params": 0, "static": -1}
        step = 10 ** 0.038
        from matplotlib.colors import to_rgba
        for mode, d, o, pm in pts:
            col = T.ds_style(d)["color"]
            ox = o * step ** dodge.get(mode, 0)
            if mode == "static":
                ax.plot([ox], [pm], ls="none", marker="D", ms=3.6, mfc="none", mec=col, mew=0.8, zorder=5)
            else:
                ax.plot([ox], [pm], ls="none", marker=MODE_MARK.get(mode, "o"), ms=5.0 if mode == "params" else 3.3,
                        mfc=to_rgba(col, 0.55), mec=T.darken(col, 0.1), mew=0.55, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        from matplotlib.lines import Line2D
        modes = [m for m in MODE_MARK if any(p[0] == m for p in pts)]
        hs = [Line2D([], [], ls="none", marker=MODE_MARK[m], ms=5.0 if m == "params" else 3.6 if m != "static" else 4.0,
                     mfc="none" if m == "static" else to_rgba(T.MUTED, 0.55), mec=T.MUTED, mew=0.8 if m == "static" else 0.55,
                     label=MODE_LABEL[m]) for m in modes]
        # the band's note becomes a legend entry with a grey swatch (same words), so that it cannot sit on a data point
        from matplotlib.patches import Patch
        hs.append(Patch(facecolor="#E6E6E6", edgecolor="none", label="gray band: within 25%"))
        ax.set_anchor("N")
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.42, -0.34), ncol=2, fontsize=T.TINY, handletextpad=0.1, labelspacing=0.25,
                  columnspacing=0.8, borderaxespad=0.0)
        dsn = []
        for p in pts:
            if p[1] not in dsn:
                dsn.append(p[1])
        for j, d in enumerate(dsn):
            ax.text(0.97, 0.04 + 0.085 * (len(dsn) - 1 - j), T.ds_style(d)["label"], transform=ax.transAxes, ha="right", va="bottom", fontsize=T.TINY,
                    color=T.ds_style(d)["color"])
    else:
        ax.text(0.5, 0.5, "no transfer files", transform=ax.transAxes, ha="center", va="center", fontsize=T.SMALL, color=T.MUTED)
    ax.set_xlabel("observed"); ax.set_ylabel("predicted without refit")
    T.panel(ax, "abcdef"[nd + 1], "insertions per read")
    if pts:
        worst = max(pts, key=lambda p: abs(np.log(p[3] / p[2])) if p[2] > 0 and p[3] > 0 else 0)
        print("  %s: %d predictions of in-ball insertions per read; largest ratio %.2f (%s, %s)" % (NAME, len(pts), worst[3] / worst[2], worst[0], worst[1]))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--datasets", dict(default="locomo,agnews")), ("--k", dict(type=int, default=10)),
                          ("--min-events", dict(type=int, default=8, help="fewest in-ball insertions of a lag bucket that gets a ridge"))])
