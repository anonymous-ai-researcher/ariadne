"""Figure S6: dose response of the SERIALIZABLE anomaly on pgvector HNSW.

Reads     A6_ssi_soundness/hammer_grid.jsonl (one line per (m, ef_construction): events_with_duplicate out of
          events_with_rows, insert_ms_mean); when the grid file is absent, the single point of hammer.jsonl.
          Controls, when present: hammer_advisory.jsonl (advisory lock around check-then-insert),
          hammer_seqscan.jsonl (no vector index), and the hammer lines of patched_library.jsonl (written by the manual
          step with the patched pgvector).  hammer_ivfflat.jsonl is not drawn: with one probe most of its double commits
          are recall misses (a fresh search does not find both rows), not the missed conflict; tab_ssi_soundness lists it.
Shows     share of facts stored twice by two SERIALIZABLE transactions (no debugger), with Wilson 95% intervals,
          against the mean duration of an index insertion: a slower insertion widens the window of Figure S5.
          Controls are drawn as hollow marks.
Paper     appendix (dose response); panel (b) of the combined figure S8.
"""
from . import style as st, load as ld

EXP = "A6_ssi_soundness"
DEFAULT_BUILD = (16, 64)       # (M, ef_construction) of every other pgvector run
NAME = "fig_s_06_ssi_dose"
# (file, label, marker, is_control): controls remove one condition of the window and should be at zero
CONTROLS = [("hammer_advisory.jsonl", "advisory lock", "s", True), ("hammer_seqscan.jsonl", "no index (seq. scan)", "D", True),
            ("patched_library.jsonl", "patched HNSW", "o", True)]


def _pool(L):
    """Pool the lines of one configuration: (insert ms, duplicates, events)."""
    k = sum(int(r.get("events_with_duplicate", 0)) for r in L)
    n = sum(int(r.get("events_with_rows", r.get("pairs", 0))) for r in L)
    ms = sum(float(r.get("insert_ms_mean", 0)) * int(r.get("events_with_rows", r.get("pairs", 1))) for r in L) / max(n, 1)
    return ms, k, n


def points(results, suffix):
    G = [r for r in ld.rows(results, EXP, "hammer_grid.jsonl", suffix, quiet=True) if "events_with_rows" in r]
    src = "hammer_grid.jsonl"
    if not G:
        G = [r for r in ld.rows(results, EXP, "hammer.jsonl", suffix, quiet=True) if "events_with_rows" in r and r.get("mode", "plain") == "plain"]
        src = "hammer.jsonl"
        if G:
            ld.note("%s: no hammer_grid.jsonl, plotting the single point of hammer.jsonl" % EXP)
    by = {}
    for r in G:
        by.setdefault((r.get("m"), r.get("ef_construction")), []).append(r)
    P = [(key,) + _pool(L) for key, L in by.items()]
    return sorted(P, key=lambda t: t[1]), src


def place_beside(ax, items, marks, gap=4.0):
    """Label each mark (items: (x, y, text)) right beside it: east, west, below or above, whichever stays inside the axes,
    covers no mark, label or the legend, and does not end close to another mark at the same height (where it could be
    read as that mark's label)."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = ax.transData
    axb = ax.get_window_extent(r)
    leg = ax.get_legend()
    lb = leg.get_window_extent(r) if leg is not None else None
    M = [tuple(tr.transform(m)) for m in marks]
    k = 72.0 / fig.dpi                                  # pixels per point is fig.dpi / 72
    g = gap / k
    placed = []
    for x, y, text in items:
        px, py = tr.transform((x, y))
        t = ax.text(0, 0, text, fontsize=st.TINY)
        bb = t.get_window_extent(r); w, h = bb.width, bb.height
        t.remove()
        cands = [("e", px + g, py - h / 2), ("w", px - g - w, py - h / 2), ("s", px - w / 2, py - g - h), ("n", px - w / 2, py + g)]
        best = None
        for i, (d, x0, y0) in enumerate(cands):
            if d in "sn":                                # slide along the axis edge to stay inside
                x0 = min(max(x0, axb.x0 + 1), axb.x1 - 1 - w)
            box = (x0, y0, x0 + w, y0 + h)
            c = i * 0.1
            if box[0] < axb.x0 or box[2] > axb.x1 or box[1] < axb.y0 or box[3] > axb.y1:
                c += 50
            for m in M:
                if (m[0], m[1]) == (px, py):
                    continue
                if box[0] - g < m[0] < box[2] + g and box[1] - g < m[1] < box[3] + g:
                    c += 10
                elif d in "ew" and abs(m[1] - py) < 2.5 * h and (box[2] < m[0] < box[2] + 4 * g if d == "e" else box[0] - 4 * g < m[0] < box[0]):
                    c += 5                               # ends next to another mark: ambiguous
            for q in placed + ([(lb.x0, lb.y0, lb.x1, lb.y1)] if lb is not None else []):
                if not (box[2] < q[0] or q[2] < box[0] or box[3] < q[1] or q[3] < box[1]):
                    c += 20
            if best is None or c < best[0]:
                best = (c, box)
        box = best[1]
        placed.append(box)
        ax.annotate(text, (x, y), xytext=((box[0] - px) * k, (box[1] - py) * k), textcoords="offset points",
                    ha="left", va="bottom", fontsize=st.TINY, color=st.MUTED, annotation_clip=False, zorder=7)


def draw(ax, results, suffix, compact=False):
    P, _ = points(results, suffix)
    if not P:
        return False
    kw = st.policy_marker("ssi", ms=4.2)
    xs = [p[1] for p in P]
    for (m, efc), ms, k, n in P:
        p, lo, hi = ld.wilson(k, n)
        ax.errorbar([ms], [100 * p], yerr=[[max(0.0, 100 * (p - lo))], [max(0.0, 100 * (hi - p))]], fmt="none", ecolor=kw["color"], elinewidth=0.7, capsize=1.8, zorder=2)
        ax.plot([ms], [100 * p], zorder=3, **kw)
    ymax = max(100 * ld.wilson(p[2], p[3])[2] for p in P)
    seen = False
    control_xy = []
    for fn, lab, mk, is_control in CONTROLS:
        C = [r for r in ld.rows(results, EXP, fn, suffix, quiet=True) if "events_with_rows" in r
             and r.get("test", "hammer") == "hammer" and r.get("mode", "plain") in ("plain", "advisory")]
        if not C:
            continue
        ms, k, n = _pool(C)
        p, lo, hi = ld.wilson(k, n)
        ax.errorbar([ms], [100 * p], yerr=[[max(0.0, 100 * (p - lo))], [max(0.0, 100 * (hi - p))]], fmt="none", ecolor=st.MUTED, elinewidth=0.6, capsize=1.5, zorder=2)
        ax.plot([ms], [100 * p], ls="none", marker=mk, ms=3.6, mfc="white" if is_control else st.MUTED, mec=st.INK, mew=0.8, zorder=3,
                label=("control: " if is_control else "") + lab)
        xs.append(ms); seen = True
        control_xy.append((ms, 100 * p))
    if max(xs) / max(min(xs), 1e-9) > 8:
        ax.set_xscale("log"); ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35); st.log_ticks(ax, "x")
    ax.set_ylim(min(-2.0, -0.03 * ymax), max(ymax * 1.12, 5))
    ax.set_xlabel("mean duration of an index insertion (ms)")
    ax.set_ylabel("facts stored twice (%)")
    st.grid(ax, "y")
    if compact:                          # panel (b) of figure S8: unchanged
        if seen:
            ax.legend(loc="upper left", handlelength=1.0, fontsize=st.TINY)
        lab = lambda m, e: "%s, %s" % (m, e)
        prefer = ("se", "nw", "ne", "sw", "e", "w")
    else:
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], **dict(kw, ls="none"), label="HNSW ($M$, efc)")] + ax.get_legend_handles_labels()[0]
        ax.legend(handles=hs, loc="upper left", handlelength=1.0, fontsize=st.TINY, title="efc = ef_construction",
                  title_fontsize=st.TINY, alignment="left")
        lab = lambda m, e: "$M$=%s, efc=%s" % (m, e) + (" (default)" if (ld.num(m), ld.num(e)) == DEFAULT_BUILD else "")
        prefer = ("w", "e", "s", "n")    # beside its own mark, never between two marks
    items, obst = [], []
    for (m, efc), ms, k, n in P:
        p, lo, hi = ld.wilson(k, n)
        items.append((ms, 100 * p, lab(m, efc)))
        obst += [(ms, 100 * lo), (ms, 100 * hi)]
    if compact:
        st.label_points(ax, items, obstacles=obst, pad=3.5, prefer=prefer)
    else:
        marks = [(it[0], it[1]) for it in items] + [(ms_, y_) for ms_, y_ in control_xy]
        place_beside(ax, items, marks)
    if compact:
        ax.text(0.02, 0.98, "labels: m, ef_construction", transform=ax.transAxes, ha="left", va="top", fontsize=st.TINY, color=st.MUTED)
    return True


def main(results="results", suffix="", out="figures"):
    fig, ax = st.subplots(cols=1, aspect=0.60)
    if not draw(ax, results, suffix):
        st.plt.close(fig)
        return ld.skip(NAME, "no hammer_grid.jsonl or hammer.jsonl in %s" % ld.exp_dir(results, EXP, suffix))
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
