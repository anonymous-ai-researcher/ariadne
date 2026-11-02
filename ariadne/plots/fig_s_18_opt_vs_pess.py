"""Figure S18: optimistic validation against pessimistic predicate locks on the read ball.

Reads     C10_vs_pessimistic/summary.jsonl  (extra_calls_per_100, lock_wait_mean_ms, ball.deadlocks per protocol:
          av-<c> = Algorithm AV with exact validation at tolerance c, ball-<c> = ball predicate locks at tolerance c,
          lock-long, none; one case per workload).
Shows     one panel per workload: extra model calls per 100 tasks (y) against the mean time a task waits for locks
          (x), both on symmetric-log axes so that zero has a place.  Each design traces a path over its tolerance from
          strict (c = 1, large mark) to tolerant (light to dark for both designs; inner marks small); the two ends of each path are labelled with c; the numbers in parentheses
          at the ball-lock ends are deadlocks per 100 tasks (every deadlock victim re-reads and thinks again).  Horizontal
          intervals are clipped at 0 (a wait is never negative).  Caps are intervals over seeds.
Paper     main text or appendix, with the table "optimistic vs pessimistic".
"""
import copy

from . import style as st, load as ld, panels as pn

from matplotlib.colors import LinearSegmentedColormap, to_hex

# ball locks over c in the colours of every other figure (style.policy_style: c = 1 dark #A04000, c = 1.1 #CC9466),
# continued lighter for looser tolerances, so that one tolerance has one colour across the paper
BALL_CMAP = LinearSegmentedColormap.from_list("ball", ["#A04000", "#CC9466", "#EBC7A5"])


def ball_color(c):
    try:
        c = float(c)
    except (TypeError, ValueError):
        c = 1.0
    if c <= 1.0:
        return "#A04000"
    p0 = st.c_position(1.1)
    return to_hex(BALL_CMAP(min(1.0, 0.5 + 0.5 * (st.c_position(c) - p0) / max(1e-9, 1.0 - p0)) if c >= 1.1 else 0.5 * st.c_position(c) / p0))


EXP = "C10_vs_pessimistic"
NAME = "fig_s_18_opt_vs_pess"
X, Y = "lock_wait_mean_ms", "extra_calls_per_100"


def draw(ax, R, annotate=True):
    protos = st.policy_order(ld.values(R, "proto"))
    fams = {}
    for p in protos:
        fams.setdefault(st.family(p), []).append(p)
    labels, obst, segs = [], [], []
    for fam, ps in fams.items():
        ps = sorted(ps, key=lambda p: st.spec_c(p) or 0)
        pts = []
        for p in ps:
            Rp = ld.where(R, proto=p)
            sx, sy = ld.stats([ld.num(r.get(X)) for r in Rp]), ld.stats([ld.num(r.get(Y)) for r in Rp])
            if sx.n and sy.n:
                pts.append((p, sx, sy))
        if not pts:
            continue
        base = st.policy_line(pts[0][0])
        if fam in ("cadm", "cert", "ball") and len(pts) > 1:
            # the path over c on top of its marks (thin), so that it can be followed through marks that bunch together
            ax.plot([q[1].mean for q in pts], [q[2].mean for q in pts], color=st.policy_style("cadm:1.1")["color"] if fam != "ball" else ball_color(1.1),
                    lw=0.7, ls=base["ls"], zorder=3.4)
            segs += [((a[1].mean, a[2].mean), (b[1].mean, b[2].mean)) for a, b in zip(pts, pts[1:])]
        for i, (p, sx, sy) in enumerate(pts):
            c = st.spec_c(p)
            end = i == 0 or i == len(pts) - 1           # the ends of a path are full size, the inner marks small
            kw = st.policy_marker(p, ms=st.MS + (1.8 if (c is not None and c <= 1.0) else 0.3 if end else -1.4))
            if fam == "ball":
                kw.update(color=ball_color(c), mfc=ball_color(c), mec=ball_color(c))
            if sx.n > 1 and sx.lo < 0:                 # a wait is never negative: clip the interval of the mean at 0
                sx = copy.copy(sx); sx.lo = 0.0
            if end:
                st.point_ci(ax, None, sy, xs=sx, zorder=3, **kw)
            else:                                      # inner marks: lighter, thinner intervals with short caps
                ye = [[sy.mean - sy.lo], [sy.hi - sy.mean]] if sy.n > 1 and sy.hi > sy.lo else None
                xe = [[sx.mean - sx.lo], [sx.hi - sx.mean]] if sx.n > 1 and sx.hi > sx.lo else None
                if ye is not None or xe is not None:
                    ax.errorbar([sx.mean], [sy.mean], yerr=ye, xerr=xe, fmt="none", ecolor=st.lighten(kw["color"], 0.4), elinewidth=0.45,
                                capsize=0.9, zorder=2.5)
                ax.plot([sx.mean], [sy.mean], zorder=3, **kw)
            obst.append((sx.mean, sy.mean))
            if sx.n > 1 and sx.hi > sx.lo:             # the intervals, for the label placement
                segs.append(((sx.lo, sy.mean), (sx.hi, sy.mean)))
            if sy.n > 1 and sy.hi > sy.lo:
                segs.append(((sx.mean, sy.lo), (sx.mean, sy.hi)))
            dead = ""
            if fam == "ball":
                d = ld.stats([100.0 * ld.num(ld.get(r, "ball.deadlocks"), 0) / max(ld.num(r.get("tasks"), 1), 1) for r in ld.where(R, proto=p)])
                dead = (" (%.0f)" if d.mean >= 100 else " (%.3g)") % d.mean if d.n and d.mean > 0 else ""
            # only the two ends of each path (c = 1 and the most tolerant c) are labelled: the inner labels crowd
            if annotate and c is not None and (i == 0 or i == len(pts) - 1):
                labels.append((sx.mean, sy.mean, "$c$=%g%s" % (c, dead), st.INK))
    return labels, obst, segs


def place_labels(ax, items, marks, segs):
    """Labels (x, y, text, colour) near their marks without covering marks, intervals or other labels: each label tries
    eight directions at growing distances and, when it has to stand off, gets a thin leader line to its mark."""
    import math
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tr = ax.transData
    k = 72.0 / fig.dpi                                       # display pixels -> points
    axb = ax.get_window_extent(r)
    M = [tuple(tr.transform(m)) for m in marks]
    L = []                                                   # points along paths and intervals
    for a, b in segs:
        pa, pb = tr.transform(a), tr.transform(b)
        n = max(int(math.hypot(*(pb - pa)) * k / 2.0), 2)
        L += [tuple(pa + (pb - pa) * t / n) for t in range(n + 1)]
    dirs = [("ne", 1, 1), ("se", 1, -1), ("nw", -1, 1), ("sw", -1, -1), ("e", 1, 0), ("w", -1, 0), ("n", 0, 1), ("s", 0, -1)]
    placed, out = [], []
    P = [tuple(tr.transform((it[0], it[1]))) for it in items]
    order = sorted(range(len(items)), key=lambda i: -sum(1 for m in M if math.hypot(m[0] - P[i][0], m[1] - P[i][1]) * k < 30))
    res = {}
    for i in order:
        px, py = P[i]
        t = ax.text(0, 0, items[i][2], fontsize=st.TINY)
        bb = t.get_window_extent(r); w, h = bb.width, bb.height
        t.remove()
        best = None
        for ri, rad in enumerate((3.5, 9.0, 16.0, 24.0)):
            for di, (d, dx, dy) in enumerate(dirs):
                ox, oy = dx * rad / k, dy * rad / k
                if dx and dy:
                    ox, oy = ox * 0.75, oy * 0.75
                x0 = px + ox - (0 if dx > 0 else w if dx < 0 else w / 2)
                y0 = py + oy - (0 if dy > 0 else h if dy < 0 else h / 2)
                box = (x0, y0, x0 + w, y0 + h)
                c = ri * 2.0 + di * 0.1
                if box[0] < axb.x0 or box[2] > axb.x1 or box[1] < axb.y0 or box[3] > axb.y1:
                    c += 50
                g = 3.0 / k
                for j, m in enumerate(M):
                    if box[0] - g < m[0] < box[2] + g and box[1] - g < m[1] < box[3] + g and (m[0], m[1]) != (px, py):
                        c += 12
                gl = 3.8 / k                                        # clear of the caps at the ends of the intervals
                nl = sum(1 for q in L if box[0] - gl < q[0] < box[2] + gl and box[1] - gl < q[1] < box[3] + gl)
                c += (12.0 + 0.5 * min(nl, 10)) if nl else 0.0      # a label never sits on a path or an interval
                for q in placed:
                    if not (box[2] + 1 < q[0] or q[2] + 1 < box[0] or box[3] + 1 < q[1] or q[3] + 1 < box[1]):
                        c += 30
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2     # another mark nearer to the label: ambiguous
                own = math.hypot(cx - px, cy - py)
                if rad < 4 and any(math.hypot(cx - m[0], cy - m[1]) < own + 2.0 / k for m in M if (m[0], m[1]) != (px, py)):
                    # (a label that stands off has a leader line, which says which mark it belongs to)
                    c += 15
                if best is None or c < best[0]:
                    best = (c, box, rad)
        placed.append(best[1]); res[i] = best
    for i, it in enumerate(items):
        _, box, rad = res[i]
        px, py = P[i]
        arrow = dict(arrowstyle="-", lw=0.4, color=st.MUTED, shrinkA=0.5, shrinkB=2.5) if rad > 4 else None
        # the text box is anchored at its lower left corner; the leader goes from the nearest corner/edge midpoint
        out.append(ax.annotate(it[2], (it[0], it[1]), xytext=((box[0] - px) * k, (box[1] - py) * k), textcoords="offset points",
                               ha="left", va="bottom", fontsize=st.TINY, color=it[3] if len(it) > 3 else st.INK,
                               annotation_clip=False, zorder=7, arrowprops=arrow))
    return out


def scales(axes, obst):
    """Symmetric-log axes (zero has a place) with limits that hold every mark of every panel."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    xs = [o[0] for o in obst]; ys = [o[1] for o in obst]
    xhi = max(xs) * 2.4 if xs and max(xs) > 0 else 1.0
    yhi = max(ys) * 2.4 if ys and max(ys) > 0 else 10.0
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.35); ax.set_yscale("symlog", linthresh=1.0, linscale=0.35)
        ax.set_xlim(-0.03, xhi); ax.set_ylim(-0.3, yhi)
        for axis, lim, first in ((ax.xaxis, xhi, 0.1), (ax.yaxis, yhi, 1.0)):
            t, ticks = first, [0.0]
            while t <= lim:
                ticks.append(t); t *= 10
            axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v)); axis.set_minor_locator(NullLocator())
        st.grid(ax, "both")


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    cases = st.case_order(ld.values(R, "case"))
    nr, nc = pn.grid_of(len(cases), 3 if len(cases) > 2 else 2)
    fig, axes = st.subplots(nr, nc, cols=1 if nc <= 2 else 2, aspect=(0.60 if nc <= 2 else 0.34) * nr + 0.08, sharex=True, sharey=True, squeeze=False)
    flat = [a for row in axes for a in row]
    todo = []
    for i, (ax, c) in enumerate(zip(flat, cases)):
        todo.append((ax,) + draw(ax, ld.where(R, case=c)))
        title = st.dataset_label(c)
        if "agents" not in title:                  # e.g. "fan-out": give the number of concurrent agents
            n_ag = ld.values(ld.where(R, case=c), "clients")
            if len(n_ag) == 1 and n_ag[0]:
                title += ", %g agents" % ld.num(n_ag[0])
        st.panel_label(ax, "abcdefgh"[i], title)
    for ax in flat[len(cases):]:
        ax.set_visible(False)
    scales(flat[:len(cases)], [o for t in todo for o in t[2]])
    for row in axes:
        row[0].set_ylabel("extra model calls per 100 tasks")
    fig.supxlabel("mean wait for locks per task (ms)", fontsize=st.FONT)
    protos = st.policy_order(ld.values(R, "proto"))
    reps, seen = [], set()
    for p in sorted(protos, key=lambda p: (st.policy_style(p)["order"][0], abs((st.spec_c(p) or 1.1) - 1.1))):   # one entry per design
        f = st.family(p)
        if f not in seen:
            seen.add(f); reps.append(p)
    lab = {"cadm": "validation (AV), $c$ = 1 (large mark) and above", "cert": "validation (AV), $c$ = 1 (large mark) and above", "ball": "ball predicate locks, same $c$ (deadlocks per 100 tasks)"}
    from matplotlib.lines import Line2D
    extra = []
    for p in [p for p in reps if st.family(p) == "ball"]:
        extra.append(Line2D([], [], label=lab["ball"], **dict(st.policy_line(p), color=ball_color(1.1), mfc=ball_color(1.1), mec=ball_color(1.1))))
    pn.policy_legend(fig, [p for p in reps if st.family(p) != "ball"], label=lambda p: lab.get(st.family(p), st.pretty_math(p)), extra=extra,
                     handlelength=1.6)
    for ax, labels, obst, segs in todo:
        place_labels(ax, labels, obst, segs)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
