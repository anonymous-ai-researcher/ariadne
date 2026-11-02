"""Figure T1 (teaser): nearest is not repeatable.  What one concurrent insertion does to one real read.

Reads     D3a_law_real/<cell>.pkl and <cell>.law.pkl (one instrumented run; default: the first of L4, L8, L2,
          A64, A16 that has both cases), and the workload of that run (rebuilt with ariadne.audit.workload_of;
          the texts of the two items are quoted when the data set carries texts).
Shows     two real reads of the run, each with exactly one concurrent insertion inside its k-th-neighbour ball:
          (a) a SIBLING (same team, at most --sib-lag positions away in the stream) that lands at old rank 1 or 2
              and breaks c-admissibility of the held answer at c = --c (default 1.1);
          (b) a BACKGROUND insertion (another team, or at least --bg-lag positions away) that lands at a high
              rank and leaves the held answer c-admissible.
          Left of each pair, the scene around the query in two dimensions.  RADII ARE EXACT: the query is at the
          origin and every point is drawn at its true metric distance from the query.  ANGLES ARE A PROJECTION:
          the direction of a point is the direction of its difference vector to the query in the plane of the
          first two principal components of the difference vectors of the k neighbours and the inserted point
          (centred at their mean, because the component that all neighbours share carries no information; the
          picture is then rotated so that the inserted point sits at a fixed angle).  Distances between two
          neighbours are therefore not to scale, distances to the query are.  Numbers next to the neighbours are
          their ranks in the held answer and, after the arrow, in the fresh top-k.  The circle is the k-th-neighbour
          ball of the read.  The shaded disc is the conflict set of the characterization theorem for THIS
          neighbour list and this c: the distances t at which an insertion breaks c-admissibility (computed rank
          by rank with _theory.conflict_intervals; it is the disc t < r_i / c of the last gap r_i > c r_{i-1}).
          Small grey dots are the next stored items outside the ball.
          Right of each pair, the per-rank test of c-admissibility on a log distance axis: for every rank the
          fresh exact distance (ring; diamond for the inserted item), the window up to c times that distance
          (bar), and the held distance (dot); the held answer is c-admissible iff every dot is inside its bar
          (a dot outside is drawn in vermillion with the ratio held / fresh).  Ticks between the rows show the
          items that moved down one rank.  Above the pair: the texts of the two items when the data set has
          texts, and how the inserted item relates to the read.  On real data no single insertion carries a
          label "sibling" or "background": the two scenes are chosen by the stream lag and the team as stated.
Paper     first figure, across both columns.  Every mark is a real item of the run; nothing is schematic.

Options   --cell L4_s1   --c 1.1   --row-sib N --row-bg N (rows of the law file; default: chosen automatically,
          the choice is printed)   --no-text   --sib-lag 3   --bg-lag 16
"""
import html, os, re

import numpy as np

from . import _theory as T

NAME = "fig_t_01_teaser"
VSTRETCH = 1.12            # the scene is a little taller than wide: room for the corner labels above and below the ball
VBOT = 1.20                # ... and a little more below it, where rank labels must pass between the ball and the bottom corner texts
EXP = "D3a_law_real"
PREFER = ("L4", "L8", "L2", "A64", "A16")


# ------------------------------------------------------------------------------------------------ choosing the reads
def candidates(L, c, kind, sib_lag=3, bg_lag=16):
    """Rows of ev (single-insertion reads) that qualify for a scene, best first by a cheap score."""
    ev = np.asarray(L["ev"], dtype=float).reshape(-1, 10); k = int(L["k"]); rho = np.asarray(L["rho"])
    out = []
    for e in ev[ev[:, 1] == 1]:
        x, p, t, r1, rk, lag, same = int(e[0]), int(e[2]), e[3], e[5], e[6], int(e[7]), e[8] > 0.5
        r = np.asarray(L["ro"])[x, :k]
        if not np.all(np.isfinite(r)) or t <= 1e-3:
            continue
        Rc = T.conflict_radius(r[None, :], k, [c])[0, 0]
        if kind == "sib":
            if not (same and lag <= sib_lag and p <= 2 and rho[x] > c + 1e-9):
                continue
            score = (p - 1) * 1.0 + (lag - 1) * 0.3 + abs(t / r1 - 0.55) * 2.0 + abs(Rc / rk - 0.68) * 3.0
        else:
            if not (((not same) or lag >= bg_lag) and p >= max(3, k // 2 + 1) and rho[x] <= c - 1e-9 and t >= Rc):
                continue
            score = abs(t / rk - 0.92) * 6.0 + abs(Rc / rk - 0.68) * 3.0 + (0.0 if not same else 0.3)
        out.append((score, x, e))
    out.sort(key=lambda z: z[0])
    return out


def geometry(ex, wl, recs, L, x, e, extra=8):
    """Positions of the scene of law row x.  Returns a dict, or None when the run file and the law file disagree."""
    k = int(L["k"]); j = int(L["task"][x])
    use = [r for r in recs if r.get("first") is not None]
    rec = use[x] if x < len(use) and use[x]["j"] == j else next((r for r in use if r["j"] == j), None)
    if rec is None:
        return None
    s = int(rec["first"][2]); sr = min(int(rec["first"][3]), s)
    q = np.asarray(wl.X[j], dtype="float32")
    ids, dist = ex.topk(q[None, :], [sr], k + extra)
    ids, dist = ids[0], dist[0]
    ok = np.isfinite(dist)
    ids, dist = ids[ok], dist[ok]
    if len(dist) < k or not np.allclose(dist[:k], np.asarray(L["ro"])[x, :k], rtol=1e-4, atol=1e-5):
        return None
    src = int(e[9])
    V = np.stack([np.asarray(ex.vec(int(i)), dtype=float) for i in ids] + [np.asarray(wl.X[src], dtype=float)])
    M = V - q[None, :].astype(float)
    t = float(ex._dist(q[None, :], np.asarray(wl.X[src], dtype="float32")[None, :])[0, 0])
    rad = np.concatenate([dist, [t]])
    basis_rows = np.concatenate([np.arange(k), [len(ids)]])               # the k neighbours and the inserted point
    mu = M[basis_rows].mean(axis=0)                                       # centred: the common direction carries no information
    _, _, Vt = np.linalg.svd(M[basis_rows] - mu, full_matrices=False)
    P = (M - mu) @ Vt[:2].T
    ang = np.arctan2(P[:, 1], P[:, 0])
    texts = None
    if getattr(wl, "texts", None):
        texts = (wl.texts[wl.nb + j], wl.texts[wl.nb + src])
    return {"k": k, "x": x, "task": j, "src": src, "r": dist[:k].copy(), "t": t, "p": int(e[2]), "lag": int(e[7]), "same": bool(e[8] > 0.5),
            "rad": rad, "ang": ang, "n_out": len(ids) - k, "nc": int(np.asarray(L["nc"])[x]), "texts": texts,
            "rho": float(np.asarray(L["rho"])[x])}


def legibility(G, rot):
    """Smallest distance between two marks of the scene, in units of r_k (larger is better)."""
    k = G["k"]; idx = list(range(k)) + [len(G["rad"]) - 1]
    xy = np.stack([G["rad"][idx] * np.cos(G["ang"][idx] + rot), G["rad"][idx] * np.sin(G["ang"][idx] + rot)], axis=1) / G["r"][-1]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)); d[np.diag_indices(len(d))] = np.inf
    return float(d.min())


def choose(ex, wl, recs, L, c, kind, row=None, tries=30, **kw):
    cand = candidates(L, c, kind, **kw)
    if row is not None:
        ev = np.asarray(L["ev"], dtype=float).reshape(-1, 10)
        hit = [e for e in ev if int(e[0]) == int(row)]
        cand = [(0.0, int(row), hit[0])] if hit else []
    best = None
    for score, x, e in cand[:tries]:
        G = geometry(ex, wl, recs, L, x, e)
        if G is None:
            continue
        leg = legibility(G, 0.0)
        val = score - 5.0 * min(leg, 0.20)
        if best is None or val < best[0]:
            best = (val, G)
    return None if best is None else best[1]


# ------------------------------------------------------------------------------------------------ drawing
def _halo(txt, lw=1.6, fg="white"):
    import matplotlib.patheffects as pe
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground=fg)])
    return txt


def draw_scene(ax, G, c, color, target_deg):
    from matplotlib.patches import Circle, Wedge
    k = G["k"]; r = G["r"]; rk = r[-1]; t = G["t"]
    rot = np.deg2rad(target_deg) - G["ang"][-1]
    ang = G["ang"] + rot
    X, Y = G["rad"] * np.cos(ang), G["rad"] * np.sin(ang)
    lim = 1.17 * rk
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * VBOT, lim * VSTRETCH); ax.set_aspect("equal"); ax.axis("off")

    # the ball, the admissible ring and the conflict set of the theorem
    ax.add_patch(Circle((0, 0), rk, fc=T.lighten(T.C_ADM, 0.92), ec="none", zorder=0))
    iv = T.conflict_intervals(r, c)
    for lo, hi in iv:
        ax.add_patch(Wedge((0, 0), hi, 0, 360, width=hi - lo, fc=T.lighten(T.C_CONF, 0.80), ec="none", zorder=1))
        ax.add_patch(Circle((0, 0), hi, fc="none", ec=T.C_CONF, lw=0.8, ls=(0, (3, 1.6)), zorder=2))
        if lo > 0:
            ax.add_patch(Circle((0, 0), lo, fc="none", ec=T.C_CONF, lw=0.8, ls=(0, (3, 1.6)), zorder=2))
    ax.add_patch(Circle((0, 0), rk, fc="none", ec=T.INK, lw=0.9, zorder=3))

    # next stored items outside the ball
    out = slice(k, k + G["n_out"])
    keep = G["rad"][out] < lim * 0.985
    ax.plot(X[out][keep], Y[out][keep], ls="none", marker="o", ms=1.9, mfc="#BDBDBD", mec="none", zorder=3)

    # query, radius to the inserted point
    ax.plot([0, X[-1]], [0, Y[-1]], color=color, lw=0.8, zorder=4, solid_capstyle="round")
    ax.plot([0], [0], marker="o", ms=3.4, mfc=T.INK, mec="white", mew=0.5, zorder=8)
    # the k held neighbours (their rank labels are placed by place_labels, once every other mark and text exists)
    for i in range(k):
        evicted = (i == k - 1)
        ax.plot([X[i]], [Y[i]], marker="o", ms=4.3, mfc="white" if evicted else "#3A3A3A", mec="#3A3A3A" if evicted else "white",
                mew=0.8 if evicted else 0.6, zorder=6)

    # the inserted point
    ax.plot([X[-1]], [Y[-1]], marker="D", ms=5.2, mfc=color, mec="white", mew=0.7, zorder=9)
    return {"X": X, "Y": Y, "rk": rk, "lim": lim, "iv": iv, "ang": ang, "k": k, "p": G["p"]}


# ------------------------------------------------------------------------------------------------ collision-free labels
def _box(b, pad):
    return (b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)


def _tbox(t, r, pad):
    """Box of the characters of a text or annotation alone (an annotation's own extent also covers its leader)."""
    from matplotlib.text import Text
    if hasattr(t, "update_positions"):
        t.update_positions(r)
    return _box(Text.get_window_extent(t, r), pad)


def _hit_box(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _hit_circle(b, cx, cy, R, pad):
    """Does the stroke of the circle (centre cx, cy, radius R, display units) pass through box b?"""
    nx, ny = min(max(cx, b[0]), b[2]), min(max(cy, b[1]), b[3])
    dmin = np.hypot(nx - cx, ny - cy)
    dmax = max(np.hypot(x - cx, y - cy) for x in (b[0], b[2]) for y in (b[1], b[3]))
    return dmin <= R + pad and dmax >= R - pad


def _hit_poly(b, P):
    """Does the polyline P (n x 2, display units) enter box b?  Checked on a fine sampling of the segments."""
    for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 1.5) + 1)
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        if np.any((xs > b[0]) & (xs < b[2]) & (ys > b[1]) & (ys < b[3])):
            return True
    return False


def obstacles(fig, ax, r, skip=()):
    """Everything a label of the scene must not touch: every text of the figure (with the ladder axes as one block),
    the strokes of the circles, every line and leader line, every mark.  Display units."""
    pt = fig.dpi / 72.0
    boxes, polys, circles, marks = [], [], [], []
    for a in fig.axes:
        if a is ax or getattr(a, "_is_scene", False):
            continue
        parts = [a.bbox] + [t.get_window_extent(r) for t in a.get_xticklabels() + a.get_yticklabels() if t.get_text().strip()]
        parts += [a.xaxis.label.get_window_extent(r), a.yaxis.label.get_window_extent(r)]
        parts += [t.get_window_extent(r) for t in a.texts if t.get_text().strip()]
        if a.get_legend() is not None:
            parts.append(a.get_legend().get_window_extent(r))
        boxes.extend(_box(b, 1.0 * pt) for b in parts)              # the ladder: plot area, tick labels, axis labels, texts, legend
    for t in list(fig.texts) + list(ax.texts):
        if t in skip or not t.get_text().strip() or not t.get_visible():
            continue
        boxes.append(_tbox(t, r, 0.8 * pt))
        ap = getattr(t, "arrow_patch", None)
        if ap is not None:
            polys.append(ap.get_path().transformed(ap.get_transform()).vertices)
    o = ax.transData.transform((0.0, 0.0))
    for patch in ax.patches:
        if patch.get_edgecolor()[3] > 0 and hasattr(patch, "radius") and patch.get_facecolor()[3] == 0:
            rad = ax.transData.transform((patch.radius, 0.0))[0] - o[0]
            circles.append((o[0], o[1], rad))
    for ln in ax.lines:
        xy = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
        if ln.get_linestyle() not in ("None", "none", "", " "):
            polys.append(xy)
        if ln.get_marker() not in (None, "None", "none", "", " "):
            h = 0.5 * ln.get_markersize() * pt + 0.6 * pt
            marks.extend([(x - h, y - h, x + h, y + h) for x, y in xy])
    return boxes, polys, circles, marks


def _clash(b, obs, pad):
    boxes, polys, circles, marks = obs
    return (sum(_hit_box(b, q) for q in boxes) + sum(_hit_box(b, q) for q in marks)
            + sum(_hit_circle(b, cx, cy, R, pad) for cx, cy, R in circles) + sum(_hit_poly(b, P) for P in polys))


def _inside(fig, ax, b):
    W = fig.bbox
    return b[0] >= W.x0 and b[2] <= W.x1 and b[1] >= W.y0 and b[3] <= W.y1


def _cross(seg, Q):
    """Does the segment seg (2 x 2) cross the polyline Q (n x 2)?"""
    (ax_, ay), (bx, by) = seg
    def side(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    for (cx, cy), (dx, dy) in zip(Q[:-1], Q[1:]):
        d1, d2 = side(cx, cy, dx, dy, ax_, ay), side(cx, cy, dx, dy, bx, by)
        d3, d4 = side(ax_, ay, bx, by, cx, cy), side(ax_, ay, bx, by, dx, dy)
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
    return False


def _boxdist(b, x, y):
    return np.hypot(max(b[0] - x, 0.0, x - b[2]), max(b[1] - y, 0.0, y - b[3]))


def place_labels(fig, ax, info):
    """Rank labels of the k neighbours and the q label.  Each label goes to the nearest offset where its box touches
    no text, no circle stroke, no line and no mark, and is nearer to its own point than to any other point (measured
    on the rendered figure).  When no such offset exists close by, the label moves farther out with a thin leader
    line, and the leader touches no text and no mark either.  Returns the number of labels left with a clash."""
    r = fig.canvas.get_renderer(); pt = fig.dpi / 72.0
    X, Y, ang, k, p = info["X"], info["Y"], info["ang"], info["k"], info["p"]
    obs = obstacles(fig, ax, r)
    P = ax.transData.transform(np.column_stack([X, Y]))                  # the k neighbours and v, display units
    o = ax.transData.transform((0.0, 0.0))
    disk = (ax.transData.transform((info["iv"][-1][1], 0.0))[0] - o[0]) if info["iv"] else 0.0
    dirs = [0.0] + [sg * a for a in np.arange(15.0, 180.0, 15.0) for sg in (1, -1)] + [180.0]

    def put(txt, own, base_ang, near, far):
        best = None
        for d, lead in [(d, False) for d in near] + [(d, True) for d in far]:
            for turn in dirs:
                u = base_ang + np.deg2rad(turn); ux, uy = np.cos(u), np.sin(u)
                ha = "left" if ux > 0.38 else "right" if ux < -0.38 else "center"
                va = "bottom" if uy > 0.55 else "top" if uy < -0.55 else "center"
                txt.set_ha(ha); txt.set_va(va); txt.xyann = (d * ux, d * uy)
                b = _tbox(txt, r, 0.6 * pt)
                n = _clash(b, obs, 0.6 * pt) + (0 if _inside(fig, ax, b) else 5)
                if own is not None and disk > 0 and np.hypot(*(P[own] - o)) > disk:   # a point outside the conflict disk:
                    n += 0.5 * (_boxdist(b, *o) < disk)                              # its label inside it only as a last resort
                if own is not None and not lead:                         # unambiguous: nearer to its own point than to any other
                    dn = _boxdist(b, *P[own]); others = [_boxdist(b, *P[j]) for j in range(len(P)) if j != own]
                    n += int(bool(others) and min(others) < dn + 2.0 * pt)
                if lead:                                                 # the leader: from the point to the near edge of the box
                    ex, ey = min(max(P[own][0], b[0]), b[2]), min(max(P[own][1], b[1]), b[3])
                    seg = np.array([[P[own][0] + (ex - P[own][0]) * 0.12, P[own][1] + (ey - P[own][1]) * 0.12], [ex, ey]])
                    boxes, polys, circles, marks = obs
                    own_mark = [q for q in marks if q[0] <= P[own][0] <= q[2] and q[1] <= P[own][1] <= q[3]]
                    n += sum(_hit_poly(q, seg) for q in boxes) + sum(_hit_poly(q, seg) for q in marks if q not in own_mark)
                    n += sum(_cross(seg, Q) for Q in polys)                  # no leader across another line
                    n += 0.01 * d                                        # among leader positions, the shortest
                if best is None or n < best[0]:
                    best = (n, ha, va, (d * ux, d * uy), lead)
                if n == 0:
                    break
            if best[0] == 0:
                break
        n, ha, va, xy_, lead = best
        if os.environ.get("FIG_DEBUG"):
            print("   label %r: penalty %.2f, offset (%.1f, %.1f) pt, leader %s" % (txt.get_text(), n, xy_[0], xy_[1], lead))
        txt.set_ha(ha); txt.set_va(va); txt.xyann = xy_
        if getattr(txt, "arrow_patch", None) is not None:
            txt.arrow_patch.set_visible(bool(lead))
        return float(n)

    pens = []; made = []
    q = _halo(ax.annotate("$q$", (0, 0), xytext=(-2.5, -6.5), textcoords="offset points", ha="right", va="center", fontsize=T.FONT, color=T.INK, zorder=9))
    made.append(q)
    pens.append(0); pens[-1] = put(q, None, np.arctan2(-Y[-1], -X[-1]), (5.0, 6.5, 8.0), ())
    obs[0].append(_tbox(q, r, 0.6 * pt))
    for i in range(k):
        evicted = (i == k - 1)
        lab = "%d" % (i + 1) if i + 1 < p else ("%d$\\rightarrow$out" % (i + 1) if evicted else "%d$\\rightarrow$%d" % (i + 1, i + 2))
        t = _halo(ax.annotate(lab, (X[i], Y[i]), xytext=(4.4, 0), textcoords="offset points", ha="left", va="center",
                              fontsize=T.TINY, color=T.INK, zorder=7,
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="#7A7A7A", shrinkA=0.5, shrinkB=2.6)), lw=1.4)
        t.arrow_patch.set_visible(False); made.append(t)
        pens.append(0); pens[-1] = put(t, i, ang[i], (4.4, 5.6, 7.0, 8.6), (10.0, 11.5, 13.0, 14.5, 16.0, 18.0, 20.0, 22.5, 25.0, 28.0))
        tb = _tbox(t, r, 0.0)                                     # two rank labels never read as one: a clear gap between them
        obs[0].append((tb[0] - 3.0 * pt, tb[1] - 0.8 * pt, tb[2] + 3.0 * pt, tb[3] + 0.8 * pt))
        if t.arrow_patch.get_visible():
            fig.canvas.draw()
            obs[1].append(t.arrow_patch.get_path().transformed(t.arrow_patch.get_transform()).vertices)
    return (sum(v >= 0.5 for v in pens), sum(pens)), made


def draw_ladder(ax, G, c, color):
    """Per-rank test: held distance (dot) against the window [fresh, c * fresh] (bar from the ring)."""
    k = G["k"]; r = G["r"]; t = G["t"]; p = G["p"]
    new = T.new_list(r, t)
    rows = np.arange(1, k + 1)
    bad = r > c * new + 1e-9
    h = 0.30
    for i in range(k):
        ax.fill_between([new[i], c * new[i]], rows[i] - h, rows[i] + h, color="#D4D4D4" if not bad[i] else T.lighten(T.C_CONF, 0.62), lw=0, zorder=1)
    for i in range(p - 1, k - 1):                                         # item at old rank i+1 is at new rank i+2
        ax.plot([r[i], r[i]], [rows[i] + 0.12, rows[i + 1] - 0.12], color="#9A9A9A", lw=0.5, zorder=2)
    for i in range(k):
        is_ins = (i == p - 1)
        ax.plot([new[i]], [rows[i]], marker="D" if is_ins else "o", ms=4.4 if is_ins else 3.8, mfc=color if is_ins else "white",
                mec="white" if is_ins else "#3A3A3A", mew=0.6 if is_ins else 0.8, zorder=4, ls="none")
        ax.plot([r[i]], [rows[i]], marker="o", ms=3.0, mfc=T.C_CONF if bad[i] else "#3A3A3A", mec="none", zorder=5, ls="none")
    ax.set_xscale("log")
    lo, hi = min(t, r[0]) / 1.10, r[-1] * (1.30 if bad.any() else 1.12)
    ax.set_xlim(lo, hi); ax.set_ylim(k + 0.75, 0.25)
    ax.set_yticks(rows); ax.set_yticklabels(["%d" % i for i in rows])
    ax.tick_params(axis="y", length=0, pad=1.5)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    cand = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
    ticks = [v for v in cand if lo * 1.04 <= v <= hi / 1.04]
    while len(ticks) > 4:
        ticks = ticks[::2]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("distance to $q$ (log)")
    ax.set_ylabel("rank", labelpad=1.0)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    for i in np.where(bad)[0]:
        ax.annotate("$\\times$%.2f" % (r[i] / new[i]), (r[i], rows[i]), xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=T.TINY, color=T.C_CONF, zorder=6)
    return bad


def free_angle(angles, lo, hi, n=41):
    """Angle (radians) in [lo, hi] degrees that is farthest from every angle of the list (radians): where a
    leader line can reach a circle without crossing a mark or its label."""
    cand = np.deg2rad(np.linspace(lo, hi, n)); a = np.asarray(angles, dtype=float)
    d = np.abs(np.angle(np.exp(1j * (cand[:, None] - a[None, :])))).min(axis=1)
    d = d - 0.15 * np.abs(cand - np.deg2rad(0.5 * (lo + hi)))            # mild preference for the middle of the range
    return float(cand[int(np.argmax(d))])


def _plain(s):
    """Text of an item as one line of plain characters (markup removed, nothing that mathtext would parse)."""
    s = re.sub(r"<[^>]*>", " ", html.unescape(html.unescape(str(s))))
    return " ".join(s.replace("\\", "/").split())


def _wrap(fig, s, width_pt, **kw):
    """Greedy word wrap of s into lines no wider than width_pt points (measured with the renderer).  Nothing is cut."""
    r = fig.canvas.get_renderer(); probe = fig.text(0, 0, "", **kw); words = s.split(); lines = []; cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        probe.set_text(trial)
        if cur and probe.get_window_extent(r).width * 72.0 / fig.dpi > width_pt:
            lines.append(cur); cur = w
        else:
            cur = trial
    lines.append(cur); probe.remove()
    return lines


QX, TX = 0.084, 0.024                                   # figure fractions: start of the quotes and of the tags, after x0
LINE = 0.130                                            # inches between header lines


def header_lines(fig, x0, x1, title, G, show_text):
    """The header of one pair as a list of (dx, text, kind) lines: the title (wrapped if needed) and the full texts of
    the two items, wrapped under their quotation marks.  kind is title, tag-q, tag-v, quote or tail."""
    W = fig.get_figwidth() * 72.0
    out = [(TX, l, "title") for l in _wrap(fig, title, (x1 - x0 - TX) * W, fontsize=T.FONT)]
    if show_text and G["texts"]:
        for tag, txt, tail in (("read $q$", G["texts"][0], ""), ("insert $v$", G["texts"][1], relation(G))):
            q = ("\u201c" + _plain(txt) + "\u201d").replace("$", "\\$")
            glue = "\u00b7"                                 # the relation is one unbreakable token while wrapping
            ls = _wrap(fig, q + ((" (" + tail.replace(" ", glue) + ")") if tail else ""), (x1 - x0 - QX) * W - 4.0, fontsize=T.SMALL, style="italic")
            ls = [l.replace(glue, " ") for l in ls]
            for n, l in enumerate(ls):
                out.append((QX, l, ("tag-q" if tag.startswith("read") else "tag-v") if n == 0 else "quote"))
    return out


def header(fig, x0, x1, y0, letter, lines, color):
    """Draw the header lines of one pair; the relation in parentheses after the quote is set upright in ink."""
    r = fig.canvas.get_renderer(); dy = LINE / fig.get_figheight()
    fig.text(x0, y0, "(%s)" % letter, fontsize=T.FONT, fontweight="bold", ha="left", va="top", color=T.INK)
    y = y0; first_quote = True
    for n, (dx, l, kind) in enumerate(lines):
        if kind == "title":
            fig.text(x0 + dx, y, l, fontsize=T.FONT, ha="left", va="top", color=T.INK)
            y -= dy if n + 1 < len(lines) and lines[n + 1][2] == "title" else dy + 0.02 / fig.get_figheight()
            continue
        if kind in ("tag-q", "tag-v"):
            fig.text(x0 + TX, y, "read $q$" if kind == "tag-q" else "insert $v$", fontsize=T.SMALL, ha="left", va="top",
                     color=T.INK if kind == "tag-q" else color)
        tail = None
        if " (" in l and l.endswith(")") and "\u201d (" in l:
            l, tail = l.split("\u201d (", 1); l += "\u201d"; tail = "(" + tail
        elif l.startswith("(") and l.endswith(")") and n > 0 and lines[n - 1][1].endswith("\u201d"):
            l, tail = "", l
        qt = fig.text(x0 + dx, y, l, fontsize=T.SMALL, ha="left", va="top", color=T.MUTED, style="italic")
        if tail:
            xr = (qt.get_window_extent(r).x1 / fig.bbox.width + 0.006) if l else x0 + dx
            fig.text(xr, y, tail, fontsize=T.SMALL, ha="left", va="top", color=T.INK)
        y -= dy


def relation(G):
    if G["same"]:
        return "same team, %s" % ("next item" if G["lag"] == 1 else "%d items apart" % G["lag"])
    return "another team"


def _sidecar(out, results, suffix, name, c, scenes):
    """Record of the two scenes next to the figure (<out>/fig_t_01_teaser.json), read by the result checks
    (paper/checkdefs/c_g3_law.py): which run, which law rows, and the quantities the caption states."""
    import json
    rec = {"figure": NAME, "results": os.path.abspath(results), "suffix": suffix, "cell": name, "c": c, "scenes": []}
    for G, _, kind in scenes:
        r = np.asarray(G["r"], dtype=float)
        rc = float(T.conflict_radius(r[None, :], len(r), [c])[0, 0])
        rec["scenes"].append({"kind": kind, "row": int(G["x"]), "task": int(G["task"]), "src": int(G["src"]), "t": float(G["t"]),
                              "p": int(G["p"]), "r1": float(r[0]), "rk": float(r[-1]), "Rc": rc,
                              "conflict_intervals": [[float(a), float(b)] for a, b in T.conflict_intervals(r, c)],
                              "lag": int(G["lag"]), "same_team": bool(G["same"]), "rho": float(G["rho"]), "nc": int(G["nc"])})
    try:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, NAME + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
    except OSError as err:                                   # the figure does not depend on the record
        T.note("%s: could not write the record of the scenes (%s)" % (NAME, err))


def main(results="results", suffix="", out="figures", cell=None, c=1.1, row_sib=None, row_bg=None, no_text=False, sib_lag=3, bg_lag=16):
    plt = T.setup()
    from ..audit import load_run, workload_of, Exact
    laws = T.law_paths(results, EXP, suffix)
    if not laws:
        return T.skip(NAME, "no law files in %s" % T.exp_dir(results, EXP, suffix))
    order = [cell] if cell else sorted(laws, key=lambda n: (PREFER.index(T.split_cell(n)[0]) if T.split_cell(n)[0] in PREFER else 99, n))
    picked = None
    for name in order:
        run_path = os.path.join(T.exp_dir(results, EXP, suffix), name + ".pkl")
        if name not in laws or not os.path.exists(run_path):
            continue
        L = T.law_load(laws[name])
        if not candidates(L, c, "sib", sib_lag, bg_lag) and row_sib is None:
            continue
        run = load_run(run_path)
        try:
            wl = workload_of(run["cfg"])
        except (OSError, KeyError, ValueError) as err:
            T.note("%s: the workload of %s cannot be rebuilt (%s)" % (NAME, name, err))
            continue
        ex = Exact(run, wl)
        A = choose(ex, wl, run["recs"], L, c, "sib", row_sib, sib_lag=sib_lag, bg_lag=bg_lag)
        B = choose(ex, wl, run["recs"], L, c, "bg", row_bg, sib_lag=sib_lag, bg_lag=bg_lag)
        if A is not None and B is not None:
            picked = (name, A, B)
            break
        if A is not None and picked is None:
            picked = (name, A, None)
    if picked is None:
        return T.skip(NAME, "no read with one in-ball sibling insertion at rank 1 or 2 that breaks c = %g" % c)
    name, A, B = picked
    scenes = [(A, T.C_CONF, "sib")] + ([(B, T.C_ADM, "bg")] if B is not None else [])
    for G, _, kind in scenes:
        print("  %s: %s read = law row %d (task %d) of %s: inserted task %d at t = %.4f, old rank %d, r_1 = %.4f, r_k = %.4f, lag %d, %s, "
              "rho = %.3f, %d overlapping commits" % (NAME, "sibling" if kind == "sib" else "background", G["x"], G["task"], name, G["src"], G["t"],
                                                      G["p"], G["r"][0], G["r"][-1], G["lag"], "same team" if G["same"] else "other team",
                                                      G["rho"], G["nc"]))
    _sidecar(out, results, suffix, name, c, scenes)

    show_text = (not no_text) and all(G["texts"] for G, _, _ in scenes)
    W = T.COL2
    S, gap, LWd, mid, left = 2.06, 0.38, 0.86, 0.24, 0.05                 # inches: scene, gap, ladder, gap between the halves, margin
    half = S + gap + LWd
    fig = plt.figure(figsize=(W, 3.0))
    heads = []
    for n, (G, color, kind) in enumerate(scenes):
        x0 = left + n * (half + mid)
        if kind == "sib":
            title = "A sibling arrives: the held answer is no longer %g-admissible" % c
        else:
            title = "%s arrives: the held answer stays %g-admissible" % ("A distant item" if G["same"] else "A background item", c)
        if not show_text:
            title = title.replace(":", " (%s):" % relation(G), 1)
        heads.append(header_lines(fig, x0 / W + 0.002, (x0 + half) / W, title, G, show_text))
    nl = max(len(h) for h in heads)
    head = (0.24 + LINE * (nl - 1)) if show_text else 0.20 + LINE * (nl - 1)
    SH = S * 0.5 * (VSTRETCH + VBOT); xb = S * 0.5 * (VBOT - VSTRETCH)          # scene height, extra room below
    H = SH + head + 0.06
    fig.set_size_inches(W, H)
    fig.set_layout_engine("none")
    for n, (G, color, kind) in enumerate(scenes):
        x0 = left + n * (half + mid)
        ax = fig.add_axes([x0 / W, 0.03 / H, S / W, SH / H])
        info = draw_scene(ax, G, c, color, target_deg=200 if kind == "sib" else 35)
        ax._is_scene = True
        fig._scene_infos = getattr(fig, "_scene_infos", []) + [(ax, info)]
        axl = fig.add_axes([(x0 + S + gap) / W, (0.86 + xb) / H, LWd / W, (S * VSTRETCH - 1.28) / H])
        bad = draw_ladder(axl, G, c, color)
        rk = G["r"][-1]; lim = info["lim"]; ytxt = lim * (VSTRETCH - 0.085); ybot = lim * (VBOT - 0.085)

        # corner labels: the ball (top left), the conflict set (bottom right), the overlap (bottom left)
        marks = info["ang"][:G["k"] + 1]
        a0 = free_angle(marks, 106, 130)                              # near-radial leaders from the corner labels
        _halo(ax.annotate("top-$k$ ball of the read,\nradius $r_k$ = %.2f" % rk, (rk * np.cos(a0), rk * np.sin(a0)), xytext=(-lim * 0.995, ytxt),
                          textcoords="data", ha="left", va="center", fontsize=T.SMALL, color=T.INK, linespacing=1.15,
                          arrowprops=dict(arrowstyle="-", lw=0.5, color=T.INK, shrinkA=1.0, shrinkB=0.5, relpos=(0.5, 0.0))))
        if info["iv"]:
            hi = info["iv"][-1][1]
            gap_i = int(np.argmin(np.abs(G["r"] / c - hi))) + 1
            a1 = free_angle(marks, -74, -50)
            cs = _halo(ax.annotate("conflict set, $c$ = %g:\n$t < r_{%d}/c$ = %.2f" % (c, gap_i, hi), (hi * np.cos(a1), hi * np.sin(a1)),
                              xytext=(lim * 0.995, -ybot), textcoords="data", ha="right", va="center", fontsize=T.SMALL, color=T.C_CONF,
                              linespacing=1.15, arrowprops=dict(arrowstyle="-", lw=0.5, color=T.C_CONF, shrinkA=1.0, shrinkB=0.5, relpos=(0.45, 1.0))))
            fig._scene_infos[-1][1]["cs"] = (cs, hi, marks)
        ax.text(-lim * 0.995, -ybot, "%d commit%s between read\nand commit, 1 in the ball" % (G["nc"], "" if G["nc"] == 1 else "s"), ha="left",
                va="center", fontsize=T.SMALL, color=T.MUTED, linespacing=1.15)

        # label of the inserted point
        xv, yv = info["X"][-1], info["Y"][-1]
        if kind == "sib":
            # the label goes to the first side of v where it crosses neither circle and covers no mark
            radii = [rk] + ([info["iv"][-1][1]] if info["iv"] else [])
            seg = np.linspace(0.0, 1.0, 25)
            pts = np.r_[np.c_[info["X"][:-1], info["Y"][:-1]], np.c_[xv * seg, yv * seg]]          # marks and the segment q-v
            best = None
            cands = [(dx * s_, dy * s_, ha, va) for s_ in (1.0, 2.0, 3.0) for dx, dy, ha, va in
                     ((3.5, -5.5, "left", "top"), (-3.5, -5.5, "right", "top"), (3.5, 5.5, "left", "bottom"), (-3.5, 5.5, "right", "bottom"))]
            for dx, dy, ha, va in cands:
                lab = _halo(ax.annotate("$v$ at $t$ = %.2f:\nnew rank %d" % (G["t"], G["p"]), (xv, yv), xytext=(dx, dy), textcoords="offset points", ha=ha,
                                        va=va, fontsize=T.SMALL, color=color, linespacing=1.15))
                fig.canvas.draw(); bb = lab.get_window_extent().transformed(ax.transData.inverted())
                xs = np.linspace(bb.x0, bb.x1, 12); ys = np.linspace(bb.y0, bb.y1, 6); gx, gy = np.meshgrid(xs, ys); d = np.hypot(gx, gy)
                nbad = sum(1 for R in radii if d.min() < R < d.max()) + int(np.any((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)))
                if best is None or nbad < best[0]:
                    if best is not None: best[1].remove()
                    best = (nbad, lab)
                else:
                    lab.remove()
                if nbad == 0: break
        else:
            _halo(ax.annotate("$v$ at $t$ = %.2f:\nnew rank %d" % (G["t"], G["p"]), (xv, yv), xytext=(lim * 0.995, ytxt), textcoords="data",
                              ha="right", va="center", fontsize=T.SMALL, color=color, linespacing=1.15,
                              arrowprops=dict(arrowstyle="-", lw=0.5, color=color, shrinkA=1.0, shrinkB=3.5, relpos=(0.5, 0.0))))

        ranks = ", ".join("%d" % (i + 1) for i in np.where(bad)[0])
        verdict = ("fails at rank%s %s" % ("s" if bad.sum() > 1 else "", ranks)) if bad.any() else "holds at every rank"
        axl.text(0.0, 1.155, "held $\\leq$ $c$ $\\times$ fresh ?", transform=axl.transAxes, ha="left", va="bottom", fontsize=T.SMALL, color=T.INK)
        axl.text(0.0, 1.045, verdict, transform=axl.transAxes, ha="left", va="bottom", fontsize=T.SMALL, color=color, fontweight="bold")
        header(fig, x0 / W + 0.002, (x0 + half) / W, 1.0 - 0.02 / H, "ab"[n], heads[n], color)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        hs = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc="#3A3A3A", mec="none", label="held answer"),
              Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec="#3A3A3A", mew=0.8, label="fresh top-$k$"),
              Patch(fc="#D4D4D4", ec="none", label="fresh to $c$ $\\times$ fresh")]
        axl.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.20, -0.36), ncol=1, fontsize=T.TINY, handlelength=1.0, handletextpad=0.4,
                   frameon=False, borderaxespad=0.0, labelspacing=0.2)
    infos = getattr(fig, "_scene_infos", [])
    fig.canvas.draw()
    for ax, info in infos:                                   # last: every other text, line and mark exists now
        # the leader of the conflict-set label may take another anchor on the dashed circle, if that frees a rank label
        trials = [None]
        if "cs" in info:
            cs, hi, marks = info["cs"]
            trials = [((-74, -50), 0.45), ((-50, -28), 0.80), ((-28, -8), 0.92), ((-96, -74), 0.25), ((-118, -96), 0.12)]
        best = None
        for tr in trials:
            if tr is not None:
                (lo_, hi_), rp = tr
                a1 = free_angle(marks, lo_, hi_)
                cs.xy = (hi * np.cos(a1), hi * np.sin(a1)); cs.arrowprops["relpos"] = (rp, 1.0)
                cs.arrow_patch.set_positions((0, 0), (1, 1)); cs._arrow_relpos = (rp, 1.0)
                fig.canvas.draw()
            if best is not None:
                for t in best[2]:
                    t.set_visible(False)                         # the kept labels are not obstacles of the next trial
            pen, made = place_labels(fig, ax, info)
            if best is not None:
                for t in best[2]:
                    t.set_visible(True)
            if best is None or pen < best[0]:
                if best is not None:
                    for t in best[2]:
                        t.remove()
                best = (pen, tr, made)
            else:
                for t in made:
                    t.remove()
            if pen[0] == 0 and pen[1] < 0.35:
                break
        pen, tr, made = best
        if tr is not None:
            (lo_, hi_), rp = tr; a1 = free_angle(marks, lo_, hi_)
            cs.xy = (hi * np.cos(a1), hi * np.sin(a1)); cs._arrow_relpos = (rp, 1.0)
        if pen[0]:
            T.note("%s: %d rank label(s) could not be placed cleanly (inside the conflict disk or touching something)" % (NAME, pen[0]))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--cell", dict(default=None, help="cell of D3a_law_real, e.g. L4_s1 (default: automatic)")),
                          ("--c", dict(type=float, default=1.1, help="tolerance (default 1.1)")),
                          ("--row-sib", dict(type=int, default=None, help="law row of the sibling scene")),
                          ("--row-bg", dict(type=int, default=None, help="law row of the background scene")),
                          ("--no-text", dict(action="store_true", help="do not quote the texts of the items")),
                          ("--sib-lag", dict(type=int, default=3)), ("--bg-lag", dict(type=int, default=16))])
