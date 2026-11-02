"""Building blocks shared by the system-side figure modules (small multiples of lines and bars).

    lines(ax, R, x, y, by)                 one line per value of `by` (a protocol / policy key) with interval bands
    bars(ax, R, group, series, y)          grouped bars with interval caps
    policy_legend(fig, specs)              one shared legend above the panels
    ordinal(ax, values, fmt)               categorical x axis at equal steps; returns {value: position}
    grid_of(n, ncol)                       rows and columns of a small-multiples grid
"""
import math

from . import style as st, load as ld


def spec_key(R):
    return ld.policy_key(R)


def lines(ax, R, x, y, by=None, xpos=None, kw_of=st.policy_line, order=st.policy_order, band=True, xnum=None, only=None):
    """One line per value of the key `by`, the value y aggregated over seeds at each x.

    xpos: {x value: coordinate} for a categorical axis (see ordinal); xnum: maps x values to numbers otherwise.
    Returns {series value: (xs, [Stats])}."""
    by = by or spec_key(R)
    vals = order(ld.values(R, by)) if by else [None]
    if only is not None:
        vals = [v for v in vals if v in only]
    out = {}
    for v in vals:
        Rv = ld.where(R, **{by: v}) if by else R
        if xpos is not None:
            A = ld.agg(Rv, x, y)
            pts = sorted(((xpos[k], s) for k, s in A.items() if k in xpos and s.n), key=lambda t: t[0])
            xs, S = [p[0] for p in pts], [p[1] for p in pts]
        else:
            xs, S = ld.series(Rv, x, y, xnum=xnum)
            keep = [i for i, a in enumerate(xs) if a == a]
            xs, S = [xs[i] for i in keep], [S[i] for i in keep]
        if not xs:
            continue
        st.line_ci(ax, xs, S, band=band, **(kw_of(v) if v is not None else dict(color=st.INK, marker="o")))
        out[v] = (xs, S)
    return out


def bars(ax, R, group, series, y, groups=None, order=st.policy_order, kw_of=st.policy_bar, group_label=str, rotate=None, **kw):
    """Grouped bars: one group per value of `group`, one bar per value of `series`.  Returns (groups, series values)."""
    A = ld.agg(R, [group, series], y)
    groups = groups if groups is not None else ld.values(R, group)
    ser = order(ld.values(R, series))
    st.grouped_bars(ax, groups, ser, lambda g, s: A.get((g, s)), kw_of, **kw)
    ax.set_xticks(range(len(groups)))
    labs = [group_label(g) for g in groups]
    rot = rotate if rotate is not None else (28 if (len(groups) > 5 or max(len(l) for l in labs) * len(groups) > 46) else 0)
    ax.set_xticklabels(labs, rotation=rot, ha="right" if rot else "center", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    return groups, ser


def short_label(spec):
    return st.pretty_math(spec, short=True)


def policy_legend(fig, specs, kind="line", ncol=None, label=st.pretty_math, extra=(), **kw):
    """Shared legend above the panels.  extra: further handles appended after the policies."""
    specs = list(specs)
    H = st.legend_handles(specs, kind, label=label) + list(extra)
    if not H:
        return None
    n = len(H)
    width = fig.get_size_inches()[0]
    longest = max(len(h.get_label().replace("$", "")) for h in H)
    per_row = max(1, int(width / ((0.24 if kind == 'bar' else 0.30) + 0.044 * longest)))
    ncol = ncol or min(n, per_row)
    ncol = math.ceil(n / math.ceil(n / ncol))               # balance the rows
    kw.setdefault("handlelength", 1.4 if kind == "bar" else 2.2)
    return fig.legend(handles=H, loc="outside upper center", ncol=ncol, columnspacing=1.0, **kw)


def ordinal(ax, values, fmt=str, rotate=0):
    """Categorical x axis at equal steps; returns {value: position}."""
    pos = {v: i for i, v in enumerate(values)}
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([fmt(v) for v in values], rotation=rotate, ha="right" if rotate else "center", rotation_mode="anchor")
    ax.set_xlim(-0.35, max(len(values) - 0.65, 0.65))
    return pos


def grid_of(n, ncol=4):
    ncol = min(ncol, max(n, 1))
    return (n + ncol - 1) // ncol, ncol


def letters(axes, texts=None, **kw):
    for i, ax in enumerate(axes):
        st.panel_label(ax, "abcdefghijklmnopqrstuvwxyz"[i], None if texts is None else texts[i], **kw)


def merged(results, exp, suffix, audit="audit_run.jsonl"):
    """Summary rows with the fields of the audit row of the same cell (summary fields win)."""
    R = ld.rows(results, exp, suffix=suffix)
    A = ld.rows(results, exp, audit, suffix, quiet=True)
    return ld.join(R, A) if A else R


def fmt_num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1e6:
        return "%gM" % (x / 1e6)
    if abs(x) >= 1e3:
        return "%gk" % (x / 1e3)
    return "%g" % x


def zero_floor(axes, axis="y"):
    """Start the value axis at zero.  Call it after everything is plotted: with shared axes, fixing a limit earlier
    freezes the range at the data of the panels drawn so far."""
    seen = set()
    for ax in axes:
        if id(ax) in seen or not ax.get_visible():
            continue
        seen.add(id(ax))
        if axis == "y":
            ax.autoscale(enable=True, axis="y"); ax.autoscale_view(scalex=False, scaley=True)
            top = ax.get_ylim()[1]
            data_top = max([float(max(l.get_ydata())) for l in ax.lines if len(l.get_ydata())] + [p.get_height() + p.get_y() for p in ax.patches] + [0.0])
            ax.set_ylim(0, top * 1.05 if data_top > 0 else 1.0)
        else:
            ax.autoscale(enable=True, axis="x"); ax.autoscale_view(scalex=True, scaley=False)
            ax.set_xlim(0, max(ax.get_xlim()[1], 1e-9) * 1.03)
