"""Shared helpers of the theory-side figures (modules fig_t_*).

Not a figure.  Everything that the theory figures have in common lives here:

  encoding    one fixed mapping of the theory side, used by every fig_t_* module:
                observed                      black markers with 95% intervals
                exchangeable only (background) grey, dashed
                closed form                   orange, solid
                semi-parametric, 1 scale      blue, solid
                semi-parametric, 2 scales     green, dash-dotted
                sibling insertions            purple fills        background insertions: grey fills
                conflict (c-admissibility broken): vermillion     admissible: blue
              Data sets keep the colours of the system-side figures (style.dataset_style).
  files       law files (<cell>.law.pkl), analysis JSON files, cell names "<case>_<grid values>_s<seed>".
  theorem     conflict set of the characterization theorem for one neighbour list, and its closed form:
              one insertion at distance t breaks c-admissibility  iff  t < R_c,  where
                  R_c = max { r_i / c :  r_i > c r_{i-1},  i = 1..k },   r_0 = 0
              (the last gap of the neighbour list that exceeds c decides).  conflict_intervals() evaluates the
              theorem rank by rank and never uses this shortcut; conflict_radius() is the shortcut, vectorized.
  axes        the tolerance axis (log scale in c - 1, ticks labelled with c), panel labels, intervals.
  cli         the common command line of the figure modules.

Only these names of the shared modules are required: style.policy_style, style.dataset_style, style.pretty,
style.figsize, style.save, load.rows, load.agg, load.runs.  Everything else is used when present and has a
fallback here.
"""
import argparse, glob, json, math, os, pickle, re

import numpy as np

from . import style as st
from . import load as ld

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator


def _g(name, default):
    return getattr(st, name, default)


# ------------------------------------------------------------------------------------------------ look
OI = dict({"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
           "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}, **_g("OKABE_ITO", {}))
INK, MUTED, GREY, LIGHT = _g("INK", "#222222"), _g("MUTED", "#555555"), _g("GREY", "#8C8C8C"), _g("LIGHT", "#D9D9D9")
FONT, SMALL, TINY = _g("FONT", 7.5), _g("SMALL", 6.5), _g("TINY", 5.8)
LW, MS = _g("LW", 1.1), _g("MS", 3.6)
COL1, COL2 = _g("COL1", 3.33), _g("COL2", 7.0)

C_OBS = INK
C_EXCH = GREY
C_CLOSED = OI["orange"]
C_SEMI1 = OI["blue"]
C_SEMI2 = OI["green"]
C_SIB = OI["purple"]
C_BG = GREY
C_CONF = OI["vermillion"]
C_ADM = OI["blue"]

# predictors of the conflict probability: key of fits.json -> line style and label
PRED = {
    "exch":   dict(color=C_EXCH,   ls=(0, (4, 1.6)),          lw=LW,       label="exchangeable only", short="exch."),
    "closed": dict(color=C_CLOSED, ls="-",                    lw=LW + 0.2, label="closed form",       short="closed"),
    "semi1":  dict(color=C_SEMI1,  ls="-",                    lw=LW,       label="semi-param., 1 scale",  short="semi-1"),
    "semi2":  dict(color=C_SEMI2,  ls=(0, (5, 1.2, 1, 1.2)),  lw=LW,       label="semi-param., 2 scales", short="semi-2"),
}

# sequential ramps: conflict probability (white to vermillion), neutral magnitude (white to dark blue)
CONF_CMAP = LinearSegmentedColormap.from_list("conflict", ["#FFFFFF", "#FCEBD9", "#F6C99B", "#EB9552", "#D55E00", "#A84A00"])
COOL_CMAP = _g("SEQ_CMAP", LinearSegmentedColormap.from_list("seq", ["#F4F9FD", "#A9D8F5", "#56B4E9", "#0072B2", "#003A5C"]))
# ordered ramp for stream lag (near = dark, far = light)
LAG_CMAP = LinearSegmentedColormap.from_list("lag", ["#5B2A52", "#A04F8E", "#CC79A7", "#E3B4CF", "#F1DCE8"])

BUCKETS = ["1", "2-3", "4-7", "8-15", "16-31", "32+", "other"]
BUCKET_LABEL = {"1": "lag 1", "2-3": "2 to 3", "4-7": "4 to 7", "8-15": "8 to 15", "16-31": "16 to 31", "32+": "32 or more",
                "other": "other team"}
BUCKET_RANGE = {"1": (1, 1), "2-3": (2, 3), "4-7": (4, 7), "8-15": (8, 15), "16-31": (16, 31), "32+": (32, 10 ** 9)}


# transfers of the law without refitting: mode of ariadne.law transfer (and "static" for ariadne.static_kernel)
MODES = ["static", "lag", "window", "k", "time", "params"]
MODE_MARK = {"window": "o", "lag": "s", "k": "^", "time": "v", "params": "*", "static": "D"}
MODE_LABEL = {"window": "window", "lag": "lag kernel", "k": "$k$", "time": "time", "params": "other corpus", "static": "static stream"}
MODE_LONG = {"static": "static stream (no execution)", "lag": "lag kernel of another concurrency level", "window": "another concurrency level (window)",
             "k": "another $k$", "time": "first half to second half", "params": "parameters of another corpus"}


def bucket_color(b):
    if b == "other":
        return GREY
    i = BUCKETS.index(b)
    return to_hex(LAG_CMAP(i / 5.0 * 0.82))


def setup():
    """Shared rcParams (style.use when it exists)."""
    use = _g("use", None)
    if use is not None:
        use()
    return plt


def lighten(color, amount=0.5):
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def darken(color, amount=0.3):
    r, g, b = to_rgb(color)
    return to_hex((r * (1 - amount), g * (1 - amount), b * (1 - amount)))


def figsize(cols=1, aspect=0.62):
    return st.figsize(cols, aspect)


def save(fig, name, out):
    return st.save(fig, name, out)


def panel(ax, letter, text=None, dy=1.03, x=0.0, **kw):
    """Panel label "(a) text" above the top-left corner of the axes."""
    kw.setdefault("fontsize", FONT)
    lab = ax.text(x, dy, "(%s)" % letter, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", **kw)
    if text:
        ax.annotate(" " + text, xy=(1, 0), xycoords=lab, ha="left", va="bottom", fontsize=kw["fontsize"], color=INK)
    return lab


def note(msg):
    f = getattr(ld, "note", None)
    if f is not None:
        f(msg)
    else:
        print("  note: " + msg)


def skip(fig_name, why):
    """Uniform message of a figure that cannot be drawn (missing data); the caller then returns."""
    f = getattr(ld, "skip", None)
    if f is not None:
        return f(fig_name, why)
    print("  skipped %s: %s" % (fig_name, why))
    return []


# ------------------------------------------------------------------------------------------------ files
def exp_dir(results, exp, suffix=""):
    return os.path.join(results, exp + suffix)


def jload(results, exp, file, suffix="", quiet=False):
    """Parsed JSON of <results>/<exp><suffix>/<file>, or None (with one printed note) when it is missing."""
    p = os.path.join(exp_dir(results, exp, suffix), file)
    if not os.path.exists(p):
        if not quiet:
            note("%s: no %s in %s" % (exp, file, exp_dir(results, exp, suffix)))
        return None
    with open(p) as f:
        return json.load(f)


def jglob(results, exp, pattern, suffix=""):
    """{file name: parsed JSON} of the files of an experiment that match the pattern."""
    out = {}
    for p in sorted(glob.glob(os.path.join(exp_dir(results, exp, suffix), pattern))):
        try:
            with open(p) as f:
                out[os.path.basename(p)] = json.load(f)
        except ValueError:
            note("unreadable JSON: %s" % p)
    return out


def law_paths(results, exp, suffix="", pattern="*"):
    """{cell name: path} of the law files of an experiment."""
    out = {}
    for p in sorted(glob.glob(os.path.join(exp_dir(results, exp, suffix), pattern + ".law.pkl"))):
        out[os.path.basename(p)[:-len(".law.pkl")]] = p
    return out


_law_cache = {}


def law_load(path):
    if path not in _law_cache:
        if len(_law_cache) > 64:
            _law_cache.clear()
        with open(path, "rb") as f:
            _law_cache[path] = pickle.load(f)
    return _law_cache[path]


def split_cell(name):
    """ "A64_natural_s3" -> ("A64_natural", 3); names without a seed give (name, None).  A "#2" suffix added by
    the fit command for duplicate names is dropped."""
    s = re.sub(r"#\d+$", "", str(name))
    m = re.match(r"^(.*)_s(\d+)$", s)
    return (m.group(1), int(m.group(2))) if m else (s, None)


def by_case(d):
    """{case: [(seed, entry), ...]} of a JSON dict keyed by cell name (keys that start with "_" are skipped,
    entries that are not dicts as well)."""
    out = {}
    for name, e in d.items():
        if str(name).startswith("_") or not isinstance(e, dict):
            continue
        case, seed = split_cell(name)
        out.setdefault(case, []).append((seed, e))
    return out


def ds_key(x):
    """Data set key ("locomo", "agnews", ..., "synthetic") of a case name, a data path or a law file dict."""
    if isinstance(x, dict):
        wl = x.get("workload") or {}
        if wl.get("kind") == "synthetic":
            return "synthetic"
        x = wl.get("data") or x.get("name") or ""
    f = _g("dataset_key", None)
    if f is not None:
        return f(x)
    low = str(x).lower()
    for key in ("longmemeval", "locomo", "agnews", "msmarco", "qqp"):
        if key in low:
            return key
    return "locomo" if re.match(r"^L\d", str(x)) else "agnews" if re.match(r"^A\d", str(x)) else "synthetic"


def ds_style(x):
    """style.dataset_style for a case name, a data path or a law file dict; always has color, marker, ls, label."""
    if isinstance(x, dict):
        wl = x.get("workload") or {}
        x = "synthetic" if wl.get("kind") == "synthetic" else (wl.get("data") or x.get("name") or "")
    s = dict(color=MUTED, marker="o", ls="-", label=str(x))
    s.update(st.dataset_style(x))
    return s


def case_sort(cases):
    f = _g("case_order", None)
    if f is not None:
        try:
            return f(list(cases))
        except Exception:                                     # a helper of another module must not break a figure
            pass
    return sorted(cases)


def case_label(case, long=False):
    """Short label of a configuration: "A64" -> "AG News, 64 agents" (long) or "AG News / 64" (short)."""
    s = str(case)
    lab = ds_style(s)["label"]
    m = re.match(r"^(?:L|A|LME|QQP|MSM)(\d+)(.*)$", s)
    if not m:
        return s if ds_key(s) == "synthetic" else lab
    extra = m.group(2).lstrip("-_")
    mk = re.match(r"^k(\d+)$", extra)
    tail = (", $k$ = %s" % mk.group(1)) if mk else (", " + extra) if extra else ""
    if long:
        return "%s, %s agents%s" % (lab, m.group(1), tail)
    return "%s, %s%s" % (lab, m.group(1), tail)


def agents_of(case):
    m = re.match(r"^(?:L|A|LME|QQP|MSM)(\d+)", str(case))
    return int(m.group(1)) if m else None


def quick_fit(path):
    """The part of `python -m ariadne.law fit` that the figures need, for a law file that has no fits.json next to
    it.  Same library calls and formulas as law.fit_one (background share from first principles, sibling scales
    by maximum likelihood on the insertion distances, predictions on the single-insertion reads), but no
    bootstrap, no end-to-end simulation, nothing printed.  The interval of D_s is the approximation
    D_s (1 +/- 1.96 / sqrt(sibling insertions)).  The entry has the keys of a fits.json entry that the figures read,
    and "quick": True."""
    from .. import law as LW
    C = LW.LawData(path)
    k, cs = C.k, list(C.cs); ne = len(C.y)
    single = np.where(C.m == 1)[0]; n1 = len(single)
    res = {"k": k, "n": C.n, "events": ne, "mean_m": C.mean_m, "lid_median": C.lid_median, "lid_pooled": C.lid_pool, "lid_twonn": C.lid_twonn,
           "lam_bg": C.lam_bg, "w_bg": C.w_bg, "nu": C.nu, "cs": cs, "f1": None, "f2": None, "single": None, "quick": True,
           "workload": C.workload}
    if ne < LW.MIN_EV:
        return res
    f1, f2 = LW.fit_sib(C.y, C.bg_pdf, C.w_bg, 1), LW.fit_sib(C.y, C.bg_pdf, C.w_bg, 2)
    c1 = LW._comps(f1); nsib = max(ne * sum(p for p, _ in c1), 1.0)
    ds = f1["Ds"][0]
    res["f1"] = {"pi": list(f1["pi"]), "Ds": list(f1["Ds"]),
                 "Ds_ci": [[ds * (1 - 1.96 / math.sqrt(nsib)), ds * (1 + 1.96 / math.sqrt(nsib))]] if c1 else [[None, None]]}
    res["f2"] = {"pi": list(f2["pi"]), "Ds": list(f2["Ds"])}
    if n1:
        D = C.lid_pool; ps1 = sum(p for p, _ in c1); ds1 = c1[0][1] if c1 else D
        obs = C.obs_conflict(single)
        _, lo, hi = wilson(np.array(obs) / 100.0 * n1, n1)
        P = {"exch": [100 * float(C.loo[single, i].mean()) for i in range(len(cs))],
             "closed": [100 * LW.mix_conflict(c, ps1, ds1 / D, D, k) for c in cs],
             "semi1": C.pred_conflict(f1, single), "semi2": C.pred_conflict(f2, single)}
        res["single"] = {"n": n1, "obs": obs, "se": [100 * math.sqrt(max(o / 100 * (1 - o / 100), 0.0) / n1) for o in obs],
                         "ci": [[100 * float(a), 100 * float(b)] for a, b in zip(lo, hi)], **P,
                         **{"err_" + a: LW.max_err(v, obs) for a, v in P.items()}}
        res["lid_sensitivity"] = {}
        for a, Dx in (("median", C.lid_median), ("pooled", C.lid_pool), ("twonn", C.lid_twonn)):
            pr = [100 * LW.mix_conflict(c, ps1, ds1 / Dx, Dx, k) for c in cs] if np.isfinite(Dx) and Dx > 0 else [None] * len(cs)
            res["lid_sensitivity"][a] = {"D": Dx, "closed": pr, "err_closed": LW.max_err(pr, obs)}
    return res


def fits_or_quick(results, exp, suffix="", law_exp=None, fig="figure"):
    """fits.json of an experiment; when it is missing, quick_fit of every law file of law_exp (default: the same
    experiment), with one printed note.  Returns ({cell: entry}, True when the entries are quick fits)."""
    F = jload(results, exp, "fits.json", suffix, quiet=True)
    if F:
        return F, False
    paths = law_paths(results, law_exp or exp, suffix)
    if not paths:
        return {}, False
    note("%s: no fits.json in %s, fitting %d law files in process (no bootstrap; run `python -m ariadne.law fit` for intervals)"
         % (fig, exp_dir(results, exp, suffix), len(paths)))
    out = {}
    for name, p in paths.items():
        try:
            out[name] = quick_fit(p)
        except (SystemExit, KeyError, ValueError) as err:
            note("%s: %s cannot be fitted (%s)" % (fig, name, err))
    return out, True


# ------------------------------------------------------------------------------------------------ statistics
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(vals):
    """(mean, lo, hi, n) of the finite values: Student t 95% interval of the mean over seeds.  With one value
    the interval is (nan, nan): the caller then falls back to the within-run interval of the file."""
    v = np.array([float(x) for x in vals if x is not None and np.isfinite(float(x))], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(v.mean())
    if n == 1:
        return m, float("nan"), float("nan"), 1
    tq = _T975[n - 2] if n - 1 <= 30 else 1.96 + 2.4 / (n - 1)
    h = tq * float(v.std(ddof=1)) / math.sqrt(n)
    return m, m - h, m + h, n


def nmean(vals):
    """Mean of the finite values (None and nan skipped); nan when there is none.  Never warns."""
    v = [fnum(x) for x in vals]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def col_mean_ci(rows):
    """Column-wise mean_ci of a list of equally long lists (None allowed).  Returns arrays mean, lo, hi, n."""
    A = np.array([[np.nan if x is None else float(x) for x in r] for r in rows], dtype=float)
    out = [mean_ci(A[:, j]) for j in range(A.shape[1])]
    return tuple(np.array([o[i] for o in out], dtype=float) for i in range(4))


def wilson(x, n, z=1.959964):
    """Wilson interval of a proportion; arrays allowed.  Returns (p, lo, hi)."""
    x = np.asarray(x, dtype=float); n = np.maximum(np.asarray(n, dtype=float), 1e-12)
    p = x / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(np.maximum(p * (1 - p), 0) / n + z * z / (4 * n * n)) / den
    return p, np.clip(mid - h, 0, 1), np.clip(mid + h, 0, 1)


def fnum(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def arr(xs):
    """List with None -> float array with nan."""
    return np.array([np.nan if x is None else float(x) for x in xs], dtype=float)


# ------------------------------------------------------------------------------------------------ theorem
def conflict_intervals(r, c):
    """Conflict set of the characterization theorem for ONE neighbour list r (sorted old distances r_1..r_k)
    and tolerance c: the set of distances t in (0, r_k) at which one concurrent insertion makes the held exact
    answer no longer c-admissible, as a merged list of intervals [(lo, hi), ...].

    Rank by rank: an insertion with r_{p-1} <= t < r_p has old rank p; it conflicts iff r_p > c t, or
    r_i > c r_{i-1} for some p < i <= k."""
    r = np.asarray(r, dtype=float); k = len(r); out = []
    gap = np.zeros(k, dtype=bool); gap[1:] = r[1:] > c * r[:-1]            # gap[i-1]: r_i > c r_{i-1} (i >= 2)
    later = np.zeros(k, dtype=bool)                                        # later[p-1]: a gap at some rank i > p
    for p in range(k - 1, 0, -1):
        later[p - 1] = later[p] or gap[p]
    for p in range(1, k + 1):
        lo = r[p - 2] if p > 1 else 0.0; hi = r[p - 1]
        if later[p - 1]:
            seg = (lo, hi)
        else:
            seg = (lo, min(hi, max(lo, r[p - 1] / c)))
        if seg[1] > seg[0]:
            if out and abs(out[-1][1] - seg[0]) < 1e-12:
                out[-1] = (out[-1][0], seg[1])
            else:
                out.append(seg)
    return out


def conflict_radius(ro, k, cs):
    """[reads, len(cs)]: the conflict radius R_c of every read (see the module docstring), from the old
    neighbour distances ro [reads, >= k]."""
    r = np.asarray(ro, dtype=float)[:, :k]
    prev = np.concatenate([np.zeros((len(r), 1)), r[:, :-1]], axis=1)
    out = np.zeros((len(r), len(cs)))
    for j, c in enumerate(cs):
        out[:, j] = np.where(r > c * prev, r / c, 0.0).max(axis=1)
    return out


def new_list(r, t):
    """Fresh exact distances after one insertion at distance t: the k smallest of r and t."""
    r = np.asarray(r, dtype=float)
    return np.sort(np.concatenate([r, [t]]))[:len(r)]


# ------------------------------------------------------------------------------------------------ axes
C_TICKS = (1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0)


def c_axis(ax, which="x", ticks=C_TICKS, lim=None, label="tolerance $c$"):
    """Log scale in c - 1 with ticks labelled by c.  Plot against (c - 1)."""
    a = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    if lim is not None:
        (ax.set_xlim if which == "x" else ax.set_ylim)(lim[0] - 1.0, lim[1] - 1.0)
        ticks = [t for t in ticks if lim[0] - 1e-9 <= t <= lim[1] + 1e-9]
    a.set_major_locator(FixedLocator([t - 1.0 for t in ticks]))
    a.set_major_formatter(plt.FuncFormatter(lambda v, _: ("%.2f" % (1 + v)).rstrip("0").rstrip(".") if v + 1 < 10 else "%g" % (1 + v)))
    a.set_minor_locator(NullLocator()); a.set_minor_formatter(NullFormatter())
    if label:
        (ax.set_xlabel if which == "x" else ax.set_ylabel)(label)


def clabel_inside(ax, cl, margin=0.07, **kw):
    """Contour labels, without the ones that matplotlib placed at the edge of the axes (they would be cut)."""
    labs = ax.clabel(cl, **kw) or []
    inv = ax.transAxes.inverted()
    keep = []
    for t in labs:
        x, y = inv.transform(ax.transData.transform(t.get_position()))
        if margin <= x <= 1 - margin and margin <= y <= 1 - margin:
            keep.append(t)
        else:
            t.remove()
    return keep


def log_plain(ax, which="both", subs=(1.0, 2.0, 5.0)):
    """Log axis with ticks at 1, 2, 5 times powers of ten, labelled as plain numbers (0.02, 0.05, 0.1, ...)."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    for name, a, lim in (("x", ax.xaxis, ax.get_xlim()), ("y", ax.yaxis, ax.get_ylim())):
        if which in (name, "both"):
            wide = max(lim) / max(min(lim), 1e-300) > 300
            a.set_major_locator(LogLocator(base=10, subs=(1.0,) if wide else subs))
            a.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
            a.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10))); a.set_minor_formatter(NullFormatter())


def pooled_ci(means, los, his):
    """Mean of a few runs with a 95% interval.  Three or more runs: Student t interval over the runs.  Two runs:
    the within-run intervals [lo_i, hi_i] are combined (independent runs, normal approximation).  One run: its
    own interval."""
    m, lo, hi, n = mean_ci(means)
    if n >= 3 or n == 0:
        return m, lo, hi, n
    if n == 1:                                                          # the interval of the file, as it is
        ok = [(fnum(l), fnum(h)) for v, l, h in zip(means, los, his) if v is not None and np.isfinite(fnum(v))]
        return (m, ok[0][0], ok[0][1], 1) if ok else (m, float("nan"), float("nan"), 1)
    se = [(fnum(h) - fnum(l)) / 3.92 for l, h in zip(los, his)]
    se = [x for x in se if np.isfinite(x)]
    if len(se) < n:
        return m, float("nan"), float("nan"), n
    h = 1.96 * math.sqrt(sum(x * x for x in se)) / n
    return m, m - h, m + h, n


def band(ax, x, lo, hi, color, alpha=0.18, **kw):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(np.asarray(x)[ok], lo[ok], hi[ok], color=color, alpha=alpha, lw=0, **kw)


def caps(ax, x, y, lo, hi, color=INK, ms=MS, marker="o", mfc=None, zorder=5, capsize=1.5, **kw):
    """Markers with asymmetric interval caps (intervals with non-finite ends are not drawn)."""
    x, y, lo, hi = (np.asarray(v, dtype=float) for v in (x, y, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    if ok.any():
        ax.errorbar(x[ok], y[ok], yerr=[np.maximum(y[ok] - lo[ok], 0), np.maximum(hi[ok] - y[ok], 0)], fmt="none", ecolor=color,
                    elinewidth=0.6, capsize=capsize, capthick=0.6, zorder=zorder - 1)
    return ax.plot(x, y, ls="none", marker=marker, ms=ms, mfc=color if mfc is None else mfc, mec=color, mew=0.7, zorder=zorder, **kw)


def grid_layout(n, cols=None, max_cols=6):
    """(rows, columns) of a small-multiples grid for n panels."""
    if cols is None:
        cols = n if n <= 4 else 3 if n <= 6 else 4 if n <= 12 else max_cols
    cols = max(1, min(cols, n))
    return int(math.ceil(n / cols)), cols


def legend_cols(width, n):
    """Columns of a figure legend with n entries that fit a figure of this width (inches)."""
    return min(n, n if width >= 6.5 else 3 if width >= 4.6 else 2)


def hide_unused(axes, n):
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)


# ------------------------------------------------------------------------------------------------ command line
def cli(main, doc, extra=()):
    """Common command line of a figure module:  --results DIR  --suffix S  --out DIR  plus the module's options.

    extra: list of (flag, kwargs) passed to add_argument; the parsed values are given to main as keywords."""
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=doc)
    ap.add_argument("--results", default="results", help="results directory (default: results)")
    ap.add_argument("--suffix", default="", help='suffix of the experiment directories ("_smoke" for the smoke runs)')
    ap.add_argument("--out", default="figures", help="output directory (default: figures)")
    for flag, kw in extra:
        ap.add_argument(flag, **kw)
    a = vars(ap.parse_args())
    return main(a.pop("results"), a.pop("suffix"), a.pop("out"), **a)
