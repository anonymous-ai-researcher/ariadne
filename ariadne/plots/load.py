"""Loading and aggregating result files for the figure and table scripts.

    from ariadne.plots import load as ld
    R = ld.rows("results", "C1_calls_vs_policy", suffix="_smoke")              # summary.jsonl -> list of dicts
    A = ld.rows("results", "C1_calls_vs_policy", "audit_run.jsonl", "_smoke")  # any JSONL / JSON file of the run
    for key, s in ld.agg(R, ["case", "protocol.policy"], "extra_calls_per_100").items():
        s.mean, s.lo, s.hi, s.n, s.half                                       # mean and 95% interval over seeds (see Stats)
    xs, st = ld.series(ld.where(R, case="L4"), "protocol.policy", "tps")       # sorted x values and their Stats
    ld.runs("results", "C1_calls_vs_policy", "_smoke")                         # paths of the <cell>.pkl files

Results of experiment X live in <results>/X<suffix>/ (suffix "_smoke" for the smoke runs, "" for the full runs).
A missing file gives [] and one printed note, so that a figure whose data is absent can return quietly.

Values are addressed by key; a dotted key that is not a literal key of the row is resolved through nested
dicts ("divergence.dup" -> row["divergence"]["dup"]), and a callable is applied to the row.
"""
import argparse, glob, json, math, os, re

import numpy as np

_NOTED = set()


def note(msg):
    """Print a note once."""
    if msg not in _NOTED:
        _NOTED.add(msg)
        print("  note: " + msg)


def exp_dir(results, exp, suffix=""):
    return os.path.join(results, exp + suffix)


def path(results, exp, file="summary.jsonl", suffix=""):
    return os.path.join(exp_dir(results, exp, suffix), file)


def exists(results, exp, file="summary.jsonl", suffix=""):
    return os.path.exists(path(results, exp, file, suffix))


def read_jsonl(p):
    """Rows of a .jsonl file (one JSON object per line) or of a .json file (a list, or one object)."""
    out = []
    with open(p) as f:
        text = f.read()
    if p.endswith(".json"):
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else [obj]
        except ValueError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def rows(results, exp, file="summary.jsonl", suffix="", quiet=False):
    """List of dicts from <results>/<exp><suffix>/<file>; [] (and a printed note) when the file is missing.

    Rows of summary.jsonl that record a failed cell (an "error" field and no "tasks") are dropped, and when a
    cell appears several times (re-runs append) the last line wins."""
    p = path(results, exp, file, suffix)
    if not os.path.exists(p):
        if not quiet:
            note("%s: no %s in %s" % (exp, file, exp_dir(results, exp, suffix)))
        return []
    R = read_jsonl(p)
    if file == "summary.jsonl":
        R = [r for r in R if isinstance(r, dict) and not ("error" in r and "tasks" not in r)]
    R = [r for r in R if isinstance(r, dict)]
    if R and all("cell" in r for r in R):
        last = {}
        for r in R:
            last[r["cell"]] = r
        if len(last) < len(R):
            R = list(last.values())
    return R


def runs(results, exp, suffix="", pattern="*"):
    """Sorted paths of the <cell>.pkl files of an experiment (load one with ariadne.audit.load_run)."""
    return sorted(p for p in glob.glob(os.path.join(exp_dir(results, exp, suffix), pattern + ".pkl"))
                  if not p.endswith(".law.pkl"))


def npz(results, exp, suffix="", pattern="*"):
    """{cell name: path} of the <cell>.audit.npz files (per-task arrays written by the run audit)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(exp_dir(results, exp, suffix), pattern + ".audit.npz"))):
        out[os.path.basename(p)[:-len(".audit.npz")]] = p
    return out


def cell_arrays(results, exp, row, suffix=""):
    """Per-task arrays of the cell of a summary row (dict of numpy arrays), or None when there is no npz."""
    p = os.path.join(exp_dir(results, exp, suffix), str(row.get("cell")) + ".audit.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def cell_pkl(results, exp, row, suffix=""):
    p = os.path.join(exp_dir(results, exp, suffix), str(row.get("cell")) + ".pkl")
    return p if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------ access
_MISSING = object()


def get(row, key, default=None):
    """row[key]; dotted keys fall back to nested dicts; a callable key is applied to the row."""
    if callable(key):
        try:
            return key(row)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return default
    if key in row:
        return row[key]
    key = str(key)
    for i, ch in enumerate(key):                      # "a.b" -> row["a"]["b"]; the tail may itself contain dots (">1.1")
        if ch == "." and isinstance(row.get(key[:i]), dict):
            v = get(row[key[:i]], key[i + 1:], _MISSING)
            if v is not _MISSING:
                return v
    return default


def num(x, default=float("nan")):
    """float(x), or nan for None / non-numeric values."""
    if isinstance(x, bool):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def where(R, _pred=None, **eq):
    """Rows whose grid parameters match.  Keyword names use "__" for "." (protocol__policy="exact"); a value
    may be a list / set / tuple (membership) or a callable (predicate on the value).

        where(R, case="L4", protocol__policy=["exact", "cadm:1.1"])
        where(R, lambda r: r["seed"] <= 3)
    """
    out = []
    for r in R:
        ok = _pred(r) if _pred is not None else True
        if ok:
            for k, v in eq.items():
                val = get(r, k.replace("__", "."))
                if callable(v):
                    ok = bool(v(val))
                elif isinstance(v, (list, tuple, set, frozenset)):
                    ok = val in v or str(val) in {str(x) for x in v}
                else:
                    ok = val == v or str(val) == str(v)
                if not ok:
                    break
        if ok:
            out.append(r)
    return out


def values(R, key):
    """Distinct values of a grid parameter, in a natural order (numbers numerically, strings by embedded number)."""
    seen = []
    for r in R:
        v = get(r, key)
        if isinstance(v, (dict, list)):
            v = json.dumps(v, sort_keys=True)
        if v is not None and v not in seen:
            seen.append(v)
    return natsorted(seen)


def natkey(v):
    if isinstance(v, bool):
        return (0, float(v), "")
    if isinstance(v, (int, float)):
        return (0, float(v), "")
    parts = re.split(r"(\d+(?:\.\d+)?)", str(v))
    return (1, 0.0, tuple((0, float(p)) if re.fullmatch(r"\d+(?:\.\d+)?", p) else (1, p) for p in parts if p != ""))


def natsorted(vals):
    try:
        return sorted(vals, key=natkey)
    except TypeError:
        return sorted(vals, key=str)


def grid_keys(R, ignore=("seed", "cell", "name")):
    """Names of the grid parameters of an experiment: the keys that precede "tasks" in a summary row."""
    keys = []
    for r in R:
        for k in r:
            if k == "tasks":
                break
            if k not in ignore and k not in keys:
                keys.append(k)
    return keys


def varying(R, ignore=("seed", "cell", "name")):
    """Grid parameters that take more than one value."""
    return [k for k in grid_keys(R, ignore) if len(values(R, k)) > 1]


def policy_key(R):
    """The key that names the protocol in these rows ("protocol.policy", "proto", "protocol", "policy"), or None."""
    for k in ("protocol.policy", "proto", "policy", "protocol", "protocol.kind", "variant"):
        if any(k in r and not isinstance(r[k], (dict, list)) for r in R):
            return k
    return None


def spec_of(row, default="none"):
    """Protocol / policy spec of a summary row, for style.policy_style and style.pretty."""
    for k in ("protocol.policy", "proto", "policy", "protocol", "variant"):
        v = row.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return str(v)
    return default


def number_in(name, default=float("nan")):
    """First number in a case name ("A64" -> 64, "think0.2" -> 0.2, "msm-100k" -> 100000)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM]?)(?![a-zA-Z0-9.])|(\d+(?:\.\d+)?)", str(name))
    if not m:
        return default
    if m.group(1) is not None:
        return float(m.group(1)) * {"": 1, "k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6}[m.group(2)]
    return float(m.group(3))


# ------------------------------------------------------------------------------------------------ statistics
# two-sided 97.5% Student t quantiles for 1..30 degrees of freedom (no scipy dependency)
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df):
    if df < 1:
        return float("nan")
    if df <= 30:
        return _T975[int(df) - 1]
    return 1.960 + 2.4 / df


class Stats:
    """mean, interval [lo, hi], half width, standard deviation and n of a sample (n = number of seeds).

    method "auto" (default): 95% Student t interval of the mean for n >= 3; for n = 2 the interval is the range of
    the two runs (a t interval on one degree of freedom is 12.7 standard errors wide and says nothing useful);
    "t": Student t for every n >= 2; "bootstrap": percentile bootstrap of the mean.  `kind` records which one was
    used ("t", "range", "bootstrap", or "point" for a single run, where lo = hi = mean)."""
    __slots__ = ("mean", "lo", "hi", "n", "sd", "vals", "kind")

    def __init__(self, vals, method="auto", n_boot=2000, seed=0):
        v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
        self.vals = v
        self.n = int(v.size)
        self.kind = "point"
        if self.n == 0:
            self.mean = self.lo = self.hi = self.sd = float("nan")
            return
        self.mean = float(v.mean())
        self.sd = float(v.std(ddof=1)) if self.n > 1 else 0.0
        if self.n == 1 or self.sd == 0.0:
            self.lo = self.hi = self.mean
            self.kind = "point" if self.n == 1 else ("range" if (method == "auto" and self.n == 2) else method if method != "auto" else "t")
        elif method == "bootstrap":
            rng = np.random.default_rng(seed)
            m = rng.choice(v, size=(n_boot, self.n), replace=True).mean(axis=1)
            self.lo, self.hi = (float(x) for x in np.percentile(m, [2.5, 97.5]))
            self.kind = "bootstrap"
        elif method == "auto" and self.n == 2:
            self.lo, self.hi = float(v.min()), float(v.max())
            self.kind = "range"
        else:
            h = t975(self.n - 1) * self.sd / math.sqrt(self.n)
            self.lo, self.hi = self.mean - h, self.mean + h
            self.kind = "t"

    @property
    def half(self):
        return (self.hi - self.lo) / 2.0 if self.n else float("nan")

    @property
    def single(self):
        return self.n <= 1

    @property
    def err(self):
        """(mean - lo, hi - mean), the asymmetric error for ax.errorbar."""
        return (self.mean - self.lo, self.hi - self.mean)

    def __repr__(self):
        return "Stats(mean=%.4g, lo=%.4g, hi=%.4g, n=%d)" % (self.mean, self.lo, self.hi, self.n)


def stats(vals, method="auto"):
    return Stats(vals, method)


def agg(R, by, value, method="auto"):
    """Group rows by the key(s) `by` and aggregate `value` over the rows of each group (the seeds).

    by: a key or a list of keys (dotted keys and callables allowed); value: a key or a callable.
    Returns an insertion-ordered dict {group: Stats}; the group is a scalar for one key, a tuple otherwise.
    method: "auto" (Student t interval of the mean for n >= 3, the range for n = 2; see Stats), "t" or "bootstrap"."""
    single = not isinstance(by, (list, tuple))
    keys = [by] if single else list(by)
    groups = {}
    for r in R:
        g = tuple(_hashable(get(r, k)) for k in keys)
        v = num(get(r, value))
        groups.setdefault(g[0] if single else g, []).append(v)
    return {g: Stats(v, method) for g, v in groups.items()}


def _hashable(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v


def series(R, x, value, method="auto", xnum=None):
    """Aggregate `value` by `x` and sort by x.  Returns (xs, [Stats]).  xnum maps an x value to a number
    for sorting and plotting (default: the value itself when numeric, else the number in its name)."""
    A = agg(R, x, value, method)
    def f(v):
        if xnum is not None:
            return xnum(v)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else number_in(v)
    items = sorted(((f(k), s) for k, s in A.items() if s.n > 0), key=lambda t: (math.isnan(t[0]), t[0]))
    return [a for a, _ in items], [s for _, s in items]


def arrays(S):
    """(mean, lo, hi) numpy arrays of a list of Stats."""
    return (np.array([s.mean for s in S]), np.array([s.lo for s in S]), np.array([s.hi for s in S]))


def n_seeds(R):
    return len({r.get("seed") for r in R if "seed" in r}) or 1


def wilson(k, n, z=1.96):
    """Wilson score interval of a proportion k / n: (p, lo, hi)."""
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - h), min(1.0, mid + h)


def ccdf(x):
    """Empirical complementary CDF: (sorted values, P[X >= value])."""
    v = np.sort(np.asarray(x, dtype=float))
    v = v[~np.isnan(v)]
    if v.size == 0:
        return v, v
    return v, 1.0 - np.arange(v.size) / v.size


def pooled(results, exp, R, field, suffix=""):
    """Concatenate one per-task array (e.g. "rho_committed") over the cells of the rows R (pools the seeds)."""
    out = []
    for r in R:
        a = cell_arrays(results, exp, r, suffix)
        if a is not None and field in a:
            out.append(np.asarray(a[field], dtype=float).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def join(R, A, on="cell"):
    """Rows of R extended with the fields of the audit rows A of the same cell (audit fields do not overwrite)."""
    idx = {a.get(on): a for a in A}
    out = []
    for r in R:
        a = idx.get(r.get(on))
        m = dict(a) if a else {}
        m.update(r)
        out.append(m)
    return out


def cli(main, doc=None):
    """Command line of a figure module:  python -m ariadne.plots.<module> --results results --suffix _smoke --out figures"""
    ap = argparse.ArgumentParser(description=(doc or "").strip().split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="directory that holds the <experiment><suffix>/ directories")
    ap.add_argument("--suffix", default="", help='"" for the full runs, "_smoke" for the smoke runs')
    ap.add_argument("--out", default="figures", help="output directory")
    a = ap.parse_args()
    return main(a.results, a.suffix, a.out)


def skip(name, why):
    """Uniform message of a figure that is not produced; returns [] (what main returns in that case)."""
    print("  skipped %s: %s" % (name, why))
    return []
