#!/usr/bin/env python
"""Results bridge of the paper: result files -> paper/results.tex.

Every measured number in the two .tex files is written as \\res[shape]{key} (or \\ra, \\rb, \\ri). This script
resolves the keys from the result files and writes

    paper/results.tex          \\resdef{key}{value} for every key that has data (read by ariadne-preamble.tex)
    paper/results_report.md    (next to the .tex output) every key with its value, the unrounded value, the number of runs behind it and
                               its provenance; the keys without data; warnings

The key -> data map is code, in paper/resmap/*.py: one module per experiment group, each with a function
resolve(R) that calls R.put(key, value, src). The helper R (class Results below) reads rows, selects cells,
aggregates over seeds, and records what was selected so that it can be checked against the experiment grids.

    python paper/make_results.py                         # full results in results/        -> results.tex
    python paper/make_results.py --suffix _smoke         # smoke results (pipeline check)   -> results.tex with
                                                         #   provisional values (orange in the PDF, an error in a final build)
    python paper/make_results.py --selftest              # no results needed beyond the smoke runs: every cell of the
                                                         #   full grids gets a copy of a smoke row, so that every selector
                                                         #   and every field path of the map is exercised; writes
                                                         #   results_selftest.tex (not read by the paper) and the report
    python paper/make_results.py --context C1.A64        # print where the keys with this prefix occur in the text
    python paper/make_results.py --list                  # all keys of the two .tex files with their shapes

Formatting follows the placeholder shape in the text: 00.0 -> one decimal, 0.00 -> two, 0.000 -> three,
00 / NN -> integer, N -> a count in running text (spelled out up to twelve), 0{,}000 -> integer with thousands
separators, anything else -> the value as text.
Manual values (versions, machine, model names) come from paper/env.yaml.

Exit code: 0 if no resolver failed and no selector is invalid; 1 otherwise. Missing data is not an error (the
placeholder stays in the PDF) unless --require-all is given.
"""
import argparse, copy, datetime, glob, importlib.util, json, math, os, re, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import yaml                                                    # noqa: E402
from ariadne.plots import load as ld                           # noqa: E402
import run as RUN                                              # noqa: E402  (experiments/run.py: cells(), expand_env())

TEX = ["pvldb20-ariadne.tex", "pvldb20-ariadne-full.tex"]
SHORT = {"ra": "00.0", "rb": "0.00", "ri": "00"}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


# ------------------------------------------------------------------------------------------------ keys in the text
def tex_keys():
    """{key: {"shapes": set, "where": [(file, line number, line text)]}} for every \\res key in the .tex files."""
    out = {}
    pat = re.compile(r"\\(ra|rb|ri)\{([^}]*)\}|\\res(?:\[([^\]]*)\])?\{([^}]*)\}")
    for f in TEX:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p, encoding="utf-8"), 1):
            if line.lstrip().startswith("%"):
                continue
            for m in re.finditer(r"\\rarng\{([^}]*)\}\{([^}]*)\}", line):     # range of two keys (one decimal)
                for key in m.groups():
                    if key not in ("#1", "#2"):
                        e = out.setdefault(key, {"shapes": set(), "where": []})
                        e["shapes"].add(SHORT["ra"]); e["where"].append((f, i, line.rstrip()))
            for m in pat.finditer(line):
                key = m.group(2) or m.group(4); shape = SHORT[m.group(1)] if m.group(1) else (m.group(3) or "XX")
                if key in ("key", "#1", "#2"):
                    continue
                e = out.setdefault(key, {"shapes": set(), "where": []})
                e["shapes"].add(shape); e["where"].append((f, i, line.rstrip()))
    return out


def fmt(value, shape):
    """Format a value by the placeholder shape. Returns (text, warning or None)."""
    if isinstance(value, str):
        return value, None
    if isinstance(value, bool):
        return str(value), "boolean value"
    v = float(value)
    if not math.isfinite(v):
        return None, "not finite"
    warn = None
    m = re.fullmatch(r"0+\.(0+)", shape)
    if m:
        d = len(m.group(1)); s = "%.*f" % (d, abs(v))
        if float(s) == 0 and v != 0:
            warn = "rounds to zero (value %.4g)" % v
        return ("$-$" if v < 0 and float(s) != 0 else "") + s, warn
    if re.fullmatch(r"[0N]+", shape) or "{,}" in shape:
        n = int(round(v))
        if shape == "N" and 0 <= n <= 12 and abs(v - n) < 1e-9:       # a small count in running text is spelled out
            return WORDS[n], None
        if abs(v - n) > 0.02 * max(1.0, abs(v)) and abs(v - n) > 0.05:
            warn = "rounded to an integer (value %.4g)" % v
        if n == 0 and v != 0:
            warn = "rounds to zero (value %.4g)" % v
        s = "{:,}".format(abs(n)).replace(",", "{,}") if abs(n) >= 10000 or ("{,}" in shape and abs(n) >= 1000) else str(abs(n))
        return ("$-$" if n < 0 else "") + s, warn
    return ("%g" % v), "numeric value for a text placeholder"


# ------------------------------------------------------------------------------------------------ the helper
class Results:
    """What a resolver gets.  All selections go through sel() so that they can be checked against the grids."""

    def __init__(self, results, suffix="", selftest=False):
        self.results = results; self.suffix = suffix; self.selftest = selftest
        self.values = {}            # key -> {"value", "src", "n", "module"}
        self.missing = {}           # key -> reason (a resolver knows the key but found no data)
        self.problems = []          # invalid selectors, resolver exceptions
        self.module = "?"
        self._rows = {}; self._exp = {}; self._full = {}
        self.env = {}
        p = os.path.join(HERE, "env.yaml")
        if os.path.exists(p):
            self.env = yaml.safe_load(open(p, encoding="utf-8")) or {}

    # ---- experiments and their grids
    def exp(self, name):
        """The experiment file (dict), {} if there is none."""
        if name not in self._exp:
            # "gpt/C3_llm_decisions", "qwen/C4_downstream_qa": the experiment of a model directory
            # (experiments/vldb/<name>.yaml); everything else is experiments/<name>.yaml
            # "vm/C1_calls_vs_policy": the run of the 192-core server, kept next to the timing server's run (experiments/<name>.yaml)
            pre, base = (name.split("/", 1) + [""])[:2] if "/" in name else ("", name)
            p = os.path.join(ROOT, "experiments", "vldb", base + ".yaml") if pre in ("gpt", "qwen") else os.path.join(ROOT, "experiments", base + ".yaml")
            self._exp[name] = yaml.safe_load(RUN.expand_env(open(p).read())) if os.path.exists(p) else {}
        return self._exp[name]

    def is_script(self, name):
        return self.exp(name).get("kind") == "script" or not self.exp(name)

    def full_params(self, name):
        """Grid parameters of every cell of the full experiment: [{"cell", "case", <grid keys>, "seed"}]."""
        if name not in self._full:
            e = self.exp(name)
            self._full[name] = [] if self.is_script(name) else [dict(c["params"], cell=c["cell"], name=name) for c in RUN.cells(e, smoke=False)]
        return self._full[name]

    def path(self, exp, file=""):
        return os.path.join(self.results, exp + self.suffix, file)

    def exists(self, exp, file):
        return os.path.exists(self.path(exp, file))

    # ---- rows
    def rows(self, exp, file="summary.jsonl"):
        """Rows of <results>/<exp><suffix>/<file> (.jsonl or .json).  Failed cells are dropped; when a cell was run
        again the last line wins.  [] if the file does not exist.  In the self-test, files of engine experiments
        get one row per cell of the full grid (a copy of the most similar smoke row)."""
        k = (exp, file)
        if k not in self._rows:
            R = ld.rows(self.results, exp, file, self.suffix, quiet=True)
            if self.selftest and not self.is_script(exp) and R and all(isinstance(r, dict) and "cell" in r for r in R):
                R = self._synthesize(exp, R)
            self._rows[k] = R
        return self._rows[k]

    def _synthesize(self, exp, smoke_rows):
        """One row per cell of the full grid: a copy of the smoke row of the same kind of workload and protocol (so
        that it carries the same fields) that shares the most grid parameters."""
        e = self.exp(exp)
        kind = lambda c: (RUN.get_path(c, "workload.kind"), RUN.get_path(c, "protocol.kind"))
        smoke_kind = {c["cell"]: kind(c) for c in RUN.cells(e, smoke=True)}
        full_kind = {c["cell"]: kind(c) for c in RUN.cells(e, smoke=False)}
        out = []
        for par in self.full_params(exp):
            want = full_kind.get(par["cell"], (None, None))

            def score(r):
                have = smoke_kind.get(r.get("cell"), (None, None))
                return (have[0] == want[0], have[1] == want[1],
                        sum(1 for a, b in par.items() if a not in ("cell", "seed", "name") and str(r.get(a)) == str(b)))
            r = copy.deepcopy(max(smoke_rows, key=score)); r.update(par); r["_selftest"] = True
            out.append(r)
        return out

    def sel(self, exp, where=None, file="summary.jsonl", pred=None):
        """Rows of an experiment whose grid parameters match `where` ({"case": "A64", "protocol.policy": "cadm:1.1"};
        a value may be a list (any of) or a function of the value).  The selector is checked against the full
        grid of the experiment: one that can never match is reported as a problem."""
        where = dict(where or {})
        kw = {a.replace(".", "__"): b for a, b in where.items()}
        if not self.is_script(exp) and where:
            if not ld.where(self.full_params(exp), **kw):
                self.problems.append("%s: selector %s on %s matches no cell of the full grid" % (self.module, _show(where), exp))
        R = ld.where(self.rows(exp, file), pred, **kw)
        return R

    def json(self, exp, file):
        """Parsed content of a .json / .jsonl file written by a script experiment ([] if missing)."""
        p = self.path(exp, file)
        return ld.read_jsonl(p) if os.path.exists(p) else []

    # ---- numbers
    @staticmethod
    def get(row, field, default=None):
        """row[field]; dotted paths go into nested dicts; a callable is applied to the row."""
        return ld.get(row, field, default)

    def vals(self, rows, field):
        """Finite numeric values of a field over rows."""
        out = []
        for r in rows:
            v = ld.num(ld.get(r, field))
            if math.isfinite(v):
                out.append(v)
        return out

    def mean(self, rows, field):
        v = self.vals(rows, field)
        return sum(v) / len(v) if v else None

    def total(self, rows, field):
        v = self.vals(rows, field)
        return sum(v) if v else None

    def vmax(self, rows, field):
        v = self.vals(rows, field)
        return max(v) if v else None

    def vmin(self, rows, field):
        v = self.vals(rows, field)
        return min(v) if v else None

    @staticmethod
    def saved(a, ref):
        """Percentage of `ref` saved by `a`: 100 * (1 - a / ref); None if either is missing or ref is 0."""
        if a is None or ref is None or ref == 0:
            return None
        return 100.0 * (1.0 - a / ref)

    @staticmethod
    def ratio(a, b, scale=1.0):
        if a is None or b is None or b == 0:
            return None
        return scale * a / b

    # ---- output
    def put(self, key, value, src="", n=None):
        """Define a key.  value None (or nan) records the key as known but without data, with src as the reason."""
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            self.missing.setdefault(key, src or "no data")
            return None
        if key in self.values:
            self.problems.append("%s: key %s defined twice (also by %s)" % (self.module, key, self.values[key]["module"]))
        self.values[key] = {"value": value, "src": src, "n": n, "module": self.module}
        self.missing.pop(key, None)
        return value

    def stat(self, key, exp, field, where=None, file="summary.jsonl", how="mean", scale=1.0):
        """sel + aggregate over the selected rows (the seeds) + put.  how: mean | sum | max | min | count.  Returns the value."""
        R = self.sel(exp, where, file)
        f = {"mean": self.mean, "sum": self.total, "max": self.vmax, "min": self.vmin}.get(how)
        v = len(R) if how == "count" else (f(R, field) if R else None)
        if v is not None:
            v *= scale
        name = field if isinstance(field, str) else getattr(field, "__name__", "f(row)")
        src = "%s/%s %s: %s(%s)%s over %d rows" % (exp, file, _show(where), how, name, "" if scale == 1 else " x %g" % scale, len(R))
        self.put(key, v, src if v is not None else src + " -> no data", n=len(R))
        return v

    def put_range(self, prefix, values, src=""):
        """prefix.min and prefix.max over the values that are not None."""
        v = [x for x in values if x is not None and math.isfinite(x)]
        self.put(prefix + ".min", min(v) if v else None, src + " (min of %d)" % len(v), n=len(v))
        self.put(prefix + ".max", max(v) if v else None, src + " (max of %d)" % len(v), n=len(v))

    def val(self, key):
        """Raw value of a key defined earlier (None if it has no data): for derived keys."""
        e = self.values.get(key)
        return e["value"] if e else None


def _show(where):
    return "[" + ", ".join("%s=%s" % (a, b if not callable(b) else "f") for a, b in (where or {}).items()) + "]"


# ------------------------------------------------------------------------------------------------ main
def load_modules():
    mods = []
    for p in sorted(glob.glob(os.path.join(HERE, "resmap", "*.py"))):
        if os.path.basename(p).startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location("resmap_" + os.path.basename(p)[:-3], p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        mods.append((os.path.basename(p), m))
    return mods


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results")); ap.add_argument("--suffix", default="")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--context", metavar="PREFIX"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", metavar="MODULE", help="run one module of paper/resmap (for development)")
    ap.add_argument("--out", default=None, help="output .tex (default: paper/results.tex, or results_selftest.tex)")
    a = ap.parse_args()
    keys = tex_keys()
    if a.list:
        for k in sorted(keys):
            print("%-34s %s" % (k, " | ".join(sorted(keys[k]["shapes"]))))
        print("%d keys" % len(keys)); return 0
    if a.context is not None:
        for k in sorted(keys):
            if k.startswith(a.context):
                print("== %s  [%s]" % (k, " | ".join(sorted(keys[k]["shapes"]))))
                for f, i, line in keys[k]["where"]:
                    print("   %s:%d: %s" % (f.split("-")[-1], i, line.strip()))
        return 0
    if a.selftest:
        a.suffix = "_smoke"
    R = Results(os.path.abspath(a.results), a.suffix, a.selftest)
    for name, m in load_modules():
        if a.only and a.only not in name:
            continue
        R.module = name
        try:
            m.resolve(R)
        except Exception:
            R.problems.append("%s: resolver failed\n%s" % (name, traceback.format_exc()))
    provisional = bool(a.suffix) or a.selftest
    lines, rep, warns = [], [], []
    for k in sorted(R.values):
        e = R.values[k]
        if k not in keys:
            continue
        shapes = sorted(keys[k]["shapes"])
        if len({re.sub(r"N+", "0", s) for s in shapes}) > 1:
            warns.append("%s: used with different shapes %s; formatted with the first" % (k, shapes))
        text, w = fmt(e["value"], shapes[0])
        if text is None:
            R.missing[k] = "value is not finite"; continue
        if w:
            warns.append("%s = %s: %s" % (k, text, w))
        if e["n"] is not None and not provisional and isinstance(e["n"], int) and e["n"] in (1, 2) and not R.is_script(e["src"].split("/")[0]):
            warns.append("%s: %d run(s) behind this number" % (k, e["n"]))
        lines.append("\\resdef{%s}{%s}" % (k, ("\\resprov{%s}" % text) if provisional else text))
        raw = e["value"] if isinstance(e["value"], str) else "%.6g" % e["value"]
        rep.append("| `%s` | %s | %s | %s | %s | %s |" % (k, text.replace("{,}", ",").replace("$-$", "-"), raw, "" if e["n"] is None else e["n"], e["module"], e["src"].replace("|", "/")))
    unresolved = sorted(k for k in keys if k not in R.values or k in R.missing)
    unused = sorted(k for k in R.values if k not in keys)
    out = a.out or os.path.join(HERE, "results_selftest.tex" if a.selftest else "results.tex")
    kind = "SELF-TEST (copies of smoke rows on the full grids: the values mean nothing)" if a.selftest else ("PROVISIONAL (results with suffix %s)" % a.suffix if a.suffix else "final")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(out, "w", encoding="utf-8") as f:
        f.write("%% generated by paper/make_results.py on %s from %s (suffix '%s'): %s\n" % (stamp, R.results, a.suffix, kind))
        f.write("%% %d of %d keys have a value; do not edit, edit paper/resmap/*.py or paper/env.yaml\n" % (len(lines), len(keys)))
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(out)[0] + "_report.md", "w", encoding="utf-8") as f:
        f.write("# Numbers of the paper: %s\n\n" % kind)
        f.write("Generated %s from `%s` (suffix `%s`). %d keys in the text, %d with a value, %d without.\n\n" % (stamp, R.results, a.suffix, len(keys), len(lines), len(unresolved)))
        if R.problems:
            f.write("## Problems (must be fixed)\n\n" + "\n".join("- " + p.replace("\n", "\n  ") for p in R.problems) + "\n\n")
        if warns:
            f.write("## Warnings (look at each)\n\n" + "\n".join("- " + w for w in warns) + "\n\n")
        f.write("## Keys without a value (%d)\n\n" % len(unresolved))
        for k in unresolved:
            why = R.missing.get(k, "no resolver defines this key")
            f.write("- `%s`: %s\n" % (k, why))
        if unused:
            f.write("\n## Keys defined by the map but not used in the text (%d)\n\n" % len(unused) + ", ".join("`%s`" % k for k in unused) + "\n")
        f.write("\n## Values\n\n| key | in the text | unrounded | rows | module | provenance |\n|---|---|---|---|---|---|\n" + "\n".join(rep) + "\n")
    nores = [k for k in unresolved if k not in R.missing]
    print("make_results: %d keys in the text, %d with a value, %d without (%d of them have no resolver), %d problems, %d warnings -> %s"
          % (len(keys), len(lines), len(unresolved), len(nores), len(R.problems), len(warns), os.path.relpath(out, ROOT)))
    for p in R.problems:
        print("  PROBLEM " + p)
    if nores:
        print("  no resolver: " + " ".join(nores[:40]) + (" ..." if len(nores) > 40 else ""))
    bad = bool(R.problems) or (bool(nores) and not a.only) or (a.require_all and bool(unresolved))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
