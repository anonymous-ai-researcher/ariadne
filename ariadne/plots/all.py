"""Build every figure and table of the paper from the result files.

    python -m ariadne.plots.all --results results --suffix _smoke --out figures

Imports every module of this package whose name starts with "fig_" (system side: fig_s_*, theory side: fig_t_*), calls
its main(results, suffix, out), then builds the LaTeX tables (tables.py, written to <out>/tables/).  A failure in one
figure is caught and reported and does not stop the others.  At the end: what was produced, what was skipped for lack
of data, and what failed.  --only takes prefixes of module names (e.g. --only fig_s_10 fig_t); --no-tables skips the tables.
"""
import argparse, importlib, inspect, os, pkgutil, sys, time, traceback


def modules(only=None):
    import ariadne.plots as pkg
    names = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__) if m.name.startswith("fig_"))
    if only:
        names = [n for n in names if any(n.startswith(o) for o in only)]
    return names


def snapshot(out):
    snap = {}
    for root, _, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f)
            try:
                snap[p] = os.stat(p).st_mtime_ns
            except OSError:
                pass
    return snap


def call_main(mod, results, suffix, out):
    """Call main with the arguments it accepts (positional (results, suffix, out) or keywords)."""
    fn = getattr(mod, "main", None)
    if fn is None:
        raise AttributeError("module has no main()")
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = ["results", "suffix", "out"]
    if {"results", "suffix", "out"} <= set(params):
        return fn(results=results, suffix=suffix, out=out)
    return fn(results, suffix, out)


def main(results="results", suffix="", out="figures", only=None, tables=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    produced, skipped, failed = [], [], []
    for name in modules(only):
        before = snapshot(out)
        t0 = time.time()
        print("== %s" % name, flush=True)
        try:
            mod = importlib.import_module("ariadne.plots." + name)
            call_main(mod, results, suffix, out)
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append((name, "exit code %r" % (e.code,)))
        except Exception as e:
            failed.append((name, "%s: %s" % (type(e).__name__, e)))
            traceback.print_exc()
        finally:
            plt.close("all")
        after = snapshot(out)
        new = sorted(p for p, m in after.items() if before.get(p) != m)
        if any(n == name for n, _ in failed):
            continue
        if new:
            stems = sorted({os.path.splitext(os.path.relpath(p, out))[0] for p in new})
            own = [x for x in stems if os.path.basename(x).startswith(name)]      # ignore files another process wrote meanwhile
            stems = own or stems
            produced.append((name, stems, time.time() - t0))
        else:
            skipped.append(name)
    table_files = []
    if tables:
        print("== tables", flush=True)
        try:
            from . import tables as T
            table_files = T.main(results, suffix, out) or []
        except Exception as e:
            failed.append(("tables", "%s: %s" % (type(e).__name__, e)))
            traceback.print_exc()
    print("\n" + "=" * 78)
    print("produced (%d modules):" % len(produced))
    for name, stems, dt in produced:
        extra = "" if stems == [name] else "  -> " + ", ".join(stems)
        print("  %-34s %5.1fs%s" % (name, dt, extra))
    if table_files:
        print("tables (%d): %s" % (len(table_files), ", ".join(os.path.basename(p) for p in table_files)))
    print("skipped for lack of data (%d):%s" % (len(skipped), "" if skipped else " none"))
    for name in skipped:
        print("  " + name)
    print("failed (%d):%s" % (len(failed), "" if failed else " none"))
    for name, why in failed:
        print("  %-34s %s" % (name, why))
    print("output directory: %s" % os.path.abspath(out))
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results"); ap.add_argument("--suffix", default=""); ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", help="prefixes of module names to run"); ap.add_argument("--no-tables", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.results, a.suffix, a.out, a.only, not a.no_tables))
