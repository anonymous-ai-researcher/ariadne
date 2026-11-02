"""Compare a stock pgvector against one patched to repeat the SSI conflict check.

This module builds nothing on its own. It (a) prints the exact commands to build a
patched pgvector from source with patches/pgvector_ssi_recheck.patch and install it
side by side, and (b) with --run executes the hammer and hole tests against the
pgvector that is currently installed and labels every output line with the on-disk
library sha256, so a stock run and a patched run can be compared line for line.

The patch repeats CheckForSerializableConflictIn after HnswInsertTuple in
hnswinsert, closing the window hole.py and hammer.py exploit. After it, hammer
should report 0 duplicates (plain mode) and hole should abort T1 instead of
committing a duplicate.

Positive result (comparison): with the stock library, --run --expect-clean exits
nonzero (anomalies present); after swapping in the patched library and rerunning,
--run --expect-clean exits 0 (no anomalies, at the cost of more aborts).

Invocation:
    python -m ariadne.pgtests.patched                       # print build/swap instructions
    python -m ariadne.pgtests.patched --run --pairs 3000 --trials 8 --out results/A6_ssi_soundness/patched_library.jsonl
    python -m ariadne.pgtests.patched --run --expect-clean  # exit 1 if any anomaly is seen
    python -m ariadne.pgtests.patched --run --smoke
"""
import argparse, json, os, subprocess, sys
from . import common as C

PATCH_REL = "patches/pgvector_ssi_recheck.patch"


def instructions(dsn):
    ver = C.versions(dsn)
    patch = os.path.abspath(PATCH_REL)
    pgv = ver.get("pgvector") or "0.8.7"
    lib = ver.get("pgvector_lib") or "/usr/lib/postgresql/16/lib/vector.so"
    print(f"""\
# ---------------------------------------------------------------------------
# Build a patched pgvector {pgv} side by side with the stock install
# ---------------------------------------------------------------------------
# Currently installed pgvector: {pgv}
#   library : {lib}
#   sha256  : {ver.get('pgvector_lib_sha256')}
#
# 1. Fetch the matching source and apply the recheck patch:
      git clone --branch v{pgv} https://github.com/pgvector/pgvector.git /tmp/pgvector-patched
      cd /tmp/pgvector-patched
      git apply {patch}            # repeats CheckForSerializableConflictIn after HnswInsertTuple
#
# 2. Build against the running server's headers (needs the -server/-dev package and
#    the same compiler flags the stock build used):
      make clean
      make                                     # produces ./vector.so
#
# 3. Keep BOTH libraries so you can switch without reinstalling. Save the stock one
#    once, then install the patched build, and swap by copying the file you want into
#    place (stop no servers that other work shares; copy is atomic enough for a test):
      PKGLIB="$(pg_config --pkglibdir)"
      cp -n "$PKGLIB/vector.so" "$PKGLIB/vector_stock.so"     # keep the stock copy
      cp ./vector.so            "$PKGLIB/vector_patched.so"   # keep the patched copy
      # to run patched:  cp "$PKGLIB/vector_patched.so" "$PKGLIB/vector.so"
      # to run stock:    cp "$PKGLIB/vector_stock.so"   "$PKGLIB/vector.so"
      # a backend loads the library at first use per session, so start fresh
      # connections (the scratch databases these tests create are always fresh)
      # after a swap.
#
# 4. Measure: run this module with --run under each library and diff the JSONL by the
#    pgvector_lib_sha256 field:
      # stock library: written by the last command of A6 (experiments/A6_ssi_soundness.yaml)
      python -m ariadne.pgtests.patched --run --pairs 3000 --trials 8 --out results/A6_ssi_soundness/installed_library.jsonl
      #  (swap in the patched library, open new connections)
      python -m ariadne.pgtests.patched --run --pairs 3000 --trials 8 --expect-clean --out results/A6_ssi_soundness/patched_library.jsonl
      #  (swap the stock library back in)
      # paper/make_results.py takes the "with the fix" numbers from patched_library.jsonl only if its library sha256
      # differs from that of installed_library.jsonl.
#
# NOTE: --run only READS the installed library (it never installs anything); swapping the .so file is the manual step above.
# ---------------------------------------------------------------------------""")


def run_subtest(module, extra, smoke):
    cmd = [sys.executable, "-m", f"ariadne.pgtests.{module}"]
    if smoke:
        cmd.append("--smoke")
    cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    recs = []
    for line in p.stdout.splitlines():
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    if p.returncode != 0 and not recs:
        sys.stderr.write(p.stderr)
    return recs, p.returncode


def run(a):
    ver = C.versions(a.dsn)
    label = ver.get("pgvector_lib_sha256")
    out = C.JsonlWriter(a.out)
    anomalies = 0
    try:
        ham, _ = run_subtest("hammer", ["--dsn", a.dsn, "--mode", "plain", "--index", "hnsw",
                                        "--pairs", str(a.pairs)], a.smoke)
        for r in ham:
            r["test"] = "hammer"; r["lib_sha256"] = label
            out.write(r)
            anomalies += int(r.get("duplicate_rows", 0) or 0)
        hol, code = run_subtest("hole", ["--dsn", a.dsn, "--index", "hnsw", "--trials", str(a.trials)], a.smoke)
        if code == 2 and not hol:
            out.write({"test": "hole", "skipped": "gdb/ptrace unavailable", "lib_sha256": label})
        for r in hol:
            r["test"] = "hole"; r["lib_sha256"] = label
            out.write(r)
            if r.get("scenario") == "SUMMARY":
                anomalies += int(r.get("hole_duplicates", 0) or 0)
        out.write({"test": "SUMMARY", "lib_sha256": label, "anomalies": anomalies,
                   "versions": ver, "params": {"pairs": a.pairs, "trials": a.trials, "smoke": a.smoke}})
    finally:
        out.close()
    if a.expect_clean and anomalies > 0:
        print(f"patched: {anomalies} anomalies seen with library {label}; expected clean -> exit 1", file=sys.stderr)
        sys.exit(1)
    return anomalies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--run", action="store_true", help="run hammer+hole against the installed pgvector")
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--expect-clean", dest="expect_clean", action="store_true",
                    help="exit 1 if any anomaly is seen (use against a patched library)")
    a = ap.parse_args()
    if a.run:
        run(a)
    else:
        instructions(a.dsn)


if __name__ == "__main__":
    main()
