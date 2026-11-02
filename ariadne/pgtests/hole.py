"""Deterministic (debugger) reproduction of the SSI unsoundness window on HNSW.

Question: can two SERIALIZABLE transactions that each run [k-NN search; insert if
no near duplicate] both commit a near duplicate, even though every serial order
stores exactly one row?

Mechanism: index_insert() calls CheckForSerializableConflictIn(index) BEFORE
aminsert. pgvector's hnswinsert first searches the graph and only then links the
new element. A debugger pauses T1 at the entry of the AM insert function (after
PostgreSQL's conflict check, before the element is linked). T2 then searches,
inserts and commits; its relation SIREAD lock was taken after T1's check and its
scan ran before T1's element was linked, so the read-write dependency is never
recorded. T1 resumes and commits. Both commit -> a duplicate under SERIALIZABLE.

Positive result: the HOLE scenario stores 2 rows for the event (both commit); the
CONTROL scenario (T2 starts only after T1's insert returned) stores 1 row and
aborts one transaction.

--index hnsw pauses at hnswinsert; --index ivfflat pauses at ivfflatinsert (the
ivfflat insert path is a single list append with no graph walk, so the window may
not appear; the result is reported honestly either way).

Needs gdb and permission to ptrace. Exits with code 2 and a clear message if gdb
is missing or ptrace is blocked. gdb processes are always cleaned up.

Invocation:
    python -m ariadne.pgtests.hole --trials 8 --out results/T_pg/hole.jsonl
    python -m ariadne.pgtests.hole --smoke
    python -m ariadne.pgtests.hole --index ivfflat --trials 4
"""
import argparse, os, shutil, signal, subprocess, sys, tempfile, threading, time
from . import common as C

# Handshake directory between the test and the gdb process that pauses the backend (gdb writes marker files here).
# ARIADNE_PG_TMP overrides it; it is created on import and made writable for every user, since gdb may run as postgres.
TMP = os.environ.get("ARIADNE_PG_TMP") or os.path.join(tempfile.gettempdir(), "ariadne_pgtmp")
os.makedirs(TMP, exist_ok=True)
try:
    os.chmod(TMP, 0o777)
except OSError:
    pass


def preflight_gdb():
    if shutil.which("gdb") is None:
        print("hole: gdb not found on PATH; this test needs gdb. install gdb and retry.", file=sys.stderr)
        sys.exit(2)
    scope = "/proc/sys/kernel/yama/ptrace_scope"
    if os.path.exists(scope):
        try:
            if open(scope).read().strip() not in ("0",):
                print("hole: ptrace is restricted (yama ptrace_scope != 0); cannot attach gdb. "
                      "run as root or set ptrace_scope=0.", file=sys.stderr)
                sys.exit(2)
        except Exception:
            pass


def nearest(cur, vlit):
    cur.execute("SELECT event, embedding <=> %s::vector FROM mem ORDER BY embedding <=> %s::vector LIMIT 10", (vlit, vlit))
    r = cur.fetchall()
    return (int(r[0][0]), round(float(r[0][1]), 4)) if r else (None, None)


def ser(dsn, db):
    c = C.connect(dsn, db=db, iso="SERIALIZABLE")
    cur = c.cursor()
    cur.execute("SET enable_seqscan = off")        # the window is a property of the index AM; keep the read on the index
    return c, cur


def commit(c):
    try:
        c.commit(); return True, "commit"
    except Exception as e:
        c.rollback(); return False, type(e).__name__


def scenario(dsn, db, event, a_lit, b_lit, hole, bp, workdir):
    log = []
    c1, cur1 = ser(dsn, db)
    c2, cur2 = ser(dsn, db)
    cur1.execute("SELECT pg_backend_pid()"); pid = cur1.fetchone()[0]
    log.append(f"T1 search -> nearest {nearest(cur1, a_lit)}")
    ok1 = True
    gdb = None
    go = os.path.join(workdir, f"go_{pid}")
    if hole:
        gdb = subprocess.Popen(
            ["gdb", "-p", str(pid), "-batch", "-ex", f"break {bp}", "-ex", "continue",
             "-ex", f"shell touch {go}.hit; while [ ! -f {go} ]; do sleep 0.05; done", "-ex", "detach"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.5)                                   # let gdb attach and set the breakpoint
        err = []

        def do_insert():
            try:
                cur1.execute("INSERT INTO mem(event, embedding) VALUES (%s, %s::vector)", (event, a_lit))
            except Exception as e:
                err.append(e)
        th = threading.Thread(target=do_insert); th.start()
        paused = False
        for _ in range(200):
            if os.path.exists(go + ".hit"):
                paused = True; break
            time.sleep(0.05)
        log.append(f"T1 INSERT started; paused at {bp}: {paused}")
    else:
        cur1.execute("INSERT INTO mem(event, embedding) VALUES (%s, %s::vector)", (event, a_lit))
        log.append("T1 INSERT returned (not committed)")

    ok2 = True
    try:
        log.append(f"T2 search -> nearest {nearest(cur2, b_lit)}")
        cur2.execute("INSERT INTO mem(event, embedding) VALUES (%s, %s::vector)", (event, b_lit))
        ok2, why2 = commit(c2); log.append(f"T2 {why2}")
    except Exception as e:
        c2.rollback(); ok2 = False; log.append(f"T2 ABORT {type(e).__name__}")

    if hole:
        open(go, "w").close(); th.join()
        try:
            gdb.wait(timeout=10)
        except Exception:
            gdb.kill()
        gdb.communicate()
        if err:
            c1.rollback(); ok1 = False; log.append(f"T1 INSERT failed {type(err[0]).__name__}")
    if ok1:
        ok1, why1 = commit(c1); log.append(f"T1 {why1}")

    c = C.connect(dsn, db=db); cur = c.cursor()
    cur.execute("SELECT count(*) FROM mem WHERE event = %s", (event,))
    n = int(cur.fetchone()[0])
    c.close()
    for cc in (c1, c2):
        try: cc.close()
        except Exception: pass
    for f in (go, go + ".hit"):
        if os.path.exists(f):
            try: os.remove(f)
            except Exception: pass
    return (ok1 and ok2), n, log


def run(a):
    preflight_gdb()
    bp = {"hnsw": "hnswinsert", "ivfflat": "ivfflatinsert"}[a.index]
    trials = 2 if a.smoke else a.trials
    n_base = 500 if a.smoke else 5000
    ver = C.versions(a.dsn)
    db = C.scratch_name("hole")
    base, events, pairs = C.make_vectors(n_base, 2 * trials + 2, seed=a.seed)
    C.build_mem_db(a.dsn, db, base, index=a.index, m=16, ef_construction=64)
    out = C.JsonlWriter(a.out)
    gdbs_started = []
    res = {"hole": [], "control": []}
    try:
        for i in range(trials):
            for name, ev in (("hole", 2 * i), ("control", 2 * i + 1)):
                a_lit = C.vec_lit(pairs[ev][0]); b_lit = C.vec_lit(pairs[ev][1])
                both, n, log = scenario(a.dsn, db, ev, a_lit, b_lit, name == "hole", bp, TMP)
                res[name].append((both, n))
                rec = {"index": a.index, "bp": bp, "scenario": name, "trial": i, "event": ev,
                       "both_committed": both, "rows_stored": n, "duplicate": n == 2,
                       "log": log, "versions": ver,
                       "params": {"trials": trials, "n_base": n_base, "smoke": a.smoke, "seed": a.seed}}
                out.write(rec)
        summary = {"index": a.index, "bp": bp, "scenario": "SUMMARY",
                   "hole_trials": len(res["hole"]),
                   "hole_both_committed": sum(b for b, _ in res["hole"]),
                   "hole_duplicates": sum(n == 2 for _, n in res["hole"]),
                   "control_trials": len(res["control"]),
                   "control_both_committed": sum(b for b, _ in res["control"]),
                   "control_duplicates": sum(n == 2 for _, n in res["control"]),
                   "versions": ver, "params": {"trials": trials, "n_base": n_base, "smoke": a.smoke, "seed": a.seed}}
        out.write(summary)
    finally:
        out.close()
        # safeguard: reap any lingering gdb children of this process
        try:
            subprocess.run(["pkill", "-9", "-P", str(os.getpid()), "gdb"], check=False)
        except Exception:
            pass
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--index", choices=["hnsw", "ivfflat"], default="hnsw")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
