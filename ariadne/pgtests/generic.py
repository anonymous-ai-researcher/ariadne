"""Does the check-then-insert window exist on built-in index AMs too? (debugger)

Question: PostgreSQL runs CheckForSerializableConflictIn before aminsert for any
AM. For AMs whose insert physically adds the entry only after work that a
concurrent reader can slip past (or whose predicate locks are only relation level),
the same window as pgvector's HNSW should appear. Does it, and for which AMs?

Interleaving per trial (all at SERIALIZABLE, seqscan off):
    T1: SELECT WHERE k = X (0 rows, index scan); INSERT X  [paused at <am>insert entry]
    T2: SELECT WHERE k = X (0 rows); INSERT X; COMMIT
    T1: resume; COMMIT

Positive result: for an AM with the window, both T1 and T2 commit and 2 rows are
stored (a phantom under SERIALIZABLE). B-tree is the negative control (T1 aborts,
1 row). Bloom is expected to abort T1 (its scan takes tuple locks).

Each AM does a warm-up insert first so its shared library is loaded before gdb
attaches (attaching during the library load can crash the backend). If the
breakpoint is not hit within the timeout the gdb process is killed (kill-on-miss)
so a missed pause never leaves a backend stopped.

Needs gdb and ptrace permission (exits 2 otherwise). gdb processes are cleaned up.

Invocation:
    python -m ariadne.pgtests.generic --out results/T_pg/generic.jsonl
    python -m ariadne.pgtests.generic --ams spgist,brin,btree --trials 3
    python -m ariadne.pgtests.generic --smoke
"""
import argparse, os, subprocess, sys, threading, time
from . import common as C
from .hole import preflight_gdb, TMP

# am -> (coltype, USING clause, insert function breakpoint, value format, operator)
CASES = {
    "spgist": ("text",  "USING spgist (k)", "spginsert", "'key-%d'",       "="),
    "brin":   ("int",   "USING brin (k)",   "brininsert", "%d",            "="),
    "bloom":  ("int",   "USING bloom (k)",  "blinsert",  "%d",             "="),
    "hash":   ("int",   "USING hash (k)",   "hashinsert", "%d",            "="),
    "gin":    ("int[]", "USING gin (k)",    "gininsert", "ARRAY[%d]",      "@>"),
    "gist":   ("point", "USING gist (k)",   "gistinsert", "point(%d,%d)",  "~="),
    "btree":  ("int",   "USING btree (k)",  "btinsert",  "%d",             "="),
}
DEFAULT = ["spgist", "brin", "hash", "gist", "gin", "btree"]      # bloom added if the extension is present


def fmt_val(fmt, typ, x):
    return fmt % ((x, x) if typ == "point" else x)


def one_trial(dsn, db, am, trial, workdir):
    typ, using, bp, fmt, op = CASES[am]
    t = f"t_{am}"
    x = 1000000 + trial
    val = fmt_val(fmt, typ, x)
    warm = fmt_val(fmt, typ, 5)

    conns = []
    for _ in range(2):
        c = C.connect(dsn, db=db, iso="SERIALIZABLE")
        cur = c.cursor(); cur.execute("SET enable_seqscan = off")
        conns.append((c, cur))
    (c1, cur1), (c2, cur2) = conns

    cur1.execute(f"INSERT INTO {t}(k) VALUES ({warm})"); c1.rollback()   # load the AM library before attaching
    cur1.execute("SET enable_seqscan = off")
    cur1.execute("SELECT pg_backend_pid()"); pid = cur1.fetchone()[0]
    cur1.execute(f"EXPLAIN SELECT id FROM {t} WHERE k {op} {val}")
    plan = cur1.fetchall()[0][0].split("(cost")[0].strip()
    cur1.execute(f"SELECT id FROM {t} WHERE k {op} {val}"); n1 = len(cur1.fetchall())
    cur1.execute("SELECT locktype, relation::regclass::text FROM pg_locks WHERE mode='SIReadLock' AND pid=pg_backend_pid()")
    lk = [f"{a}:{b}" for a, b in cur1.fetchall()]

    go = os.path.join(workdir, f"ggo_{pid}")
    gdb = subprocess.Popen(
        ["gdb", "-p", str(pid), "-batch", "-ex", f"break {bp}", "-ex", "continue",
         "-ex", f"shell touch {go}.hit; while [ ! -f {go} ]; do sleep 0.05; done", "-ex", "detach"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    err = []

    def ins():
        try:
            cur1.execute(f"INSERT INTO {t}(k) VALUES ({val})")
        except Exception as e:
            err.append(e)
    th = threading.Thread(target=ins); th.start()
    hit = False
    for _ in range(200):
        if os.path.exists(go + ".hit"):
            hit = True; break
        time.sleep(0.05)

    ok2 = True
    try:
        cur2.execute(f"SELECT id FROM {t} WHERE k {op} {val}"); cur2.fetchall()
        cur2.execute(f"INSERT INTO {t}(k) VALUES ({val})"); c2.commit()
    except Exception:
        c2.rollback(); ok2 = False

    open(go, "w").close()
    if not hit:                                   # kill-on-miss: never leave a stopped backend
        gdb.kill()
    th.join()
    try:
        gdb.wait(timeout=10)
    except Exception:
        gdb.kill()
    gdb.communicate()

    ok1 = not err
    if ok1:
        try:
            c1.commit()
        except Exception:
            c1.rollback(); ok1 = False
    else:
        c1.rollback()
    n = int(C.admin(dsn, f"SELECT count(*) FROM {t} WHERE k {op} {val}", db=db)[0][0])
    for cc in (c1, c2):
        try: cc.close()
        except Exception: pass
    for f in (go, go + ".hit"):
        if os.path.exists(f):
            try: os.remove(f)
            except Exception: pass
    return {"paused": hit, "t1_rows_seen": n1, "t1_committed": ok1, "t2_committed": ok2,
            "rows_stored": n, "both_committed": ok1 and ok2, "plan": plan, "siread": lk}


def run(a):
    preflight_gdb()
    trials = 2 if a.smoke else a.trials
    ver = C.versions(a.dsn)
    db = C.scratch_name("generic")
    C.make_scratch_db(a.dsn, db)
    ams = [x for x in (a.ams.split(",") if a.ams else DEFAULT) if x]
    have_bloom = bool(C.admin(a.dsn, "SELECT 1 FROM pg_available_extensions WHERE name='bloom'", db=db))
    if not a.ams and have_bloom:
        ams = ["spgist", "brin", "bloom", "hash", "gist", "gin", "btree"]
    if "bloom" in ams and have_bloom:
        C.admin(a.dsn, "CREATE EXTENSION IF NOT EXISTS bloom", db=db)
    out = C.JsonlWriter(a.out)
    try:
        for am in ams:
            if am not in CASES:
                out.write({"am": am, "error": "unknown am", "versions": ver}); continue
            if am == "bloom" and not have_bloom:
                out.write({"am": am, "error": "bloom extension not available", "versions": ver}); continue
            typ, using, bp, fmt, op = CASES[am]
            t = f"t_{am}"
            C.admin(a.dsn, f"DROP TABLE IF EXISTS {t}", db=db)
            C.admin(a.dsn, f"CREATE TABLE {t} (id serial PRIMARY KEY, k {typ})", db=db)
            gen = {"text": "'key-' || g", "point": "point(g, g)", "int[]": "ARRAY[g]"}.get(typ, "g")
            nrows = 3000 if a.smoke else 20000
            C.admin(a.dsn, f"INSERT INTO {t}(k) SELECT {gen} FROM generate_series(1, {nrows}) g", db=db)
            C.admin(a.dsn, f"CREATE INDEX {t}_k ON {t} {using}", db=db)
            C.admin(a.dsn, f"ANALYZE {t}", db=db)
            trs = []
            for i in range(trials):
                try:
                    trs.append(one_trial(a.dsn, db, am, i, TMP))
                except Exception as e:
                    trs.append({"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"})
            both = sum(1 for r in trs if r.get("both_committed"))
            dups = sum(1 for r in trs if r.get("rows_stored") == 2)
            out.write({"am": am, "bp": bp, "trials": trials, "both_committed": both, "duplicates": dups,
                       "detail": trs, "versions": ver,
                       "params": {"trials": trials, "nrows": nrows, "smoke": a.smoke}})
    finally:
        out.close()
        try:
            subprocess.run(["pkill", "-9", "-P", str(os.getpid()), "gdb"], check=False)
        except Exception:
            pass
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--ams", default="", help="comma list; default spgist,brin,(bloom),hash,gist,gin,btree")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
