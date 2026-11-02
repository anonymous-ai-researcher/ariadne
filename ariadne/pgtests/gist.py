"""Built-in GiST misses a conflict under SERIALIZABLE (no debugger).

Question: GiST declares ampredlocks=true, so it registers predicate locks and
should be safe under SSI. But it transfers predicate locks only on page split
(PredicateLockPageSplit), not when an insertion merely enlarges a bounding box.
Can two SERIALIZABLE transactions that each ask "is there a point equal to / near
X?", see none, and insert X, both commit, storing two rows?

Interleaving (no debugger): T1 read, T2 read, T1 insert X, T2 insert X, T1 commit,
T2 commit (and the variant r1 r2 w1 c1 w2 c2). Every serial order would have the
second reader see the first row, so at most one may commit.

Positive result: for GiST equality and range searches whose point X lies OUTSIDE
all existing bounding boxes, both transactions commit (missed conflict). The
B-tree control and GiST/SP-GiST kNN do not. On PostgreSQL 16.15 the paper reports
40 of 40 both-commit trials for GiST equality and range outside the data, 2 of 40
inside the populated region, and 0 for kNN ordering and for B-tree.

Cells: btree equality (control), gist equality, gist range (circle), gist kNN,
spgist equality, spgist kNN; each in two regions (outside all data, inside the
data region), each with --trials repeats.

Invocation:
    python -m ariadne.pgtests.gist --out results/T_pg/gist.jsonl
    python -m ariadne.pgtests.gist --trials 10
    python -m ariadne.pgtests.gist --smoke
"""
import argparse, random
import psycopg2
from . import common as C

PT = "point(%(x)s,%(y)s)"
CASES = [
    ("btree  equality x = X",            "t_btree",  "SELECT id FROM t_btree WHERE x = %(xi)s",
     "INSERT INTO t_btree(x) VALUES (%(xi)s)", "SELECT count(*) FROM t_btree WHERE x = %(xi)s"),
    ("gist   equality k ~= X",           "t_gist",   f"SELECT id FROM t_gist WHERE k ~= {PT}",
     f"INSERT INTO t_gist(k) VALUES ({PT})", f"SELECT count(*) FROM t_gist WHERE k ~= {PT}"),
    ("gist   range k <@ circle(X,r)",    "t_gist",   f"SELECT id FROM t_gist WHERE k <@ circle({PT}, %(r)s)",
     f"INSERT INTO t_gist(k) VALUES ({PT})", f"SELECT count(*) FROM t_gist WHERE k ~= {PT}"),
    ("gist   kNN nearest within r",      "t_gist",   f"SELECT id FROM (SELECT id, k <-> {PT} AS d FROM t_gist ORDER BY k <-> {PT} LIMIT 1) s WHERE d < %(r)s",
     f"INSERT INTO t_gist(k) VALUES ({PT})", f"SELECT count(*) FROM t_gist WHERE k ~= {PT}"),
    ("spgist equality k ~= X",           "t_spgist", f"SELECT id FROM t_spgist WHERE k ~= {PT}",
     f"INSERT INTO t_spgist(k) VALUES ({PT})", f"SELECT count(*) FROM t_spgist WHERE k ~= {PT}"),
    ("spgist kNN nearest within r",      "t_spgist", f"SELECT id FROM (SELECT id, k <-> {PT} AS d FROM t_spgist ORDER BY k <-> {PT} LIMIT 1) s WHERE d < %(r)s",
     f"INSERT INTO t_spgist(k) VALUES ({PT})", f"SELECT count(*) FROM t_spgist WHERE k ~= {PT}"),
]
ORDERS = {"r1 r2 w1 w2 c1 c2": ["r1", "r2", "w1", "w2", "c1", "c2"],
          "r1 r2 w1 c1 w2 c2": ["r1", "r2", "w1", "c1", "w2", "c2"]}


def trial(dsn, db, q, ins, cnt, p, order):
    cs = []
    for _ in range(2):
        c = C.connect(dsn, db=db, iso="SERIALIZABLE")
        cur = c.cursor(); cur.execute("SET enable_seqscan = off"); c.commit()
        cs.append((c, cur))
    seen, ok, plan, locks = [], [True, True], None, None
    for step in order:
        kind, i = step[0], int(step[1]) - 1
        if not ok[i]:
            continue
        c, cur = cs[i]
        try:
            if kind == "r":
                if plan is None:
                    cur.execute("EXPLAIN " + q, p)
                    plan = " / ".join(r[0].split("(cost")[0].strip(" ->") for r in cur.fetchall() if "Scan" in r[0])
                cur.execute(q, p); seen.append(len(cur.fetchall()))
                if locks is None:
                    cur.execute("SELECT locktype || ':' || relation::regclass::text, count(*) FROM pg_locks "
                                "WHERE mode='SIReadLock' AND pid=pg_backend_pid() GROUP BY 1 ORDER BY 1")
                    locks = [f"{a}={b}" for a, b in cur.fetchall()]
            elif kind == "w":
                cur.execute(ins, p)
            else:
                c.commit()
        except psycopg2.errors.SerializationFailure:
            c.rollback(); ok[i] = False
    for c, _ in cs:
        c.rollback(); c.close()
    n = int(C.admin(dsn, cnt % p, db=db)[0][0])
    return seen, ok, n, plan, locks


def run(a):
    trials = 3 if a.smoke else a.trials
    nrows = 5000 if a.smoke else 50000
    ver = C.versions(a.dsn)
    db = C.scratch_name("gist")
    C.make_scratch_db(a.dsn, db)
    C.admin(a.dsn, "CREATE EXTENSION IF NOT EXISTS btree_gist", db=db)
    rng = random.Random(a.seed)
    for am in ("gist", "spgist"):
        C.admin(a.dsn, f"CREATE TABLE t_{am} (id serial PRIMARY KEY, k point)", db=db)
        C.admin(a.dsn, f"INSERT INTO t_{am}(k) SELECT point(random()*1000, random()*1000) FROM generate_series(1, {nrows})", db=db)
        C.admin(a.dsn, f"CREATE INDEX t_{am}_k ON t_{am} USING {am} (k)", db=db)
        C.admin(a.dsn, f"ANALYZE t_{am}", db=db)
    C.admin(a.dsn, "CREATE TABLE t_btree (id serial PRIMARY KEY, x int)", db=db)
    C.admin(a.dsn, f"INSERT INTO t_btree(x) SELECT g * 10 FROM generate_series(1, {nrows}) g", db=db)
    C.admin(a.dsn, "CREATE INDEX t_btree_x ON t_btree (x)", db=db)
    C.admin(a.dsn, "ANALYZE t_btree", db=db)
    gist_pages = C.admin(a.dsn, "SELECT relpages FROM pg_class WHERE relname='t_gist_k'", db=db)[0][0]

    out = C.JsonlWriter(a.out)
    cases = CASES if not a.smoke else [c for c in CASES if "kNN" not in c[0] or "gist   kNN" in c[0]]
    orders = ORDERS if not a.smoke else {"r1 r2 w1 w2 c1 c2": ORDERS["r1 r2 w1 w2 c1 c2"]}
    slot = 0
    try:
        for region in ("outside all data", "inside data region"):
            for name, tbl, q, ins, cnt in cases:
                for oname, order in orders.items():
                    both = empty = 0
                    plan = locks = None
                    for _ in range(trials):
                        slot += 1
                        if region.startswith("outside"):
                            p = {"x": 100000 + 1000 * slot, "y": 100000 + 1000 * slot, "r": 5, "xi": 10000000 + 1000 * slot}
                        else:
                            p = {"x": round(rng.uniform(100, 900), 4), "y": round(rng.uniform(100, 900), 4),
                                 "r": 0.3, "xi": 10 * rng.randrange(1000, 49000) + 5}
                        seen, ok, n, plan, locks = trial(a.dsn, db, q, ins, cnt, p, order)
                        if seen == [0, 0]:
                            empty += 1
                            both += (ok[0] and ok[1] and n == 2)
                    out.write({"region": region, "case": name.strip(), "order": oname, "trials": trials,
                               "both_saw_nothing": empty, "both_committed_duplicate": both,
                               "plan": plan, "siread": locks, "gist_index_pages": int(gist_pages),
                               "versions": ver, "params": {"trials": trials, "nrows": nrows, "smoke": a.smoke, "seed": a.seed}})
    finally:
        out.close()
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
