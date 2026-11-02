"""Coarseness of SERIALIZABLE over pgvector: two transactions whose queries are almost antipodal abort each other.

Question: Section 3.2 says that one similarity read takes a predicate lock on the whole index, so that two
SERIALIZABLE transactions that read and then insert conflict however far apart their queries are.  Do they?

Interleaving per trial (both at SERIALIZABLE, seqscan off, a fresh pair of vectors per trial):
    T1: top-10 read for q            (index scan; no neighbour within TAU)
    T2: top-10 read for q' ~ -q      (cosine distance between q and q' close to 2: almost antipodal)
    T1: INSERT q        T2: INSERT q'        T1: COMMIT        T2: COMMIT
Neither read can return the other's vector (they are at the two ends of the sphere, and the base rows are in between),
so any serial order commits both.  With a relation-level SIREAD lock on the index, each insertion conflicts with the
other transaction's read (an rw-dependency in each direction), which is a dangerous structure: one of the two must abort.

Control (--index btree, run by default after hnsw): the same interleaving on a scalar key with a B-tree, keys far apart
(different leaf pages): page-level predicate locks do not overlap, so both commit.

Positive result: hnsw: exactly one of the two transactions aborts in every trial (one_aborted == trials); btree control:
both commit in every trial (both_committed == trials).

Output: one JSON line per index with the counts, the SIREAD locks of T1's read (locktype:relation), the plan, the
cosine distance between the two queries, and the per-trial outcomes.

Invocation:
    python -m ariadne.pgtests.antipodal --trials 20 --out results/A6_ssi_soundness/antipodal.jsonl
    python -m ariadne.pgtests.antipodal --smoke
"""
import argparse
import numpy as np
import psycopg2
from . import common as C

ORDER = ["r1", "r2", "w1", "w2", "c1", "c2"]
NOISE = 0.05          # the two queries are q + noise and -q + noise: cosine distance about 2 - O(NOISE^2)


def _unit(v):
    return (v / max(float(np.linalg.norm(v)), 1e-12)).astype("float32")


def _locks(cur):
    cur.execute("SELECT locktype || ':' || coalesce(relation::regclass::text, '?'), count(*) FROM pg_locks "
                "WHERE mode='SIReadLock' AND pid=pg_backend_pid() GROUP BY 1 ORDER BY 1")
    return ["%s=%d" % (a, b) for a, b in cur.fetchall()]


def trial_vector(dsn, db, q1, q2, ev):
    """One interleaving on the vector table; returns the per-trial record."""
    cs = []
    for _ in range(2):
        c = C.connect(dsn, db=db, iso="SERIALIZABLE")
        cur = c.cursor(); cur.execute("SET enable_seqscan = off"); c.commit()
        cs.append((c, cur))
    lits = [C.vec_lit(q1), C.vec_lit(q2)]
    ok, step_failed, nearest, locks = [True, True], [None, None], [None, None], None
    for step in ORDER:
        kind, i = step[0], int(step[1]) - 1
        if not ok[i]:
            continue
        c, cur = cs[i]
        try:
            if kind == "r":
                cur.execute("SELECT embedding <=> %s::vector FROM mem ORDER BY embedding <=> %s::vector LIMIT 10", (lits[i], lits[i]))
                r = cur.fetchall(); nearest[i] = round(float(r[0][0]), 4) if r else None
                if i == 0:
                    locks = _locks(cur)
            elif kind == "w":
                if nearest[i] is None or nearest[i] >= C.TAU:          # check-then-insert: no neighbour within TAU
                    cur.execute("INSERT INTO mem(event, embedding) VALUES (%s, %s::vector)", (ev + i, lits[i]))
            else:
                c.commit()
        except (psycopg2.errors.SerializationFailure, psycopg2.errors.DeadlockDetected) as e:
            c.rollback(); ok[i] = False; step_failed[i] = "%s at %s" % (type(e).__name__, step)
    for c, _ in cs:
        try:
            c.rollback(); c.close()
        except Exception:
            pass
    stored = int(C.admin(dsn, "SELECT count(*) FROM mem WHERE event IN (%d, %d)" % (ev, ev + 1), db=db)[0][0])
    return {"t1_committed": ok[0], "t2_committed": ok[1], "failed": step_failed, "nearest": nearest, "rows_stored": stored,
            "siread_t1": locks}


def trial_btree(dsn, db, x1, x2):
    cs = []
    for _ in range(2):
        c = C.connect(dsn, db=db, iso="SERIALIZABLE")
        cur = c.cursor(); cur.execute("SET enable_seqscan = off"); c.commit()
        cs.append((c, cur))
    xs = [x1, x2]
    ok, step_failed, seen, locks = [True, True], [None, None], [None, None], None
    for step in ORDER:
        kind, i = step[0], int(step[1]) - 1
        if not ok[i]:
            continue
        c, cur = cs[i]
        try:
            if kind == "r":
                cur.execute("SELECT id FROM kmem WHERE key = %s LIMIT 1", (xs[i],)); seen[i] = len(cur.fetchall())
                if i == 0:
                    locks = _locks(cur)
            elif kind == "w":
                if not seen[i]:
                    cur.execute("INSERT INTO kmem(key) VALUES (%s)", (xs[i],))
            else:
                c.commit()
        except (psycopg2.errors.SerializationFailure, psycopg2.errors.DeadlockDetected) as e:
            c.rollback(); ok[i] = False; step_failed[i] = "%s at %s" % (type(e).__name__, step)
    for c, _ in cs:
        try:
            c.rollback(); c.close()
        except Exception:
            pass
    stored = int(C.admin(dsn, "SELECT count(*) FROM kmem WHERE key IN (%d, %d)" % (x1, x2), db=db)[0][0])
    return {"t1_committed": ok[0], "t2_committed": ok[1], "failed": step_failed, "rows_stored": stored, "siread_t1": locks}


def summarize(index, trs, extra, ver, params):
    one = sum(1 for t in trs if t.get("t1_committed") != t.get("t2_committed"))
    both = sum(1 for t in trs if t.get("t1_committed") and t.get("t2_committed"))
    none_ = sum(1 for t in trs if "error" not in t and not t.get("t1_committed") and not t.get("t2_committed"))
    errors = sum(1 for t in trs if "error" in t)
    locks = sorted({x for t in trs for x in (t.get("siread_t1") or [])})
    return {"test": "antipodal", "index": index, "order": " ".join(ORDER), "trials": len(trs), "one_aborted": one,
            "both_committed": both, "both_aborted": none_, "errors": errors, "siread_t1": locks, **extra,
            "detail": trs, "versions": ver, "params": params}


def run(a):
    trials = 3 if a.smoke else a.trials
    n_base = 500 if a.smoke else a.base
    ver = C.versions(a.dsn)
    out = C.JsonlWriter(a.out)
    rng = np.random.default_rng([a.seed, 7919])           # a stream of its own: make_vectors(seed) draws the base rows
    params = {"trials": trials, "n_base": n_base, "dim": C.DIM, "noise": NOISE, "seed": a.seed, "smoke": a.smoke}
    idx = [x for x in a.indexes.split(",") if x]
    try:
        if "hnsw" in idx:
            db = C.scratch_name("antipodal")
            try:
                base, _, _ = C.make_vectors(n_base, 1, seed=a.seed)
                C.build_mem_db(a.dsn, db, base, index="hnsw", m=16, ef_construction=64)
                cur = C.connect(a.dsn, db=db, autocommit=True).cursor()
                cur.execute("SET enable_seqscan = off")
                lit = C.vec_lit(base[0])
                cur.execute("EXPLAIN SELECT embedding <=> %s::vector FROM mem ORDER BY embedding <=> %s::vector LIMIT 10", (lit, lit))
                plan = " / ".join(r[0].strip().split("(cost")[0].strip(" ->") for r in cur.fetchall() if "Scan" in r[0])
                cur.connection.close()
                trs, dists = [], []
                for i in range(trials):
                    q = rng.normal(size=C.DIM).astype("float32")
                    q1 = _unit(_unit(q) + NOISE * rng.normal(size=C.DIM).astype("float32") / np.sqrt(C.DIM))
                    q2 = _unit(-_unit(q) + NOISE * rng.normal(size=C.DIM).astype("float32") / np.sqrt(C.DIM))
                    dists.append(float(1.0 - q1 @ q2))
                    try:
                        trs.append(trial_vector(a.dsn, db, q1, q2, 1000 + 2 * i))
                    except Exception as e:
                        trs.append({"error": "%s: %s" % (type(e).__name__, str(e).splitlines()[0])})
                out.write(summarize("hnsw", trs, {"plan": plan, "query_cos_dist_min": round(min(dists), 4),
                                                  "query_cos_dist_max": round(max(dists), 4)}, ver, params))
            finally:
                C.drop_scratch_db(a.dsn, db)
        if "btree" in idx:
            db = C.scratch_name("antipodal_bt")
            try:
                C.make_scratch_db(a.dsn, db)
                C.admin(a.dsn, "CREATE TABLE kmem (id bigserial PRIMARY KEY, key bigint NOT NULL)", db=db)
                C.admin(a.dsn, "INSERT INTO kmem(key) SELECT (random() * 4611686018427387904)::bigint FROM generate_series(1, %d)"
                        % (2000 if a.smoke else 20000), db=db)
                C.admin(a.dsn, "CREATE INDEX kmem_key ON kmem (key)", db=db)
                C.admin(a.dsn, "ANALYZE kmem", db=db)
                trs = []
                for i in range(trials):
                    x1 = int(rng.integers(1, 2 ** 40)) * 2 + 1                  # odd keys near the low end ...
                    x2 = 4611686018427387904 - x1                               # ... and their mirror at the high end
                    try:
                        trs.append(trial_btree(a.dsn, db, x1, x2))
                    except Exception as e:
                        trs.append({"error": "%s: %s" % (type(e).__name__, str(e).splitlines()[0])})
                out.write(summarize("btree", trs, {}, ver, params))
            finally:
                C.drop_scratch_db(a.dsn, db)
    finally:
        out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--base", type=int, default=5000, help="rows in the vector table")
    ap.add_argument("--indexes", default="hnsw,btree", help="comma list of hnsw, btree")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
