"""What does SERIALIZABLE actually lock for an index read? Inspect pg_locks.

Question: when a transaction at SERIALIZABLE isolation performs a read that uses a
given index access method, which predicate (SIREAD) locks does it hold, and at
what granularity (relation, page, tuple)?

Positive result (the coarseness finding): a pgvector k-NN read over HNSW holds a
single relation level SIREAD lock covering the whole index (ampredlocks=false),
whereas a B-tree key lookup holds only page or tuple level locks. A relation
level lock means every concurrent insertion into that index conflicts.

For each configured index type we build a small table with the matching column and
operator, run a SERIALIZABLE read that the planner serves from the index, and
report the SIREAD locks grouped by locktype, plus amname and ampredlocks.

Invocation:
    python -m ariadne.pgtests.locks --out results/T_pg/locks.jsonl
    python -m ariadne.pgtests.locks --smoke
    python -m ariadne.pgtests.locks --indexes hnsw,btree,gist
"""
import argparse, numpy as np
from . import common as C

# name -> (coltype, create-index USING clause, build generator SQL, read query, needs vector base)
# {V} is substituted with a vector literal for the vector cases.
SPECS = {
    "hnsw":    ("vector", "USING hnsw (k vector_cosine_ops)",
                None, "ORDER BY k <=> '{V}'::vector LIMIT 10", True),
    "ivfflat": ("vector", "USING ivfflat (k vector_cosine_ops) WITH (lists=50)",
                None, "ORDER BY k <=> '{V}'::vector LIMIT 10", True),
    "btree":   ("bigint", "USING btree (k)",
                "(random()*4611686018427387904)::bigint", "WHERE k = 12345 LIMIT 1", False),
    "hash":    ("bigint", "USING hash (k)",
                "(random()*4611686018427387904)::bigint", "WHERE k = 12345 LIMIT 1", False),
    "gist":    ("point", "USING gist (k)",
                "point(random()*1000, random()*1000)", "WHERE k ~= point(12345,12345) LIMIT 1", False),
    "spgist":  ("point", "USING spgist (k)",
                "point(random()*1000, random()*1000)", "WHERE k ~= point(12345,12345) LIMIT 1", False),
    "brin":    ("int", "USING brin (k)",
                "(g)::int", "WHERE k = 12345 LIMIT 1", False),
    "gin":     ("int[]", "USING gin (k)",
                "ARRAY[g % 997, (g*7) % 997]", "WHERE k @> ARRAY[12345] LIMIT 1", False),
}
DEFAULT = ["hnsw", "ivfflat", "btree", "gist", "spgist", "brin", "hash", "gin"]


def am_props(dsn, db, am):
    rows = C.admin(dsn, f"SELECT amname, amtype FROM pg_am WHERE amname='{am}'", db=db)
    if not rows:
        return {"amname": am, "exists": False}
    out = {"amname": am, "exists": True}
    # ampredlocks is set in each AM's amroutine in C and is not exposed in any catalog,
    # so we label it from the known PostgreSQL / pgvector source values (observe the actual
    # SIREAD granularity in siread_locks). ampredlocks=true: btree, hash, gist, gin.
    # ampredlocks=false: spgist, brin, bloom, and pgvector's hnsw and ivfflat.
    predlock_true = {"btree", "hash", "gist", "gin"}
    out["ampredlocks"] = (am in predlock_true) if am not in ("hnsw", "ivfflat") else False
    return out


def one_index(dsn, db, name, dim, vlit):
    coltype, using, gen, where, needs_vec = SPECS[name]
    tbl = f"t_{name}"
    typ = f"vector({dim})" if coltype == "vector" else coltype
    C.admin(dsn, f"DROP TABLE IF EXISTS {tbl}", db=db)
    C.admin(dsn, f"CREATE TABLE {tbl} (id bigserial PRIMARY KEY, k {typ} NOT NULL)", db=db)
    n = 3000 if coltype == "vector" else 20000
    if coltype == "vector":
        base, _, _ = C.make_vectors(n, 1, dim=dim, seed=7)
        c = C.connect(dsn, db=db)
        cur = c.cursor()
        for i in range(0, n, 1000):
            args = b",".join(cur.mogrify("(%s::vector)", (C.vec_lit(v),)) for v in base[i:i + 1000])
            cur.execute(b"INSERT INTO " + tbl.encode() + b"(k) VALUES " + args)
        c.commit(); c.close()
    else:
        C.admin(dsn, f"INSERT INTO {tbl}(k) SELECT {gen} FROM generate_series(1, {n}) g", db=db)
    C.admin(dsn, f"CREATE INDEX {tbl}_k ON {tbl} {using}", db=db)
    C.admin(dsn, f"ANALYZE {tbl}", db=db)

    q = where.replace("{V}", vlit)
    rel = C.admin(dsn, f"SELECT relpages FROM pg_class WHERE relname='{tbl}_k'", db=db)
    idx_pages = int(rel[0][0]) if rel else None

    c = C.connect(dsn, db=db, iso="SERIALIZABLE")
    cur = c.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"EXPLAIN SELECT id FROM {tbl} {q}")
    plan = " / ".join(r[0].strip().split("(cost")[0].strip() for r in cur.fetchall() if "Scan" in r[0])
    used_index = "Index" in plan or "Bitmap" in plan
    cur.execute(f"SELECT id FROM {tbl} {q}")
    cur.fetchall()
    cur.execute("""SELECT locktype, count(*) FROM pg_locks
                   WHERE mode='SIReadLock' AND pid=pg_backend_pid() GROUP BY 1 ORDER BY 1""")
    locks = {lt: int(n) for lt, n in cur.fetchall()}
    c.rollback(); c.close()
    C.admin(dsn, f"DROP TABLE {tbl}", db=db)
    return {"index": name, "plan": plan, "used_index": used_index, "index_pages": idx_pages,
            "siread_locks": locks, "siread_total": sum(locks.values()), **am_props(dsn, db, name)}


def run(a):
    dim = 32 if a.smoke else C.DIM
    names = [x for x in (a.indexes.split(",") if a.indexes else DEFAULT) if x]
    if a.smoke:
        names = [x for x in names if x in ("hnsw", "btree", "gist")] or ["hnsw", "btree"]
    ver = C.versions(a.dsn)
    db = C.scratch_name("locks")
    C.make_scratch_db(a.dsn, db)
    if any(SPECS.get(x, (None,))[0] == "vector" for x in names):
        C.admin(a.dsn, "CREATE EXTENSION IF NOT EXISTS vector", db=db)
    base, _, pairs = C.make_vectors(10, 1, dim=dim, seed=3)
    vlit = C.vec_lit(pairs[0][0])
    out = C.JsonlWriter(a.out)
    try:
        for name in names:
            if name not in SPECS:
                out.write({"index": name, "error": "unknown index type", "versions": ver})
                continue
            try:
                rec = one_index(a.dsn, db, name, dim, vlit)
            except Exception as e:
                rec = {"index": name, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
            rec["versions"] = ver
            rec["params"] = {"dim": dim, "smoke": a.smoke}
            out.write(rec)
    finally:
        out.close()
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--indexes", default="", help="comma list; default all known index types")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
