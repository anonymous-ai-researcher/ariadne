"""Natural (no debugger) occurrence of duplicates under SERIALIZABLE on pgvector.

Question: without any debugger, how often do two SERIALIZABLE clients that each run
[k-NN search; insert if no neighbour within TAU; commit] both commit a duplicate,
and how does that rate respond to the width of the insert window?

Each trial gives two (or --threads) clients near-duplicate vectors of one new fact.
The second client starts a random 0..6 ms after the first. Any serial order stores
exactly one row; an extra row is a race duplicate. Serialization failures are
retried. --mode plain runs SERIALIZABLE with no extra coordination (the anomaly
path); --mode advisory wraps the check-then-insert in pg_advisory_xact_lock (the
control, expected 0 duplicates). --index none drops the vector index so the k-NN
read falls back to a seqscan (whole-relation SIREAD, also a control).

Dose-response: a slower insertion (larger m, ef_construction) widens the window
and raises the duplicate rate. --grid sweeps (m, ef_construction) over
(8,32),(16,64),(32,200),(32,400),(64,400) and writes one line per point.

After each point, fresh_search re-runs the clients' top-10 query (index scan) in the quiescent final state with the
vector of every copy of every fact stored twice: found_both = dup_events means that both rows are reachable through the
index, so the duplicates are missed conflicts and not recall errors.

Positive result: --mode plain reports a nonzero duplicate rate that grows with
m/ef_construction, while --mode advisory, --index none and the recheck patch
(ariadne.pgtests.patched) report 0. On PostgreSQL 16.15 / pgvector 0.8.7 the paper reports 9.4% of tightly
timed pairs, up to 61.2% at a mean insertion time of 7.9 ms, and 0 of 30,000 pairs with the patch.

Invocation:
    python -m ariadne.pgtests.hammer --pairs 600 --out results/T_pg/hammer.jsonl
    python -m ariadne.pgtests.hammer --grid --pairs 600 --out results/T_pg/hammer_grid.jsonl
    python -m ariadne.pgtests.hammer --mode advisory --pairs 600
    python -m ariadne.pgtests.hammer --smoke
"""
import argparse, threading, time
import numpy as np
import psycopg2
from . import common as C

GRID = [(8, 32), (16, 64), (32, 200), (32, 400), (64, 400)]
DMAX = 0.006        # second client starts 0..DMAX s after the first


def rebuild_index(dsn, db, m, efc, index):
    C.admin(dsn, "DROP INDEX IF EXISTS mem_vec", db=db)
    if index == "none":
        return
    ops = "vector_cosine_ops"
    if index == "hnsw":
        C.admin(dsn, f"CREATE INDEX mem_vec ON mem USING hnsw (embedding {ops}) "
                     f"WITH (m = {int(m)}, ef_construction = {int(efc)})", db=db)
    elif index == "ivfflat":
        C.admin(dsn, f"CREATE INDEX mem_vec ON mem USING ivfflat (embedding {ops}) WITH (lists = 100)", db=db)
    C.admin(dsn, "ANALYZE mem", db=db)


def run_point(dsn, db, pairs, mode, index, m, efc, seed):
    E = psycopg2.errors
    rng = np.random.default_rng(seed)
    n = len(pairs)
    threads = len(pairs[0])
    conns = [C.connect(dsn, db=db) for _ in range(threads)]
    for c in conns:
        c.set_session(isolation_level="SERIALIZABLE", autocommit=False)
        c.cursor().execute("SET enable_seqscan = off"); c.commit()
    stat = {"aborts": 0, "ins_ms": []}

    def client(conn, ev, vec, delay):
        cur = conn.cursor()
        time.sleep(delay)
        lit = C.vec_lit(vec)
        while True:
            try:
                if mode == "advisory":
                    cur.execute("SELECT pg_advisory_xact_lock(4242)")
                cur.execute("SELECT embedding <=> %s::vector FROM mem ORDER BY embedding <=> %s::vector LIMIT 10", (lit, lit))
                r = cur.fetchall()
                if not (r and float(r[0][0]) < C.TAU):
                    t0 = time.time()
                    cur.execute("INSERT INTO mem(event, embedding) VALUES (%s, %s::vector)", (ev, lit))
                    stat["ins_ms"].append(1000 * (time.time() - t0))
                conn.commit()
                return
            except (E.SerializationFailure, E.DeadlockDetected):
                conn.rollback(); stat["aborts"] += 1

    t_start = time.time()
    for i in range(n):
        ths = [threading.Thread(target=client, args=(conns[j], i, pairs[i][j], 0 if j == 0 else rng.uniform(0, DMAX)))
               for j in range(threads)]
        for t in ths: t.start()
        for t in ths: t.join()
    for c in conns:
        c.close()
    rows = C.admin(dsn, "SELECT event, count(*) FROM mem WHERE event >= 0 GROUP BY event", db=db)
    dup_events = sum(1 for _, k in rows if k > 1)
    dup_rows = sum(int(k) - 1 for _, k in rows if k > 1)
    p, lo, hi = C.wilson(dup_events, n)
    rec = {"mode": mode, "index": index, "m": m, "ef_construction": efc, "pairs": n, "threads": threads,
           "events_with_rows": len(rows), "events_with_duplicate": dup_events, "duplicate_rows": dup_rows,
           "dup_rate": p, "dup_rate_lo95": lo, "dup_rate_hi95": hi, "aborts": stat["aborts"],
           "insert_ms_mean": round(float(np.mean(stat["ins_ms"])), 3) if stat["ins_ms"] else None,
           "wall_s": round(time.time() - t_start, 2)}
    rec["fresh_search"] = fresh_search(dsn, db, pairs, {int(e): int(k) for e, k in rows})
    return rec


def fresh_search(dsn, db, pairs, count):
    """After the run, in the quiescent final state: search again (same index, same top-10 query as the clients, seqscan
    off) with the vector of every copy of every fact stored twice, and count the facts for which every such search
    returns all rows of the fact within TAU.  This tells a missed conflict (the rows are reachable, the reader simply did
    not wait for the writer) from a recall error of the index (a row that a search cannot find).  As a baseline, the same
    search with the first copy's vector for every fact with a row: how often does a search return all of its rows?"""
    c = C.connect(dsn, db=db, autocommit=True)
    try:
        cur = c.cursor()
        cur.execute("SET enable_seqscan = off")
        q = "SELECT event, embedding <=> %s::vector FROM mem ORDER BY embedding <=> %s::vector LIMIT 10"
        lit0 = C.vec_lit(pairs[0][0])
        cur.execute("EXPLAIN " + q, (lit0, lit0))
        plan = " / ".join(r[0].strip().split("(cost")[0].strip(" ->") for r in cur.fetchall() if "Scan" in r[0])

        def found_all(ev, vec):
            lit = C.vec_lit(vec)
            cur.execute(q, (lit, lit))
            hits = sum(1 for e, d in cur.fetchall() if int(e) == ev and float(d) < C.TAU)
            return hits >= count.get(ev, 0)

        dup = [ev for ev, k in count.items() if k > 1]
        both = sum(1 for ev in dup if all(found_all(ev, v) for v in pairs[ev]))
        base = [ev for ev, k in count.items() if k >= 1]
        base_ok = sum(1 for ev in base if found_all(ev, pairs[ev][0]))
        return {"dup_events": len(dup), "found_both": both, "facts_searched": len(base), "facts_all_rows_found": base_ok,
                "plan": plan, "k": 10, "tau": C.TAU}
    finally:
        c.close()


def run(a):
    pairs_n = 20 if a.smoke else a.pairs
    n_base = 800 if a.smoke else a.base
    dim = a.dim
    ver = C.versions(a.dsn)
    points = ([(8, 32), (16, 64)] if a.smoke else GRID) if a.grid else [(a.m, a.ef_construction)]
    db = C.scratch_name("hammer")
    out = C.JsonlWriter(a.out)
    try:
        for m, efc in points:
            # fresh base + index per grid point so earlier inserts do not carry over
            base, _, pairs = C.make_vectors(n_base, pairs_n, dim=dim, seed=a.seed, copies=a.threads)
            C.build_mem_db(a.dsn, db, base, index=(a.index if a.index != "none" else "none"),
                           m=m, ef_construction=efc)
            if a.index == "none":
                C.admin(a.dsn, "DROP INDEX IF EXISTS mem_vec", db=db)
            rec = run_point(a.dsn, db, pairs, a.mode, a.index, m, efc, a.seed)
            rec["versions"] = ver
            rec["params"] = {"pairs": pairs_n, "n_base": n_base, "dim": dim, "mode": a.mode, "index": a.index,
                             "threads": a.threads, "seed": a.seed, "smoke": a.smoke, "grid": a.grid}
            out.write(rec)
    finally:
        out.close()
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--threads", type=int, default=2, help="concurrent clients per event (near-duplicate copies)")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", dest="ef_construction", type=int, default=64)
    ap.add_argument("--dim", type=int, default=C.DIM)
    ap.add_argument("--base", type=int, default=5000)
    ap.add_argument("--mode", choices=["plain", "advisory"], default="plain")
    ap.add_argument("--index", choices=["hnsw", "ivfflat", "none"], default="hnsw")
    ap.add_argument("--grid", action="store_true", help="sweep (m, ef_construction)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
