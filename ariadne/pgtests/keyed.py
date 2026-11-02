"""Key-based control: check-then-insert on a scalar key with a plain B-tree.

Question: the duplicate/abort behaviour of the same check-then-insert workload when
the "is this already stored?" test is an exact B-tree key lookup instead of a
similarity read. This is the control that isolates what the vector index costs:
under SERIALIZABLE a B-tree key lookup takes only page/tuple predicate locks, so
aborts are rare and nearly all real, whereas the pgvector version aborts massively
because of its relation level lock.

Workload (event fan-out, the same shape as the similarity tests but with keys):
a stream of facts; each is reported by 1, 2 or 4 agents (p = 0.5, 0.3, 0.2) that
arrive within a jitter window; every agent runs SELECT event WHERE key = K; think;
INSERT if not found. Two table sizes (small and large B-tree) and two isolation
levels (READ COMMITTED, SERIALIZABLE). A correct execution stores one row per
event; duplicates come from the check-then-insert race, not from the index.

Positive result: at READ COMMITTED duplicates appear and aborts are ~0; at
SERIALIZABLE duplicates are 0 and aborts per task are small (a fraction), most of
them on multi-agent events (true conflicts). Contrast with hammer.py, where the
vector version's SERIALIZABLE abort rate is tens per task. Self-contained (does
not import the ariadne engine).

Invocation:
    python -m ariadne.pgtests.keyed --out results/T_pg/keyed.jsonl
    python -m ariadne.pgtests.keyed --events 600 --conc 16
    python -m ariadne.pgtests.keyed --smoke
"""
import argparse, math, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psycopg2
from . import common as C

FANOUT = ((1, 0.5), (2, 0.3), (4, 0.2))
MODES = {"krc_s": ("READ COMMITTED", "kmem_s"), "kser_s": ("SERIALIZABLE", "kmem_s"),
         "krc_l": ("READ COMMITTED", "kmem_l"), "kser_l": ("SERIALIZABLE", "kmem_l")}


def make_tasks(n_events, conc, think, jitter, seed):
    rng = np.random.default_rng(seed)
    keys = rng.integers(1, 2 ** 62, size=n_events)
    fs, ps = zip(*FANOUT)
    mean_f = sum(f * p for f, p in FANOUT)
    rate = conc / (mean_f * think)
    t, tasks = 0.0, []
    for e in range(n_events):
        t += rng.exponential(1.0 / rate)
        f = int(rng.choice(fs, p=ps))
        for c in range(f):
            tasks.append({"t": t + rng.uniform(0, jitter), "event": e, "fan": f, "key": int(keys[e])})
    tasks.sort(key=lambda x: x["t"])
    return tasks


def setup_keyed(dsn, db, n_small, n_large):
    C.make_scratch_db(dsn, db)
    for tbl, n in (("kmem_s", n_small), ("kmem_l", n_large)):
        C.admin(dsn, f"CREATE TABLE {tbl} (id bigserial PRIMARY KEY, event int NOT NULL, key bigint NOT NULL)", db=db)
        C.admin(dsn, f"INSERT INTO {tbl}(event, key) SELECT -1, (random() * 4611686018427387904)::bigint "
                     f"FROM generate_series(1, {n})", db=db)
        C.admin(dsn, f"CREATE INDEX {tbl}_key ON {tbl} (key)", db=db)
        C.admin(dsn, f"ANALYZE {tbl}", db=db)


def run_mode(dsn, db, tasks, iso, tbl, think, seed):
    E = psycopg2.errors
    C.admin(dsn, f"DELETE FROM {tbl} WHERE event >= 0", db=db)
    tl = threading.local()
    rng = np.random.default_rng(seed)
    think_lat = rng.lognormal(mean=math.log(think) - 0.045, sigma=0.3, size=200000)
    ti = [0]
    tlock = threading.Lock()

    def think_fn():
        with tlock:
            d = float(think_lat[ti[0] % len(think_lat)]); ti[0] += 1
        time.sleep(d)

    recs = [{"event": t["event"], "fan": t["fan"], "aborts": 0, "outcome": None} for t in tasks]

    def conn():
        c = getattr(tl, "c", None)
        if c is None or c.closed:
            c = C.connect(dsn, db=db); tl.c = c
        c.rollback(); c.set_session(isolation_level=iso, autocommit=False)
        return c

    def work(i):
        task = tasks[i]; rec = recs[i]
        c = conn(); cur = c.cursor()
        while True:
            try:
                cur.execute(f"SELECT event FROM {tbl} WHERE key = %s LIMIT 1", (task["key"],))
                found = cur.fetchone()
                think_fn()
                if found:
                    rec["outcome"] = "noop"
                else:
                    cur.execute(f"INSERT INTO {tbl}(event, key) VALUES (%s, %s)", (task["event"], task["key"]))
                    rec["outcome"] = "insert"
                c.commit(); return
            except (E.SerializationFailure, E.DeadlockDetected):
                c.rollback(); rec["aborts"] += 1
                if rec["aborts"] > 200:
                    rec["outcome"] = "gave_up"; return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=320) as ex:
        futs = []
        for i, t in enumerate(tasks):
            dt = t["t"] - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            futs.append(ex.submit(work, i))
        for f in futs:
            f.result()
    wall = time.time() - t0
    rows = C.admin(dsn, f"SELECT event, count(*) FROM {tbl} WHERE event >= 0 GROUP BY event", db=db)
    n_events = len({r["event"] for r in recs})
    dup = sum(int(k) - 1 for _, k in rows if k > 1)
    single = [r for r in recs if r["fan"] == 1]
    tbl_rows = int(C.admin(dsn, f"SELECT count(*) FROM {tbl}", db=db)[0][0])
    leaf = int(C.admin(dsn, f"SELECT relpages FROM pg_class WHERE relname='{tbl}_key'", db=db)[0][0])
    return {"iso": iso, "table": tbl, "table_rows": tbl_rows, "index_leaf_pages": leaf,
            "tasks": len(recs), "events": n_events, "rows_stored": sum(int(k) for _, k in rows),
            "duplicates": dup, "dup_per_100_events": round(100 * dup / n_events, 2),
            "events_with_dup": sum(1 for _, k in rows if k > 1),
            "aborts": sum(r["aborts"] for r in recs), "aborts_per_task": round(sum(r["aborts"] for r in recs) / len(recs), 3),
            "aborts_on_singleton_events": sum(r["aborts"] for r in single), "singleton_tasks": len(single),
            "gave_up": sum(1 for r in recs if r["outcome"] == "gave_up"), "wall_s": round(wall, 2)}


def run(a):
    n_events = 40 if a.smoke else a.events
    n_small = 2000 if a.smoke else a.base
    n_large = 20000 if a.smoke else 300000
    modes = [m for m in (a.modes.split(",") if a.modes else list(MODES)) if m]
    ver = C.versions(a.dsn)
    db = C.scratch_name("keyed")
    tasks = make_tasks(n_events, a.conc, a.think, a.jitter * a.think, a.seed)
    setup_keyed(a.dsn, db, n_small, n_large)
    out = C.JsonlWriter(a.out)
    try:
        for mode in modes:
            if mode not in MODES:
                out.write({"mode": mode, "error": "unknown mode", "versions": ver}); continue
            iso, tbl = MODES[mode]
            rec = run_mode(a.dsn, db, tasks, iso, tbl, a.think, a.seed)
            rec["mode"] = mode
            rec["versions"] = ver
            rec["params"] = {"events": n_events, "n_small": n_small, "n_large": n_large, "conc": a.conc,
                             "think_s": a.think, "jitter_x": a.jitter, "seed": a.seed, "smoke": a.smoke}
            out.write(rec)
    finally:
        out.close()
        C.drop_scratch_db(a.dsn, db)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--modes", default="", help="comma list of krc_s,kser_s,krc_l,kser_l")
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--conc", type=float, default=16)
    ap.add_argument("--think", type=float, default=0.1)
    ap.add_argument("--jitter", type=float, default=2.0, help="copy arrival window in units of think time")
    ap.add_argument("--base", type=int, default=15000, help="rows in the small table")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
