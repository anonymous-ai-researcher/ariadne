"""Shared helpers for ariadne.pgtests: DSN handling, scratch databases, versions,
vector literals, a synthetic near-duplicate workload, JSONL output and argparse.

DSN: a libpq string such as "dbname=root". The admin DSN connects to the server;
scratch databases are created next to it (same host/user/port, different dbname).
Default DSN: environment ARIADNE_PG_DSN, else PG_DSN (the variable the experiment files use), else "dbname=root".

Scratch databases: every database these tests create or drop has a name that
starts with "ariadne_t_". The drop helper refuses any other name, so a run can
never touch the databases other work keeps on the same cluster.
"""
import argparse, hashlib, json, os, subprocess, time
import numpy as np
import psycopg2

DIM = 300
TAU = 0.15        # cosine distance below which two memories are "the same fact"
SEP = 0.25        # events are at least this far from the base and from each other
SIGMA = 0.20      # paraphrase noise; two copies of one event end up about 0.04 apart
TOPK = 10
SCRATCH_PREFIX = "ariadne_t_"


# --------------------------------------------------------------------------- DSN
def default_dsn():
    return os.environ.get("ARIADNE_PG_DSN") or os.environ.get("PG_DSN") or "dbname=root"


def dsn_for(dsn, db):
    """Return `dsn` with its dbname replaced by `db` (other fields kept)."""
    parts = [p for p in dsn.split() if not p.startswith("dbname=")]
    return " ".join(parts + [f"dbname={db}"])


def connect(dsn, db=None, iso=None, autocommit=False):
    c = psycopg2.connect(dsn if db is None else dsn_for(dsn, db))
    if autocommit:
        c.autocommit = True
    if iso:
        c.set_session(isolation_level=iso, autocommit=autocommit)
    return c


def admin(dsn, sql, db=None):
    """Run one statement on an autocommit connection, returning rows or None."""
    c = connect(dsn, db=db, autocommit=True)
    try:
        cur = c.cursor()
        cur.execute(sql)
        return cur.fetchall() if cur.description else None
    finally:
        c.close()


# --------------------------------------------------------------------- scratch db
def _check_scratch(name):
    if not name.startswith(SCRATCH_PREFIX):
        raise ValueError(f"refusing to touch database {name!r}: scratch names must start with {SCRATCH_PREFIX!r}")
    return name


def make_scratch_db(dsn, name, template=None):
    """Drop (if present) and recreate a scratch database. Name must be a scratch name."""
    _check_scratch(name)
    admin(dsn, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    if template:
        admin(dsn, f"CREATE DATABASE {name} TEMPLATE {template}")
    else:
        admin(dsn, f"CREATE DATABASE {name}")
    return name


def drop_scratch_db(dsn, name):
    _check_scratch(name)
    admin(dsn, f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def scratch_name(tag):
    """A unique scratch name for this process, e.g. ariadne_t_hammer_12345."""
    return f"{SCRATCH_PREFIX}{tag}_{os.getpid()}"


# ---------------------------------------------------------------------- versions
def _pgvector_libpath():
    """Best effort: the on-disk path of the pgvector shared library and its sha256."""
    path = None
    try:
        pkglib = subprocess.check_output(["pg_config", "--pkglibdir"], text=True).strip()
        cand = os.path.join(pkglib, "vector.so")
        if os.path.exists(cand):
            path = cand
    except Exception:
        for cand in ("/usr/lib/postgresql/16/lib/vector.so",):
            if os.path.exists(cand):
                path = cand
                break
    checksum = None
    if path:
        try:
            with open(path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            pass
    return path, checksum


def versions(dsn, db=None):
    """PostgreSQL version, pgvector extension version, pgvector library path+checksum."""
    v = admin(dsn, "SHOW server_version", db=db)[0][0]
    pgv = None
    try:                                  # version actually installed in this db, if any
        e = admin(dsn, "SELECT extversion FROM pg_extension WHERE extname='vector'", db=db)
        pgv = e[0][0] if e else None
    except Exception:
        pgv = None
    if pgv is None:                       # else the library version available on the cluster
        try:
            e = admin(dsn, "SELECT default_version FROM pg_available_extensions WHERE name='vector'", db=db)
            pgv = e[0][0] if e else None
        except Exception:
            pgv = None
    path, checksum = _pgvector_libpath()
    return {"postgresql": v, "pgvector": pgv, "pgvector_lib": path, "pgvector_lib_sha256": checksum}


# ------------------------------------------------------------------ vector helper
def vec_lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# -------------------------------------------------------------------- workload gen
def make_vectors(n_base, n_events, dim=DIM, sigma=SIGMA, seed=1, copies=2):
    """Synthetic near-duplicate workload, self-contained (no external vectors needed).

    Returns (base, events, pairs):
      base   float32 [n_base, dim]  unit vectors, the initial index contents.
      events float32 [n_events, dim] unit vectors, one centroid per new fact.
      pairs  list of length n_events, each a list of `copies` unit vectors that are
             near duplicates of that event (about 0.04 cosine apart).
    Random unit vectors in a few hundred dimensions are near orthogonal, so every
    event is far (cosine distance near 1.0, well above SEP) from the base and from
    every other event, while the copies of one event sit far inside TAU. A correct
    execution stores exactly one row per event; any extra row is a race duplicate.
    """
    rng = np.random.default_rng(seed)

    def unit(a):
        return (a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)).astype("float32")

    base = unit(rng.normal(size=(n_base, dim)))
    events = unit(rng.normal(size=(n_events, dim)))
    pairs = []
    for e in range(n_events):
        cs = []
        for _ in range(copies):
            v = events[e] + sigma * rng.normal(size=dim).astype("float32") / np.sqrt(dim)
            cs.append(unit(v))
        pairs.append(cs)
    return base, events, pairs


def build_mem_db(dsn, db, base, index="hnsw", m=16, ef_construction=64, lists=100,
                 metric="cosine", template=False):
    """Create scratch database `db` with a `mem(id, event, embedding)` table holding
    `base` (event=-1) and a vector index of the requested kind. Returns db name."""
    make_scratch_db(dsn, db)
    dim = base.shape[1]
    ops = "vector_cosine_ops" if metric == "cosine" else "vector_l2_ops"
    c = connect(dsn, db=db)
    try:
        cur = c.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE TABLE mem (id bigserial PRIMARY KEY, event int NOT NULL, embedding vector({dim}) NOT NULL)")
        for i in range(0, len(base), 1000):
            args = b",".join(cur.mogrify("(-1,%s::vector)", (vec_lit(v),)) for v in base[i:i + 1000])
            cur.execute(b"INSERT INTO mem(event, embedding) VALUES " + args)
        c.commit()
        if index == "hnsw":
            cur.execute(f"CREATE INDEX mem_vec ON mem USING hnsw (embedding {ops}) "
                        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})")
        elif index == "ivfflat":
            cur.execute(f"CREATE INDEX mem_vec ON mem USING ivfflat (embedding {ops}) WITH (lists = {int(lists)})")
        elif index == "none":
            pass
        else:
            raise ValueError(f"unknown index {index!r}")
        c.commit()
        cur.execute("ANALYZE mem")
        c.commit()
    finally:
        c.close()
    return db


# -------------------------------------------------------------------- JSONL output
class JsonlWriter:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "a") if path else None

    def write(self, rec):
        line = json.dumps(rec)
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# --------------------------------------------------------------- Wilson interval
def wilson(k, n, z=1.96):
    """Wilson score 95% interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = float(z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d)
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


# ----------------------------------------------------------------------- argparse
def add_common_args(ap):
    ap.add_argument("--dsn", default=default_dsn(), help='libpq DSN, e.g. "dbname=root"')
    ap.add_argument("--out", default="", help="append one JSON line per result to FILE.jsonl")
    ap.add_argument("--trials", type=int, default=8, help="number of trials")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast run for a sanity check")
    return ap
