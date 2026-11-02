"""PostgreSQL + pgvector adapter (real transactions, isolation levels, advisory commit lock).

config:
    kind: pgvector
    dsn: "dbname=root"                 # admin connection; databases are created next to it
    index: hnsw | ivfflat | none       (default hnsw)
    M: 16, ef_construction: 64, ef_search: 40, lists: 100, probes: 10
    metric: cosine | l2
The initial memory is loaded once into a template database (named after a hash of its contents and
index parameters) and every reset() clones it, so repeated runs start from an identical index.
"""
import hashlib, threading
import numpy as np
from .base import Store, Session


def vec_lit(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgSession(Session):
    def __init__(self, store):
        super().__init__(store)
        self.pg = store.pg
        self.conn = self.pg.connect(store.dsn_for("run"))
        self.cur = self.conn.cursor()
        self._session_params()
        self.conn.commit()
        self.op = "<=>" if store.metric == "cosine" else "<->"

    def _session_params(self):
        st = self.store
        if st.index == "hnsw":
            self.cur.execute(f"SET hnsw.ef_search = {int(st.cfg.get('ef_search', 40))}")
        elif st.index == "ivfflat":
            self.cur.execute(f"SET ivfflat.probes = {int(st.cfg.get('probes', 10))}")

    def begin(self, isolation=None):
        self.conn.rollback()
        self.conn.set_session(isolation_level=isolation or "READ COMMITTED", autocommit=False)

    def search(self, vec, k, ef=None):
        lit = vec_lit(vec)
        if ef and self.store.index == "hnsw":
            self.cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
        self.cur.execute(f"SELECT item, embedding {self.op} %s::vector AS d FROM mem ORDER BY embedding {self.op} %s::vector LIMIT %s",
                         (lit, lit, k))
        return [(int(r[0]), self.store.to_metric(float(r[1]))) for r in self.cur.fetchall()]

    def exact_search(self, vec, k):
        lit = vec_lit(vec)
        self.cur.execute("SET LOCAL enable_indexscan = off"); self.cur.execute("SET LOCAL enable_bitmapscan = off")
        self.cur.execute(f"SELECT item, embedding {self.op} %s::vector AS d FROM mem ORDER BY embedding {self.op} %s::vector LIMIT %s",
                         (lit, lit, k))
        out = [(int(r[0]), self.store.to_metric(float(r[1]))) for r in self.cur.fetchall()]
        self.cur.execute("SET LOCAL enable_indexscan = on"); self.cur.execute("SET LOCAL enable_bitmapscan = on")
        return out

    def insert(self, item, vec):
        self.cur.execute("INSERT INTO mem(item, embedding) VALUES (%s, %s::vector)", (int(item), vec_lit(vec)))

    def delete(self, item):
        self.cur.execute("DELETE FROM mem WHERE item = %s", (int(item),))

    def exists(self, ids):
        if not ids:
            return set()
        self.cur.execute("SELECT item FROM mem WHERE item = ANY(%s)", (list(map(int, ids)),))
        return {int(r[0]) for r in self.cur.fetchall()}

    def lock(self, key=0):
        self.cur.execute("SELECT pg_advisory_xact_lock(%s)", (int(key),))

    def unlock(self):
        pass

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


class PgVector(Store):
    name = "pgvector"
    has_txn = True

    def __init__(self, cfg):
        super().__init__(cfg)
        import psycopg2
        self.pg = psycopg2
        self.errors = psycopg2.errors
        self.dsn = self.cfg.get("dsn", "dbname=root")
        self.index = self.cfg.get("index", "hnsw")
        self.rundb = self.cfg.get("rundb", "ariadne_run")
        self.tpl = None

    def dsn_for(self, which):
        db = self.rundb if which == "run" else which
        parts = [p for p in self.dsn.split() if not p.startswith("dbname=")]
        return " ".join(parts + [f"dbname={db}"])

    def admin(self, sql, db=None):
        c = self.pg.connect(self.dsn if db is None else self.dsn_for(db))
        c.autocommit = True
        cur = c.cursor(); cur.execute(sql)
        out = cur.fetchall() if cur.description else None
        c.close()
        return out

    def setup(self, base):
        self.base = base
        n, dim = base.shape
        sig = hashlib.sha1(base[: min(n, 2000)].tobytes()).hexdigest()[:10]
        par = f"{self.index}_{self.cfg.get('M', 16)}_{self.cfg.get('ef_construction', 64)}_{self.cfg.get('lists', 100)}_{self.metric}"
        self.tpl = f"ariadne_tpl_{n}_{dim}_{sig}_{hashlib.sha1(par.encode()).hexdigest()[:6]}"
        if self.admin(f"SELECT 1 FROM pg_database WHERE datname='{self.tpl}'"):
            return
        self.admin(f"CREATE DATABASE {self.tpl}")
        c = self.pg.connect(self.dsn_for(self.tpl)); cur = c.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE TABLE mem (id bigserial PRIMARY KEY, item int NOT NULL, embedding vector({dim}) NOT NULL)")
        for i in range(0, n, 1000):
            args = b",".join(cur.mogrify("(%s,%s::vector)", (i + j, vec_lit(v))) for j, v in enumerate(base[i:i + 1000]))
            cur.execute(b"INSERT INTO mem(item, embedding) VALUES " + args)
        cur.execute("CREATE INDEX mem_item ON mem (item)")
        ops = "vector_cosine_ops" if self.metric == "cosine" else "vector_l2_ops"
        if self.index == "hnsw":
            cur.execute(f"CREATE INDEX mem_vec ON mem USING hnsw (embedding {ops}) WITH (m = {int(self.cfg.get('M', 16))}, ef_construction = {int(self.cfg.get('ef_construction', 64))})")
        elif self.index == "ivfflat":
            cur.execute(f"CREATE INDEX mem_vec ON mem USING ivfflat (embedding {ops}) WITH (lists = {int(self.cfg.get('lists', 100))})")
        c.commit(); cur.execute("ANALYZE mem"); c.commit(); c.close()

    def reset(self):
        self.admin(f"DROP DATABASE IF EXISTS {self.rundb} WITH (FORCE)")
        self.admin(f"CREATE DATABASE {self.rundb} TEMPLATE {self.tpl}")

    def session(self):
        return PgSession(self)

    def count_extra(self):
        return int(self.admin(f"SELECT count(*) FROM mem WHERE item >= {len(self.base)}", self.rundb)[0][0])

    def versions(self):
        v = self.admin("SHOW server_version")[0][0]
        e = self.admin("SELECT extversion FROM pg_extension WHERE extname='vector'", self.rundb)
        return {"postgresql": v, "pgvector": e[0][0] if e else None}
