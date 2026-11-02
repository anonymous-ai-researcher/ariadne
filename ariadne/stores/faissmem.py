"""In-process store on FAISS: fast, no server, exact and approximate indexes.

Use it for experiments that need scale or many repetitions without a database in the loop, and as the
exact reference (index: flat).  A readers-writer latch makes search/insert physically atomic; it does
NOT provide isolation.

config:
    kind: faissmem
    index: hnsw | flat | ivf          (default hnsw)
    M: 16, ef_construction: 64, ef_search: 40, nlist: 256, nprobe: 8
    metric: cosine | l2
"""
import threading
import numpy as np
import faiss
from .base import Store, Session


class RWLatch:
    def __init__(self):
        self.c = threading.Condition()
        self.readers = 0
        self.writer = False

    def rlock(self):
        with self.c:
            while self.writer:
                self.c.wait()
            self.readers += 1

    def runlock(self):
        with self.c:
            self.readers -= 1
            if self.readers == 0:
                self.c.notify_all()

    def wlock(self):
        with self.c:
            while self.writer or self.readers:
                self.c.wait()
            self.writer = True

    def wunlock(self):
        with self.c:
            self.writer = False
            self.c.notify_all()


class FaissSession(Session):
    def search(self, vec, k, ef=None):
        st = self.store
        st.latch.rlock()
        try:
            return st._search(vec, k, ef)
        finally:
            st.latch.runlock()

    def exact_search(self, vec, k):
        st = self.store
        st.latch.rlock()
        try:
            n = st.n
            X = st.buf[:n]
            if st.metric == "cosine":
                d = np.sqrt(np.maximum(2.0 - 2.0 * (X @ vec), 0.0))
            else:
                d = np.sqrt(np.maximum(((X - vec) ** 2).sum(1), 0.0))
            if st.deleted:
                d[np.fromiter(st.deleted_rows, dtype=int)] = np.inf
            idx = np.argpartition(d, min(k, n - 1))[:k]
            idx = idx[np.argsort(d[idx])]
            return [(int(st.row2item[i]), float(d[i])) for i in idx if np.isfinite(d[i])]
        finally:
            st.latch.runlock()

    def insert(self, item, vec):
        st = self.store
        st.latch.wlock()
        try:
            st._add(item, vec)
        finally:
            st.latch.wunlock()

    def delete(self, item):
        st = self.store
        st.latch.wlock()
        try:
            if item in st.item2row and item not in st.deleted:
                st.deleted.add(item); st.deleted_rows.add(st.item2row[item])
        finally:
            st.latch.wunlock()

    def exists(self, ids):
        st = self.store
        return {i for i in ids if i in st.item2row and i not in st.deleted}


class FaissMem(Store):
    name = "faissmem"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.latch = RWLatch()
        self.kind = self.cfg.get("index", "hnsw")

    def _new_index(self, dim):
        mt = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        if self.kind == "flat":
            return faiss.IndexFlat(dim, mt)
        if self.kind == "ivf":
            q = faiss.IndexFlat(dim, mt)
            ix = faiss.IndexIVFFlat(q, dim, int(self.cfg.get("nlist", 256)), mt)
            ix.nprobe = int(self.cfg.get("nprobe", 8))
            return ix
        ix = faiss.IndexHNSWFlat(dim, int(self.cfg.get("M", 16)), mt)
        ix.hnsw.efConstruction = int(self.cfg.get("ef_construction", 64))
        ix.hnsw.efSearch = int(self.cfg.get("ef_search", 40))
        return ix

    def setup(self, base):
        self.base = np.ascontiguousarray(base, dtype="float32")
        self.reset()

    def reset(self):
        n, dim = self.base.shape
        self.buf = np.zeros((n + 1024, dim), dtype="float32")
        self.buf[:n] = self.base
        self.n = n
        self.row2item = list(range(n))
        self.item2row = {i: i for i in range(n)}
        self.deleted, self.deleted_rows = set(), set()
        self.ix = self._new_index(dim)
        if self.kind == "ivf":
            self.ix.train(self.base)
        self.ix.add(self.base)
        self.n_base = n

    def _add(self, item, vec):
        if self.n == len(self.buf):
            self.buf = np.vstack([self.buf, np.zeros_like(self.buf)])
        self.buf[self.n] = vec
        self.row2item.append(item); self.item2row[item] = self.n
        self.ix.add(vec.reshape(1, -1).astype("float32"))
        self.n += 1

    def _search(self, vec, k, ef):
        if self.kind == "hnsw" and ef:
            self.ix.hnsw.efSearch = int(ef)
        extra = len(self.deleted_rows)
        D, I = self.ix.search(vec.reshape(1, -1).astype("float32"), min(k + extra, self.n))
        out = []
        for d, i in zip(D[0], I[0]):
            if i < 0 or i in self.deleted_rows:
                continue
            dist = float(np.sqrt(max(2.0 - 2.0 * d, 0.0))) if self.metric == "cosine" else float(np.sqrt(max(d, 0.0)))
            out.append((int(self.row2item[i]), dist))
            if len(out) == k:
                break
        return out

    def session(self):
        return FaissSession(self)

    def count_extra(self):
        return sum(1 for it in self.row2item[self.n_base:] if it not in self.deleted)
