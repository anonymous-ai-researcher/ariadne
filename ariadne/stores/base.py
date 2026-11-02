"""Store adapter interface.

Every adapter exposes the same small surface so that protocols and workloads are store-agnostic:

    store = make_store(cfg)        # cfg: dict from the YAML `store:` section
    store.setup(base)              # load the initial memory (items 0..n-1) and build the index
    store.reset()                  # restore the initial memory before each run
    s = store.session()            # one per client thread

    s.search(vec, k, ef=None)      -> [(item_id, metric_distance)] sorted ascending
    s.exact_search(vec, k)         -> same, exhaustive (NotImplementedError if unsupported)
    s.insert(item_id, vec); s.delete(item_id); s.exists(ids) -> set of present ids
    s.lock(key=0)                  # commit lock; released by s.commit() / s.rollback()
    s.commit(); s.rollback()
    s.begin(isolation)             # engines with transactions only

Distances returned by sessions are METRIC distances: Euclidean for metric="l2", the chord distance
sqrt(2 * cosine distance) for metric="cosine" (vectors are unit-normalized by the workload).

Stores without transactions get the commit lock from the adapter (a process-local lock).  With clients
in several processes or machines an external lock service is needed; set `lock: file:<path>` in the
store config to use an inter-process file lock.
"""
import fcntl, os, threading
import numpy as np


class ProcessLocks:
    """Named commit locks shared by all sessions of one store (thread-level, optionally file-level)."""

    def __init__(self, spec=None):
        self.mu = threading.Lock()
        self.locks = {}
        self.file_dir = spec[5:] if spec and spec.startswith("file:") else None
        if self.file_dir:
            os.makedirs(self.file_dir, exist_ok=True)

    def get(self, key):
        with self.mu:
            if key not in self.locks:
                self.locks[key] = threading.Lock()
            return self.locks[key]


class Session:
    def __init__(self, store):
        self.store = store
        self._held = None
        self._fh = None

    # -- commit lock for engines without transactions
    def lock(self, key=0):
        lk = self.store.locks.get(key)
        lk.acquire()
        self._held = lk
        if self.store.locks.file_dir:
            self._fh = open(os.path.join(self.store.locks.file_dir, f"lock_{key}"), "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)

    def unlock(self):
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN); self._fh.close(); self._fh = None
        if self._held is not None:
            lk, self._held = self._held, None
            lk.release()

    def commit(self):
        self.unlock()

    def rollback(self):
        self.unlock()

    def begin(self, isolation=None):
        pass

    def exact_search(self, vec, k):
        raise NotImplementedError

    def exists(self, ids):
        raise NotImplementedError

    def close(self):
        pass


class Store:
    name = "base"
    has_txn = False

    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.metric = self.cfg.get("metric", "cosine")
        self.locks = ProcessLocks(self.cfg.get("lock"))
        self.base = None

    def to_metric(self, raw):
        """Convert the engine's native distance to the metric distance."""
        if self.metric == "cosine":
            return float(np.sqrt(max(2.0 * raw, 0.0)))
        return float(raw)

    def setup(self, base):
        self.base = base

    def reset(self):
        raise NotImplementedError

    def session(self):
        raise NotImplementedError

    def count_extra(self):
        raise NotImplementedError

    def close(self):
        pass


def make_store(cfg):
    kind = cfg.get("kind", "faissmem")
    if kind == "pgvector":
        from .pgvector import PgVector as C
    elif kind == "faissmem":
        from .faissmem import FaissMem as C
    elif kind == "qdrant":
        from .qdrant import Qdrant as C
    elif kind == "chroma":
        from .chroma import Chroma as C
    elif kind == "milvus":
        from .milvus import Milvus as C
    elif kind == "lancedb":
        from .lancedb_store import LanceDB as C
    elif kind == "weaviate":
        from .weaviate_store import Weaviate as C
    else:
        raise ValueError(f"unknown store kind '{kind}'")
    return C(cfg)
