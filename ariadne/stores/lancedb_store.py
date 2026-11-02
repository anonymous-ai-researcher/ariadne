"""LanceDB adapter (embedded; no transactions; optimistic commits per table version).

config: kind: lancedb, path: ./lancedb_store, index: none | ivf_pq | hnsw
Note: without an ANN index LanceDB scans exhaustively, so search() is exact.
"""
import numpy as np
from .base import Store, Session


class LanceSession(Session):
    def search(self, vec, k, ef=None):
        st = self.store
        q = st.tbl.search(vec.astype("float32")).metric("cosine" if st.metric == "cosine" else "l2").limit(k)
        rows = q.to_list()
        return [(int(r["item"]), st.to_metric(float(r["_distance"])) if st.metric == "cosine" else float(np.sqrt(max(r["_distance"], 0.0))))
                for r in rows]

    def insert(self, item, vec):
        self.store.tbl.add([{"item": int(item), "vector": vec.astype("float32").tolist()}])

    def delete(self, item):
        self.store.tbl.delete(f"item = {int(item)}")

    def exists(self, ids):
        ids = [int(i) for i in ids]
        if not ids:
            return set()
        rows = self.store.tbl.search().where("item IN (" + ",".join(map(str, ids)) + ")").limit(len(ids)).to_list()
        return {int(r["item"]) for r in rows}


class LanceDB(Store):
    name = "lancedb"

    def __init__(self, cfg):
        super().__init__(cfg)
        import lancedb
        self.db = lancedb.connect(self.cfg.get("path", "./lancedb_store"))

    def reset(self):
        data = [{"item": i, "vector": v.astype("float32").tolist()} for i, v in enumerate(self.base)]
        self.tbl = self.db.create_table("mem", data=data, mode="overwrite")
        kind = self.cfg.get("index", "none")
        if kind == "ivf_pq":
            self.tbl.create_index(metric="cosine" if self.metric == "cosine" else "l2", index_type="IVF_PQ")
        elif kind == "hnsw":
            self.tbl.create_index(metric="cosine" if self.metric == "cosine" else "l2", index_type="IVF_HNSW_SQ")

    def session(self):
        return LanceSession(self)

    def count_extra(self):
        return self.tbl.count_rows(f"item >= {len(self.base)}")
