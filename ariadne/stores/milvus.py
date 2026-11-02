"""Milvus adapter through pymilvus.MilvusClient: a file path selects Milvus Lite, a URI a server.

config: kind: milvus, uri: ./milvus_lite.db | http://localhost:19530, consistency: Strong,
        index: AUTOINDEX | HNSW | IVF_FLAT | FLAT (server only), M, ef_construction, ef_search
"""
from .base import Store, Session


class MilvusSession(Session):
    def search(self, vec, k, ef=None):
        st = self.store
        params = {"params": {"ef": int(ef or st.cfg["ef_search"])}} if (ef or st.cfg.get("ef_search")) else {}
        r = st.client.search("mem", data=[vec.tolist()], limit=k, output_fields=["item"],
                             consistency_level=st.cfg.get("consistency", "Strong"), search_params=params or None)
        out = []
        for h in r[0]:
            raw = float(h["distance"])
            out.append((int(h["entity"]["item"]), st.to_metric(1.0 - raw) if st.metric == "cosine" else raw ** 0.5))
        return out

    def insert(self, item, vec):
        self.store.client.insert("mem", [{"id": int(item), "vector": vec.tolist(), "item": int(item)}])

    def delete(self, item):
        self.store.client.delete("mem", ids=[int(item)])

    def exists(self, ids):
        r = self.store.client.get("mem", ids=[int(i) for i in ids], output_fields=["item"])
        return {int(x["item"]) for x in r}


class Milvus(Store):
    name = "milvus"

    def __init__(self, cfg):
        super().__init__(cfg)
        from pymilvus import MilvusClient
        self.client = MilvusClient(self.cfg.get("uri", "./milvus_lite.db"))

    def reset(self):
        c = self.client
        if c.has_collection("mem"):
            c.drop_collection("mem")
        c.create_collection("mem", dimension=int(self.base.shape[1]), metric_type="COSINE" if self.metric == "cosine" else "L2",
                            auto_id=False, consistency_level=self.cfg.get("consistency", "Strong"))
        for i in range(0, len(self.base), 2000):
            c.insert("mem", [{"id": i + j, "vector": v.tolist(), "item": i + j} for j, v in enumerate(self.base[i:i + 2000])])

    def session(self):
        return MilvusSession(self)

    def count_extra(self):
        r = self.client.query("mem", filter=f"item >= {len(self.base)}", output_fields=["count(*)"],
                              consistency_level=self.cfg.get("consistency", "Strong"))
        return int(r[0]["count(*)"])
