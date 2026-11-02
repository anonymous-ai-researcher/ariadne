"""Qdrant adapter over the REST API (no multi-statement transactions; commit lock from the adapter).

config:
    kind: qdrant
    url: http://127.0.0.1:6333
    binary: /path/to/qdrant            # optional: start a local server if the URL does not answer
    storage: /path/to/storage
    indexing_threshold: 1000, hnsw_ef: null, M: null, ef_construction: null
"""
import os, subprocess, threading, time, uuid
from .base import Store, Session


class QdrantSession(Session):
    def __init__(self, store):
        super().__init__(store)
        import requests
        self.s = requests.Session()
        self.u = store.url + "/collections/mem"

    def search(self, vec, k, ef=None):
        body = {"vector": vec.tolist(), "limit": k, "with_payload": True}
        if ef or self.store.cfg.get("hnsw_ef"):
            body["params"] = {"hnsw_ef": int(ef or self.store.cfg["hnsw_ef"])}
        r = self.s.post(self.u + "/points/search", json=body); r.raise_for_status()
        return [(int(p["payload"]["item"]), self.store.to_metric(1.0 - p["score"] if self.store.metric == "cosine" else p["score"]))
                for p in r.json()["result"]]

    def exact_search(self, vec, k):
        body = {"vector": vec.tolist(), "limit": k, "with_payload": True, "params": {"exact": True}}
        r = self.s.post(self.u + "/points/search", json=body); r.raise_for_status()
        return [(int(p["payload"]["item"]), self.store.to_metric(1.0 - p["score"] if self.store.metric == "cosine" else p["score"]))
                for p in r.json()["result"]]

    def insert(self, item, vec):
        pt = {"id": int(item), "vector": vec.tolist(), "payload": {"item": int(item)}}
        self.s.put(self.u + "/points?wait=true", json={"points": [pt]}).raise_for_status()

    def delete(self, item):
        self.s.post(self.u + "/points/delete?wait=true", json={"points": [int(item)]}).raise_for_status()

    def exists(self, ids):
        r = self.s.post(self.u + "/points", json={"ids": list(map(int, ids)), "with_payload": False, "with_vector": False})
        r.raise_for_status()
        return {int(p["id"]) for p in r.json()["result"]}


class Qdrant(Store):
    name = "qdrant"

    def __init__(self, cfg):
        super().__init__(cfg)
        import requests
        self.rq = requests
        self.url = self.cfg.get("url", "http://127.0.0.1:6333")
        try:
            requests.get(self.url + "/collections", timeout=1)
        except Exception:
            if not self.cfg.get("binary"):
                raise
            d = self.cfg.get("storage", "./qdrant_storage"); os.makedirs(d, exist_ok=True)
            env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=os.path.abspath(d), QDRANT__TELEMETRY_DISABLED="true", QDRANT__LOG_LEVEL="WARN")
            subprocess.Popen([self.cfg["binary"]], cwd=d, env=env, stdout=open(os.path.join(d, "log.txt"), "a"), stderr=subprocess.STDOUT)
            for _ in range(80):
                try:
                    requests.get(self.url + "/collections", timeout=1); break
                except Exception:
                    time.sleep(0.5)

    def reset(self):
        s = self.rq.Session(); u = self.url + "/collections/mem"
        s.delete(u)
        body = {"vectors": {"size": int(self.base.shape[1]), "distance": "Cosine" if self.metric == "cosine" else "Euclid"},
                "optimizers_config": {"indexing_threshold": int(self.cfg.get("indexing_threshold", 1000))}}
        if self.cfg.get("M") or self.cfg.get("ef_construction"):
            body["hnsw_config"] = {kk: int(v) for kk, v in (("m", self.cfg.get("M")), ("ef_construct", self.cfg.get("ef_construction"))) if v}
        s.put(u, json=body).raise_for_status()
        for i in range(0, len(self.base), 500):
            pts = [{"id": i + j, "vector": v.tolist(), "payload": {"item": i + j}} for j, v in enumerate(self.base[i:i + 500])]
            s.put(u + "/points?wait=true", json={"points": pts}).raise_for_status()
        last, stable = -1, 0
        for _ in range(1200):                                # wait until the optimizer has finished building the index:
            info = s.get(u, timeout=30).json()["result"]     # green and an indexed count that no longer moves (small
            n_ix = info.get("indexed_vectors_count") or 0    # segments below the indexing threshold stay unindexed)
            stable = stable + 1 if n_ix == last else 0
            last = n_ix
            if info["status"] == "green" and (n_ix >= 0.9 * len(self.base) or stable >= 6):
                break
            time.sleep(0.5)
        self.info = {kk: info.get(kk) for kk in ("status", "indexed_vectors_count", "points_count")}

    def session(self):
        return QdrantSession(self)

    def count_extra(self):
        r = self.rq.post(self.url + "/collections/mem/points/count",
                         json={"exact": True, "filter": {"must": [{"key": "item", "range": {"gte": len(self.base)}}]}})
        return int(r.json()["result"]["count"])
