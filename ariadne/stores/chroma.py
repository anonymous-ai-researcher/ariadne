"""Chroma adapter (embedded PersistentClient or HTTP server; no transactions).

config: kind: chroma, path: ./chroma_store | host: localhost, port: 8000
"""
import numpy as np
from .base import Store, Session


class ChromaSession(Session):
    def search(self, vec, k, ef=None):
        r = self.store.col.query(query_embeddings=[vec.tolist()], n_results=k, include=["distances"])
        # chroma returns cosine distance (1 - cos) for hnsw:space=cosine and squared L2 for l2
        return [(int(i), self.store.to_metric(float(d)) if self.store.metric == "cosine" else float(np.sqrt(max(d, 0.0))))
                for i, d in zip(r["ids"][0], r["distances"][0])]

    def insert(self, item, vec):
        self.store.col.add(ids=[str(int(item))], embeddings=[vec.tolist()])

    def delete(self, item):
        self.store.col.delete(ids=[str(int(item))])

    def exists(self, ids):
        r = self.store.col.get(ids=[str(int(i)) for i in ids], include=[])
        return {int(i) for i in r["ids"]}


class Chroma(Store):
    name = "chroma"

    def __init__(self, cfg):
        super().__init__(cfg)
        import chromadb
        if self.cfg.get("host"):
            self.client = chromadb.HttpClient(host=self.cfg["host"], port=int(self.cfg.get("port", 8000)))
        else:
            self.client = chromadb.PersistentClient(path=self.cfg.get("path", "./chroma_store"))

    def reset(self):
        try:
            self.client.delete_collection("mem")
        except Exception:
            pass
        self.col = self.client.create_collection("mem", metadata={"hnsw:space": "cosine" if self.metric == "cosine" else "l2"})
        for i in range(0, len(self.base), 2000):
            ch = self.base[i:i + 2000]
            self.col.add(ids=[str(i + j) for j in range(len(ch))], embeddings=ch.tolist())

    def session(self):
        return ChromaSession(self)

    def count_extra(self):
        return self.col.count() - len(self.base)
