"""Weaviate adapter (v4 Python client).  NOT exercised in the smoke tests of this repository:
no server was available where the framework was written.  Check against your deployment first.

config: kind: weaviate, host: localhost, port: 8080, grpc_port: 50051
"""
import uuid
from .base import Store, Session

NS = uuid.UUID("00000000-0000-0000-0000-00000000a71a")


def uid(item):
    return str(uuid.uuid5(NS, str(int(item))))


class WeaviateSession(Session):
    def search(self, vec, k, ef=None):
        from weaviate.classes.query import MetadataQuery
        r = self.store.col.query.near_vector(near_vector=vec.tolist(), limit=k, return_metadata=MetadataQuery(distance=True))
        return [(int(o.properties["item"]), self.store.to_metric(float(o.metadata.distance))) for o in r.objects]

    def insert(self, item, vec):
        self.store.col.data.insert(properties={"item": int(item)}, vector=vec.tolist(), uuid=uid(item))

    def delete(self, item):
        self.store.col.data.delete_by_id(uid(item))

    def exists(self, ids):
        return {int(i) for i in ids if self.store.col.data.exists(uid(i))}


class Weaviate(Store):
    name = "weaviate"

    def __init__(self, cfg):
        super().__init__(cfg)
        import weaviate
        self.client = weaviate.connect_to_local(host=self.cfg.get("host", "localhost"), port=int(self.cfg.get("port", 8080)),
                                                grpc_port=int(self.cfg.get("grpc_port", 50051)))

    def reset(self):
        from weaviate.classes.config import Configure, Property, DataType, VectorDistances
        if self.client.collections.exists("Mem"):
            self.client.collections.delete("Mem")
        self.col = self.client.collections.create(
            "Mem", properties=[Property(name="item", data_type=DataType.INT)],
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE if self.metric == "cosine" else VectorDistances.L2_SQUARED))
        with self.col.batch.dynamic() as b:
            for i, v in enumerate(self.base):
                b.add_object(properties={"item": i}, vector=v.tolist(), uuid=uid(i))

    def session(self):
        return WeaviateSession(self)

    def count_extra(self):
        from weaviate.classes.query import Filter
        r = self.col.aggregate.over_all(filters=Filter.by_property("item").greater_or_equal(len(self.base)), total_count=True)
        return int(r.total_count)

    def close(self):
        self.client.close()
