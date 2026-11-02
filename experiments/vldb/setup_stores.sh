#!/usr/bin/env bash
# The other stores of the portability experiment F1 on the performance machine (Section 6.7 and the full version).
# Run once, as the login user, before `bash experiments/vldb/run_perf.sh`; prints the two exports that run_perf.sh needs.
# Chroma, Milvus Lite, LanceDB and FAISS run in process (requirements.txt).  Qdrant: the server binary 1.15.4.  Weaviate: the official image in Docker, ports 8080 and 50051.  The versions go into ENV/stores.txt.
set -euo pipefail
OUT=${OUT:-results_vldb_perf}; mkdir -p "$OUT/ENV" ~/stores
cd ~/stores
if [ ! -x qdrant ]; then
  curl -fsSL -o qdrant.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.15.4/qdrant-x86_64-unknown-linux-musl.tar.gz
  tar xzf qdrant.tar.gz && rm qdrant.tar.gz
fi
command -v docker >/dev/null || { sudo apt-get install -y docker.io && sudo usermod -aG docker "$USER"; echo "log out and in again, then rerun"; exit 1; }
docker rm -f ariadne-weaviate >/dev/null 2>&1 || true
docker run -d --name ariadne-weaviate -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:${WEAVIATE_VERSION:-latest} >/dev/null
cd - >/dev/null
pip install "weaviate-client>=4,<5" >/dev/null
sleep 10
{ echo "qdrant $(~/stores/qdrant --version 2>/dev/null)"
  echo "weaviate image $(docker inspect --format '{{.Config.Image}} {{index .RepoDigests 0}}' "$(docker inspect --format '{{.Image}}' ariadne-weaviate)" 2>/dev/null)"
  curl -fsS localhost:8080/v1/meta 2>/dev/null | python3 -c "import json,sys; print('weaviate server', json.load(sys.stdin).get('version'))" || echo "weaviate server not answering"
  python3 -m pip freeze | grep -iE "^(weaviate-client|chromadb|pymilvus|milvus-lite|lancedb|faiss-cpu|qdrant-client)=" ; } | tee "$OUT/ENV/stores.txt"
echo "export QDRANT_BIN=$HOME/stores/qdrant WEAVIATE_HOST=localhost"
