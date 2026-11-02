#!/usr/bin/env bash
# =============================================================================
# matrix.sh  --  run the pgtests across PostgreSQL major versions 14..18 using
#                the official pgvector/pgvector:pgNN Docker images.
#
# Not used for the paper, which tests PostgreSQL 16.15 only (experiment A10 is the
# optional extension to other major versions). Read it before running and adjust
# the image tags, ports and paths to your machine. It is deliberately
# conservative (set -euo pipefail, explicit container names, cleanup trap).
#
# It runs locks, hammer and gist (the tests that need no debugger) inside a fresh
# container per major version and copies the JSONL outputs to RESULTS_DIR. The
# gdb-based tests (hole, generic) are intentionally skipped: ptrace inside a
# container needs --cap-add=SYS_PTRACE and a matching gdb, which varies by host.
#
# Prerequisites on the host: docker, and a Python 3 with psycopg2 and numpy on
# PATH (the ariadne package is bind-mounted and run with the host interpreter via
# the container's exposed port; no Python is installed in the container).
#
# Usage:
#   ariadne/pgtests/matrix.sh                       # versions 14 15 16 17 18
#   PG_VERSIONS="16 17" ariadne/pgtests/matrix.sh   # a subset
#   RESULTS_DIR=results/T_pg_matrix ariadne/pgtests/matrix.sh
#   ariadne/pgtests/matrix.sh results/A10_versions       # output directory as the first argument (what run.py passes)
#   PGVECTOR_TAG=0.8.0 ariadne/pgtests/matrix.sh          # image pgvector/pgvector:<tag>-pgNN instead of pgvector/pgvector:pgNN
#                                                        # (the pgvector version of every line is recorded in its "versions" field)
# =============================================================================
set -euo pipefail

PG_VERSIONS="${PG_VERSIONS:-14 15 16 17 18}"
RESULTS_DIR="${1:-${RESULTS_DIR:-results/T_pg_matrix}}"
HOST_PORT_BASE="${HOST_PORT_BASE:-55432}"      # host port for pg14 = base, pg15 = base+1, ...
PGPASSWORD_VALUE="${PGPASSWORD_VALUE:-ariadne}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$RESULTS_DIR" in /*) OUT_ROOT="$RESULTS_DIR" ;; *) OUT_ROOT="$REPO_ROOT/$RESULTS_DIR" ;; esac
mkdir -p "$OUT_ROOT"

cleanup() {
  for v in $PG_VERSIONS; do
    docker rm -f "ariadne_pg${v}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

idx=0
for v in $PG_VERSIONS; do
  name="ariadne_pg${v}"
  port=$((HOST_PORT_BASE + idx))
  idx=$((idx + 1))
  echo "### PostgreSQL ${v}: starting ${name} on host port ${port}"

  docker rm -f "$name" >/dev/null 2>&1 || true
  # The pgvector image ships pgvector preinstalled for that PostgreSQL major.
  docker run -d --name "$name" \
    -e POSTGRES_PASSWORD="$PGPASSWORD_VALUE" \
    -e POSTGRES_DB=root \
    -p "${port}:5432" \
    "pgvector/pgvector:${PGVECTOR_TAG:+${PGVECTOR_TAG}-}pg${v}" >/dev/null

  # Wait for the server to accept connections.
  echo -n "    waiting for readiness"
  for _ in $(seq 1 60); do
    if docker exec "$name" pg_isready -U postgres -d root >/dev/null 2>&1; then
      echo " ok"; break
    fi
    echo -n "."; sleep 1
  done

  DSN="host=127.0.0.1 port=${port} user=postgres password=${PGPASSWORD_VALUE} dbname=root"
  OUT="$OUT_ROOT/pg${v}"
  mkdir -p "$OUT"

  # Run the debugger-free tests against this container from the host interpreter.
  # (ariadne.pgtests only needs psycopg2 + numpy; the DSN points at the container.)
  ( cd "$REPO_ROOT"
    python3 -m ariadne.pgtests.locks  --dsn "$DSN" --out "$OUT/locks.jsonl"         || echo "    locks failed on pg${v}"
    python3 -m ariadne.pgtests.hammer --dsn "$DSN" --pairs 600 --out "$OUT/hammer.jsonl"      || echo "    hammer failed on pg${v}"
    python3 -m ariadne.pgtests.hammer --dsn "$DSN" --grid --pairs 600 --out "$OUT/hammer_grid.jsonl" || echo "    hammer --grid failed on pg${v}"
    python3 -m ariadne.pgtests.gist   --dsn "$DSN" --out "$OUT/gist.jsonl"          || echo "    gist failed on pg${v}"
  )

  echo "    results in $OUT"
  docker rm -f "$name" >/dev/null 2>&1 || true
done

echo "### done. per-version JSONL under $OUT_ROOT"
