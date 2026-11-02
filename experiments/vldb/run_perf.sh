#!/usr/bin/env bash
# Performance experiments: everything whose numbers in the main text are times or rates.
# bash experiments/vldb/run_perf.sh            (from the repository root, on the dedicated machine, nothing else running)
set -uo pipefail
OUT=${OUT:-results_vldb_perf}; mkdir -p "$OUT" logs
bash experiments/vldb/record_env.sh "$OUT"
ulimit -n 65536 || true
R() { echo "### $(date -u +%FT%TZ) $*" | tee -a logs/vldb_perf.log; python3 experiments/run.py "$@" --out "$OUT" 2>&1 | tee -a logs/vldb_perf.log; }
# Patched pgvector (Section 3.2), 30,000 pairs.  Needs the manual library swap that the first command prints:
#   python3 -m ariadne.pgtests.patched            (prints the build, install and restore commands; follow them, then)
#   PATCHED=1 bash experiments/vldb/run_perf.sh   (runs only this step; restore the stock library afterwards)
if [ "${PATCHED:-0}" = 1 ]; then
  python3 -m ariadne.pgtests.patched --run --pairs 30000 --trials 8 --expect-clean \
          --out "$OUT/A6_ssi_soundness/patched_library.jsonl" 2>&1 | tee -a logs/vldb_perf.log
  exit 0
fi
# Smoke check of the store and of PostgreSQL at 256 connections (minutes)
R experiments/C5_latency_throughput.yaml --only workload.agents=256 --only proto=av-cadm1.1 --only think=0.2s --repeat 1
# Main runs, in this order (the first three carry the main-text numbers)
R experiments/C5_latency_throughput.yaml          # Section 6.6, Figure 7, both thinking times; about 59 h
# C1 (Sections 6.2 and 6.3, Table 4), the same cells as the paper: all predicates on eight workloads, eight seeds on L2,
# L4 and A64; MS MARCO (MSM64, now on pgvector) with none, exact and c in {1.05, 1.1, 1.2}; about 22 h plus the
# index build of 10^6 rows per MS MARCO run
for c in L2 L4 L8 A16 A64 A64k50 LME4 QQP64; do R experiments/C1_calls_vs_policy.yaml --only case=$c; done
for c in L2 L4 A64; do R experiments/C1_calls_vs_policy.yaml --repeat 8 --only case=$c; done
for p in none exact cadm:1.05 cadm:1.1 cadm:1.2; do R experiments/C1_calls_vs_policy.yaml --only case=MSM64 --only protocol.policy=$p; done
R experiments/C10_vs_pessimistic.yaml             # Section 6.6, ball locks against validation; about 9 h
R experiments/A5_ssi_cost.yaml                    # Table 3 (aborts, latency); about 6 h
R experiments/A6_ssi_soundness.yaml               # Section 3.2 (stress test, insertion time; needs gdb and the patched build)
R experiments/G1_critical_section.yaml            # Section 4.3, critical-section breakdown; about 1 h
R experiments/E4_validation_budget.yaml           # Table 7, critical-section time; under 1 h
R experiments/E1_predicate_components.yaml       # Table 7, critical-section column of the predicate rows; about 1 h
R experiments/E2_lock_scope.yaml                 # Table 7, critical-section column of the lock rows; under 1 h
R experiments/F4_client_processes.yaml            # threads against processes; about 1 h
# the remaining experiments whose figures or tables show times or rates (full version), so that every time and rate of
# the paper comes from this machine; about 6 h
R experiments/B3_multi_read.yaml                 # several reads per transaction (full version)
R experiments/C13_read_mix.yaml                  # read-only share (full version)
R experiments/E3_retry_strategy.yaml             # full version, Table A6 and the retry figure
R experiments/E5_isolation_of_phases.yaml        # full version, Table A6
R experiments/E6_per_decision_tolerance.yaml     # full version, Table A6
R experiments/F6_failure_injection.yaml          # full version: throughput and tail latency under client failures
: "${QDRANT_BIN:?run experiments/vldb/setup_stores.sh first and export QDRANT_BIN and WEAVIATE_HOST}"
R experiments/F1_stores.yaml                     # full version: portability table and store figure
# The timing figures of the full version (think time, open loop, scale, long run); about 18 h.  With them every
# time and rate of the paper and of the full version comes from this machine.  WITH_APPENDIX=0 skips them.
if [ "${WITH_APPENDIX:-1}" = 1 ]; then
  R experiments/C8_think_and_arrival.yaml; R experiments/C8b_open_loop.yaml; R experiments/C9_scalability.yaml
  R experiments/F5_long_run.yaml
fi
# The count experiments, run on this machine with the same code, so that every number of the paper comes from one
# code version; about 30 h.  WITH_RECOUNT=0 skips them.
if [ "${WITH_RECOUNT:-1}" = 1 ]; then
  for e in C6_index_operating_points C7_choosing_c B2_separation C12_k_tau_sensitivity F2_embedders_and_datasets \
           F3_seeds_and_variance C11_starvation A3_race_realtext A4_anomaly_taxonomy B6_completeness B7_characterization \
           D2a_planted_law D2b_planted_analysis D5_heterogeneous_dimension D5b_heterogeneous_analysis G2_uncontended_overhead \
           G3_micro; do R experiments/$e.yaml; done
fi
# F2 and D6 with the three other embedding models, each into ${OUT}_<tag>/ (tau_cos 0.1, as in every other run);
# needs data/raw/train.csv (AG News) and data/raw/locomo10.json, and OPENAI_API_KEY for the OpenAI embedder; about 15 h
# including the embedding of the data.  WITH_EMBEDDERS=0 skips them.
if [ "${WITH_EMBEDDERS:-1}" = 1 ]; then
  [ -f data/raw/train.csv ] || { echo "embedders: data/raw/train.csv (AG News) is missing" | tee -a logs/vldb_perf.log; exit 2; }
  : "${OPENAI_API_KEY:?the embedder step needs OPENAI_API_KEY for openai:text-embedding-3-small}"
  for spec in st:BAAI/bge-small-en-v1.5 st:sentence-transformers/all-MiniLM-L6-v2 openai:text-embedding-3-small; do
    tag=$(echo "$spec" | sed 's#:#-#; s#/#_#g')
    echo "### $(date -u +%FT%TZ) prepare data with $spec" | tee -a logs/vldb_perf.log
    RAW=data/raw EMB_SPEC=$spec bash experiments/prepare_data.sh 2>&1 | tee -a logs/vldb_perf.log
    mkdir -p "${OUT}_$tag"; cp -r "$OUT/ENV" "${OUT}_$tag/" 2>/dev/null || true
    for e in F2_embedders_and_datasets D6_embedders_and_k D6b_embedders_analysis; do
      echo "### $(date -u +%FT%TZ) EMB=$tag $e" | tee -a logs/vldb_perf.log
      EMB=$tag python3 experiments/run.py experiments/$e.yaml --out "${OUT}_$tag" 2>&1 | tee -a logs/vldb_perf.log
    done
  done
fi
echo "### done $(date -u +%FT%TZ)" | tee -a logs/vldb_perf.log
