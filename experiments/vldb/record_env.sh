#!/usr/bin/env bash
# Records the machine and software of a run directory: bash experiments/vldb/record_env.sh <results dir>
set -u
D=${1:?results directory}; mkdir -p "$D/ENV"; E="$D/ENV"
{ date -u +%FT%TZ; hostname; uname -a; cat /proc/cmdline; } > "$E/host.txt"
lscpu > "$E/lscpu.txt"; free -g > "$E/memory.txt"; df -h . > "$E/disk.txt"
cat /sys/devices/system/cpu/smt/active > "$E/smt_active.txt" 2>/dev/null
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$E/governor.txt" 2>/dev/null
cat /sys/devices/system/cpu/intel_pstate/no_turbo > "$E/no_turbo.txt" 2>/dev/null
cat /sys/devices/system/cpu/cpufreq/boost > "$E/boost.txt" 2>/dev/null
python3 --version > "$E/python.txt" 2>&1; python3 -m pip freeze > "$E/pip_freeze.txt" 2>/dev/null
python3 -m ariadne.provenance code > "$E/code_sha256.txt" 2>/dev/null
psql "${PG_DSN:-dbname=root}" -Atc "select version()" > "$E/postgres_version.txt" 2>/dev/null
psql "${PG_DSN:-dbname=root}" -Atc "select extversion from pg_extension where extname='vector'" > "$E/pgvector_version.txt" 2>/dev/null
psql "${PG_DSN:-dbname=root}" -Atc "select name||'='||setting from pg_settings where source not in ('default','override')" > "$E/postgres_settings.txt" 2>/dev/null
nvidia-smi > "$E/nvidia_smi.txt" 2>/dev/null
echo "recorded in $E"
