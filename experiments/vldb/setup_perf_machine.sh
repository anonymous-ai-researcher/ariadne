#!/usr/bin/env bash
# One-time setup of the dedicated performance machine (run as root once; general Linux commands, not part of the framework).
# Ubuntu 24.04, PostgreSQL 16 from the Ubuntu archive, pgvector 0.8.7 built from source.
set -euo pipefail
apt-get update
apt-get install -y build-essential git gdb mdadm linux-tools-common linux-tools-$(uname -r) python3-venv
# local NVMe of c6id.metal as one RAID-0 volume for the PostgreSQL data directory (the root volume is EBS)
DEVS=$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1}')
if [ -n "$DEVS" ] && [ ! -e /dev/md0 ]; then
  mdadm --create /dev/md0 --level=0 --raid-devices=$(echo $DEVS | wc -w) $DEVS --run
  mkfs.ext4 -F /dev/md0 && mkdir -p /data && mount /dev/md0 /data
fi
# PostgreSQL 16.15, as in the paper; if this exact build is gone, install the current 16.x and record it
apt-get install -y postgresql-16=16.15-0ubuntu0.24.04.1 postgresql-client-16=16.15-0ubuntu0.24.04.1 postgresql-server-dev-16=16.15-0ubuntu0.24.04.1 \
  || apt-get install -y postgresql-16 postgresql-client-16 postgresql-server-dev-16
if mountpoint -q /data; then
  systemctl stop postgresql; mkdir -p /data/pg; rsync -a /var/lib/postgresql/ /data/pg/
  sed -i "s|^data_directory.*|data_directory = '/data/pg/16/main'|" /etc/postgresql/16/main/postgresql.conf; systemctl start postgresql
fi
git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git /opt/pgvector && make -C /opt/pgvector && make -C /opt/pgvector install
# CPU: fixed frequency, no turbo, no SMT (SMT can also be disabled in the BIOS; 'nosmt' on the kernel command line survives reboots)
cpupower frequency-set -g performance || true
[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ] && echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo || true
[ -e /sys/devices/system/cpu/cpufreq/boost ] && echo 0 > /sys/devices/system/cpu/cpufreq/boost || true
echo off > /sys/devices/system/cpu/smt/control || true
sysctl -w vm.swappiness=1 kernel.yama.ptrace_scope=0
# PostgreSQL settings; memory values for a 256 GB machine
sudo -u postgres psql -c "ALTER SYSTEM SET max_connections = 1000"
sudo -u postgres psql -c "ALTER SYSTEM SET shared_buffers = '64GB'"
sudo -u postgres psql -c "ALTER SYSTEM SET effective_cache_size = '192GB'"
sudo -u postgres psql -c "ALTER SYSTEM SET maintenance_work_mem = '16GB'"
sudo -u postgres psql -c "ALTER SYSTEM SET max_parallel_maintenance_workers = 8"
sudo -u postgres psql -c "ALTER SYSTEM SET log_min_messages = 'fatal'"
sudo -u postgres psql -c "ALTER SYSTEM SET log_min_error_statement = 'panic'"
systemctl restart postgresql
sudo -u postgres psql -c "CREATE ROLE $SUDO_USER SUPERUSER LOGIN" || true
sudo -u postgres createdb -O "$SUDO_USER" root || true
echo "ulimit: add '$SUDO_USER - nofile 65536' to /etc/security/limits.conf and log in again"
