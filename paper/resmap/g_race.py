"""A1.* A2.* A3.* A4.* A9.*: the race without protection (Section 3.1, Table tab:race, and its appendix paragraph).

A1.<store>.<proto>      A1_race_stores: duplicates per 100 facts at 16 concurrent tasks (the level of Table tab:race)
A1.none.min / .max      range of the "no protection" column of that table over its eight rows
A1.nstores              number of distinct stores behind that range (pgvector at two isolation levels is one store)
A1.pgrc.none.c4 / .c32  the pgvector read-committed cell at 4 and 32 concurrent tasks
A2.jitter025 / 100      A2_race_sensitivity: duplicates per 100 facts at an arrival spread of 0.25 and of 100 thinking times
A3.qqp.<proto>          A3_race_realtext: the Quora Question Pairs row of Table tab:race (rows beyond the serial execution)
A4.<proto>.diff         A4_anomaly_taxonomy: share of tasks whose action differs from the serial execution in commit order
A9.*                    A9_memory_frameworks: Mem0 with a real model (mem0_real.jsonl; the mock run is never used)
A11.*                   A11_simple_baselines: write first, then verify (fan-out workload of Table tab:race)

Fields are those of ariadne/plots/tables.py (tab_race, tab_race_by_concurrency) and of fig_s_01, fig_s_02, fig_s_04,
fig_s_34: dup_per_100_events of summary.jsonl; excess_stored_per_100_events and different_action_per_100 of
audit_serial.jsonl.
"""

A1, A2, A3, A4, A9 = "A1_race_stores", "A2_race_sensitivity", "A3_race_realtext", "A4_anomaly_taxonomy", "A9_memory_frameworks"
A11 = "A11_simple_baselines"
DUP = "dup_per_100_events"

# key tag -> case of A1_race_stores, in the row order of Table tab:race
STORES = (("pgrc", "pgvector-rc"), ("pgrr", "pgvector-rr"), ("qdrant", "qdrant"), ("chroma", "chroma"),
          ("milvus", "milvus"), ("lancedb", "lancedb"), ("weaviate", "weaviate"), ("faiss", "faiss"))
# key tag -> value of the grid parameter "proto" (A1 and A3), in the column order of the table
PROTOS = (("none", "none"), ("reread", "reread"), ("lock", "reread+lock"))
TABLE_CONC = 16            # caption of Table tab:race: "at 16 concurrent tasks" (workload.conc of the full grid: 4, 16, 32)
MIX_CASE = "A64mix"        # the 60 / 20 / 10 / 10 operation mix of the A4 sentence on the main workload (see resolve_a4)
MEM0_FILE = "mem0_real.jsonl"


def resolve(R):
    resolve_a1(R)
    resolve_a2(R)
    resolve_a3(R)
    resolve_a4(R)
    resolve_a9(R)
    resolve_a11(R)


# ------------------------------------------------------------------------------------------------ A1: every store
def store_kinds(R):
    """{case: store kind} from the experiment file, so that the two pgvector cases count as one store."""
    return {c.get("_name"): (c.get("store") or {}).get("kind") for c in (R.exp(A1).get("cases") or [])}


def resolve_a1(R):
    unprotected = {}                                          # case -> mean over seeds, no protection, table level
    for tag, case in STORES:
        for ptag, proto in PROTOS:
            v = R.stat("A1.%s.%s" % (tag, ptag), A1, DUP, {"case": case, "proto": proto, "workload.conc": TABLE_CONC})
            if ptag == "none":
                unprotected[case] = v
    have = [c for _, c in STORES if unprotected[c] is not None]
    sel = "%s/summary.jsonl [proto=none, workload.conc=%d]" % (A1, TABLE_CONC)
    R.put_range("A1.none", [unprotected[c] for c in have],
                "%s: per case mean(%s) over seeds, then the range over the cases %s" % (sel, DUP, ", ".join(have) or "(none)"))
    kinds = store_kinds(R)
    unknown = [c for c in have if not kinds.get(c)]
    if not have:
        R.put("A1.nstores", None, "%s: no case has data" % sel)
    elif unknown:
        R.put("A1.nstores", None, "experiments/%s.yaml gives no store.kind for %s" % (A1, ", ".join(unknown)))
    else:
        stores = sorted({kinds[c] for c in have})
        R.put("A1.nstores", float(len(stores)),
              "%s: distinct store.kind (experiments/%s.yaml) of the %d cases with data: %s" % (sel, A1, len(have), ", ".join(stores)), n=len(have))
    for conc in (4, 32):
        R.stat("A1.pgrc.none.c%d" % conc, A1, DUP, {"case": "pgvector-rc", "proto": "none", "workload.conc": conc})


# ------------------------------------------------------------------------------------------------ A2, A3: factors, real text
def resolve_a2(R):
    # the case name carries the arrival spread of the copies in units of the thinking time (workload.jitter)
    R.stat("A2.jitter025", A2, DUP, {"case": "jitter0.25"})
    R.stat("A2.jitter100", A2, DUP, {"case": "jitter100"})


def resolve_a3(R):
    # Real paraphrases of one question can be farther apart than the threshold, so a serial execution also stores several
    # rows per cluster.  The row of the table is therefore the excess over the serial execution of the same tasks in
    # commit order (audit_serial.jsonl), which is 0 for every serial execution like the other rows; tab_race and
    # fig_s_04 use the same field.
    for ptag, proto in PROTOS:
        R.stat("A3.qqp.%s" % ptag, A3, "excess_stored_per_100_events", {"case": "qqp", "proto": proto}, file="audit_serial.jsonl")


# ------------------------------------------------------------------------------------------------ A4: mixed operations
def resolve_a4(R):
    # "% of the tasks take an action that no serial execution in commit order takes": tasks with a different action per
    # 100 tasks in the serial audit.  proto none is the instrumented run without validation, objects the validation that
    # the objects read still exist.  The sentence names the mix but not the stream: A64mix (AG News, the main workload).
    for proto in ("none", "objects"):
        R.stat("A4.%s.diff" % proto, A4, "different_action_per_100", {"case": MIX_CASE, "proto": proto}, file="audit_serial.jsonl")


# ------------------------------------------------------------------------------------------------ A9: Mem0
def mem0_runs(R):
    """{mode: rows, one per seed} of the real-model runs of ariadne.ext.mem0_race.  The script appends one line per
    mode and invocation: the last line of a (mode, seed) wins.  Lines of the mock model server are not results."""
    last = {}
    for r in R.json(A9, MEM0_FILE):
        if not isinstance(r, dict) or r.get("framework") != "mem0" or r.get("mode") is None:
            continue
        if R.get(r, "endpoint.mock"):
            continue
        last[(r["mode"], r.get("seed"))] = r
    modes = {}
    for (mode, _), r in last.items():
        modes.setdefault(mode, []).append(r)
    return modes


def resolve_a9(R):
    keys = (("A9.concurrent.dup", "concurrent", DUP), ("A9.serial.dup", "serial", DUP), ("A9.lock.dup", "lock", DUP),
            ("A9.adds", "concurrent", "add_calls"), ("A9.events", "concurrent", "events"))
    where = "%s/%s" % (A9, MEM0_FILE)
    if not R.exists(A9, MEM0_FILE):
        for key, _, _ in keys:
            R.put(key, None, "%s does not exist (written by the real-model command only; mem0_mock.jsonl is a pipeline check and is not used)" % where)
        return
    modes = mem0_runs(R)
    for key, mode, field in keys:
        rows = modes.get(mode, [])
        errors = R.total(rows, "add_errors")
        src = "%s [framework=mem0, mode=%s, real model]: mean(%s) over %d runs (last line per seed; add_errors in these runs: %s)" % (
            where, mode, field, len(rows), "n/a" if errors is None else "%d" % errors)
        v = R.mean(rows, field)
        R.put(key, v, src if v is not None else src + " -> no data", n=len(rows))


# ------------------------------------------------------------------------------------------------ A11: write first, then verify
def resolve_a11(R):
    # "write-verify-id": insert, then look for a near duplicate and delete the own item if the other one has the smaller id;
    # "write-verify-any": delete the own item whenever a near duplicate is seen (both copies can disappear: lost facts).
    for tag, proto in (("wvid", "write-verify-id"), ("wvany", "write-verify-any")):
        R.stat("A11.%s.dup" % tag, A11, DUP, {"case": "fan", "proto": proto})
    rows = R.sel(A11, {"case": "fan", "proto": "write-verify-any"})
    v = R.mean(rows, lambda r: 100.0 * r["lost_events"] / r["events"] if r.get("events") else None)
    R.put("A11.wvany.lost", v, "%s/summary.jsonl [case=fan, proto=write-verify-any]: mean(100 x lost_events / events) over %d rows%s"
          % (A11, len(rows), "" if v is not None else " -> no data"), n=len(rows))
