"""PostgreSQL SERIALIZABLE over pgvector (Section 3.2, 3.3 and the appendix on PostgreSQL).

    A5.*  A5_ssi_cost (engine): Table "tab:ssi" and the sentences around it, all at the case fan-c16
    A6.*  A6_ssi_soundness (script): predicate locks, debugger reproduction, stress test, controls, patched library
    A7.*  A7_gist_missed_conflict (script): missed conflicts of GiST without a debugger
    A8.*  A8_keyed_control (script): the control row "B-tree key, serializable" of Table "tab:ssi"

Fields are those of the generated tables tab_serializable and tab_ssi_soundness (ariadne/plots/tables.py) and of
fig_s_06 / fig_s_07: aborts_per_task, llm_calls_per_task, lat_p50_s, lat_p99_s, dup_per_100_events, gave_up;
events_with_duplicate / events_with_rows with ld.wilson; hole_both_committed / hole_trials; both_committed / trials.
"""
from ariadne.plots import load as ld                 # wilson(): the interval that tab_ssi_soundness and fig_s_06 print

A5, A6, A7, A8 = "A5_ssi_cost", "A6_ssi_soundness", "A7_gist_missed_conflict", "A8_keyed_control"

# ---- A5: Table "tab:ssi" is "at 16 concurrent tasks (three seeds)" with a column "dupl./100": the fan-out case
# with workload.conc = 16.  (The stream cases A16 and L4 have no dup_per_100_events at all.)
A5_CASE = "fan-c16"
A5_MODES = (("rc", "rc"), ("ssi", "ssi"), ("ssibo", "ssi-backoff"), ("ssishort", "ssi-short"),
            ("locklong", "lock-long"), ("lockshort", "lock-short"), ("av110", "av-cadm1.1"))
# aborts / aborts1 / aborts2 are one quantity printed with 0 / 1 / 2 decimals (a key has one shape in the text)
A5_FIELDS = (("dup", "dup_per_100_events"), ("aborts", "aborts_per_task"), ("aborts1", "aborts_per_task"),
             ("aborts2", "aborts_per_task"), ("calls", "llm_calls_per_task"), ("p50", "lat_p50_s"), ("p99", "lat_p99_s"))

# ---- A6: the stress test of the text is the default index of the experiment command (pgvector defaults)
HNSW_DEFAULT = {"mode": "plain", "index": "hnsw", "m": 16, "ef_construction": 64}

# ---- A7: the sentence describes "both search, both insert, both commit": this order.  The script also runs the
# variant "r1 r2 w1 c1 w2 c2" (same number of trials); its count goes to the provenance only.
A7_ORDER = "r1 r2 w1 w2 c1 c2"
A7_VARIANT = "r1 r2 w1 c1 w2 c2"
A7_CELLS = (("out.eq", "outside all data", "gist equality"), ("out.range", "outside all data", "gist range"),
            ("out.btree", "outside all data", "btree equality"), ("out.knn", "outside all data", "gist knn"),
            ("in.eq", "inside data region", "gist equality"))

# ---- A8: keyed.jsonl has SERIALIZABLE lines for two table sizes, the text has one row.  The small table
# (kmem_s: --base 10000 in the experiment file) has the size of the store of A5 (n_base 10 000).
A8_TABLE = "kmem_s"


# ------------------------------------------------------------------------------------------------ helpers
def _show(where):
    return "[" + ", ".join("%s=%s" % (a, b) for a, b in (where or {}).items() if not callable(b)) + "]"


def _last(R, exp, file, where=None, pred=None, label=None):
    """Last line of a script output that matches `where` (a run that was appended twice: the last line wins).
    Returns (row or None, text); text is the start of the provenance, or the reason when there is no row."""
    tag = "%s/%s %s" % (exp, file, label or _show(where))
    if not R.exists(exp, file):
        return None, tag + ": the file does not exist"
    rows = R.sel(exp, where, file, pred)
    if not rows:
        return None, tag + ": no line matches"
    return rows[-1], tag + (" (last of %d matching lines)" % len(rows) if len(rows) > 1 else "")


def _num(R, row, field):
    """Numeric value of a field of a line, None when the line or the field is missing."""
    if row is None:
        return None
    v = ld.num(R.get(row, field))
    return v if v == v else None


def _field(R, key, row, field, src, note=""):
    """Put one field of a line; the reason names the field when it is missing."""
    v = _num(R, row, field)
    if row is None:
        return R.put(key, None, src)
    if v is None:
        err = (" (the line records an error: %s)" % row["error"]) if row.get("error") else ""
        return R.put(key, None, "%s: no field %s%s" % (src, field, err))
    return R.put(key, v, "%s: %s%s" % (src, field, note), n=1)


def _sha(R, row):
    """sha256 of the pgvector library that produced a line (label of patched.py, else the versions block)."""
    return None if row is None else (row.get("lib_sha256") or R.get(row, "versions.pgvector_lib_sha256"))


def _common(values):
    """The single value shared by all entries that are not None; (None, reason) otherwise."""
    have = {k: v for k, v in values.items() if v is not None}
    if not have:
        return None, "no line with a field trials"
    if len(set(have.values())) > 1:
        return None, "the lines disagree on trials (%s): one denominator cannot serve all of them" % ", ".join(
            "%s: %g" % kv for kv in sorted(have.items()))
    return next(iter(have.values())), "trials, the same in the %d lines used (%s)" % (len(have), ", ".join(sorted(have)))


# ------------------------------------------------------------------------------------------------ A5
def _a5(R):
    for mode, proto in A5_MODES:
        where = {"case": A5_CASE, "proto": proto}
        for name, field in A5_FIELDS:                         # mean over the seeds, as in tab_serializable; latency in seconds
            R.stat("A5.%s.%s" % (mode, name), A5, field, where)
    # "<gaveup> of about <tasks> tasks give up": both per run (mean over the seeds), the 'gave up' column of tab_serializable
    where = {"case": A5_CASE, "proto": "ssi"}
    R.stat("A5.ssi.gaveup", A5, "gave_up", where)
    R.stat("A5.ssi.tasks", A5, "tasks", where)


# ------------------------------------------------------------------------------------------------ A6
def _hammer(R, file, where):
    """(k, n, row, src) of the last matching line of a hammer file: facts stored twice and facts with a row,
    the two fields behind the "both commit" column of tab_ssi_soundness and the points of fig_s_06."""
    row, src = _last(R, A6, file, where)
    if row is None:
        return None, None, None, src
    k, n = _num(R, row, "events_with_duplicate"), _num(R, row, "events_with_rows")
    if k is None or not n:
        return None, None, row, src + ": no field events_with_duplicate / events_with_rows (or no fact stored)"
    asked = _num(R, row, "pairs")
    if asked is not None and asked != n:
        src += " (pairs started: %g, facts with a row: %g)" % (asked, n)
    return k, n, row, src


def _pick(t):
    """(k, n, row, src) -> (k, n, src)."""
    return t[0], t[1], t[3]


def _counts(R, prefix, k, n, src):
    R.put(prefix + ".dup", k, src + (": events_with_duplicate" if k is not None else ""), n=1)
    R.put(prefix + ".pairs", n, src + (": events_with_rows" if n is not None else ""), n=1)


def _a6_hammer(R):
    """Stress test at the default index, its two controls, and the largest rate of the (m, ef_construction) grid."""
    k, n, stock, src = _hammer(R, "hammer.jsonl", HNSW_DEFAULT)
    p, lo, hi = ld.wilson(k, n) if n else (None, None, None)
    how = "(k = events_with_duplicate = %g, n = events_with_rows = %g; ld.wilson as in tab_ssi_soundness), percent" % (k, n) if n else ""
    R.put("A6.hammer.rate", None if p is None else 100 * p, src + (": 100 * k / n " + how if n else ""), n=1)
    R.put("A6.hammer.lo", None if p is None else 100 * lo, src + (": lower end of the Wilson 95% interval " + how if n else ""), n=1)
    R.put("A6.hammer.hi", None if p is None else 100 * hi, src + (": upper end of the Wilson 95% interval " + how if n else ""), n=1)
    R.put("A6.hammer.pairs", n, src + (": events_with_rows (the denominator of the rate)" if n else ""), n=1)

    _counts(R, "A6.advisory", *_pick(_hammer(R, "hammer_advisory.jsonl", dict(HNSW_DEFAULT, mode="advisory"))))
    _counts(R, "A6.seqscan", *_pick(_hammer(R, "hammer_seqscan.jsonl", {"mode": "plain", "index": "none"})))

    # ---- grid: the largest rate over the (m, ef_construction) points and the mean insertion time at that point
    file, where = "hammer_grid.jsonl", {"mode": "plain", "index": "hnsw"}
    tag = "%s/%s %s" % (A6, file, _show(where))
    points = {}
    for r in R.sel(A6, where, file):
        points[(r.get("m"), r.get("ef_construction"))] = r    # a point measured twice: the last line wins
    cand = []
    for (m, efc), r in points.items():
        kk, nn = _num(R, r, "events_with_duplicate"), _num(R, r, "events_with_rows")
        if kk is not None and nn:
            cand.append((100.0 * kk / nn, _num(R, r, "insert_ms_mean"), m, efc, kk, nn))
    if not cand:
        why = tag + (": no line with events_with_duplicate / events_with_rows" if R.exists(A6, file) else ": the file does not exist")
        R.put("A6.grid.max.rate", None, why); R.put("A6.grid.max.ms", None, why)
        return stock
    rate, ms, m, efc, kk, nn = max(cand, key=lambda t: (t[0], -1.0 if t[1] is None else t[1]))
    at = "m=%s, ef_construction=%s (%g of %g)" % (m, efc, kk, nn)
    R.put("A6.grid.max.rate", rate, "%s: max over %d (m, ef_construction) points of 100 * events_with_duplicate / events_with_rows, at %s"
          % (tag, len(cand), at), n=len(cand))
    slow = max((t for t in cand if t[1] is not None), key=lambda t: t[1], default=None)
    note = "" if slow is None or (slow[2], slow[3]) == (m, efc) else \
        "; NOTE the longest insertion of the grid is %g ms at m=%s, ef_construction=%s (rate %.1f%%)" % (slow[1], slow[2], slow[3], slow[0])
    R.put("A6.grid.max.ms", ms, "%s: insert_ms_mean (ms) of the point with the largest rate, %s%s%s"
          % (tag, at, note, "" if ms is not None else ": no field insert_ms_mean"), n=1)
    return stock


def _a6_patched(R, stock):
    """A6.patched.*: the stress test with the patched library, from the manual step only (patched_library.jsonl).
    Refused unless that library is provably another one than the installed (stock) library."""
    keys = ("A6.patched.dup", "A6.patched.pairs")
    file = "patched_library.jsonl"
    k, n, row, src = _hammer(R, file, dict(HNSW_DEFAULT, test="hammer"))
    why = None
    if not R.exists(A6, file):
        why = "%s/%s does not exist: it is written by the manual step with the patched library (see the doc of %s)" % (A6, file, A6)
    elif n is None:
        why = src
    else:
        sha = _sha(R, row)
        inst = sorted({s for s in (_sha(R, r) for r in R.rows(A6, "installed_library.jsonl")) if s})
        if not sha:
            why = src + ": the line records no library sha256, so it cannot be told from the stock library"
        elif not inst:
            why = "%s/installed_library.jsonl is missing or records no sha256: cannot show that %s was measured with another library" % (A6, file)
        elif sha in inst:
            why = "%s: refused, its library sha256 %s.. is the one of installed_library.jsonl (the stock library measured twice)" % (src, sha[:12])
        elif sha == _sha(R, stock):
            why = "%s: refused, its library sha256 %s.. is the one behind A6.hammer.* in hammer.jsonl" % (src, sha[:12])
        else:
            hole, _ = _last(R, A6, file, {"test": "hole", "index": "hnsw", "scenario": "SUMMARY"})
            paused = "no hole summary in the file" if hole is None else "paused scenario with this library: both committed in %s of %s" % (
                hole.get("hole_both_committed"), hole.get("hole_trials"))
            src += " (library sha256 %s.., installed %s..; %s)" % (sha[:12], inst[-1][:12], paused)
    if why:
        for key in keys:
            R.put(key, None, why)
        return
    _counts(R, "A6.patched", k, n, src)


def _a6_debugger(R):
    """Predicate locks of one read, the paused reproduction on HNSW, and the same pause in other access methods."""
    row, src = _last(R, A6, "locks.jsonl", {"index": "hnsw"})
    _field(R, "A6.locks.hnsw.pages", row, "index_pages", src,
           "" if row is None else " (relpages of the HNSW index of the lock test; siread_locks of the read: %s)" % (row.get("siread_locks"),))

    # tab_ssi_soundness prints <x>_both_committed; the sentence says "stores both rows" (<x>_duplicates): they must agree
    row, src = _last(R, A6, "hole.jsonl", {"index": "hnsw", "scenario": "SUMMARY"})
    for key, pre in (("hole", "hole"), ("control", "control")):
        both, dup = _num(R, row, pre + "_both_committed"), _num(R, row, pre + "_duplicates")
        if both is not None and dup is not None and both != dup:
            R.put("A6.%s.dup" % key, None, "%s: %s_both_committed = %g but %s_duplicates = %g: the table (both commit) and the sentence "
                  "(stores both rows) would differ" % (src, pre, both, pre, dup))
        else:
            _field(R, "A6.%s.dup" % key, row, pre + "_both_committed", src, " (equal to %s_duplicates)" % pre if dup is not None else "")
        _field(R, "A6.%s.trials" % key, row, pre + "_trials", src)

    trials = {}
    for am in ("spgist", "brin", "btree"):
        row, src = _last(R, A6, "generic.jsonl", {"am": am})
        det = [d for d in (row.get("detail") or []) if isinstance(d, dict)] if row else []
        _field(R, "A6.generic." + am, row, "both_committed", src,
               " (trials: %s; breakpoint hit in %d)" % (row.get("trials"), sum(1 for d in det if d.get("paused"))) if row else "")
        trials[am] = _num(R, row, "trials")
    v, how = _common(trials)
    R.put("A6.generic.trials", v, "%s/generic.jsonl [am=spgist, brin, btree]: %s" % (A6, how), n=1)


# ------------------------------------------------------------------------------------------------ A7
def _a7(R):
    def case_is(name):
        return lambda v: " ".join(str(v).lower().split()).startswith(name)
    trials = {}
    for key, region, case in A7_CELLS:
        label = "[region=%s, case=%s*, order=%s]" % (region, case, A7_ORDER)
        row, src = _last(R, A7, "gist.jsonl", {"region": region, "order": A7_ORDER, "case": case_is(case)}, label=label)
        var, _ = _last(R, A7, "gist.jsonl", {"region": region, "order": A7_VARIANT, "case": case_is(case)})
        note = "" if row is None else " (of trials = %s, both_saw_nothing = %s; variant order %s, not used: %s)" % (
            row.get("trials"), row.get("both_saw_nothing"), A7_VARIANT,
            "not run" if var is None else "%s of %s" % (var.get("both_committed_duplicate"), var.get("trials")))
        _field(R, "A7." + key, row, "both_committed_duplicate", src, note)
        trials[key] = _num(R, row, "trials")
    v, how = _common(trials)
    R.put("A7.trials", v, "%s/gist.jsonl [order=%s]: %s" % (A7, A7_ORDER, how), n=1)


# ------------------------------------------------------------------------------------------------ A8
def _a8(R):
    # one line per (isolation level, table, seed); a seed that was run again: the last line wins; mean over the seeds
    where = {"iso": "SERIALIZABLE", "table": A8_TABLE}
    tag = "%s/keyed.jsonl %s" % (A8, _show(where))
    if not R.exists(A8, "keyed.jsonl"):
        for key in ("A8.kser.dup", "A8.kser.aborts"):
            R.put(key, None, tag + ": the file does not exist")
        return
    last = {}
    for r in R.sel(A8, where, "keyed.jsonl"):
        last[R.get(r, "params.seed")] = r
    rows = list(last.values())
    note = " over %d seeds (%s); table_rows about %s, think %s s, arrival window %s thinking times" % (
        len(rows), ", ".join(str(x) for x in sorted(last, key=str)), rows[0].get("table_rows") if rows else "n/a",
        R.get(rows[0], "params.think_s") if rows else "n/a", R.get(rows[0], "params.jitter_x") if rows else "n/a")
    for key, field in (("A8.kser.dup", "dup_per_100_events"), ("A8.kser.aborts", "aborts_per_task")):
        v = R.mean(rows, field)
        R.put(key, v, "%s: mean(%s)%s" % (tag, field, note) if v is not None else tag + ": no line matches", n=len(rows))


def resolve(R):
    _a5(R)
    stock = _a6_hammer(R)
    _a6_patched(R, stock)
    _a6_debugger(R)
    _a7(R)
    _a8(R)
