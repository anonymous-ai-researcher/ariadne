"""env.* (manual values, paper/env.yaml) and B5.* (fault injection into the logs of correct runs, B5b_selftest)."""

FAULTS = ("skip", "reorder", "ghost", "distance")


def resolve(R):
    # ---- manual values
    for k in ("pg", "pgvector", "mem0", "machine", "vm", "gpu", "embedder", "llm", "ndatasets", "nbaselines", "nembedders", "nllms"):
        v = R.env.get(k)
        R.put("env." + k, None if v in (None, "") else (v if isinstance(v, str) else float(v)), "paper/env.yaml: %s" % k if v not in (None, "") else "paper/env.yaml: %s is empty" % k)
    # ---- B5b: planted faults and how many the auditor found, summed over the audited logs
    S = R.json("B5b_selftest", "selftest.jsonl")
    last = {}
    for r in S:                                               # a log audited twice: the last line wins
        if isinstance(r, dict) and "run" in r:
            last[r["run"]] = r
    for k in FAULTS:
        rows = [r for r in last.values() if isinstance(r.get(k), dict)]
        for what in ("planted", "detected"):
            v = sum(int(r[k].get(what, 0)) for r in rows) if rows else None
            R.put("B5.%s.%s" % (k, what), None if v is None else float(v),
                  "B5b_selftest/selftest.jsonl: sum of %s.%s over %d logs" % (k, what, len(rows)), n=len(rows))
