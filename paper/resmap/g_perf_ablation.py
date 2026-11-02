"""C5.*, C10.*, C11.* (Section 7.6, throughput and pessimistic alternatives) and E1.* .. E4.* (Section 7.7, Table tab:abl).

    C5.*   C5_latency_throughput    64 and 256 agents at thinking time 0.2 s (the setting named in the caption of fig:scale)
    C10.*  C10_vs_pessimistic       case A64: ball predicate locks (ball-<c>) against certified validation (av-<c> = cert:<c>);
                                    C10.A64.<design>.cut = extra calls at c = 1.0 / extra calls at c = 1.1
    C11.*  C11_starvation           case fan16-hot, round limit 30: largest number of rounds of any task
    E1.*   E1_predicate_components  case A64, one row per predicate at tolerance 1.1
    E2.*   E2_lock_scope            case A64rr8 (AG News in eight teams, the block header of the table), cadm:1.1, one row per lock scope
    E3.*   E3_retry_strategy        case A64, cadm:1.1, round limit 30: extra calls of "reread" beyond "rethink"
    E4.*   E4_validation_budget     case A64 (policy cadm:1.1 is fixed by the experiment), one row per validation budget

Fields are those of ariadne/plots/tables.py: the ablation columns Calls, Dup., link and rho_99 are the columns of
Table tab:cost (tab_cost: extra_calls_per_100, dup_written_per_100, div_committed_vs_exact_%.link, rho_committed.p99),
"c.s." is cs_mean_ms (tab_ablation).  Every value is the mean over the seeds of one cell, except C11 (a maximum).
"""

SUMMARY, AUDIT = "summary.jsonl", "audit_run.jsonl"

# column of Table tab:abl -> (file, field); units: per 100 tasks, per 100 tasks, percent, ratio, milliseconds
ABL = (("extra", SUMMARY, "extra_calls_per_100"),
       ("dupw", AUDIT, "dup_written_per_100"),
       ("link", AUDIT, "div_committed_vs_exact_%.link"),
       ("rho99", AUDIT, "rho_committed.p99"),
       ("cs", SUMMARY, "cs_mean_ms"))

C5 = "C5_latency_throughput"
C5_THINK = "0.2s"
C5_PROTO = {"none": "none", "ssibo": "ssi-backoff", "exact": "av-exact", "c110": "av-cadm1.1"}

C10 = "C10_vs_pessimistic"
# Lock scope: the AG News stream split round robin into eight teams of eight agents on one shared store, so that
# writers of different teams produce items about the same thing (on LoCoMo the teams are different conversations).
E2_CASE = "A64rr8"
C10_PROTO = {"ball10": "ball-1.0", "ball11": "ball-1.1", "av10": "av-1.0", "av11": "av-1.1"}

E1_POLICY = {"c110": "cadm:1.1", "radius110": "radius:1.1", "rank1110": "rank1:1.1", "sum110": "sumdist:1.1"}
E2_LOCK = {"global": "global", "group": "group", "nolock": "none"}
E4_VAL = ("ef10", "ef40", "ef400", "exact")


def show(where):
    return "[" + ", ".join("%s=%s" % kv for kv in where.items()) + "]"


def seeds_of(R, exp, where, file=SUMMARY):
    """Rows of one cell (its seeds).  Audit rows are kept only for the cells that summary.jsonl has, because the
    generated tables join the audit to the summary by cell."""
    rows = R.sel(exp, where, file)
    if file != SUMMARY:
        cells = {r.get("cell") for r in R.sel(exp, where, SUMMARY)}
        rows = [r for r in rows if r.get("cell") in cells]
    return rows


def aggregate(R, key, exp, where, field, file=SUMMARY, how="mean", scale=1.0, note="", put=True):
    """Aggregate `field` over the seeds of the cell `where` and define `key`.  Returns (value, number of values)."""
    rows = seeds_of(R, exp, where, file)
    v = R.vals(rows, field)
    src = "%s/%s %s: %s(%s)%s over %d of %d rows%s" % (exp, file, show(where), how, field, "" if scale == 1 else " x %g" % scale,
                                                    len(v), len(rows), "; " + note if note else "")
    value = None
    if len({r.get("seed") for r in rows}) < len(rows):          # a grid parameter that the selector does not fix
        src += " -> the selector covers more than one cell (a seed occurs twice): fix the new grid parameter in the selector"
    elif not rows:
        src += " -> no rows"
    elif not v:
        src += " -> the field is missing or not a number in every row"
    else:
        value = scale * (max(v) if how == "max" else sum(v) / len(v))
    if put:
        R.put(key, value, src, n=len(v))
    return value, len(v)


def ablation_row(R, prefix, exp, where):
    for col, file, field in ABL:
        aggregate(R, "%s.%s" % (prefix, col), exp, where, field, file)


def resolve(R):
    # ---- C5: scaling (fig_s_11 plots tps and lat_p99_s per number of agents, protocol and thinking time)
    for name, proto in C5_PROTO.items():
        aggregate(R, "C5.a64.%s.tps" % name, C5, {"workload.agents": 64, "proto": proto, "think": C5_THINK}, "tps")
    for name in ("exact", "c110"):
        aggregate(R, "C5.a64.%s.p99" % name, C5, {"workload.agents": 64, "proto": C5_PROTO[name], "think": C5_THINK}, "lat_p99_s",
                  note="seconds, as in the text")
    for name, proto in (("c110", C5_PROTO["c110"]), ("cert110", "av-cert1.1")):
        aggregate(R, "C5.a256.%s.held" % name, C5, {"workload.agents": 256, "proto": proto, "think": C5_THINK}, "lock_held_fraction",
                  scale=100.0, note="fraction of the wall time -> percent")

    # ---- C10: optimistic against pessimistic (tab_opt_vs_pess, fig_s_18: extra_calls_per_100 and lock_wait_mean_ms)
    c10 = {}
    for name, proto in C10_PROTO.items():
        c10[name] = aggregate(R, "C10.A64.%s.extra" % name, C10, {"case": "A64", "proto": proto}, "extra_calls_per_100")
    # factor by which tolerance (c = 1.1 against c = 1.0) divides the extra calls of each design, from the cell means
    for design in ("ball", "av"):
        (strict, n0), (tol, n1) = c10[design + "10"], c10[design + "11"]
        src = "C10.A64.%s10.extra / C10.A64.%s11.extra (ratio of the two cell means)" % (design, design)
        if strict is None or tol is None:
            R.put("C10.A64.%s.cut" % design, None, src + " -> no data")
        elif tol <= 0:
            R.put("C10.A64.%s.cut" % design, None, src + " -> the tolerant cell has no extra calls, the factor is unbounded")
        else:
            R.put("C10.A64.%s.cut" % design, strict / tol, src, n=min(n0, n1))
    aggregate(R, "C10.A64.ball11.wait", C10, {"case": "A64", "proto": C10_PROTO["ball11"]}, "lock_wait_mean_ms",
              note="milliseconds; mean over all tasks of the time spent in the ball-lock manager at the write")

    # ---- C11: starvation (fig_s_19 pools the per-task rounds over seeds; its largest x is this maximum)
    for name, policy in (("exact", "exact"), ("c120", "cadm:1.2")):
        aggregate(R, "C11.fan16.%s.maxrounds" % name, "C11_starvation",
                  {"case": "fan16-hot", "protocol.policy": policy, "protocol.max_rounds": 30}, "max_rounds_seen", how="max",
                  note="largest over the seeds, not a mean")

    # ---- Table tab:abl
    for name, policy in E1_POLICY.items():
        ablation_row(R, "E1." + name, "E1_predicate_components", {"case": "A64", "protocol.policy": policy})
    for name, lock in E2_LOCK.items():
        ablation_row(R, "E2." + name, "E2_lock_scope", {"case": E2_CASE, "protocol.policy": "cadm:1.1", "protocol.lock": lock})
    for val in E4_VAL:
        ablation_row(R, "E4." + val, "E4_validation_budget", {"case": "A64", "val": val})

    # ---- E3: extra calls per 100 tasks of reading again outside the lock beyond thinking again on the validation answer
    exp, field = "E3_retry_strategy", "extra_calls_per_100"
    cell = {"case": "A64", "protocol.policy": "cadm:1.1", "protocol.max_rounds": 30}
    reread, n1 = aggregate(R, None, exp, dict(cell, **{"protocol.on_invalid": "reread"}), field, put=False)
    rethink, n2 = aggregate(R, None, exp, dict(cell, **{"protocol.on_invalid": "rethink"}), field, put=False)
    src = "%s/%s %s: mean(%s) of protocol.on_invalid=reread (%d rows) minus that of rethink (%d rows)" % (exp, SUMMARY, show(cell), field, n1, n2)
    if reread is None or rethink is None:
        R.put("E3.reread.more", None, src + " -> no data for %s" % " and ".join(s for s, x in (("reread", reread), ("rethink", rethink)) if x is None))
    else:
        R.put("E3.reread.more", reread - rethink, src, n=min(n1, n2))
