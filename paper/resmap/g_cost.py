"""C1.*, C2.*, C6.*, B1.*, B7.*, F1.*, F2.*, G1.*: what tolerant validation saves and what it costs.

  C1.<case>.<pol>.<metric>   C1_calls_vs_policy, the cell (case, protocol.policy), mean over seeds.  Same rows and
                             fields as tab_cost / tab_cost_all (ariadne/plots/tables.py):
                               extra   summary.jsonl    extra_calls_per_100             (calls per 100 tasks)
                               p99     summary.jsonl    lat_p99_s                       (seconds)
                               cs      summary.jsonl    cs_mean_ms                      (milliseconds, per task)
                               dupw    audit_run.jsonl  dup_written_per_100
                               tag, link, top3, set     div_committed_vs_exact_%.<proxy>   (percent of committed tasks)
                               rho99   audit_run.jsonl  rho_committed.p99
                               rhoF99  audit_run.jsonl  rho_fresh.p99
                               viol    audit_run.jsonl  rho_committed.">c"  (percent; c = tolerance of the policy)
                               violF   audit_run.jsonl  rho_fresh.">c"
  C1.exact.extra.min/max     range over the six main configurations (MAIN6) of the seed mean of extra at exact
  C1.save<ccc>.min/max       range over MAIN6 of 100 * (1 - extra[cadm:c] / extra[exact]), seed means per case
  C1.<case>.save110          the same saving for one case
  C1.linkadd<ccc>.min/max    range over MAIN6 of link[cadm:c] - link[exact] (percentage points; the row "exact" of
                             the cost table is the index floor)
  C2.changed / physonly      C2_counterfactual_pareto/audit_first.jsonl: changed_set_%, physical_only_change_%;
                             range over all cases of the experiment of the seed mean
  C6.A64.ef<n>.recall        C6_index_operating_points/audit_run.jsonl: recall_fresh of the runs that validate with
                             the read budget (val = same), HNSW M = 16 (the default index of the paper)
  C6.A64.ef10.save110        saving of cadm:1.1 (val = same) against re-execution (val = reexec) at the same budget
  B1.sepviol                 B1_safety_theorem/audit_sep.jsonl: stored pairs closer than tau / c in the certified insert-only runs
  B1.checked / violations    B1_safety_theorem/audit_run.jsonl: totals of safety_theorem.checked / .violations over
                             all audited runs (the "total" row of tab_safety)
  B7.reads / ok / tot        B7_characterization/audit_law.jsonl: totals of single_insertion_reads and of the two
                             numbers of "characterization" (agreeing / tested) over all audited runs
  F1.nstores / violations    F1_stores/audit_run.jsonl: number of store systems (store.kind of the case) with a
                             safety check, and the total of safety_theorem.violations (tab_portability)
  F2.save110.min/max         F2_embedders_and_datasets/summary.jsonl of every embedder: range over (embedder, case)
  F2.other.save110.min/max   the same range over the embedders other than the main one; F2.nother their number
  C6.A64.ef10.dsave          |saving at ef10 - saving at ef40| in C6 (A64, hnsw-M16), points
  G1.vread / pred / write    G1_critical_section/summary.jsonl, 16 agents, k = 10, ef = 40: cs_vread_mean_ms,
                             cs_predicate_mean_ms, cs_write_mean_ms (milliseconds)
"""
import glob, os

from ariadne.plots import load as ld                  # only for the extra result roots of F2 (see _f2)

SUM, RUN = "summary.jsonl", "audit_run.jsonl"
C1 = "C1_calls_vs_policy"

# predicate tag of the keys -> protocol.policy of the grid
POLICY = {"none": "none", "ov90": "overlap:0.9", "exact": "exact", "c102": "cadm:1.02", "c105": "cadm:1.05", "c110": "cadm:1.1",
          "c120": "cadm:1.2", "c150": "cadm:1.5", "cert110": "cert:1.1", "olink": "oracle:link"}
# metric tag -> (file, field)
FIELDS = {"extra": (SUM, "extra_calls_per_100"), "p99": (SUM, "lat_p99_s"), "cs": (SUM, "cs_mean_ms"),
          "dupw": (RUN, "dup_written_per_100"), "rho99": (RUN, "rho_committed.p99"), "rhoF99": (RUN, "rho_fresh.p99"),
          "tag": (RUN, "div_committed_vs_exact_%.tag"), "link": (RUN, "div_committed_vs_exact_%.link"),
          "top3": (RUN, "div_committed_vs_exact_%.top3"), "set": (RUN, "div_committed_vs_exact_%.set")}

# The LoCoMo and AG News configurations of Section 7.1.  The ranges of Sections 7.2 and 7.3 are taken over these six
# ("across the six configurations"); LME4, QQP64 and MSM64 are quoted one by one in the sentence that follows the ranges.
MAIN6 = ("L2", "L4", "L8", "A16", "A64", "A64k50")

# Table "tab:cost" (A64): rows and columns
TABLE_POLS = ("none", "ov90", "exact", "c102", "c105", "c110", "c120", "c150", "cert110")
TABLE_COLS = ("extra", "dupw", "tag", "link", "top3", "set", "rho99")
# single cells quoted in the text of Sections 7.2 and 7.3
CELLS = [("A64", "c105", "viol"), ("A64", "c105", "violF"), ("A64", "c110", "viol"), ("A64", "c110", "violF"),
         ("L4", "c110", "viol"), ("L4", "c110", "violF"), ("A64", "c110", "cs"), ("A64", "cert110", "cs"),
         ("A64", "exact", "p99"), ("A64", "c110", "p99"), ("L8", "exact", "p99"), ("L8", "c110", "p99"),
         ("A64", "exact", "rhoF99"), ("L4", "exact", "rhoF99"), ("L4", "exact", "dupw"), ("A64", "olink", "extra")]


# ------------------------------------------------------------------------------------------------ helpers
def _tolerance(spec):
    """c of cadm:c / cert:c, None for the other predicates."""
    name, _, arg = str(spec).partition(":")
    try:
        return float(arg) if name in ("cadm", "cert") else None
    except ValueError:
        return None


def _field(metric, spec):
    """(file, field) of a metric tag; (None, reason) when the metric is not defined for the predicate."""
    if metric in FIELDS:
        return FIELDS[metric]
    if metric in ("viol", "violF"):
        c = _tolerance(spec)
        if c is None:
            return None, "the share above the tolerance is defined for cadm:c and cert:c only, not for %s" % spec
        return RUN, ("rho_committed" if metric == "viol" else "rho_fresh") + ".>%s" % c     # the audit writes ">1.05", ">1.1"
    return None, "unknown metric %s" % metric


def _cell(R, exp, where, field, file=SUM):
    """(mean over the seeds of one cell, number of rows); the mean is None when the cell has no rows or no such field."""
    rows = R.sel(exp, where, file)
    return (R.mean(rows, field) if rows else None), len(rows)


def _cases(R, exp):
    """Case names of the full grid of an experiment, in the order of the experiment file."""
    out = []
    for p in R.full_params(exp):
        if p.get("case") is not None and p["case"] not in out:
            out.append(p["case"])
    return out


def _show(vals):
    return " ".join("%s=%s" % (c, "n/a" if v is None else "%.4g" % v) for c, v in vals.items())


def _range(R, key, vals, what, only_max=False):
    """key.min / key.max over the per-case values {case: value or None}; the per-case values go to the provenance."""
    src = "%s; per case: %s" % (what, _show(vals))
    if only_max:
        v = [x for x in vals.values() if x is not None]
        R.put(key + ".max", max(v) if v else None, src + " (max of %d)" % len(v), n=len(v))
    else:
        R.put_range(key, list(vals.values()), src)


def _coverage(R, exp, rows):
    return "%d rows, the full grid has %d cells" % (len(rows), len(R.full_params(exp)))


def _saving(R, exp, where, spec, ref="exact", key="protocol.policy"):
    """Percentage of the extra calls of the reference predicate that `spec` saves in one case: seed means first.
    Returns (value or None, smaller number of rows of the two cells)."""
    a, na = _cell(R, exp, dict(where, **{key: spec}), "extra_calls_per_100")
    b, nb = _cell(R, exp, dict(where, **{key: ref}), "extra_calls_per_100")
    return R.saved(a, b), min(na, nb)


# ------------------------------------------------------------------------------------------------ C1
def _c1(R):
    cells = [("A64", p, m) for p in TABLE_POLS for m in TABLE_COLS] + CELLS
    for case, pol, metric in cells:
        key = "C1.%s.%s.%s" % (case, pol, metric)
        file, field = _field(metric, POLICY[pol])
        if file is None:
            R.put(key, None, field)
            continue
        R.stat(key, C1, field, {"case": case, "protocol.policy": POLICY[pol]}, file)

    # extra calls of re-execution validation, and the share of them that a tolerance saves
    extra = {c: _cell(R, C1, {"case": c, "protocol.policy": "exact"}, "extra_calls_per_100")[0] for c in MAIN6}
    _range(R, "C1.exact.extra", extra, "%s/%s [protocol.policy=exact]: mean(extra_calls_per_100) over seeds" % (C1, SUM))
    for tag in ("c105", "c110", "c120"):
        spec = POLICY[tag]
        sav = {c: _saving(R, C1, {"case": c}, spec)[0] for c in MAIN6}
        _range(R, "C1.save" + tag[1:], sav,
               "%s/%s: 100 * (1 - mean(extra_calls_per_100)[%s] / mean(extra_calls_per_100)[exact]), means over seeds" % (C1, SUM, spec))
    for case in ("LME4", "QQP64", "MSM64"):
        v, n = _saving(R, C1, {"case": case}, "cadm:1.1")
        R.put("C1.%s.save110" % case, v, "%s/%s [case=%s]: 100 * (1 - mean(extra_calls_per_100)[cadm:1.1] / mean(extra_calls_per_100)[exact]), "
              "means over seeds (%d rows per cell)%s" % (C1, SUM, case, n, "" if v is not None else " -> no data"), n=n)

    # link decisions that differ, in points above the row "exact" (the index floor of the cost table)
    link = FIELDS["link"][1]
    floor = {c: _cell(R, C1, {"case": c, "protocol.policy": "exact"}, link, RUN)[0] for c in MAIN6}
    for tag in ("c102", "c105", "c110"):
        spec = POLICY[tag]
        add = {}
        for c in MAIN6:
            v = _cell(R, C1, {"case": c, "protocol.policy": spec}, link, RUN)[0]
            add[c] = None if v is None or floor[c] is None else v - floor[c]
        _range(R, "C1.linkadd" + tag[1:], add, "%s/%s: mean(%s)[%s] - mean(%s)[exact], means over seeds, percentage points"
               % (C1, RUN, link, spec, link), only_max=(tag == "c102"))


# ------------------------------------------------------------------------------------------------ C2, C6
def _c2(R):
    exp, file = "C2_counterfactual_pareto", "audit_first.jsonl"
    for key, field in (("C2.changed", "changed_set_%"), ("C2.physonly", "physical_only_change_%")):
        vals = {c: _cell(R, exp, {"case": c}, field, file)[0] for c in _cases(R, exp)}
        _range(R, key, vals, "%s/%s: mean(%s) over seeds, all cases of the experiment" % (exp, file, field))


def _c6(R):
    exp = "C6_index_operating_points"
    cell = {"case": "A64", "index": "hnsw-M16"}
    for b in ("ef40", "ef10"):
        R.stat("C6.A64.%s.recall" % b, exp, "recall_fresh", dict(cell, budget=b, val="same"), RUN)
    where = dict(cell, budget="ef10")
    v, n = _saving(R, exp, where, "same", ref="reexec", key="val")
    R.put("C6.A64.ef10.save110", v, "%s/%s [case=A64, index=hnsw-M16, budget=ef10]: 100 * (1 - mean(extra_calls_per_100)[val=same, cadm:1.1] / "
          "mean(extra_calls_per_100)[val=reexec, exact]), means over seeds (%d rows per cell)%s" % (exp, SUM, n, "" if v is not None else " -> no data"), n=n)
    v40, n40 = _saving(R, exp, dict(cell, budget="ef40"), "same", ref="reexec", key="val")
    d = abs(v - v40) if v is not None and v40 is not None else None
    R.put("C6.A64.ef10.dsave", d, "%s/%s [case=A64, index=hnsw-M16]: |saving at budget ef10 - saving at budget ef40| in points, each saving "
          "100 * (1 - mean(extra_calls_per_100)[val=same, cadm:1.1] / mean(extra_calls_per_100)[val=reexec, exact]) over seeds%s"
          % (exp, SUM, "" if d is not None else " -> no data"), n=min(n, n40))


# ------------------------------------------------------------------------------------------------ B1, B7
# cases of B1_safety_theorem whose writes are all insertions (A64mix also replaces and deletes)
B1_INSERT_ONLY = ("L8", "A64", "A64multi")


def _b1(R):
    exp = "B1_safety_theorem"
    rows = R.sel(exp, None, RUN)
    for key, field in (("B1.checked", "safety_theorem.checked"), ("B1.violations", "safety_theorem.violations")):
        n = len(R.vals(rows, field))
        R.put(key, R.total(rows, field), "%s/%s: sum(%s) over the %d audited runs that have it (%s); the total row of tab_safety%s"
              % (exp, RUN, field, n, _coverage(R, exp, rows), "" if n else " -> no data"), n=n)


    # Proposition (separation): with exact validation reads no two stored items are closer than tau / c.  Certified runs
    # only, and only the cases whose writes are all (tau-deduplicating) insertions: the premise of the proposition.  A64mix
    # also replaces and deletes items (workload.ops), which the proposition does not cover; the text says "whose writes are
    # all insertions".
    sep = R.sel(exp, {"protocol.policy": lambda p: str(p).startswith("cert:"), "case": list(B1_INSERT_ONLY)}, "audit_sep.jsonl")
    n = len(R.vals(sep, "separation_proposition.violations"))
    R.put("B1.sepviol", R.total(sep, "separation_proposition.violations"),
          "%s/audit_sep.jsonl [protocol.policy=cert:*, case in %s]: sum(separation_proposition.violations) over %d certified "
          "insert-only runs%s" % (exp, "/".join(B1_INSERT_ONLY), n, "" if n else " -> no data"), n=n)


def _b7(R):
    exp, file = "B7_characterization", "audit_law.jsonl"
    rows = R.sel(exp, None, file)
    reads = ok = tot = n = 0
    for r in rows:
        m = str(r.get("characterization", "")).split("/")            # "agreeing/tested" over the single-insertion reads and the c grid
        s = R.vals([r], "single_insertion_reads")
        if len(m) != 2 or not (m[0].isdigit() and m[1].isdigit()) or not s:
            continue
        reads += int(s[0]); ok += int(m[0]); tot += int(m[1]); n += 1
    src = "%s/%s: sum of %%s over %d audited runs (%s)%s" % (exp, file, n, _coverage(R, exp, rows), "" if n else " -> no data")
    R.put("B7.reads", float(reads) if n else None, src % "single_insertion_reads", n=n)
    R.put("B7.ok", float(ok) if n else None, src % "the first number of characterization (agreeing)", n=n)
    R.put("B7.tot", float(tot) if n else None, src % "the second number of characterization (tested = reads x tolerances)", n=n)


# ------------------------------------------------------------------------------------------------ F1, F2
def _f1(R):
    exp = "F1_stores"
    e = R.exp(exp)
    kind = {}                                                     # case -> store system (pgvector and FAISS have several cases)
    for c in (e.get("cases") or []) + ((e.get("smoke") or {}).get("cases") or []):
        if c.get("_name") and (c.get("store") or {}).get("kind"):
            kind[c["_name"]] = c["store"]["kind"]
    rows = R.sel(exp, None, RUN)
    checked = [r for r in rows if (R.vals([r], "safety_theorem.checked") or [0])[0] > 0]
    cases = sorted({str(r.get("case")) for r in checked})
    unknown = [c for c in cases if c not in kind]
    stores = sorted({kind[c] for c in cases if c in kind})
    src = "%s/%s: distinct store.kind (experiments/%s.yaml) of the cases with safety_theorem.checked > 0: %s; cases: %s (%s)" \
          % (exp, RUN, exp, ", ".join(stores) or "none", ", ".join(cases) or "none", _coverage(R, exp, rows))
    if unknown:
        R.put("F1.nstores", None, src + " -> cases without a store in the experiment file: " + ", ".join(unknown))
    else:
        R.put("F1.nstores", float(len(stores)) if stores else None, src + ("" if stores else " -> no data"), n=len(checked))
    field = "safety_theorem.violations"
    n = len(R.vals(rows, field))
    R.put("F1.violations", R.total(rows, field), "%s/%s: sum(%s) over the %d audited runs that have it (%s); the column of tab_portability%s"
          % (exp, RUN, field, n, _coverage(R, exp, rows), "" if n else " -> no data"), n=n)


def _f2(R):
    """F2 is run once per embedding model, each into its own results directory (see the doc of the experiment):
    EMB=<tag> python experiments/run.py experiments/F2_embedders_and_datasets.yaml --out results_<tag>.
    The rows of the main results directory go through R.sel (the default embedder; this also checks the selectors
    against the grid); the directories <results>_<tag> next to it are read with the same selectors."""
    exp, f = "F2_embedders_and_datasets", "extra_calls_per_100"
    cases = _cases(R, exp)
    vals, roots = {}, []
    main = os.path.basename(R.results)
    for c in cases:
        v, n = _saving(R, exp, {"case": c}, "cadm:1.1")
        if n:
            vals["%s:%s" % (main, c)] = v
    if R.rows(exp):
        roots.append(main)
    for d in sorted(glob.glob(glob.escape(R.results) + "_*")):
        if not os.path.isfile(os.path.join(d, exp + R.suffix, SUM)):
            continue
        rows = ld.rows(d, exp, SUM, R.suffix, quiet=True)
        roots.append(os.path.basename(d))
        for c in cases:
            a = ld.where(rows, case=c, protocol__policy="cadm:1.1"); b = ld.where(rows, case=c, protocol__policy="exact")
            if a and b:
                vals["%s:%s" % (os.path.basename(d), c)] = R.saved(R.mean(a, f), R.mean(b, f))
    _range(R, "F2.save110", vals, "%s/%s: 100 * (1 - mean(%s)[cadm:1.1] / mean(%s)[exact]), means over seeds, per (results directory "
           "= embedder, case); %d embedder director%s with data: %s" % (exp, SUM, f, f, len(roots), "y" if len(roots) == 1 else "ies", ", ".join(roots) or "none"))
    # the other embedding models only (the default model is reported by C1 in the same paragraph of the text)
    others = [r for r in roots if r != main]
    _range(R, "F2.other.save110", {k: v for k, v in vals.items() if not k.startswith(main + ":")},
           "%s/%s: as F2.save110, over the embedder directories other than the main one: %s" % (exp, SUM, ", ".join(others) or "none"))
    R.put("F2.nother", float(len(others)) if others else None, "%s: number of embedder directories other than the main one with F2 rows: %s"
          % (exp, ", ".join(others) or "none"), n=len(others))


# ------------------------------------------------------------------------------------------------ G1
def _g1(R):
    exp = "G1_critical_section"
    where = {"workload.agents": 16, "kb": "k10-ef40"}             # one top-10 read at the default search budget, little contention
    for key, field in (("G1.vread", "cs_vread_mean_ms"), ("G1.pred", "cs_predicate_mean_ms"), ("G1.write", "cs_write_mean_ms")):
        R.stat(key, exp, field, where)


def resolve(R):
    _c1(R)
    _c2(R)
    _c6(R)
    _b1(R)
    _b7(R)
    _f1(R)
    _f2(R)
    _g1(R)
