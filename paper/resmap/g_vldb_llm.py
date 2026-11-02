"""C3.<m>.*, C4.<m>.*, C5.t2.*, X.machines.* (two models, two benchmarks, the timing machine).

Directories (paper/assemble_results.py builds the results root):
    gpt/   results_vldb_gpt4omini   GPT-4o mini decides       experiments/vldb/C3_llm_decisions.yaml, C4_downstream_qa.yaml
    qwen/  results_vldb_qwen7b      Qwen2.5-7B-Instruct decides (same files)
    vm/    the base results root (the 192-core server; vm/C1_calls_vs_policy is compared with C1 of the timing server)
    the experiments of results_vldb_perf replace those of the base root at the top level (C1, C5, C10, ...).

C3 (per model m in gpt, qwen; case in L4, L8, LME4; pol in none, ov90, exact, c105, c110, c120; five seeds, mean over seeds)
    C3.<m>.<case>.<pol>.extra        extra model calls per 100 tasks             summary.jsonl: extra_calls_per_100
    C3.<m>.<case>.<pol>.tok          tokens per task (prompt + completion)        summary.jsonl: decider.*_tokens / tasks
    C3.<m>.<case>.<pol>.changed      decisions that differ from the decision on the exact answer (%)   audit_replay.jsonl
    C3.<m>.<case>.save110            100 (1 - extra[c110] / extra[exact])
    C3.<m>.<case>.<pol>.tokover      100 (tok[pol] / tok[none] - 1), for exact and c110
    C3.<m>.<case>.tokratio           (tok[c110] - tok[none]) / (tok[exact] - tok[none])
    C3.<m>.<case>.<pol>.chgx         changed[pol] - changed[exact], seed-paired mean (points), for c110 and none;
                                     .hi one-sided 95% upper bound (c110), .lo two-sided 95% lower bound (none), t over seeds
    C3.<m>.unparsable                unparsable answers, % of the model calls of all cells
    C3.<m>.lat                       mean latency of a model call (s)
    C3.replay.n                      tasks replayed per run (the same in every cell of both models)
    ranges over the six (model, case) pairs:
    C3.save110.min/max, C3.tokover.exact.min/max, C3.tokover.c110.min/max, C3.tokratio.max, C3.unparsable.max,
    C3.chg.c110.excess.max, C3.chg.c110.excess.hi.max, C3.chg.none.excess.min/max, C3.chg.none.excess.lo.min
C4 (per model m; benchmark d: locomo = runs of case L4, lme = runs of case LME4; answering and judging model fixed)
    C4.<m>.<d>.<pol>.f1 / .judge     token F1 / judge accuracy (%), mean over the seeds   qa.jsonl run lines
    C4.<m>.<d>.<pol>.diff/.lo/.hi    F1 of pol minus F1 of exact, paired per (seed, question), the five seeds pooled,
                                     bootstrap 95% interval over the pairs (F1 points)      qa_details.jsonl
    C4.<m>.<d>.<pol>.jdiff/.jlo/.jhi the same for the judge (points)
                                     without qa_details.jsonl: the compare lines of the seeds, mean weighted by n, interval
                                     from the per-seed intervals (normal approximation)
    C4.<m>.<d>.none.duppairs         near-duplicate pairs per question, none minus exact, mean over the seeds
    C4.c110.absdiff.max              largest |diff| at c110 over the four (model, benchmark) pairs
    C4.c110.ci.max                   largest |bound| of the c110 intervals over the four pairs
    C4.c110.jci.max                  the same for the judge
    C4.none.diff.min/max             F1 points lost without isolation (exact minus none) over the four pairs
    C4.none.diff.hi.max              largest upper bound of none minus exact (must stay below 0)
    C4.none.jdiff.min/max            judge points lost without isolation
    C4.questions, C4.lme.questions   questions answered per run (LoCoMo, LongMemEval), the same in every run of both models
C5 at 2 s of thinking (C5_latency_throughput, think = 2s)
    C5.t2.a256.c110.held             commit lock held at 256 agents, c = 1.1 (%)
    C5.t2.a256.exact.rel             throughput of re-execution at 256 agents relative to no isolation (%)
    C5.t2.c110.rel.min               smallest throughput of c = 1.1 relative to no isolation over 1 to 256 agents (%)
X (the timing server against the 192-core server, C1_calls_vs_policy, the same cells)
    X.machines.extra.maxdev          largest relative difference of extra calls per 100 tasks (%), cells with >= 1 call
    X.machines.cells                 cells compared
"""
import math, os
from collections import defaultdict

import numpy as np

MODELS = {"gpt": "gpt", "qwen": "qwen"}            # key tag -> directory prefix
CASES = ("L4", "L8", "LME4")
BENCH = {"locomo": "L4", "lme": "LME4"}
POL = {"none": "none", "ov90": "overlap:0.9", "exact": "exact", "c105": "cadm:1.05", "c110": "cadm:1.1", "c120": "cadm:1.2"}
EXTRA = "extra_calls_per_100"
CHANGED = "decision_changed_committed_vs_exact_%"
REPLAY = "audit_replay.jsonl"
B = 2000
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}


def _exp(m, name):
    return "%s/%s" % (MODELS[m], name)


def _w(case, pol):
    return {"case": case, "protocol.policy": pol}


def _tok(r):
    d = r.get("decider") or {}
    if not d.get("tokens_include_cached"):
        return float("nan")
    return (d.get("prompt_tokens", 0) + d.get("completion_tokens", 0)) / max(r.get("tasks") or 0, 1)


def _range(R, prefix, vals, src):
    R.put_range(prefix, [v for v in vals if v is not None], src)


def _tbound(d, side):
    """Mean of the per-seed differences and its t bound: side 'hi1' one-sided upper 95%, 'lo2' two-sided lower 95%."""
    d = [x for x in d if x is not None and math.isfinite(x)]
    if len(d) < 2:
        return (d[0] if d else None), None
    n = len(d); m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); se = sd / math.sqrt(n)
    return m, (m + T95.get(n - 1, 1.645) * se if side == "hi1" else m - T975.get(n - 1, 1.96) * se)


# ------------------------------------------------------------------------------------------------ C3
def _c3(R):
    rng = defaultdict(list)
    sampled = []
    for m in MODELS:
        e = _exp(m, "C3_llm_decisions")
        allrows = R.sel(e, None)
        for case in CASES:
            per = {}
            for tag, pol in POL.items():
                w = _w(case, pol)
                R.stat("C3.%s.%s.%s.extra" % (m, case, tag), e, EXTRA, w)
                rows = R.sel(e, w)
                tk = R.mean(rows, _tok)
                R.put("C3.%s.%s.%s.tok" % (m, case, tag), tk, "%s/summary.jsonl %s: mean((decider.prompt_tokens + decider.completion_tokens) / tasks), "
                      "cached calls counted with the tokens of the original call, %d rows" % (e, w, len(rows)), n=len(rows))
                R.stat("C3.%s.%s.%s.changed" % (m, case, tag), e, CHANGED, w, file=REPLAY)
                per[tag] = {"extra": R.val("C3.%s.%s.%s.extra" % (m, case, tag)), "tok": tk,
                            "chg": {r.get("seed"): R.get(r, CHANGED) for r in R.sel(e, w, REPLAY)}}
            src = "%s case=%s" % (e, case)
            sv = R.saved(per["c110"]["extra"], per["exact"]["extra"]) if per["exact"]["extra"] else None
            R.put("C3.%s.%s.save110" % (m, case), sv, src + ": 100 (1 - extra[cadm:1.1] / extra[exact])")
            rng["save"].append(sv)
            tn, te, tc = per["none"]["tok"], per["exact"]["tok"], per["c110"]["tok"]
            ok = all(x is not None and math.isfinite(x) for x in (tn, te, tc)) and tn > 0
            for tag, t in (("exact", te), ("c110", tc)):
                v = 100.0 * (t / tn - 1.0) if ok else None
                R.put("C3.%s.%s.%s.tokover" % (m, case, tag), v, src + ": 100 (tok[%s] / tok[none] - 1)" % tag)
                rng["tokover." + tag].append(v)
            v = (tc - tn) / (te - tn) if ok and te != tn else None
            R.put("C3.%s.%s.tokratio" % (m, case), v, src + ": (tok[c110] - tok[none]) / (tok[exact] - tok[none])")
            rng["tokratio"].append(v)
            ex = per["exact"]["chg"]
            for tag, side in (("c110", "hi1"), ("none", "lo2")):
                cur = per[tag]["chg"]
                d = [cur[s] - ex[s] for s in sorted(set(cur) & set(ex)) if cur[s] is not None and ex[s] is not None]
                mu, bound = _tbound(d, side)
                s2 = src + ": changed[%s] - changed[exact], paired over %d seeds" % (tag, len(d))
                R.put("C3.%s.%s.%s.chgx" % (m, case, tag), mu, s2, n=len(d))
                R.put("C3.%s.%s.%s.chgx.%s" % (m, case, tag, "hi" if side == "hi1" else "lo"), bound,
                      s2 + (": one-sided 95%% upper bound (t)" if side == "hi1" else ": two-sided 95%% lower bound (t)"), n=len(d))
                rng["chg.%s" % tag].append(mu); rng["chg.%s.bound" % tag].append(bound)
        calls, bad = R.total(allrows, "decider.calls"), R.total(allrows, "decider.unparsable")
        u = 100.0 * bad / calls if calls else None
        R.put("C3.%s.unparsable" % m, u, "%s/summary.jsonl all cells: 100 sum(decider.unparsable) / sum(decider.calls)" % e)
        rng["unparsable"].append(u)
        R.put("C3.%s.lat" % m, R.mean(allrows, "decider.llm_latency_mean_s"), "%s/summary.jsonl all cells: mean(decider.llm_latency_mean_s)" % e)
        sampled += R.vals(R.sel(e, None, REPLAY), "sampled")
        # decisions that act on one specific memory (UPDATE or DELETE name their target, and a changed target counts as a
        # changed decision in the replay): share of all decisions under re-execution, pooled over cases and seeds
        acts = {}
        for r in R.sel(e, {"protocol.policy": "exact"}):
            for a, k in (r.get("actions") or {}).items():
                acts[a] = acts.get(a, 0) + int(k)
        tot = sum(acts.values())
        v = 100.0 * (acts.get("UPDATE", 0) + acts.get("DELETE", 0)) / tot if tot else None
        R.put("C3.%s.targeted" % m, v, "%s/summary.jsonl [protocol.policy=exact]: 100 (UPDATE + DELETE) / all decisions, summed over cases "
              "and seeds (%s)" % (e, ", ".join("%s %d" % kv for kv in sorted(acts.items())) or "no actions"))
        rng["targeted"].append(v)
    _range(R, "C3.save110", rng["save"], "six (model, case) pairs: save110")
    _range(R, "C3.targeted", rng["targeted"], "two models: share of decisions that update or delete a specific memory")
    _range(R, "C3.tokover.exact", rng["tokover.exact"], "six pairs: tokover at exact")
    _range(R, "C3.tokover.c110", rng["tokover.c110"], "six pairs: tokover at cadm:1.1")
    v = [x for x in rng["tokratio"] if x is not None]
    R.put("C3.tokratio.max", max(v) if v else None, "six pairs: max tokratio")
    v = [x for x in rng["unparsable"] if x is not None]
    R.put("C3.unparsable.max", max(v) if v else None, "two models: max unparsable %")
    v = [x for x in rng["chg.c110"] if x is not None]
    R.put("C3.chg.c110.excess.max", max(v) if v else None, "six pairs: max changed[c110] - changed[exact]")
    v = [x for x in rng["chg.c110.bound"] if x is not None]
    R.put("C3.chg.c110.excess.hi.max", max(v) if v else None, "six pairs: max one-sided upper bound of changed[c110] - changed[exact]")
    _range(R, "C3.chg.none.excess", rng["chg.none"], "six pairs: changed[none] - changed[exact]")
    v = [x for x in rng["chg.none.bound"] if x is not None]
    R.put("C3.chg.none.excess.lo.min", min(v) if v else None, "six pairs: min lower bound of changed[none] - changed[exact]")
    lo, hi = (min(sampled), max(sampled)) if sampled else (None, None)
    if lo is not None and lo != hi:
        R.put("C3.replay.n", None, "audit_replay.jsonl of both models: sampled differs between cells (%g to %g)" % (lo, hi))
    else:
        R.put("C3.replay.n", lo, "audit_replay.jsonl of both models: sampled, the same in all %d cells" % len(sampled), n=len(sampled))


# ------------------------------------------------------------------------------------------------ C4
def _boot(d, seed=0):
    d = np.asarray(d, dtype=float); n = len(d)
    if n == 0:
        return None, None, None
    g = np.random.default_rng(seed)
    means = d[g.integers(0, n, size=(B, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def _cboot(per, seed=0):
    """mean and 95 % percentile interval, resampling questions with all their paired differences"""
    qs = list(per)
    if not qs:
        return None, None, None
    S = np.array([sum(per[q]) for q in qs], float); C = np.array([len(per[q]) for q in qs], float)
    g = np.random.default_rng(seed); idx = g.integers(0, len(qs), size=(B, len(qs)))
    means = S[idx].sum(1) / C[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(S.sum() / C.sum()), float(lo), float(hi)


def _tint(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (float(x.mean()) if n else None), None, None
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def _qa(R, m):
    """({cell: run line}, {run path: compare line}, {cell: {qid: (f1, judge)}}) of one model."""
    e = _exp(m, "C4_downstream_qa")
    runs, cmps, path2cell = {}, {}, {}
    for r in R.json(e, "qa.jsonl"):
        if not isinstance(r, dict):
            continue
        if r.get("compare"):
            cmps[r.get("run")] = r
        elif r.get("cell"):
            runs[r["cell"]] = r; path2cell[r.get("run")] = r["cell"]
    det = defaultdict(dict)
    for r in R.json(e, "qa_details.jsonl"):
        c = path2cell.get(r.get("run")) if isinstance(r, dict) else None
        if c is not None:
            det[c][r.get("qid")] = (r.get("f1"), r.get("judge"))
    return runs, cmps, det


def _c4(R):
    absd, jabsd, ci, jci, loss, jloss, nonehi = [], [], [], [], [], [], []
    nq = {"locomo": [], "lme": []}
    for m in MODELS:
        e3 = _exp(m, "C3_llm_decisions"); e4 = _exp(m, "C4_downstream_qa")
        runs, cmps, det = _qa(R, m)
        for d, case in BENCH.items():
            cells = {}
            for tag, pol in POL.items():
                cl = sorted(r["cell"] for r in R.sel(e3, _w(case, pol)) if r.get("cell") in runs)
                cells[tag] = cl
                L = [runs[c] for c in cl]
                for what, field in (("f1", "f1"), ("judge", "judge_acc")):
                    v = R.mean(L, field)
                    R.put("C4.%s.%s.%s.%s" % (m, d, tag, what), None if v is None else 100.0 * v,
                          "%s/qa.jsonl run lines of %s %s: mean(%s) x 100 over %d seeds" % (e4, e3, _w(case, pol), field, len(L)), n=len(L))
                nq[d] += [runs[c].get("questions") for c in cl]
            seed = lambda c: c.rsplit("_s", 1)[-1]
            ref = {seed(c): c for c in cells["exact"]}
            for tag in POL:
                if tag == "exact":
                    continue
                pairs = [(c, ref[seed(c)]) for c in cells[tag] if seed(c) in ref]
                src = "%s %s %s vs exact, %d seeds" % (e4, d, tag, len(pairs))
                res = {}
                if pairs and all(det.get(c) and det.get(r) for c, r in pairs):
                    for k, idx in (("", 0), ("j", 1)):
                        per = defaultdict(list)          # question -> paired differences of the seeds
                        for c, r in pairs:
                            for q, val in det[c].items():
                                rv = det[r].get(q)
                                if rv is not None and val[idx] is not None and rv[idx] is not None:
                                    per[q].append(val[idx] - rv[idx])
                        n = sum(len(v) for v in per.values())
                        if d == "lme":                   # few questions: t interval over the per-question means
                            res[k] = _tint([sum(v) / len(v) for v in per.values()]) + (len(per), "qa_details.jsonl: per question the "
                                     "mean of the paired differences of the seeds, t interval over the questions (reported, not tested)")
                        else:                            # bootstrap over questions, a question carries all its seeds
                            res[k] = _cboot(per) + (n, "qa_details.jsonl: paired per (seed, question), bootstrap over the questions "
                                     "(a resampled question carries the differences of all its seeds)")
                else:
                    C = [cmps.get(runs[c].get("run")) for c, r in pairs]
                    C = [x for x, (c, r) in zip(C, pairs) if x and x.get("reference") == runs[r].get("run")]
                    for k, fd, fc in (("", "diff_f1", "ci95"), ("j", "diff_judge", "ci95_judge")):
                        ok = [x for x in C if x.get(fd) is not None and (x.get(fc) or [None])[0] is not None]
                        if not ok:
                            res[k] = (None, None, None, 0, "no compare lines"); continue
                        W = sum(x["n"] for x in ok); mu = sum(x[fd] * x["n"] for x in ok) / W
                        se = math.sqrt(sum(((x[fc][1] - x[fc][0]) / 3.92 * x["n"] / W) ** 2 for x in ok))
                        res[k] = (mu, mu - 1.96 * se, mu + 1.96 * se, W, "compare lines of %d seeds (normal approximation)" % len(ok))
                for k, (mu, lo, hi, n, how) in res.items():
                    sc = lambda x: None if x is None else 100.0 * x
                    names = ("diff", "lo", "hi") if k == "" else ("jdiff", "jlo", "jhi")
                    for nm, v in zip(names, (mu, lo, hi)):
                        R.put("C4.%s.%s.%s.%s" % (m, d, tag, nm), sc(v), "%s: %s x 100 (n=%d)" % (src, how, n), n=n)
                if tag == "c110" and d == "locomo":
                    absd.append(R.val("C4.%s.%s.c110.diff" % (m, d))); jabsd.append(R.val("C4.%s.%s.c110.jdiff" % (m, d)))
                    lo, hi = R.val("C4.%s.%s.c110.lo" % (m, d)), R.val("C4.%s.%s.c110.hi" % (m, d))
                    ci.append(None if lo is None else max(abs(lo), abs(hi)))
                    lo, hi = R.val("C4.%s.%s.c110.jlo" % (m, d)), R.val("C4.%s.%s.c110.jhi" % (m, d))
                    jci.append(None if lo is None else max(abs(lo), abs(hi)))
                if tag == "none" and d == "locomo":
                    v = R.val("C4.%s.%s.none.diff" % (m, d)); loss.append(None if v is None else -v)
                    v = R.val("C4.%s.%s.none.jdiff" % (m, d)); jloss.append(None if v is None else -v)
                    nonehi.append(R.val("C4.%s.%s.none.hi" % (m, d)))
                if tag == "none":
                    C = [cmps.get(runs[c].get("run")) for c, _ in pairs]
                    v = [x.get("diff_dup_pairs_per_q") for x in C if x and x.get("diff_dup_pairs_per_q") is not None]
                    R.put("C4.%s.%s.none.duppairs" % (m, d), sum(v) / len(v) if v else None,
                          "%s compare lines: mean(diff_dup_pairs_per_q) over %d seeds" % (e4, len(v)), n=len(v))
    f = lambda v: [x for x in v if x is not None]
    R.put("C4.c110.absdiff.max", max(map(abs, f(absd))) if f(absd) else None, "LoCoMo, both models: max |diff| at cadm:1.1 (F1 points)")
    R.put("C4.c110.jabsdiff.max", max(map(abs, f(jabsd))) if f(jabsd) else None, "LoCoMo, both models: max |judge diff| at cadm:1.1 (points)")
    R.put("C4.c110.ci.max", max(f(ci)) if f(ci) else None, "LoCoMo, both models: max |bound| of the 95% intervals at cadm:1.1 (F1 points)")
    R.put("C4.c110.jci.max", max(f(jci)) if f(jci) else None, "LoCoMo, both models: max |bound| of the judge intervals at cadm:1.1 (points)")
    R.put_range("C4.none.diff", f(loss), "LoCoMo, both models: F1 points lost without isolation (exact minus none)")
    R.put_range("C4.none.jdiff", f(jloss), "LoCoMo, both models: judge points lost without isolation (exact minus none)")
    R.put("C4.none.diff.hi.max", max(f(nonehi)) if f(nonehi) else None, "LoCoMo, both models: max upper bound of none minus exact (F1 points)")
    for key, d in (("C4.questions", "locomo"), ("C4.lme.questions", "lme")):
        v = [x for x in nq[d] if x is not None]
        if v and min(v) != max(v):
            R.put(key, None, "qa.jsonl run lines of both models (%s): questions differ between runs (%g to %g)" % (d, min(v), max(v)))
        else:
            R.put(key, v[0] if v else None, "qa.jsonl run lines of both models (%s): questions, the same in all %d runs" % (d, len(v)), n=len(v))


# ------------------------------------------------------------------------------------------------ C5 at 2 s, machines
def _c5(R):
    C5 = "C5_latency_throughput"
    tps = lambda a, p: R.mean(R.sel(C5, {"workload.agents": a, "proto": p, "think": "2s"}), "tps")
    R.stat("C5.t2.a256.c110.held", C5, "lock_held_fraction", {"workload.agents": 256, "proto": "av-cadm1.1", "think": "2s"}, scale=100.0)
    b = tps(256, "none"); x = tps(256, "av-exact")
    R.put("C5.t2.a256.exact.rel", 100.0 * x / b if x is not None and b else None, "%s [256 agents, 2s]: 100 tps[av-exact] / tps[none]" % C5)
    rel = []
    for a in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        b, c = tps(a, "none"), tps(a, "av-cadm1.1")
        if b and c is not None:
            rel.append(100.0 * c / b)
    R.put("C5.t2.c110.rel.min", min(rel) if rel else None, "%s [2s]: min over agents of 100 tps[av-cadm1.1] / tps[none] (%d agent counts)" % (C5, len(rel)), n=len(rel))


def _machines(R):
    """C1 on the timing machine against vm/C1, cell by cell (case x predicate, seed means), MSM64 (FAISS on the 192-core server,
    pgvector on the timing machine) and 'none' excluded, cells with >= 1 extra call per 100 tasks on vm/.  Noise model per
    machine and case: Gamma GLM of the seed variance (_xv.py); DerSimonian-Laird random-effects mean b of
    log(P / V), 90 % interval (equivalence margin 5 %); maxdev over the cells with >= 5 per 100 on vm/ (descriptive)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _xv as XV
    def cells(rows):
        g = defaultdict(list)
        for r in rows:
            v, t = R.get(r, EXTRA), r.get("tasks")
            if v is not None and t:
                g[(r.get("case"), r.get("protocol.policy"))].append((v * t / 100.0, t))
        return g
    new, old = R.sel("C1_calls_vs_policy", None), R.sel("vm/C1_calls_vs_policy", None)
    if not new or not old:
        for k in ("X.machines.extra.maxdev", "X.machines.cells", "X.machines.offset.bound"):
            R.put(k, None, "C1_calls_vs_policy on the timing machine and vm/C1_calls_vs_policy: one of them is missing")
        return
    P, V = cells(new), cells(old)
    keep = lambda case, pol, per100: case != "MSM64" and pol != "none" and per100 >= 1.0
    mP, _ = XV.machine_models(P, keep); mV, _ = XV.machine_models(V, keep)
    C = []
    for k in sorted(set(P) & set(V)):
        ev, eP = [e for e, _ in V[k]], [e for e, _ in P[k]]
        v100 = 100 * sum(ev) / sum(t for _, t in V[k]); p100 = 100 * sum(eP) / sum(t for _, t in P[k])
        if not keep(k[0], k[1], v100) or sum(eP) <= 0 or not mV.get(k[0]):
            continue
        muV, muP = sum(ev) / len(ev), sum(eP) / len(eP)
        vV, _ = XV.predict(mV[k[0]], muV); vP, _ = XV.predict(mP.get(k[0]) or mV[k[0]], muP)
        C.append(dict(pol=k[1], y=math.log(muP / muV), v=vP / (len(eP) * muP ** 2) + vV / (len(ev) * muV ** 2), p100=p100, v100=v100))
    src = "C1 timing machine vs vm/C1, %d cells (MSM64 and none excluded, >= 1 extra call per 100 on vm/)" % len(C)
    R.put("X.machines.cells", float(len(C)) if C else None, src, n=len(C))
    if not C:
        return
    b, se, tau2, Q, df = XV.dl([(c["y"], c["v"]) for c in C])
    q = XV.tq(XV.T95, len(C) - 1)
    lo, hi = 100 * (math.exp(b - q * se) - 1), 100 * (math.exp(b + q * se) - 1)
    R.put("X.machines.offset", 100 * (math.exp(b) - 1), src + ": random-effects mean of log(P/V), as a percentage", n=len(C))
    R.put("X.machines.offset.lo", lo, src + ": lower end of the 90 % interval (%)", n=len(C))
    R.put("X.machines.offset.hi", hi, src + ": upper end of the 90 % interval (%)", n=len(C))
    R.put("X.machines.offset.bound", max(abs(lo), abs(hi)), src + ": max |end| of the 90 % interval of the mean offset (%); equivalence margin 5", n=len(C))
    E = XV.dl([(c["y"], c["v"]) for c in C if c["pol"] == "exact"]); Cc = XV.dl([(c["y"], c["v"]) for c in C if c["pol"] == "cadm:1.1"])
    if E and Cc:
        k = min(sum(1 for c in C if c["pol"] == "exact"), sum(1 for c in C if c["pol"] == "cadm:1.1"))
        d, s = E[0] - Cc[0], math.sqrt(E[1] ** 2 + Cc[1] ** 2); qq = XV.tq(XV.T95, max(k - 1, 1))
        R.put("X.machines.offset.ec110", 100 * (math.exp(d) - 1), src + ": offset of re-execution minus that of c = 1.1 (%)", n=k)
        R.put("X.machines.offset.ec110.bound", 100 * max(abs(math.exp(d - qq * s) - 1), abs(math.exp(d + qq * s) - 1)),
              src + ": max |end| of its 90 % interval (%)", n=k)
    heavy = [c for c in C if c["v100"] >= 5]
    R.put("X.machines.extra.maxdev", max(100 * abs(c["p100"] / c["v100"] - 1) for c in heavy) if heavy else None,
          "C1: max over the %d cells with >= 5 extra calls per 100 tasks on vm/ (MSM64 excluded) of 100 |P - V| / V" % len(heavy), n=len(heavy))


def resolve(R):
    _c3(R)
    _c4(R)
    _c5(R)
    _machines(R)
