"""Keys for numbers that the PVLDB version states in sentences of the text.

    C2.dense.*          C2 dense counterfactual replay (first_dense.jsonl): tasks, additional duplicate writes of cadm:1.1
                        over exact (as S-safe-dup-none counts them), the same per 10,000 tasks, and those on MSM64
    C1.A64.dupw.maxdiff largest |dup_written_per_100[cadm:c] - dup_written_per_100[exact]| over c <= 1.2 (A64, means over seeds)
    C1.A64.tag.maxdiff  the same for the tag decision (div_committed_vs_exact_%.tag)
    C5.a64.*.rel        throughput at 64 agents relative to no isolation (percent), C5 at thinking time 0.2 s
    D3.rank1.maxdiff    largest |observed - predicted| share of in-ball insertions at rank one (points), pooled per
                        configuration over seeds and weighted by events (semi1 on LoCoMo, semi2 on AG News, as fitted)
    D3.L.ks.max         largest one-scale KS distance over the LoCoMo law runs (L2, L4, L8)
    D3.L4.ksp.min       smallest KS p value of the one-scale fit over the L4 runs
    D3.A64k50.ksp.max   largest KS p value of the two-scale fit over the A64k50 runs
    D4.A64.exch.dev     100 x |exchangeable in-ball rate / background rate - 1| (A64, means over seeds)
    T.static.inball.maxdev  100 x max |pred / obs - 1| of the in-ball rate predicted from the static stream (L4, L8, A64)
    B5a.fan.correct.dupw    duplicates per 100 tasks of the correct protocol on fan (the index floor)
    B5a.A64rr8.{group,correct}.dupw  duplicates per 100 tasks of one lock per team and of the correct protocol on A64rr8
    B5a.rrcs.excess.*   points by which rr-cs exceeds the correct protocol in rho_fresh > 1.1, per case (min, max)
    F4.little.min       smallest tps x lat_mean_s / clients over the F4 runs
"""
import json, math, os



def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def RF(r):
    return _f((r.get("rho_fresh") or {}).get(">1.1"))


def resolve(R):
    # ---- C2 dense replay
    rows = R.json("C2_counterfactual_pareto", "first_dense.jsonl")
    extra = n = msm = 0
    for r in rows:
        P = r.get("policies") or {}
        a = _f((P.get("cadm:1.1") or {}).get("dup_written_per_100")); b = _f((P.get("exact") or {}).get("dup_written_per_100"))
        t = _f(r.get("tasks"))
        if a is None or b is None or not t:
            continue
        d = max(int(round((a - b) * t / 100.0)), 0); extra += d; n += t
        if str(r.get("cell", "")).startswith("MSM64"):
            msm += d
    src = "C2_counterfactual_pareto/first_dense.jsonl: %d rows; round((dupw[cadm:1.1] - dupw[exact]) x tasks / 100) summed" % len(rows)
    R.put("C2.dense.tasks", n if n else None, src + " (tasks)", n=len(rows))
    R.put("C2.dense.extra", extra if n else None, src, n=len(rows))
    R.put("C2.dense.msm.extra", msm if n else None, src + " (MSM64 rows)", n=len(rows))
    R.put("C2.dense.per10k", 1e4 * extra / n if n else None, src + " per 10,000 tasks", n=len(rows))

    # ---- C1 A64: differences to exact up to c = 1.2
    for key, field, file in (("dupw", "dup_written_per_100", "audit_run.jsonl"), ("tag", "div_committed_vs_exact_%.tag", "audit_run.jsonl")):
        ref = R.mean(R.sel("C1_calls_vs_policy", {"case": "A64", "protocol.policy": "exact"}, file), field)
        ds = []
        for c in ("cadm:1.02", "cadm:1.05", "cadm:1.1", "cadm:1.2"):
            v = R.mean(R.sel("C1_calls_vs_policy", {"case": "A64", "protocol.policy": c}, file), field)
            if v is not None and ref is not None:
                ds.append(abs(v - ref))
        R.put("C1.A64.%s.maxdiff" % key, max(ds) if len(ds) == 4 else None,
              "C1_calls_vs_policy/%s [A64]: max over c in 1.02..1.2 of |mean(%s)[c] - mean(%s)[exact]|" % (file, field, field), n=len(ds))

    # ---- C5: throughput at 64 and 256 agents relative to no isolation
    def tps(proto, agents=64):
        return R.mean(R.sel("C5_latency_throughput", {"workload.agents": agents, "proto": proto, "think": "0.2s"}), "tps")
    for agents, protos in ((64, (("c110", "av-cadm1.1"), ("exact", "av-exact"))),
                           (256, (("c110", "av-cadm1.1"), ("exact", "av-exact"), ("cert110", "av-cert1.1")))):
        base = tps("none", agents)
        for name, proto in protos:
            v = tps(proto, agents)
            R.put("C5.a%d.%s.rel" % (agents, name), 100.0 * v / base if v is not None and base else None,
                  "C5_latency_throughput [%d agents, think 0.2s]: 100 x mean(tps)[%s] / mean(tps)[none]" % (agents, proto))

    # ---- C5: certified validation, whose exact scan sits inside the commit lock
    def cell(agents, think, field, scale=1.0):
        v = R.mean(R.sel("C5_latency_throughput", {"workload.agents": agents, "proto": "av-cert1.1", "think": think}), field)
        return None if v is None else scale * v
    for agents, think, key in ((16, "0.2s", "C5.a16.cert110.held"), (64, "2s", "C5.t2.a64.cert110.held"), (256, "2s", "C5.t2.a256.cert110.held")):
        R.put(key, cell(agents, think, "lock_held_fraction", 100.0),
              "C5_latency_throughput [%d agents, av-cert1.1, think %s]: 100 x mean(lock_held_fraction)" % (agents, think))
    R.put("C5.a256.cert110.tps", cell(256, "0.2s", "tps"), "C5_latency_throughput [256 agents, av-cert1.1, think 0.2s]: mean(tps)")

    # ---- D3 law fits
    p = R.path("D3b_law_analysis", "fits.json")
    F = json.load(open(p)) if os.path.exists(p) else {}
    by = {}
    for cell, e in F.items():
        by.setdefault(cell.rsplit("_s", 1)[0], []).append(e)
    diffs = []
    for case in ("L2", "L4", "L8", "A16", "A64", "A64k50"):
        E = by.get(case) or []
        pred = "pred1" if case.startswith("L") else "pred2"
        w = [_f(e.get("events")) or 0 for e in E]
        ob = [_f((e.get("rank") or {}).get("obs", [None])[0]) for e in E]
        pr = [_f((e.get("rank") or {}).get(pred, [None])[0]) for e in E]
        if E and None not in ob and None not in pr and sum(w):
            o = sum(a * b for a, b in zip(w, ob)) / sum(w); q = sum(a * b for a, b in zip(w, pr)) / sum(w)
            diffs.append(100 * abs(o - q))
    R.put("D3.rank1.maxdiff", max(diffs) if len(diffs) == 6 else None,
          "D3b_law_analysis/fits.json: max over six configurations of 100 x |pooled rank.obs[0] - pooled rank.pred[0]| (event-weighted)", n=len(diffs))
    ks = [_f((e.get("f1") or {}).get("ks")) for c in ("L2", "L4", "L8") for e in by.get(c, [])]
    R.put("D3.L.ks.max", max(x for x in ks if x is not None) if ks and None not in ks else None,
          "D3b_law_analysis/fits.json: max f1.ks over the L2, L4, L8 runs", n=len(ks))
    kp = [_f((e.get("f1") or {}).get("ks_p")) for e in by.get("L4", [])]
    R.put("D3.L4.ksp.min", min(kp) if kp and None not in kp else None, "D3b_law_analysis/fits.json: min f1.ks_p over the L4 runs", n=len(kp))
    kq = [_f((e.get("f2") or {}).get("ks_p")) for e in by.get("A64k50", [])]
    R.put("D3.A64k50.ksp.max", max(kq) if kq and None not in kq else None,
          "D3b_law_analysis/fits.json: max f2.ks_p over the A64k50 runs", n=len(kq))

    # ---- D4: exchangeable split against the background rate (keys D4.A64.exch.inball and D4.A64.exch.bg)
    a, b = R.val("D4.A64.exch.inball"), R.val("D4.A64.exch.bg")
    R.put("D4.A64.exch.dev", 100 * abs(a / b - 1) if a is not None and b else None, "100 x |D4.A64.exch.inball / D4.A64.exch.bg - 1|")

    # ---- static kernel: in-ball rate predicted / observed (keys T.static.<case>.pred and .obs, unrounded)
    devs = []
    for case in ("L4", "L8", "A64"):
        pv, ov = R.val("T.static.%s.pred" % case), R.val("T.static.%s.obs" % case)
        if pv is not None and ov:
            devs.append(100 * abs(pv / ov - 1))
    R.put("T.static.inball.maxdev", max(devs) if len(devs) == 3 else None,
          "100 x max over L4, L8, A64 of |T.static.<case>.pred / T.static.<case>.obs - 1|", n=len(devs))

    # ---- B5a: correct protocol on fan, and the repeatable-read variant
    R.stat("B5a.fan.correct.dupw", "B5a_broken_variants", "dup_written_per_100", {"case": "fan", "proto": "correct"}, file="audit_run.jsonl")
    R.stat("B5a.fan.nolock.dupw", "B5a_broken_variants", "dup_written_per_100", {"case": "fan", "proto": "no-lock"}, file="audit_run.jsonl")
    for name, proto in (("group", "group-lock"), ("correct", "correct")):
        R.stat("B5a.A64rr8.%s.dupw" % name, "B5a_broken_variants", "dup_written_per_100", {"case": "A64rr8", "proto": proto}, file="audit_run.jsonl")
    ex = []
    for case in ("fan", "L8", "A64rr8"):
        rr = R.mean(R.sel("B5a_broken_variants", {"case": case, "proto": "rr-cs"}, "audit_run.jsonl"), RF)
        ok = R.mean(R.sel("B5a_broken_variants", {"case": case, "proto": "correct"}, "audit_run.jsonl"), RF)
        if rr is not None and ok is not None:
            ex.append(rr - ok)
    R.put_range("B5a.rrcs.excess", ex if len(ex) == 3 else [],
                "B5a_broken_variants/audit_run.jsonl: mean(rho_fresh['>1.1'])[rr-cs] - mean(...)[correct] per case (fan, L8, A64rr8)")

    # ---- F4: Little's law ratio
    lit = []
    for r in R.sel("F4_client_processes"):
        t, l, c = _f(r.get("tps")), _f(r.get("lat_mean_s")), _f(r.get("clients"))
        if t and l and c:
            lit.append(t * l / c)
    R.put("F4.little.min", min(lit) if lit else None, "F4_client_processes/summary.jsonl: min tps x lat_mean_s / clients", n=len(lit))
