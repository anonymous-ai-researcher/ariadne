"""Statistics helpers for the cross-machine comparison of the cost experiment (paper/resmap/g_vldb_llm.py, keys X.machines.*).

C1 is run on the timing machine and repeated on a second machine; the comparison is cell by cell (case x predicate, seed
means).  Noise model per machine and case: Gamma GLM of the seed variance of the per-run event counts; the cells are pooled
with a DerSimonian-Laird random-effects mean of log(timing / second machine).  Only the standard library is used.
"""


import argparse, json, math, os, subprocess, sys, tempfile


from collections import defaultdict


PKG = "0a7aeeb0c505"


cnt = {"OK": 0, "WARN": 0, "STOP": 0, "MISSING": 0, "INFO": 0}


def say(st, what, detail=""):
    cnt[st] += 1
    print("%-7s %s%s" % (st, what, ("   [" + detail + "]") if detail else ""))


def rows(d, exp, f="summary.jsonl"):
    p = os.path.join(d, exp, f) if d else None
    if not p or not os.path.exists(p):
        return None
    R = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            if isinstance(r, dict) and not ("error" in r and "tasks" not in r):
                R[r.get("cell", len(R))] = r
    return list(R.values())


def mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(v) / len(v) if v else None


def group(R, keys, field):
    g = defaultdict(list)
    for r in R:
        g[tuple(r.get(k) for k in keys)].append(r.get(field))
    return {k: mean(v) for k, v in g.items()}


def wilson_lo(k, n, z=1.96):
    if not n:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d


def f(x, nd=2):
    return "None" if x is None else "%.*f" % (nd, x)


EXCLUDE_CASES = {"MSM64"}


DELTA = 0.05


T3 = {2: 19.207, 3: 9.219, 4: 6.62, 5: 5.507, 6: 4.904, 7: 4.53, 8: 4.277, 9: 4.094, 10: 3.957, 12: 3.764, 14: 3.636, 16: 3.544,
      18: 3.475, 20: 3.422, 25: 3.33, 30: 3.27, 40: 3.199, 50: 3.157, 60: 3.13, 80: 3.097, 100: 3.077, 150: 3.051, 200: 3.038,
      300: 3.025, 500: 3.015, 1000: 3.008, 10 ** 9: 3.0}


T95 = {1: 6.314, 2: 2.92, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.86, 9: 1.833, 10: 1.812, 12: 1.782, 14: 1.761,
       16: 1.746, 18: 1.734, 20: 1.725, 25: 1.708, 30: 1.697, 40: 1.684, 50: 1.676, 60: 1.671, 80: 1.664, 100: 1.66, 150: 1.655,
       200: 1.653, 300: 1.65, 500: 1.648, 1000: 1.646, 10 ** 9: 1.645}


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145,
        16: 2.12, 18: 2.101, 20: 2.086, 25: 2.06, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.0, 80: 1.99, 100: 1.984, 150: 1.976,
        200: 1.972, 300: 1.968, 500: 1.965, 1000: 1.962, 10 ** 9: 1.96}


def tq(tab, df):
    """quantile from a table, linear in 1/df between the tabulated degrees of freedom"""
    df = max(df, min(tab))
    ks = sorted(tab)
    for lo, hi in zip(ks, ks[1:]):
        if lo <= df <= hi:
            w = (1 / lo - 1 / df) / (1 / lo - 1 / hi)
            return tab[lo] + w * (tab[hi] - tab[lo])
    return tab[ks[-1]]


def digamma(x):
    r = 0.0
    while x < 6:
        r -= 1 / x; x += 1
    f = 1 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi2_sf(q, df):                   # Wilson-Hilferty
    if df <= 0:
        return None
    z = ((q / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return norm_sf(z)


def poisson_sf(k, lam):               # P(X >= k)
    if k <= 0:
        return 1.0
    p = math.exp(-lam); c = p
    for i in range(1, k):
        p *= lam / i; c += p
    return max(0.0, 1 - c)


def c1_cells(d):
    """{(case, policy): [events per run of each seed]} of C1 in directory d (events = extra calls)"""
    R = rows(d, "C1_calls_vs_policy")
    if not R:
        return None
    g = defaultdict(list)
    for r in R:
        if r.get("extra_calls_per_100") is not None and r.get("tasks"):
            g[(r.get("case"), r.get("protocol.policy"))].append((r["extra_calls_per_100"] * r["tasks"] / 100.0, r["tasks"]))
    return g


def gamma_fit(cells):
    """cells: list of (mu, s2, nu).  Gamma GLM, log link, known shape nu/2: s2 ~ Gamma(nu/2, mean exp(a) mu^beta).
    Returns (a, beta, cov 2x2) or None."""
    pts = [(math.log(m), s2, nu) for m, s2, nu in cells if m > 0 and nu >= 1]
    if len(pts) < 3:
        return None
    a, b = math.log(max(1e-9, sum(s2 for _, s2, _ in pts) / sum(math.exp(x) for x, _, _ in pts))), 1.0
    for _ in range(100):
        S00 = S01 = S11 = r0 = r1 = 0.0
        for x, s2, nu in pts:
            eta = a + b * x; m = math.exp(eta); w = nu / 2.0; z = eta + (s2 - m) / m
            S00 += w; S01 += w * x; S11 += w * x * x; r0 += w * z; r1 += w * z * x
        det = S00 * S11 - S01 * S01
        if det <= 0:
            return None
        na, nb = (S11 * r0 - S01 * r1) / det, (S00 * r1 - S01 * r0) / det
        done = abs(na - a) < 1e-10 and abs(nb - b) < 1e-10
        a, b = na, nb
        if done:
            break
    return a, b, ((S11 / det, -S01 / det), (-S01 / det, S00 / det))


def predict(fit, mu):
    """(predicted variance per run, effective degrees of freedom of that prediction)"""
    a, b, C = fit; x = math.log(mu)
    v = C[0][0] + 2 * x * C[0][1] + x * x * C[1][1]
    return math.exp(a + b * x), (2.0 / v if v > 0 else 1e9)


def machine_models(g, keep):
    """per case: Gamma fit on the cells in keep(case, policy, per100) with >= 2 seeds"""
    by = defaultdict(list)
    for (case, pol), v in g.items():
        ev = [e for e, _ in v]; n = len(ev)
        if n < 2 or not keep(case, pol, 100 * sum(ev) / sum(t for _, t in v)):
            continue
        mu = sum(ev) / n; s2 = sum((x - mu) ** 2 for x in ev) / (n - 1)
        by[case].append((mu, s2, n - 1))
    return {c: gamma_fit(v) for c, v in by.items()}, by


def self_check(name, g, models, keep):
    """leave-one-seed-out deviations standardized by the model: share beyond the 3-sigma critical value"""
    n = k = 0; lam = 0.0
    for (case, pol), v in g.items():
        ev = [e for e, _ in v]; s = len(ev)
        if s < 3 or not models.get(case) or not keep(case, pol, 100 * sum(ev) / sum(t for _, t in v)):
            continue
        mu = sum(ev) / s; var, nu = predict(models[case], mu); crit = tq(T3, nu)
        for j in range(s):
            rest = ev[:j] + ev[j + 1:]; dlt = ev[j] - sum(rest) / (s - 1)
            z = dlt / math.sqrt(var * (1 + 1 / (s - 1)))
            n += 1; k += abs(z) > crit
        lam += s * 0.0027
    if not n:
        say("MISSING", "X-2 self-check %s" % name); return
    p = poisson_sf(k, lam)
    say("OK" if p >= 0.01 else "WARN", "X-2 self-check on %s alone: leave-one-seed-out deviations beyond the critical value near 0.27 %%" % name,
        "%d of %d (expected %.1f), P(>= observed) = %.3f" % (k, n, lam, p))


def dl(items):
    """DerSimonian-Laird on [(y, v)]: (mean, se, tau2, Q, df)"""
    k = len(items)
    if k == 0:
        return None
    w = [1 / v for _, v in items]; sw = sum(w)
    yf = sum(wi * y for wi, (y, _) in zip(w, items)) / sw
    Q = sum(wi * (y - yf) ** 2 for wi, (y, _) in zip(w, items))
    C = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (Q - (k - 1)) / C) if k > 1 and C > 0 else 0.0
    ws = [1 / (v + tau2) for _, v in items]; s = sum(ws)
    return sum(wi * y for wi, (y, _) in zip(ws, items)) / s, 1 / math.sqrt(s), tau2, Q, k - 1


def equiv(label, est, se, df, decide=True):
    lo, hi = est - tq(T95, max(df, 1)) * se, est + tq(T95, max(df, 1)) * se
    rl, rh = math.exp(lo) - 1, math.exp(hi) - 1
    ok = -DELTA <= rl and rh <= DELTA
    say(("OK" if ok else "WARN") if decide else "INFO",
        "X-2 offset %s: 90 %% interval within +-%d %%" % (label, round(100 * DELTA)),
        "%+.2f %% [%+.2f %%, %+.2f %%]" % (100 * (math.exp(est) - 1), 100 * rl, 100 * rh))


VMEXP = ["C7_choosing_c", "F3_seeds_and_variance", "B4_deletes_updates", "B6_completeness", "B7_characterization",
         "B2_separation", "C11_starvation", "C12_k_tau_sensitivity", "C6_index_operating_points", "F2_embedders_and_datasets",
         "A3_race_realtext", "A4_anomaly_taxonomy", "A2_race_sensitivity", "A11_simple_baselines", "D5_heterogeneous_dimension",
         "D5b_heterogeneous_analysis", "G2_uncontended_overhead", "A1_race_stores"]
