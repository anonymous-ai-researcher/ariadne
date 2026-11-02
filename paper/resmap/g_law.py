"""D1.* D2.* D3.* D4.* T.* L.*: the conflict law (Section 5, Section 7.5 with Table "tab:law", Appendix C).

    D1.*  closed form against Monte Carlo               D1_closed_form/mc*.json
    D2.*  planted streams: recovery, background checks  D2b_planted_analysis/planted.json, bgrank.json
    D3.*  the law on real runs                          D3a_law_real/audit_law.jsonl (in-ball share, all seeds),
                                                        D3b_law_analysis/fits.json, strata_m.json, strata_lid.json
    D4.*  controls on the same items                    D4a_exchangeable_controls/audit_law.jsonl
    T.*   predictions without fitting on the target     D3b_law_analysis/transfer_*.json, static_*.json
    L.*   lag kernel                                    D3b_law_analysis/lag.json

The analysis files are dicts keyed by cell name ("L4_s1").  A configuration is all cells of that case; a value
is the mean over these cells (the seeds), as in ariadne/plots/fig_t_*.py (_theory.by_case).  The commands of
D3b_law_analysis analyse the files *_s1 only, so today every such mean is one cell.

Conventions taken from the text (each is repeated in the provenance of the key):
  * Table "tab:law", "In-ball": share of reads with one or more concurrent in-ball insertions (percent), not
    insertions per read; "bg.": 100 x w_bg, the share of the in-ball insertions that the first-principles rate
    explains.
  * Scales: one sibling scale on LoCoMo (f1, err_semi1, e2e.m1), two on AG News (f2, err_semi2, e2e.m2), as the
    D_s column of the table shows.  The closed form is the one-scale closed form with the pooled dimension
    (err_closed, the orange curve of fig_t_03) for every configuration.
  * Ranges, maxima and means "over the configurations" need all configurations: with one missing, the key has no
    value.  The exception is D3.strata.m*.err, where a stratum that is too small in a configuration is left out
    (reads with three or more in-ball insertions hardly exist at low concurrency); its provenance lists the rest.
  * Counts and pooled quantities follow the figure that shows them: D1.* pools the rows of all seeds (fig_t_14),
    D2.total counts the runs that fig_t_07 draws, L.*.h1 pools the counts of the seeds (fig_t_05).
"""
import glob, math, os, re

D1, D2B, D3A, D3B, D4A = "D1_closed_form", "D2b_planted_analysis", "D3a_law_real", "D3b_law_analysis", "D4a_exchangeable_controls"
LOCOMO, AGNEWS = ("L2", "L4", "L8"), ("A16", "A64", "A64k50")
TABLE = LOCOMO + AGNEWS                                       # the six rows of Table "tab:law"
SCALES = {c: (1 if c in LOCOMO else 2) for c in TABLE}        # sibling scales of the row (its D_s column)
# D3.lid.*: both sentences that quote it (setup: "in the first six configurations"; Section 7.5: the sibling
# dimensions of the table rows "are far below the intrinsic dimension of ...") are about the six rows of the table
LID_CASES = TABLE
MIN_TEST = 30                                                 # ariadne.law.MIN_EV: strata with fewer test reads are left out of its summaries


# ------------------------------------------------------------------------------------------------ helpers
def _num(x):
    """float(x), or None for None, booleans, text and non-finite numbers."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _dig(o, path):
    """o[path[0]][path[1]]... through dicts and lists; None when a step is missing."""
    for p in path:
        if isinstance(o, dict):
            o = o.get(p)
        elif isinstance(o, (list, tuple)) and isinstance(p, int) and -len(o) <= p < len(o):
            o = o[p]
        else:
            return None
    return o


def _split(name):
    """ "A64k50_s1" -> ("A64k50", 1); a "#2" added by `law fit` for a repeated name is dropped (as _theory.split_cell)."""
    s = re.sub(r"#\d+$", "", str(name))
    m = re.match(r"^(.*)_s(\d+)$", s)
    return (m.group(1), int(m.group(2))) if m else (s, None)


def _doc(R, exp, file):
    """The JSON object of a script output (the last one if the file holds several), or None."""
    d = [x for x in R.json(exp, file) if isinstance(x, dict)]
    return d[-1] if d else None


def _max_err(pred, obs):
    """Largest absolute difference of two curves over the c grid (ariadne.law.max_err)."""
    if not isinstance(pred, list) or not isinstance(obs, list):
        return None
    d = [abs(a - b) for a, b in zip((_num(x) for x in pred), (_num(x) for x in obs)) if a is not None and b is not None]
    return max(d) if d else None


class Cells:
    """A dict keyed by cell name (fits.json, lag.json, strata_*.json, the predictions of static_*.json), by case."""

    def __init__(self, exp, file, doc):
        self.where = "%s/%s" % (exp, file); self.found = isinstance(doc, dict); self.by = {}
        for name, e in (doc or {}).items() if self.found else []:
            if not str(name).startswith("_") and isinstance(e, dict):
                self.by.setdefault(_split(name)[0], []).append((str(name), e))
        for v in self.by.values():
            v.sort(key=lambda t: (_split(t[0])[1] or 0, t[0]))

    def mean(self, case, field, label=None, scale=1.0):
        """Mean over the cells (seeds) of a case.  field: a path into the entry, or a function of the entry.
        Returns (value or None, provenance or reason, number of cells with a value)."""
        label = label or ("f(entry)" if callable(field) else ".".join(str(p) for p in field))
        if not self.found:
            return None, "%s: file missing" % self.where, 0
        E = self.by.get(case, [])
        if not E:
            return None, "%s: no cell of case %s" % (self.where, case), 0
        v = [_num(field(e) if callable(field) else _dig(e, field)) for _, e in E]
        ok = [x for x in v if x is not None]
        cells = ", ".join(n for n, _ in E)
        if not ok:
            return None, "%s [%s]: %s is null (too few in-ball insertions or reads for this block?)" % (self.where, cells, label), 0
        return (scale * sum(ok) / len(ok),
                "%s [%s]: mean of %s%s over %d of %d cells" % (self.where, cells, label, "" if scale == 1 else " x %g" % scale, len(ok), len(E)), len(ok))

    def put(self, R, key, case, field, label=None, scale=1.0):
        v, src, n = self.mean(case, field, label, scale)
        R.put(key, v, src, n=n)
        return v


def _over(R, key, cases, vals, src, how):
    """A range ("range": key.min, key.max), maximum or mean over ALL the given cases of per-case values."""
    miss = [c for c in cases if vals.get(c) is None]
    src = "%s; per case the mean over its cells, then the %s over %s" % (src, how, ", ".join(cases))
    if miss:
        for k in ([key + ".min", key + ".max"] if how == "range" else [key]):
            R.put(k, None, "%s: no value for %s" % (src, ", ".join(miss)))
    elif how == "range":
        R.put_range(key, [vals[c] for c in cases], src)
    else:
        v = [vals[c] for c in cases]
        R.put(key, max(v) if how == "max" else sum(v) / len(v), src, n=len(v))


# ------------------------------------------------------------------------------------------------ D1
def _d1(R):
    """Pooled over the rows of all mc*.json (one file per seed), exactly as fig_t_14: tested = not "low", z finite."""
    files = sorted(os.path.basename(p) for p in glob.glob(R.path(D1, "mc*.json")))
    rows = [r for f in files for d in R.json(D1, f) if isinstance(d, dict) for r in (d.get("rows") or []) if isinstance(r, dict)]
    where = "%s/mc*.json (%d files, %d rows)" % (D1, len(files), len(rows))
    if not rows:
        for k in ("cells", "maxz", "above2"):
            R.put("D1." + k, None, "%s: no rows" % where)
        return
    z = [abs(_num(r.get("z"))) for r in rows if not r.get("low") and _num(r.get("z")) is not None]
    settings = {(r.get("k"), r.get("D"), r.get("alpha"), r.get("c")) for r in rows}
    R.put("D1.cells", float(len(settings)), "%s: distinct (k, D, alpha, c); seeds are not settings; %d (setting, seed) z scores tested" % (where, len(z)), n=len(files))
    R.put("D1.maxz", max(z) if z else None, "%s: max |z| over the %d tested (setting, seed) cells" % (where, len(z)), n=len(files))
    R.put("D1.above2", 100.0 * sum(1 for x in z if x > 2) / len(z) if z else None,
          "%s: 100 x share of |z| > 2 among the %d tested (setting, seed) cells" % (where, len(z)), n=len(files))


# ------------------------------------------------------------------------------------------------ D2
def _d2(R):
    # ---- recovery: the runs that fig_t_07 (a) draws (planted share > 0, a fitted D_s); one run = one configuration and seed
    P = _doc(R, D2B, "planted.json")
    where = "%s/planted.json" % D2B
    if P is None:
        R.put("D2.total", None, where + ": file missing"); R.put("D2.inside", None, where + ": file missing")
    else:
        planted = [e for e in P.values() if isinstance(e, dict) and (_num(_dig(e, ("planted", "pi"))) or 0) > 0 and _num(_dig(e, ("planted", "Ds"))) is not None]
        runs = [e for e in planted if _num(_dig(e, ("fit", "Ds"))) is not None]
        inside = [e for e in runs if _dig(e, ("fit", "planted_inside_ci")) is True]
        undecided = [e for e in runs if _dig(e, ("fit", "planted_inside_ci")) is None]
        src = "%s: runs (configuration x seed) with planted.pi > 0 and a fitted D_s: %d of %d planted runs" % (where, len(runs), len(planted))
        R.put("D2.total", float(len(runs)) if runs else None, src, n=len(runs))
        R.put("D2.inside", float(len(inside)) if runs else None,
              "%s; of them fit.planted_inside_ci is true (%d runs have no interval and count as not inside)" % (src, len(undecided)), n=len(runs))
    # ---- background insertions: the pooled block of bgrank.json (one k: the one with the most insertions)
    B = _doc(R, D2B, "bgrank.json")
    p = (B or {}).get("pooled") if isinstance((B or {}).get("pooled"), dict) else None
    where = "%s/bgrank.json pooled" % D2B
    if p is None:
        why = where + (": file missing" if B is None else ": no pooled block (no file with identified background insertions)")
        for k in ("ratio", "rank1", "rank1.lo", "rank1.hi", "uniform"):
            R.put("D2.bg." + k, None, why)
        return
    nf = len(p.get("files") or [])
    tag = "%s (k = %s, %d files, %s insertions, %s distinct pairs; pooled_by_k has k = %s)" % (
        where, p.get("k"), nf, p.get("n_all"), p.get("n"), ", ".join(sorted((B.get("pooled_by_k") or {}), key=lambda s: _num(s) or 0)))
    R.put("D2.bg.ratio", _num(p.get("rate_ratio")), tag + ": rate_ratio = observed / first-principles count", n=nf)
    r1 = _num(p.get("rank1"))
    R.put("D2.bg.rank1", None if r1 is None else 100 * r1, tag + ": 100 x rank1", n=nf)
    kk = _num(p.get("k"))
    R.put("D2.bg.uniform", None if not kk else 100.0 / kk, tag + ": 100 / k, the rank-one share of uniform ranks at the k of the pooled block", n=nf)
    for j, k in ((0, "lo"), (1, "hi")):
        v = _num(_dig(p, ("rank1_ci", j)))
        R.put("D2.bg.rank1." + k, None if v is None else 100 * v, tag + ": 100 x rank1_ci[%d] (Wilson 95%%)" % j, n=nf)
    # ---- the exchangeable control on a real stream (D4b bgrank.json pooled): its in-ball insertions have uniform ranks
    E = _doc(R, "D4b_controls_analysis", "bgrank.json")
    q = (E or {}).get("pooled") if isinstance((E or {}).get("pooled"), dict) else None
    w = "D4b_controls_analysis/bgrank.json pooled"
    r1 = _num((q or {}).get("rank1"))
    R.put("D4.exch.rank1", None if r1 is None else 100 * r1, w + ": 100 x rank1")
    for j, k in ((0, "lo"), (1, "hi")):
        v = _num(_dig(q or {}, ("rank1_ci", j)))
        R.put("D4.exch.rank1." + k, None if v is None else 100 * v, w + ": 100 x rank1_ci[%d] (Wilson 95%%)" % j)
    R.put("D4.exch.chip", _num((q or {}).get("chi2_p")), w + ": p of the chi-square test against uniform ranks")
    kk = _num((q or {}).get("k"))
    R.put("D4.exch.uniform", None if not kk else 100.0 / kk, w + ": 100 / k")


# ------------------------------------------------------------------------------------------------ D3
def _stratum(label, field):
    """Field of the stratum with this label of a strata_m.json entry, when the stratum has MIN_TEST or more test reads."""
    def f(e):
        if e.get("by") != "m":
            return None
        for s in e.get("strata") or []:
            if isinstance(s, dict) and s.get("label") == label and (_num(s.get("n_test")) or 0) >= MIN_TEST:
                return s.get(field)
        return None
    return f


def _d3(R):
    F = Cells(D3B, "fits.json", _doc(R, D3B, "fits.json"))
    bg, lid, errx, errc, errs, e2e, e2e_bg = {}, {}, {}, {}, {}, {}, {}
    for c in TABLE:
        j = SCALES[c]
        # in-ball share: the audit of every run of the case (all seeds); the fit columns come from the cells in fits.json
        R.stat("D3.%s.inball" % c, D3A, "reads_with_in_ball_insertion_%", {"case": c}, file="audit_law.jsonl")
        bg[c] = F.put(R, "D3.%s.bg" % c, c, ("w_bg",), scale=100.0)
        if j == 1:
            F.put(R, "D3.%s.Ds1" % c, c, ("f1", "Ds", 0))
        else:                                                  # two scales, ascending: a = the smaller dimension
            F.put(R, "D3.%s.Ds2a" % c, c, ("f2", "Ds", 0)); F.put(R, "D3.%s.Ds2b" % c, c, ("f2", "Ds", 1))
        errx[c] = F.put(R, "D3.%s.errx" % c, c, ("single", "err_exch"))
        errc[c] = F.put(R, "D3.%s.errc" % c, c, ("single", "err_closed"))
        errs[c] = F.put(R, "D3.%s.errs" % c, c, ("single", "err_semi%d" % j))
        e2e[c] = F.mean(c, lambda e, j=j: _max_err(_dig(e, ("e2e", "m%d" % j)), _dig(e, ("e2e", "obs"))))[0]
        e2e_bg[c] = F.mean(c, lambda e: _max_err(_dig(e, ("e2e", "bg")), _dig(e, ("e2e", "obs"))))[0]
    for c in LID_CASES:
        lid[c] = F.mean(c, ("lid_pooled",))[0]
    F.put(R, "D3.A64.tail", "A64", ("tail", "obs", "1.5-2", 0), label="tail.obs[1.5-2][0] (minus the log-log slope of the observed conflict probability)")
    w = F.where
    _over(R, "D3.bg", TABLE, bg, w + ": 100 x w_bg", "range")
    _over(R, "D3.lid", LID_CASES, lid, w + ": lid_pooled", "range")
    _over(R, "D3.errx", TABLE, errx, w + ": single.err_exch", "range")
    _over(R, "D3.errs", TABLE, errs, w + ": single.err_semi1 (LoCoMo) / err_semi2 (AG News)", "range")
    _over(R, "D3.errc.L", LOCOMO, errc, w + ": single.err_closed", "range")
    _over(R, "D3.errc.A", [c for c in TABLE if c not in LOCOMO], errc, w + ": single.err_closed (AG News)", "range")
    _over(R, "D3.e2e.maxerr", TABLE, e2e, w + ": max over c of |e2e.m1 - e2e.obs| (LoCoMo) / |e2e.m2 - e2e.obs| (AG News)", "max")
    _over(R, "D3.e2e.bg.maxerr", TABLE, e2e_bg, w + ": max over c of |e2e.bg - e2e.obs|", "max")
    # ---- strata by the number of in-ball insertions: end-to-end simulation with the file's one-scale fit
    S = Cells(D3B, "strata_m.json", _doc(R, D3B, "strata_m.json"))
    for key, label in (("m1", "m = 1"), ("m2", "m = 2"), ("m3", "m >= 3")):
        v = {c: S.mean(c, _stratum(label, "err_semi_global"))[0] for c in TABLE}
        use = [c for c in TABLE if v[c] is not None]
        src = "%s: err_semi_global of the stratum \"%s\"; per case the mean over its cells, then the max over the cases whose stratum has %d or more test reads" % (
            S.where, label, MIN_TEST)
        R.put("D3.strata.%s.err" % key, max(v[c] for c in use) if use else None,
              src + (" (%d of %d: %s)" % (len(use), len(TABLE), ", ".join(use)) if use else ": no such stratum (file missing, or all too small)"), n=len(use))
    # ---- strata by local intrinsic dimension: the numbers of fig_t_15 (b)
    S = Cells(D3B, "strata_lid.json", _doc(R, D3B, "strata_lid.json"))
    for key, pred in (("closed", "closed_pooled"), ("semi", "semi_global")):
        v = {c: S.mean(c, lambda e, pred=pred: _dig(e, ("summary", "mean_err_common", pred)) if e.get("by") == "lid" else None)[0] for c in TABLE}
        _over(R, "D3.strata.lid." + key, TABLE, v, "%s: summary.mean_err_common.%s" % (S.where, pred), "mean")


# ------------------------------------------------------------------------------------------------ D4
def _d4(R):
    f = "audit_law.jsonl"
    R.stat("D4.A64.natural.inball", D4A, "in_ball_insertions_per_read", {"case": "A64", "order": "natural"}, file=f)
    R.stat("D4.A64.shuffle.inball", D4A, "in_ball_insertions_per_read", {"case": "A64", "order": "shuffle"}, file=f)
    R.stat("D4.A64.exch.inball", D4A, "in_ball_insertions_per_read", {"case": "A64", "order": "exchangeable"}, file=f)
    R.stat("D4.A64.exch.bg", D4A, "expected_background_per_read", {"case": "A64", "order": "exchangeable"}, file=f)


# ------------------------------------------------------------------------------------------------ T
T_FIELDS = (("pred", "pred_m", "in-ball insertions per read, predicted"), ("obs", "obs_m", "in-ball insertions per read, observed"),
            ("err", "err", "max over c of |predicted - observed| conflict probability, points"))


def _transfer(R, mode, src, dst, parts=("pred", "obs", "err")):
    """T.<mode>.<src>.<dst>.*: one transfer_<mode>_<src>_<dst>.json (the command names the two law files)."""
    file = "transfer_%s_%s_%s.json" % (mode, src, dst)
    t = _doc(R, D3B, file); where = "%s/%s" % (D3B, file)
    why = None
    if t is None:
        why = where + ": file missing"
    elif t.get("mode") != mode or _split(t.get("src"))[0] != src or _split(t.get("dst"))[0] != dst:
        why = "%s: holds %s -> %s, mode %s (expected %s -> %s, mode %s)" % (where, t.get("src"), t.get("dst"), t.get("mode"), src, dst, mode)
    for part, field, what in T_FIELDS:
        if part in parts:
            v = None if why else _num(t.get(field))
            notes = "; ".join(t.get("notes") or []) if (t and not why) else ""
            R.put("T.%s.%s.%s.%s" % (mode, src, dst, part), v,
                  why or "%s (%s -> %s): %s (%s)%s%s" % (where, t.get("src"), t.get("dst"), field, what, "" if v is not None else " is null", "; notes: " + notes if notes else ""), n=1)


def _t(R):
    _transfer(R, "lag", "L8", "L4"); _transfer(R, "lag", "A64", "A16"); _transfer(R, "window", "L4", "L8")
    # toward more agents (Figure fig:lag d): relative error of the predicted in-ball insertions per read, percent
    for src, dst in (("L2", "L8"), ("A16", "A64")):
        _transfer(R, "lag", src, dst)
        pr, ob = R.val("T.lag.%s.%s.pred" % (src, dst)), R.val("T.lag.%s.%s.obs" % (src, dst))
        R.put("T.lag.%s.%s.dev" % (src, dst), 100.0 * (pr / ob - 1.0) if pr is not None and ob else None,
              "%s/transfer_lag_%s_%s.json: 100 (pred_m / obs_m - 1), percent" % (D3B, src, dst), n=1)
    _transfer(R, "params", "L4", "A64")
    err = {}
    for file, cases in (("static_locomo.json", ("L4", "L8")), ("static_agnews.json", ("A64",))):
        S = _doc(R, D3B, file)
        P = Cells(D3B, file + " predictions", S.get("predictions") if S is not None else None)
        for c in cases:
            for part, field, what in T_FIELDS:
                v = P.put(R, "T.static.%s.%s" % (c, part), c, (field,), label="%s (%s)" % (field, what))
                if part == "err":
                    err[c] = v
    _over(R, "T.static.err", ("L4", "L8", "A64"), err, "%s/static_locomo.json, static_agnews.json predictions: err (the three static rows of the table)" % D3B, "range")


# ------------------------------------------------------------------------------------------------ L
def _h1(e):
    b = _dig(e, ("buckets", "1")) or {}
    return _num(b.get("inball")), _num(b.get("overlap"))


def _l(R):
    """Lag bucket "1" (same team, lag one) of lag.json.  h is pooled over the cells of a case as in fig_t_05 (c):
    sum of in-ball insertions / sum of overlapping commits; D and the exchangeable reference are means."""
    G = Cells(D3B, "lag.json", _doc(R, D3B, "lag.json"))
    for c in LOCOMO:
        E = [(n, _h1(e)) for n, e in G.by.get(c, [])]
        E = [(n, a, b) for n, (a, b) in E if a is not None and b]
        ov = sum(b for _, _, b in E)
        why = "%s: %s" % (G.where, "file missing" if not G.found else "no cell of case %s with overlapping commits at lag 1" % c)
        R.put("L.%s.h1" % c, 100.0 * sum(a for _, a, _ in E) / ov if ov else None,
              "%s [%s]: 100 x sum(buckets.1.inball) / sum(buckets.1.overlap) over %d cells" % (G.where, ", ".join(n for n, _, _ in E), len(E)) if ov else why, n=len(E))
        G.put(R, "L.%s.D1" % c, c, ("buckets", "1", "D"))
    G.put(R, "L.L4.hx", "L4", ("exchangeable_h",), scale=100.0)


def resolve(R):
    _d1(R); _d2(R); _d3(R); _d4(R); _t(R); _l(R)
