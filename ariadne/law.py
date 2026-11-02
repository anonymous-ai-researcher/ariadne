"""Conflict law for c-approximate top-k (similarity) reads: closed forms, fits, transfers, lag kernel.

Setting.  A read at query q sees its k nearest neighbours at metric distances r_1 <= ... <= r_k.  A
concurrent insertion v with t = d(q, v) < r_k ("in-ball") lands at old rank p = 1 + #{i : r_i < t}.
Characterization theorem: the stale answer stays c-admissible  iff  r_p <= c t  and  r_i <= c r_{i-1}
for i = p+1..k.  Given the neighbour distances of a read, a conflict is therefore a deterministic
function of t, and every statement below is about the distribution of t.

Model.  An in-ball concurrent insertion is a mixture of
  BACKGROUND  exchangeable with the stored points around q: its position is that of a randomly chosen
              existing neighbour.  No parameter.  Expected number per read from first principles:
              lam = n_c * k / (N + 1)   (n_c overlapping commits, N stored items at the read).
  SIBLING(S)  correlated with the read: radial density proportional to t^(D_s - 1) inside the ball, i.e.
              y = ln(r_k / t) ~ Exp(rate D_s).  D_s is the sibling dimension; a workload may need more
              than one scale.  Only the sibling part is estimated, from insertion positions alone
              (the background share is fixed at w_bg = min(lam / observed insertions per read, 1));
              ranks and conflict outcomes are never used in a fit and serve as tests.

Closed form (stored points Poisson in volume coordinates, local dimension D, x = c^(-D), alpha = D_s / D;
alpha = 1 is the background law, alpha > 1 is allowed and means "more spread out than the background"):
  rank pmf           q_alpha(p; k) = alpha * Gamma(p-1+alpha)/Gamma(p) * Gamma(k)/Gamma(k+alpha)
  P(adm and rank p)  = alpha/(p-1+alpha) * (1 - x^(p-1+alpha)) * prod_{i=p}^{k-1} i/(i+alpha) * (1 - x^(i+alpha))
  A sibling that becomes the new nearest neighbour violates its own rank with probability exactly c^(-D_s).
Semi-parametric form: the same two components evaluated on every read's actual neighbour distances
(cor_conflict, cor_rank_pmf for a sibling scale; leave-one-out for the background), so that no assumption
on the stored data is needed.
Lag kernel: siblings resolved by stream lag.  For a lag bucket b, h_b = P(an overlapping commit at that lag
falls inside the ball) and D_b = 1 / mean(y) its effective dimension.  The kernel does not depend on the
concurrency level; the concurrency level only decides how many commits of each lag overlap a read.

Input: self-contained law files <run>.law.pkl written by `python -m ariadne.audit law` (format documented
in ariadne.audit.law_extract).  The c grid is the `cs` of the law file (values <= 1 dropped) unless --cs.

Commands (all print a plain-text report and write the JSON given by --out):

  python -m ariadne.law mc --out mc.json [--n 60000]
        closed form against Monte Carlo on a grid of k, D, alpha, c
  python -m ariadne.law grid --out grid.json
        closed-form conflict probability on a dense (c, D_s) grid for heat maps
  python -m ariadne.law fit A.law.pkl B.law.pkl ... --out fits.json [--boot 200] [--boot-ks B] [--reps 30]
        first-principles background, sibling spectrum (1 and 2 scales), goodness of fit, rank test,
        conflict predictions, end-to-end simulation, tail exponent, LID sensitivity
  python -m ariadne.law transfer --src A.law.pkl --dst B.law.pkl --mode window|k|lag|time|params --out T.json [--scales 1]
        predict dst from src without refitting on dst (time: --src only, first half -> second half)
  python -m ariadne.law lag A.law.pkl ... --out lag.json
        lag kernel table per file and invariance summary across files of the same workload
  python -m ariadne.law planted S.law.pkl ... --out planted.json [--boot 200] [--wl-seed S]
        synthetic workloads with known parameters (and exchangeable controls): planted vs recovered
  python -m ariadne.law strata A.law.pkl ... --by lid|rk|team|time|m [--bins 3] [--reps 30] --out strata.json
        the same test inside strata of the reads: does one file-level D_s suffice, does a heterogeneous
        intrinsic dimension break the closed form
  python -m ariadne.law bgrank X.law.pkl ... --out bgrank.json [--wl-seed S]
        pooled test (several files and seeds) that BACKGROUND in-ball insertions land at uniform old ranks
  python -m ariadne.law report fits.json
        compact table of the main numbers of a fit file
Common options: --cs 1.02,1.05,...   --seed 0

JSON written (non-finite numbers become null; conflict probabilities and their errors are in PERCENT,
shares, h and rank probabilities are fractions in [0, 1]):

  fit       {name: {"k", "n", "events", "mean_m", "lid_median", "lid_pooled", "lid_twonn", "lam_bg", "w_bg", "nu",
                    "f1": {"pi": [..], "Ds": [..], "Ds_ci": [[lo, hi]], "bic", "ks", "ks_p", "rank_chi2_p"},
                    "f2": {"pi": [..], "Ds": [..], "bic", "ks", "ks_p", "rank_chi2_p"},
                    "free1": {"pi": [..], "Ds": [..]},
                    "cs": [...],
                    "single": {"n", "obs", "se", "ci": [[lo, hi], ...], "exch", "closed", "semi1", "semi2",
                               "err_exch", "err_closed", "err_semi1", "err_semi2"},
                    "rank": {"obs": [k shares], "pred1": [...], "pred2": [...], "exch": [...]},
                    "e2e": {"obs": [...], "m1": [...], "m2": [...], "bg": [...]},
                    "tail": {"obs": {"1.1-1.2": [slope, lo, hi], "1.2-1.5": [...], "1.5-2": [...]}, "Ds1": .., "Ds2": [..]},
                    "lid_sensitivity": {"median": {"D", "err_closed"}, "pooled": {...}, "twonn": {...}},
                    "y_hist": {"edges": [...], "density": [...]}, "y_model1": [...], "y_model2": [...], "y_bg": [...]}}
            extra keys: "bg" (background-only fit and the in-ball rate check), "f1"/"f2" "ll" and "pi_ci",
            "single" "closed2"/"err_closed2", "tail" "exch"/"model1"/"model2", "y_hist" "centres", "char", "notes".
            With fewer than 30 in-ball insertions the fitted blocks are null and "notes" says why.
  lag       {name: {"buckets": {bucket: {"overlap", "inball", "h", "D", "share", "rank1", "h_se", "D_se"}},
                    "exchangeable_h": mean k / (N + 1), "k", "n"},
             "_invariance": {group: {"files": [...], "buckets": {bucket: {"files", "h", "D", "h_ratio", "D_min", "D_max"}}}}}
            buckets: other (other teams), 1, 2-3, 4-7, 8-15, 16-31, 32+ (same team, lag in stream order)
  transfer  {"src", "dst", "mode", "pred_m", "obs_m", "ks", "rank1_pred", "rank1_obs", "cs", "obs", "pred", "err",
             "se", "n_single", "n_events", "scales", "model": {"pi", "Ds"}, "ref": {"exch", "err_exch", "self", "err_self"}, "notes"}
  mc        {"n", "rows": [{"k", "D", "alpha", "c", "closed", "mc", "se", "z", "low", "rank1_closed", "rank1_mc", "rank1_z",
                            "cond1_closed", "cond1_mc", "cond1_z", "own_rank"}], "max_abs_z", "max_abs_z_rank1", "max_abs_z_cond1", ...}
  grid      {"c": [60], "Ds": [60], "D": [8, 16, 32], "k": [10, 50], "pi": [0, .25, .5, .75, 1],
             "conflict": nested list indexed [D][k][pi][Ds][c] (probabilities), "background": [D][k][c]}
  planted   {name: {"planted": {"Ds", "pi", "D", "R", "lag_p", ...}, "fit": {"Ds", "Ds_ci", "pi", "ks", "planted_inside_ci"},
                    "sib": {"n", "pairs", "share", "share_first_principles", "Ds_mle", "Ds_se", "z", "ks", "ks_p"},
                    "bg": {"n", "obs", "lam_bg", "ratio", "z", ...}, "cs", "single": {"n", "obs", "se", "pred_fit", "pred_planted",
                    "closed_planted", "exch", "err_fit", "err_planted", "err_closed", "err_exch"}, "notes"}}
            ("sib": in-ball insertions whose source is the planted parent or child of the reading task; a concurrent
            pair seen from both sides counts once in Ds_se and in the KS test.  Exchangeable controls: pi = 0.)
  strata    {name: {"by", "k", "n", "events", "cs", "global": {"Ds", "pi", "w_bg", "lam_bg", "mean_m", "lid_pooled", "lid_median"},
                    "strata": [{"label", "n", "events", "n_single", "n_test", "mean_m", "lam_bg", "w_bg", "Ds", "Ds_se",
                                "lid_median", "lid_pooled", "cs", "obs", "se", "exch", "semi_own", "semi_global", "semi_global_ds",
                                "closed_pooled", "closed_own", "err_exch", "err_semi_own", "err_semi_global",
                                "err_semi_global_ds", "err_closed_pooled", "err_closed_own", "note"}],
                    "summary": {"mean_err": {predictor: ..}, "mean_err_common": {predictor: ..}, "strata_compared",
                                "strata_common", "Ds": [...], "lr": {"stat", "dof", "p", "Ds_common"}}, "notes"}}
            strata: lid / rk = equal-count bins of the per-read LID / of r_k; time = contiguous bins of the row order;
            team = one per team (12 largest); m = reads with 1, 2, 3 or more in-ball insertions.
            Ds: one scale fitted on the in-ball insertions of the stratum with the stratum's own background share
            min(lam_bg / mean_m, 1) and its own background shape (by m: the file-level share); Ds_se is the
            approximation Ds / sqrt(sibling insertions).  obs .. closed_own are conflict probabilities (%) of the
            n_test single-insertion reads of the stratum: semi_own / semi_global = semi-parametric form with the
            stratum's / the file's fit (share and D_s); semi_global_ds = the file's D_s with the first-principles
            share of the stratum, which isolates the question whether one D_s suffices; closed_pooled / closed_own =
            closed form with the file's sibling share and D_s and D = pooled LID of the file / median LID of the
            stratum, so that the two differ in D only.  With --by m the test set is ALL reads of
            the stratum, obs is the share whose stale answer is not c-admissible, exch / semi_* are end-to-end
            simulations (simulate_reads) and closed_* the idealized model with that many insertions (Monte Carlo).
            mean_err: max |err| of a predictor averaged over the strata with 30 or more test reads (weighted by reads)
            in which it exists; mean_err_common: the same over the strata in which every predictor exists, so that
            the predictors can be compared.  lr: likelihood-ratio test of a common D_s across the strata that have
            a fit (in-ball insertions of one burst are not independent, so p is too small rather than too large).
  bgrank    {"files": {name: {"k", "n_reads", "kind", "note", "n", "rank_hist", "rank1", "rank1_ci", "rank1_p", "chi2",
                              "chi2_p", "rate_obs", "rate_exp", "rate_ratio", "rate_z", "by_m": {"1": {...}, "2+": {...}},
                              "by_lag": {"<=8": {...}, ">8": {...}}}},
             "pooled": {"k", "files", "n", "n_all", "rank_hist", "rank1", "rank1_ci", "rank1_p", "chi2", "chi2_p",
                        "rate_obs", "rate_exp", "rate_ratio", "rate_z", "by_m", "by_lag", "replicates",
                        "workloads": [{"files", "n", "rank1"}], "rank1_se_cluster", "rank1_p_cluster"},
             "pooled_by_k": {k: {...}}}
            Background insertions: all in-ball insertions of exchangeable controls and of synthetic files with pi = 0,
            the non-sibling ones of planted synthetic files; other files are skipped ("n" null, "note" says why).
            rank1_ci = Wilson 95% interval of the rank-1 share (uniform ranks give 1 / k), rank1_p = exact two-sided
            binomial test, chi2_p = chi-square against uniform ranks (neighbouring ranks merged to expected >= 5).
            rate_obs / rate_exp are COUNTS: in-ball background insertions against sum of n_c k / (N + 1); rate_z is
            the Poisson z-score.  by_m splits by the number of in-ball insertions of the read, by_lag by the stream
            distance between the source task and the reading task.  Files are pooled per k ("pooled" = the k with
            the most insertions).  Runs with the same vectors (same data or generator settings and workload seed; for
            synthetic workloads whatever the sibling parameters, which do not move the base or the non-sibling
            items) see the same pairs: a (reader, source) pair is counted once in the pooled rank tests ("n" against
            "n_all") and the variance of the pooled rate is inflated by n_all / n.  In-ball insertions of one run are
            not independent (shared readers, sources and neighbourhoods), so rank1_p and chi2_p are too small;
            rank1_se_cluster / rank1_p_cluster test the pooled rank-1 share across the independent workloads
            (ratio estimator, Student t, 3 or more workloads needed).

Conventions kept from the first analysis scripts: distances of in-ball insertions are floored at TMIN
(exact duplicates), the background shape is a 3-exponential mixture fitted to the static neighbour lists
(y = ln(r_{k+1} / r_i), i <= k) and then frozen, bootstrap p-values are (1 + #{boot >= obs}) / (B + 1).
"""
import argparse, json, math, os, pickle, re
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import binomtest, chi2, kstest, t as student_t

TMIN = 1e-3            # floor for the distance of an in-ball insertion (exact duplicates have t = 0)
MIN_EV = 30            # fewer in-ball insertions than this: no fit, null in the output
MIN_BUCKET = 8         # fewer in-ball insertions in a lag bucket: no effective dimension
MIN_OV = 100           # fewer overlapping commits in a lag bucket of the source: h is borrowed from a neighbouring bucket
BG_MAX = 100000        # the background shape is fitted on at most this many static neighbour distances
LAG_MAX = 4096
LAG_NEAR = 8           # bgrank: source task within this many stream positions of the reader, or further
MIN_STATIC = 300       # strata: fewer static neighbour distances than this, the file-level background shape is used
DS_BOUNDS = (0.05, 60.0)   # search interval of a sibling dimension
CS = (1.02, 1.05, 1.1, 1.2, 1.5, 2.0)
TAIL = ((1.1, 1.2), (1.2, 1.5), (1.5, 2.0))
BUCKETS = [("other", None), ("1", (1, 1)), ("2-3", (2, 3)), ("4-7", (4, 7)), ("8-15", (8, 15)), ("16-31", (16, 31)), ("32+", (32, LAG_MAX))]


# ----------------------------------------------------------------------------- closed forms
def rank_pmf(alpha, k):
    """P(old rank = p), p = 1..k, of one in-ball insertion with tilt alpha = D_s / D."""
    p = np.arange(1, k + 1)
    return np.exp(math.log(alpha) + gammaln(p - 1 + alpha) - gammaln(p) + gammaln(k) - gammaln(k + alpha))


def adm_joint(c, alpha, D, k):
    """Vector over p = 1..k of P(admissible and rank = p) for one in-ball insertion with tilt alpha."""
    x = float(c) ** (-float(D))
    i = np.arange(1, k)                                           # existing ratios r_{i+1} / r_i, i = 1..k-1
    f = i / (i + alpha) * (1.0 - x ** (i + alpha))
    suffix = np.concatenate([np.cumprod(f[::-1])[::-1], [1.0]])   # suffix[p-1] = prod_{i=p}^{k-1} f_i
    p = np.arange(1, k + 1)
    return alpha / (p - 1 + alpha) * (1.0 - x ** (p - 1 + alpha)) * suffix


def conflict(c, alpha, D, k):
    """P(the stale answer is not c-admissible) after one in-ball insertion with tilt alpha."""
    return 1.0 - float(adm_joint(c, alpha, D, k).sum())


def conflict_given_rank(c, alpha, D, k):
    return 1.0 - adm_joint(c, alpha, D, k) / rank_pmf(alpha, k)


def mix_rank_pmf(pi, alpha, k):
    return pi * rank_pmf(alpha, k) + (1 - pi) / k


def mix_conflict(c, pi, alpha, D, k):
    """Sibling with probability pi (tilt alpha), background otherwise."""
    return pi * conflict(c, alpha, D, k) + (1 - pi) * conflict(c, 1.0, D, k)


def mix_conflict_given_rank(c, pi, alpha, D, k):
    num = pi * (rank_pmf(alpha, k) - adm_joint(c, alpha, D, k)) + (1 - pi) * (1.0 / k - adm_joint(c, 1.0, D, k))
    return num / mix_rank_pmf(pi, alpha, k)


def simulate(D, k, n, rng, m_bg=0, m_cor=0, alpha=1.0, counts=None, pi=None):
    """Monte Carlo of the idealized model.  Either fixed numbers of background / sibling in-ball insertions
    per read, or `counts` (in-ball insertions per read) with each insertion a sibling with probability pi.
    Returns (rho, ranks): rho[r] = smallest c for which the stale answer of read r is still admissible
    (conflict at c  <=>  rho > c), and the old ranks of all insertions in read order."""
    rho = np.ones(n); ranks = []
    M = (m_bg + m_cor) if counts is None else (int(np.max(counts)) if n else 0)
    if M == 0:
        return rho, np.zeros(0, dtype=int)
    step = max(1, int(4e6 // (k * M)))
    for a in range(0, n, step):
        b = min(n, a + step); nn = b - a
        g = np.cumsum(rng.exponential(size=(nn, k)), axis=1)            # volumes of the balls of the k neighbours
        if counts is None:
            live = np.ones((nn, M), dtype=bool); cor = np.broadcast_to(np.arange(M) >= m_bg, (nn, M))
        else:
            live = np.arange(M)[None, :] < np.asarray(counts[a:b]).astype(int)[:, None]
            cor = rng.uniform(size=(nn, M)) < pi
        u = rng.uniform(size=(nn, M))
        w = np.where(live, g[:, -1:] * np.where(cor, u ** (1.0 / alpha), u), np.inf)
        ranks.append((1 + (g[:, None, :] < w[:, :, None]).sum(2))[live])
        new = np.sort(np.concatenate([g, w], axis=1), axis=1)[:, :k]
        rho[a:b] = np.max(g / new, axis=1) ** (1.0 / D)                 # volume ratio -> distance ratio
    return rho, np.concatenate(ranks)


# ----------------------------------------------------------------------------- per-read (semi-parametric) forms
def cor_rank_pmf(ro, k, Ds):
    """[reads, k]: P(old rank = p | read) of one in-ball insertion with radial density ~ t^(Ds-1)."""
    a = (ro[:, :k] / ro[:, k - 1:k]) ** Ds
    return np.diff(np.concatenate([np.zeros((len(ro), 1)), a], axis=1), axis=1)


def cor_conflict(ro, k, Ds, c):
    """[reads]: P(conflict at c | read) of one in-ball insertion with radial density ~ t^(Ds-1), from the
    read's actual neighbour distances (characterization theorem integrated over t)."""
    a = (ro[:, :k] / ro[:, k - 1:k]) ** Ds                            # P(t < r_p)
    lo = np.concatenate([np.zeros((len(ro), 1)), a[:, :-1]], axis=1)  # P(t < r_{p-1})
    ratio = ro[:, 1:k] / np.maximum(ro[:, :k - 1], 1e-9)              # r_{i+1} / r_i, i = 1..k-1
    viol = np.zeros((len(ro), k), dtype=bool)                         # viol[:, p-1]: some ratio after rank p exceeds c
    if k > 1:
        viol[:, :k - 1] = np.maximum.accumulate(ratio[:, ::-1], axis=1)[:, ::-1] > c
    b = np.maximum(lo, a * float(c) ** (-Ds))                         # P(t < r_p / c)
    return np.where(viol, a - lo, b - lo).sum(1)


# ----------------------------------------------------------------------------- exponential mixtures and fits
def mixexp_pdf(y, w, rates):
    return sum(wi * ri * np.exp(-ri * y) for wi, ri in zip(w, rates))


def mixexp_cdf(y, w, rates):
    return sum(wi * (1 - np.exp(-ri * y)) for wi, ri in zip(w, rates))


def fit_mixexp(y, J, iters=400, tol=1e-9):
    """EM for a J-component exponential mixture.  Returns (logL, weights, rates), rates ascending."""
    y = np.asarray(y, dtype=float); best = None
    for seed in range(4):
        r = np.random.default_rng(seed)
        rates = np.sort(1.0 / (y.mean() * np.exp(r.uniform(-2, 2, size=J))))
        w = np.full(J, 1.0 / J); last = -np.inf
        for _ in range(iters):
            dens = w[:, None] * rates[:, None] * np.exp(-rates[:, None] * y[None, :]) + 1e-300
            tot = dens.sum(0); ll = float(np.log(tot).sum())
            g = dens / tot
            w = g.mean(1); rates = g.sum(1) / np.maximum((g * y).sum(1), 1e-12)
            if ll - last < tol * abs(ll):
                break
            last = ll
        ll = float(np.sum(np.log(mixexp_pdf(y, w, rates) + 1e-300)))
        if best is None or ll > best[0]:
            o = np.argsort(rates); best = (ll, w[o], rates[o])
    return best


def fit_sib(y, bg_pdf, w_bg, J):
    """Sibling spectrum with the background share FIXED at w_bg: maximum likelihood of
    w_bg * background + (1 - w_bg) * sum_j a_j Exp(D_j), J = 1 or 2.  Returns shares (of all in-ball
    insertions) and dimensions, ascending in D."""
    y = np.asarray(y, dtype=float); fb = bg_pdf(y) if len(y) else np.zeros(0); tot = 1.0 - float(w_bg)
    if len(y) < 2 or tot <= 1e-9:                                   # no sibling mass (or no data): nothing to estimate
        return {"pi": [max(tot, 0.0)] + [0.0] * (J - 1), "Ds": [float("nan")] * J, "ll": float(np.sum(np.log(fb + 1e-300)))}

    def dens(th):
        if J == 1:
            return w_bg * fb + tot * th[0] * np.exp(-th[0] * y)
        a, d1, d2 = th
        return w_bg * fb + tot * (a * d1 * np.exp(-d1 * y) + (1 - a) * d2 * np.exp(-d2 * y))
    nll = lambda th: -float(np.sum(np.log(dens(th) + 1e-300)))
    best = None
    starts = [[1.0], [3.0], [8.0]] if J == 1 else [[0.3, 1.5, 6.0], [0.5, 1.0, 4.0], [0.2, 2.5, 9.0], [0.7, 3.0, 12.0]]
    bounds = [DS_BOUNDS] if J == 1 else [(0.0, 1.0), DS_BOUNDS, DS_BOUNDS]
    for s0 in starts:
        r = minimize(nll, s0, bounds=bounds, method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    if J == 1:
        return {"pi": [tot], "Ds": [float(best.x[0])], "ll": -float(best.fun)}
    a, d1, d2 = best.x
    comp = sorted([(float(d1), float(a * tot)), (float(d2), float((1 - a) * tot))])
    return {"pi": [c[1] for c in comp], "Ds": [c[0] for c in comp], "ll": -float(best.fun)}


def fit_cor(y, bg_pdf, J=1):
    """Free fit: maximum likelihood of  sum_j pi_j Exp(D_j) + (1 - sum_j pi_j) * background  (shares fitted)."""
    y = np.asarray(y, dtype=float); fb = bg_pdf(y)

    def nll(th):
        pis, ds = np.array(th[:J]), np.array(th[J:])
        if pis.sum() > 1: return 1e12
        dens = (1 - pis.sum()) * fb
        for p, d in zip(pis, ds):
            dens = dens + p * d * np.exp(-d * y)
        return -float(np.sum(np.log(dens + 1e-300)))
    best = None
    starts = [([0.3], [2.0]), ([0.6], [4.0]), ([0.8], [1.0]), ([0.1], [8.0])] if J == 1 else \
             [([0.3, 0.3], [1.5, 6.0]), ([0.2, 0.5], [1.0, 4.0]), ([0.5, 0.2], [2.5, 9.0]), ([0.1, 0.6], [0.7, 5.0])]
    for p0, d0 in starts:
        r = minimize(nll, p0 + d0, bounds=[(0.0, 1.0)] * J + [DS_BOUNDS] * J, method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    pis, ds = np.array(best.x[:J]), np.array(best.x[J:]); o = np.argsort(ds)
    return {"pi": pis[o].tolist(), "Ds": ds[o].tolist(), "ll": -float(best.fun)}


def ks_stat(y, cdf):
    """Kolmogorov distance between the sample y and the model cdf."""
    z = np.sort(np.asarray(y, dtype=float)); n = len(z)
    if n == 0:
        return float("nan")
    F = cdf(z)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(0, n) / n)))


def chisq(cnt, exp, npar=0):
    """Chi-square of observed against expected counts; neighbouring cells are merged until every expected
    count is at least 5.  Returns (chi2, dof, p) or None if nothing can be tested."""
    O, E, o, e = [], [], 0.0, 0.0
    for i in range(len(cnt)):
        o += cnt[i]; e += exp[i]
        if e >= 5:
            O.append(o); E.append(e); o = e = 0.0
    if e > 0 and O:
        O[-1] += o; E[-1] += e
    if len(O) < 2:
        return None
    O, E = np.array(O), np.array(E)
    x2 = float(np.sum((O - E) ** 2 / E)); dof = max(len(O) - 1 - npar, 1)
    return round(x2, 1), dof, float(chi2.sf(x2, dof))


def twonn(ro, drop=0.1):
    """TwoNN-style intrinsic dimension from the ratio of the two nearest stored neighbours of every query:
    slope through the origin of -ln(1 - F(mu)) against ln(mu), mu = r_2 / r_1, largest `drop` share discarded."""
    r1, r2 = ro[:, 0], ro[:, 1]; ok = (r1 > 0) & np.isfinite(r2) & (r2 >= r1)
    mu = np.sort(r2[ok] / r1[ok]); n = len(mu)
    if n < 20:
        return float("nan")
    keep = max(int(n * (1 - drop)), 2)
    x = np.log(mu[:keep]); yv = -np.log(1.0 - np.arange(1, keep + 1) / n)
    return float((x * yv).sum() / max((x * x).sum(), 1e-300))


def _comps(f):
    """Sibling components (share, dimension) of a fit that carry mass."""
    if not f:
        return []
    return [(float(p), float(d)) for p, d in zip(f["pi"], f["Ds"]) if p is not None and d is not None and p > 0 and np.isfinite(d)]


# ----------------------------------------------------------------------------- data
class LawData:
    """One law file, with everything the fits and tests need.

    k, n, cs           scalars and the c grid;  name, meta, workload, task (stream task id of every read)
    lid                per read: maximum-likelihood local intrinsic dimension
    ro, m, rho         per read: old neighbour distances [n, k+1], in-ball insertions, admissibility ratio
    nc, N, lam         per read: overlapping commits, store size, first-principles background n_c k / (N + 1)
    ej, em, ep, et     per in-ball insertion: row of its read, m of that read, old rank, distance (floored at TMIN)
    elag, esame, esrc  per in-ball insertion: stream lag, same team, source task
    y                  ln(r_k / t) of every in-ball insertion
    loo  [n, len(cs)]  P(conflict at c | read) of an insertion exchangeable with the read's neighbours
    ys                 static reference: ln(r_{k+1} / r_i), i <= k, all reads pooled
    bg_pdf, bg_cdf     background shape of y (3-exponential mixture fitted to ys, frozen), bg_ks its KS distance
    lid_median, lid_pool, lid_twonn   local intrinsic dimension: median per-read MLE, 1 / mean(ys), TwoNN
    mean_m, lam_bg, w_bg, nu          insertions per read: observed, background expected, background share
                                      min(lam_bg / mean_m, 1), and the rest (siblings) mean_m - lam_bg
    """

    def __init__(self, src, cs=None, bg_J=3):
        if isinstance(src, dict):
            L = src; self.path = None
        else:
            self.path = str(src)
            with open(src, "rb") as f:
                L = pickle.load(f)
        self.L = L
        self.name = L.get("name") or (re.sub(r"(\.law)?\.pkl$", "", os.path.basename(self.path)) if self.path else "run")
        k = self.k = int(L["k"]); self.n = int(L["n"])
        self.cs = [float(c) for c in (cs if cs else L.get("cs", CS)) if float(c) > 1] or list(CS)
        self.meta = L.get("meta") or {}; self.workload = L.get("workload") or {}
        self.workload_seed = L.get("workload_seed")
        self.ro = np.asarray(L["ro"], dtype=float); self.m = np.asarray(L["m"]).astype(int); self.rho = np.asarray(L["rho"], dtype=float)
        self.task = np.asarray(L["task"]).astype(int) if "task" in L else np.arange(self.n)
        self.nc = np.asarray(L["nc"], dtype=float); self.N = np.asarray(L["N"], dtype=float)
        ev = np.asarray(L["ev"], dtype=float).reshape(-1, 10)
        self.ej = ev[:, 0].astype(int); self.em = ev[:, 1].astype(int); self.ep = ev[:, 2].astype(int)
        self.et = np.maximum(ev[:, 3], TMIN)
        self.elag = ev[:, 7].astype(int); self.esame = ev[:, 8] > 0.5; self.esrc = ev[:, 9].astype(int)
        self.y = np.maximum(np.log(np.maximum(self.ro[self.ej, k - 1], TMIN) / self.et), 0.0)
        self.loo_suf = np.asarray(L["loo_suf"], dtype=float)
        self.loo = (self.loo_suf[:, :, None] > np.array(self.cs)[None, None, :]).mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ys = np.log(self.ro[:, k:k + 1] / np.maximum(self.ro[:, :k], 1e-6)).ravel()
        self.ys = ys[np.isfinite(ys) & (ys >= 0)]
        self.bg_J = bg_J; self._bgfit = None
        self.lid = np.asarray(L["lid"], dtype=float); lid = self.lid[np.isfinite(self.lid)]
        self.lid_median = float(np.median(lid)) if len(lid) else float("nan")
        self.lid_pool = float(1.0 / self.ys.mean()) if len(self.ys) and self.ys.mean() > 0 else float("nan")
        self.lid_twonn = twonn(self.ro)
        self.lam = self.nc * k / (self.N + 1.0)
        self.mean_m = float(self.m.mean()); self.lam_bg = float(self.lam.mean())
        self.w_bg = min(self.lam_bg / self.mean_m, 1.0) if self.mean_m > 0 else 1.0
        self.nu = self.mean_m - self.lam_bg

    # ---- background shape of y: fitted on first use (the lag kernel does not need it)
    @property
    def _bg(self):
        if self._bgfit is None:
            if len(self.ys) < 10:
                raise SystemExit(f"{self.name}: too few static neighbour distances to fit the background shape")
            sub = self.ys if len(self.ys) <= BG_MAX else np.random.default_rng(0).choice(self.ys, size=BG_MAX, replace=False)
            _, w, r = fit_mixexp(sub, self.bg_J)
            self._bgfit = (w, r, ks_stat(self.ys, lambda t: mixexp_cdf(t, w, r)))
        return self._bgfit

    bg_w = property(lambda self: self._bg[0]); bg_rates = property(lambda self: self._bg[1]); bg_ks = property(lambda self: self._bg[2])

    @property
    def bg_pdf(self):
        w, r, _ = self._bg
        return lambda t: mixexp_pdf(t, w, r)

    @property
    def bg_cdf(self):
        w, r, _ = self._bg
        return lambda t: mixexp_cdf(t, w, r)

    # ---- model built from a fit f = {"pi": [...], "Ds": [...]}; the rest of the mass is background
    def model_cdf(self, f):
        cp = _comps(f); wb = 1.0 - sum(p for p, _ in cp)
        return lambda t: wb * self.bg_cdf(t) + sum(p * (1 - np.exp(-d * t)) for p, d in cp)

    def model_pdf(self, f):
        cp = _comps(f); wb = 1.0 - sum(p for p, _ in cp)
        return lambda t: wb * self.bg_pdf(t) + sum(p * d * np.exp(-d * t) for p, d in cp)

    def _rows(self, reads):
        reads = np.asarray(reads)
        return np.where(reads)[0] if reads.dtype == bool else reads.astype(int)

    def pred_conflict(self, f, reads, cs=None):
        """Mean predicted conflict probability (%) over the given reads, one in-ball insertion each."""
        reads = self._rows(reads); cs = self.cs if cs is None else cs
        if len(reads) == 0:
            return [None] * len(cs)
        cp = _comps(f); wb = 1.0 - sum(p for p, _ in cp); out = []
        for c in cs:
            pr = wb * (self.loo_suf[reads] > c).mean(1)
            for p, d in cp:
                pr = pr + p * cor_conflict(self.ro[reads], self.k, d, c)
            out.append(100 * float(pr.mean()))
        return out

    def pred_rank_pmf(self, f, reads):
        """Predicted distribution of the old rank of an in-ball insertion, averaged over the given reads."""
        reads = self._rows(reads); cp = _comps(f)
        pm = np.full(self.k, (1.0 - sum(p for p, _ in cp)) / self.k)
        for p, d in cp:
            pm = pm + p * cor_rank_pmf(self.ro[reads], self.k, d).mean(0)
        return pm

    def obs_conflict(self, sel, cs=None):
        r = self.rho[self._rows(sel)]
        return [100 * float(np.mean(r > c)) if len(r) else None for c in (self.cs if cs is None else cs)]

    def lag_table(self):
        """Lag kernel: per bucket the overlapping commits, the in-ball insertions among them, h, D, shares."""
        same = np.asarray(self.L["conc_same"]); other = int(self.L["conc_other"]); ne = len(self.y); out = {}
        for name, rg in BUCKETS:
            if rg is None:
                sel = ~self.esame; ov = other
            else:
                sel = self.esame & (self.elag >= rg[0]) & (self.elag <= rg[1]); ov = int(same[rg[0]:rg[1] + 1].sum())
            nb = int(sel.sum()); h = nb / ov if ov else None
            ym = float(self.y[sel].mean()) if nb else 0.0
            D = 1.0 / ym if (nb >= MIN_BUCKET and ym > 0) else None
            out[name] = {"overlap": ov, "inball": nb, "h": h, "D": D, "share": nb / ne if ne else None,
                         "rank1": float(np.mean(self.ep[sel] == 1)) if nb >= MIN_BUCKET else None,
                         "h_se": math.sqrt(max(h * (1 - h), 0.0) / ov) if ov else None, "D_se": D / math.sqrt(nb) if D else None}
        return out


def simulate_reads(C, f, rng, reps=30, reads=None):
    """End to end: for each read with m > 0 observed in-ball insertions, draw the insertions from the model
    around the read's actual neighbour list (siblings from their scale, background insertions from the
    pooled static distribution) and compute the admissibility ratio.  Returns the share (%) of ALL reads
    (of the given reads, if `reads` is set) whose stale answer is not c-admissible, for every c of the grid."""
    cp = _comps(f); k = C.k; cs = np.array(C.cs); exceed = np.zeros(len(cs))
    pc = np.array([p for p, _ in cp] + [max(1.0 - sum(p for p, _ in cp), 0.0)]); pc = pc / pc.sum()
    rows = np.arange(C.n) if reads is None else C._rows(reads); tot = len(rows)
    if len(C.ys) == 0 or tot == 0:
        return [None] * len(cs)
    for j in rows[C.m[rows] > 0]:
        r = C.ro[j, :k]; mj = int(C.m[j])
        comp = rng.choice(len(pc), size=(reps, mj), p=pc); u = rng.uniform(size=(reps, mj))
        t = r[-1] * np.exp(-rng.choice(C.ys, size=(reps, mj)))             # exchangeable with existing neighbours
        for i, (_, d) in enumerate(cp):
            t = np.where(comp == i, r[-1] * u ** (1.0 / d), t)
        new = np.sort(np.concatenate([np.broadcast_to(r, (reps, k)), t], axis=1), axis=1)[:, :k]
        rho = np.max(np.maximum(r, 1e-6)[None, :] / np.maximum(new, 1e-6), axis=1)
        exceed += (rho[:, None] > cs[None, :]).sum(0)
    return (100 * exceed / (reps * tot)).tolist()


# ----------------------------------------------------------------------------- helpers
def fmt(xs, w=6, d=1):
    return " ".join(f"{x:{w}.{d}f}" if _ok(x) else " " * (w - 1) + "-" for x in xs)


def _ok(x):
    return x is not None and isinstance(x, (int, float, np.integer, np.floating)) and math.isfinite(x)


def _s(x, spec=".2f"):
    return format(x, spec) if _ok(x) else "-"


def _l(xs, spec=".2f"):
    return "[" + ", ".join(_s(x, spec) for x in xs) + "]"


def max_err(pred, obs):
    d = [abs(a - b) for a, b in zip(pred, obs) if _ok(a) and _ok(b)]
    return max(d) if d else None


def plain(o):
    """numpy -> lists and floats, non-finite numbers -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(a): plain(b) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(a) for a in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def save(path, obj):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plain(obj), f, default=float)
    print(f"\nwrote {path}")


def bic(ll, npar, n):
    return -2.0 * ll + npar * math.log(max(n, 1))


def pctl(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if len(v) >= 10 else [None, None]


def sample_y(C, f, n, rng):
    """n draws of y from the model f (background part from the empirical static distribution)."""
    yb = rng.choice(C.ys, size=n); u = rng.uniform(size=n); lo = 0.0
    for p, d in _comps(f):
        sel = (u >= lo) & (u < lo + p); yb[sel] = rng.exponential(1.0 / d, size=int(sel.sum())); lo += p
    return yb


def ks_boot(C, f, J, B, rng):
    """KS distance of the fixed-background fit and its parametric-bootstrap p-value (refit on every replicate)."""
    k1 = ks_stat(C.y, C.model_cdf(f)); n = len(C.y)
    if B <= 0:
        return k1, None
    ksb = []
    for _ in range(B):
        yb = sample_y(C, f, n, rng)
        ksb.append(ks_stat(yb, C.model_cdf(fit_sib(yb, C.bg_pdf, C.w_bg, J))))
    return k1, float((1 + np.sum(np.array(ksb) >= k1)) / (B + 1))


def boot_ds(C, B, rng):
    """Cluster bootstrap over reads of the one-scale fit; the background share is recomputed on every replicate.
    Returns [B', 2] (sibling share, D_s)."""
    out = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(C.n, size=C.n), minlength=C.n)
        mm = float((cnt * C.m).sum())
        if mm < 5:
            continue
        w = min(float((cnt * C.lam).sum()) / mm, 1.0)
        if w >= 1.0:
            continue
        f = fit_sib(np.repeat(C.y, cnt[C.ej]), C.bg_pdf, w, 1); out.append((f["pi"][0], f["Ds"][0]))
    return np.array(out).reshape(-1, 2)


def slopes(r, loo_suf=None):
    """-d log P(conflict) / d log c on the TAIL windows, from admissibility ratios r (or leave-one-out maxima)."""
    out = []
    for c1, c2 in TAIL:
        p1, p2 = (float(np.mean(r > c1)), float(np.mean(r > c2))) if loo_suf is None else (float(np.mean(loo_suf > c1)), float(np.mean(loo_suf > c2)))
        out.append(math.log(p1 / p2) / math.log(c2 / c1) if min(p1, p2) > 0 else float("nan"))
    return out


def slope_of(v):
    """Same from predicted probabilities v at (1.1, 1.2, 1.5, 2.0)."""
    cc = (1.1, 1.2, 1.5, 2.0)
    return [math.log(v[i] / v[i + 1]) / math.log(cc[i + 1] / cc[i]) if (_ok(v[i]) and _ok(v[i + 1]) and min(v[i], v[i + 1]) > 0) else float("nan") for i in range(3)]


def tail_key(w):
    return f"{w[0]:g}-{w[1]:g}"


def evaluate(C, g, reads=None):
    """Test a model g = {"pi", "Ds"} on the reads of C (boolean mask, default all) without fitting anything."""
    rm = np.ones(C.n, dtype=bool) if reads is None else np.asarray(reads, dtype=bool)
    em = rm[C.ej]; ne = int(em.sum()); k = C.k
    single = np.where(rm & (C.m == 1))[0]; n1 = len(single)
    cnt = np.bincount(C.ep[em], minlength=k + 1)[1:k + 1].astype(float)
    obs = C.obs_conflict(single); pred = C.pred_conflict(g, single)
    return {"obs_m": float(C.m[rm].mean()) if rm.any() else None, "n_events": ne, "n_single": n1,
            "ks": ks_stat(C.y[em], C.model_cdf(g)) if ne >= MIN_EV else None,
            "rank1_pred": float(C.pred_rank_pmf(g, C.ej[em])[0]) if ne else None, "rank1_obs": float(cnt[0] / cnt.sum()) if ne else None,
            "cs": list(C.cs), "obs": obs, "se": [100 * math.sqrt(max(o / 100 * (1 - o / 100), 0.0) / n1) if _ok(o) else None for o in obs],
            "pred": pred, "err": max_err(pred, obs)}


# ----------------------------------------------------------------------------- command: mc
def cmd_mc(a):
    rng = np.random.default_rng(a.seed); n = a.n; rows = []
    cs = [float(x) for x in a.cs.split(",")] if a.cs else list(CS)
    print(f"closed form vs Monte Carlo: one in-ball insertion per read, {n} reads per (k, D, alpha)")
    print("   k   D alpha     c |  closed      MC      se      z | rank-1 closed     MC      z | conflict given rank 1: closed     MC      z   c^-Ds")
    for k in (5, 10, 20, 50):
        for D in (8, 16, 32):
            for alpha in (1.0, 0.5, 0.2, 0.1):
                rho, ranks = simulate(D, k, n, rng, m_bg=1) if alpha == 1.0 else simulate(D, k, n, rng, m_cor=1, alpha=alpha)
                q1 = float(rank_pmf(alpha, k)[0]); r1 = ranks == 1; n1 = int(r1.sum())
                zq = (float(r1.mean()) - q1) / math.sqrt(q1 * (1 - q1) / n)
                for c in cs:
                    cf = conflict(c, alpha, D, k); sim = rho > c; mc = float(sim.mean())
                    se = math.sqrt(max(cf * (1 - cf), 0.0) / n); low = n * cf * (1 - cf) < 10       # normal approximation unusable
                    z = (mc - cf) / se if se > 0 else float("nan")
                    c1 = float(conflict_given_rank(c, alpha, D, k)[0]); m1 = float(sim[r1].mean()) if n1 else float("nan")
                    s1 = math.sqrt(max(c1 * (1 - c1), 0.0) / n1) if n1 else 0.0; low1 = (n1 * c1 * (1 - c1) < 10)
                    z1 = (m1 - c1) / s1 if s1 > 0 else float("nan")
                    rows.append({"k": k, "D": D, "alpha": alpha, "c": c, "closed": cf, "mc": mc, "se": se, "z": z, "low": bool(low),
                                 "rank1_closed": q1, "rank1_mc": float(r1.mean()), "rank1_z": zq,
                                 "cond1_closed": c1, "cond1_mc": m1, "cond1_z": z1, "cond1_low": bool(low1), "own_rank": c ** (-alpha * D)})
                    print(f"  {k:2d}  {D:2d}  {alpha:4.2f}  {c:4.2f} | {cf:7.4f} {mc:7.4f} {se:7.4f} {_s(z, '6.2f'):>6s}{'*' if low else ' '}|"
                          f"       {q1:7.4f} {r1.mean():7.4f} {zq:6.2f} |                        {c1:7.4f} {m1:7.4f} {_s(z1, '6.2f'):>6s}{'*' if low1 else ' '} {c ** (-alpha * D):7.4f}")
    zs = np.array([r["z"] for r in rows if not r["low"]]); z1 = np.array([r["cond1_z"] for r in rows if not r["cond1_low"]])
    zq = np.array(sorted({(r["k"], r["D"], r["alpha"]): r["rank1_z"] for r in rows}.values()))
    worst = max((r for r in rows if not r["low"]), key=lambda r: abs(r["z"]))
    res = {"n": n, "rows": rows, "cells": len(rows), "cells_tested": int(len(zs)), "max_abs_z": float(np.abs(zs).max()),
           "share_abs_z_above_2": float(np.mean(np.abs(zs) > 2)), "max_abs_z_rank1": float(np.abs(zq).max()),
           "max_abs_z_cond1": float(np.abs(z1).max()) if len(z1) else None, "cond1_cells_tested": int(len(z1)),
           "max_abs_diff": float(max(abs(r["closed"] - r["mc"]) for r in rows))}
    print("  (* = fewer than 10 expected conflicts or non-conflicts: z is not meaningful and the cell is left out of the summary;")
    print("     se = binomial standard error of the Monte Carlo estimate under the closed form; the c values of one (k, D, alpha) share one simulation)")
    print(f"summary: {len(rows)} cells, {len(zs)} tested: largest |z| = {res['max_abs_z']:.2f} (k={worst['k']} D={worst['D']} alpha={worst['alpha']} c={worst['c']}), "
          f"|z| > 2 in {100 * res['share_abs_z_above_2']:.1f}% of the cells (5% expected), largest |closed - MC| = {res['max_abs_diff']:.4f}")
    print(f"         rank-1 share: largest |z| = {res['max_abs_z_rank1']:.2f} over {len(zq)} simulations; conflict given rank 1: largest |z| = {_s(res['max_abs_z_cond1'])} over {len(z1)} cells")
    save(a.out, res)


# ----------------------------------------------------------------------------- command: grid
def cmd_grid(a):
    cs = np.geomspace(1.005, 2.0, 60); dss = np.linspace(0.5, 20.0, 60)
    Dl, kl, pil = [8, 16, 32], [10, 50], [0.0, 0.25, 0.5, 0.75, 1.0]
    out = np.zeros((len(Dl), len(kl), len(pil), len(dss), len(cs))); bgs = np.zeros((len(Dl), len(kl), len(cs)))
    for iD, D in enumerate(Dl):
        for ik, k in enumerate(kl):
            bg = np.array([conflict(c, 1.0, D, k) for c in cs]); bgs[iD, ik] = bg
            sib = np.array([[conflict(c, ds / D, D, k) for c in cs] for ds in dss])
            for ip, pi in enumerate(pil):
                out[iD, ik, ip] = pi * sib + (1 - pi) * bg[None, :]
    print("closed-form conflict probability of one in-ball insertion: P = pi * sibling(D_s) + (1 - pi) * background(D)")
    print(f"grid: c = geomspace(1.005, 2, 60), D_s = linspace(0.5, 20, 60), D in {Dl}, k in {kl}, pi in {pil}; index order [D][k][pi][Ds][c]")
    ci = [int(np.argmin(np.abs(cs - c))) for c in (1.02, 1.05, 1.1, 1.2, 1.5, 2.0)]
    di = [int(np.argmin(np.abs(dss - d))) for d in (1, 2, 4, 8, 16)]
    for iD, D in enumerate(Dl):
        for ik, k in enumerate(kl):
            print(f"D={D} k={k}: conflict probability (%), pure siblings (pi = 1);   c = " + " ".join(f"{cs[i]:6.3f}" for i in ci))
            for i in di:
                print(f"      D_s = {dss[i]:5.2f}                                                " + fmt(100 * out[iD, ik, -1, i, ci]))
            print("      background (pi = 0)                                        " + fmt(100 * bgs[iD, ik, ci]))
    save(a.out, {"c": cs, "Ds": dss, "D": Dl, "k": kl, "pi": pil, "index": "conflict[D][k][pi][Ds][c], background[D][k][c]",
                 "conflict": np.round(out, 6), "background": np.round(bgs, 6)})


# ----------------------------------------------------------------------------- command: fit
def fit_one(C, rng, boot=200, boot_ks=None, reps=30):
    """All analyses of one law file.  Prints the report, returns the JSON entry."""
    k, n, cs = C.k, C.n, C.cs; ne = len(C.y); boot_ks = boot if boot_ks is None else boot_ks
    single = np.where(C.m == 1)[0]; n1 = len(single); notes = []
    cnt = np.bincount(C.ep, minlength=k + 1)[1:k + 1].astype(float)
    res = {"k": k, "n": n, "events": ne, "mean_m": C.mean_m, "lid_median": C.lid_median, "lid_pooled": C.lid_pool, "lid_twonn": C.lid_twonn,
           "lam_bg": C.lam_bg, "w_bg": C.w_bg, "nu": C.nu, "cs": list(cs), "notes": notes,
           "char": [int(C.L.get("char_ok", 0)), int(C.L.get("char_tot", 0))],
           "f1": None, "f2": None, "free1": None, "single": None, "rank": None, "e2e": None, "tail": None, "lid_sensitivity": None,
           "y_hist": None, "y_model1": None, "y_model2": None, "y_bg": None}
    zbg = (C.m.sum() - C.lam.sum()) / math.sqrt(max(C.lam.sum(), 1e-12))
    res["bg"] = {"shape_ks": C.bg_ks, "shape_w": C.bg_w, "shape_rates": C.bg_rates, "rate_obs": C.mean_m, "rate_exp": C.lam_bg, "rate_z": zbg}
    print(f"\n================ {C.name}: k={k}, reads {n}, in-ball concurrent insertions {ne} ({C.mean_m:.4f} per read, reads with one or more {100 * np.mean(C.m > 0):.2f}%), "
          f"single-insertion reads {n1}; characterization agrees {res['char'][0]}/{res['char'][1]}")
    print(f"   intrinsic dimension of the stored neighbours: median per-read MLE {C.lid_median:.1f}, pooled {C.lid_pool:.1f}, TwoNN {_s(C.lid_twonn, '.1f')};"
          f" background shape (3 exponentials): weights {_l(C.bg_w)} rates {_l(C.bg_rates, '.1f')}, KS {C.bg_ks:.3f}")
    print(f"A. first-principles background: overlapping commits per read {C.nc.mean():.1f}, store size {C.N.mean():.0f} -> expected background insertions per read {C.lam_bg:.4f}"
          f" | observed {C.mean_m:.4f} (z = {zbg:+.1f} against background only) -> background share {100 * C.w_bg:.1f}%, sibling insertions per read {C.nu:.4f}")
    if ne < MIN_EV:
        notes.append(f"only {ne} in-ball insertions (fewer than {MIN_EV}): no fit")
        print(f"   {notes[-1]}")
        return res

    # ---- B. sibling spectrum, background share fixed
    y = C.y; ll0 = float(np.sum(np.log(C.bg_pdf(y) + 1e-300)))
    res["bg"].update({"ll": ll0, "bic": bic(ll0, 0, ne), "ks": ks_stat(y, C.bg_cdf)})
    f1, f2 = fit_sib(y, C.bg_pdf, C.w_bg, 1), fit_sib(y, C.bg_pdf, C.w_bg, 2)
    free = fit_cor(y, C.bg_pdf, 1)
    if not _comps(f1):
        notes.append("the first-principles background accounts for all in-ball insertions: no sibling mass to fit")
    bs = boot_ds(C, boot, rng) if (boot > 0 and _comps(f1)) else np.zeros((0, 2))
    f1["Ds_ci"] = [pctl(bs[:, 1])]; f1["pi_ci"] = [pctl(bs[:, 0])]
    print("B. sibling spectrum with the background share fixed (only insertion positions y = ln(r_k / t) are used)")
    print(f"   background only : logL {ll0:9.1f} BIC {bic(ll0, 0, ne):9.1f} KS {res['bg']['ks']:.3f}")
    for lab, f, npar in (("1 scale ", f1, 1), ("2 scales", f2, 3)):
        J = len(f["Ds"]); f["bic"] = bic(f["ll"], npar, ne)
        f["ks"], f["ks_p"] = ks_boot(C, f, J, boot_ks, rng)
        pm = C.pred_rank_pmf(f, C.ej); x2 = chisq(cnt, pm * cnt.sum(), 0)
        f["rank_chi2"] = list(x2[:2]) if x2 else None; f["rank_chi2_p"] = x2[2] if x2 else None; f["rank_pmf"] = pm
        ci = f" (95% CI {_l(f['Ds_ci'][0])})" if J == 1 else ""
        print(f"   {lab}        : shares {_l(f['pi'], '.3f')} D_s {_l(f['Ds'])}{ci} | logL {f['ll']:9.1f} BIC {f['bic']:9.1f} | KS {f['ks']:.3f} (bootstrap p = {_s(f['ks_p'], '.3f')})"
              f" | ranks (not fitted): rank-1 share observed {100 * cnt[0] / cnt.sum():.1f}% predicted {100 * pm[0]:.1f}%, chi2 {x2[0] if x2 else '-'}/{x2[1] if x2 else '-'} p = {_s(x2[2] if x2 else None, '.2g')}")
    print(f"   free fit, share fitted (1 scale): sibling share {_l(free['pi'], '.3f')} D_s {_l(free['Ds'])} | logL {free['ll']:9.1f} BIC {bic(free['ll'], 2, ne):9.1f} | KS {ks_stat(y, C.model_cdf(free)):.3f}")
    for lab, f in (("1 scale", f1), ("2 scales", f2)):
        if any(d <= DS_BOUNDS[0] * 1.01 or d >= DS_BOUNDS[1] * 0.99 for _, d in _comps(f)):
            notes.append(f"{lab}: a fitted sibling dimension sits at a bound of the search interval {DS_BOUNDS} (component not identified)")
            print(f"   note: {notes[-1]}")
    res["f1"] = {a_: f1[a_] for a_ in ("pi", "Ds", "Ds_ci", "pi_ci", "ll", "bic", "ks", "ks_p", "rank_chi2", "rank_chi2_p")}
    res["f2"] = {a_: f2[a_] for a_ in ("pi", "Ds", "ll", "bic", "ks", "ks_p", "rank_chi2", "rank_chi2_p")}
    res["free1"] = {"pi": free["pi"], "Ds": free["Ds"], "ll": free["ll"], "bic": bic(free["ll"], 2, ne), "ks": ks_stat(y, C.model_cdf(free))}
    res["rank"] = {"obs": cnt / cnt.sum(), "pred1": f1["rank_pmf"], "pred2": f2["rank_pmf"], "exch": np.full(k, 1.0 / k)}

    # ---- densities of y for plotting
    edges = np.linspace(0.0, max(float(np.percentile(y, 99.5)), 1e-6), 41); mid = 0.5 * (edges[1:] + edges[:-1])
    res["y_hist"] = {"edges": edges, "centres": mid, "density": np.histogram(y, edges)[0] / (ne * np.diff(edges))}
    res["y_model1"], res["y_model2"], res["y_bg"] = C.model_pdf(f1)(mid), C.model_pdf(f2)(mid), C.bg_pdf(mid)

    # ---- C. single-insertion reads
    D = C.lid_pool; ps1 = sum(p for p, _ in _comps(f1)); ds1 = _comps(f1)[0][1] if _comps(f1) else D
    closed_at = lambda Dx, cc=None: [100 * mix_conflict(c, ps1, ds1 / Dx, Dx, k) for c in (cs if cc is None else cc)]
    closed2_at = lambda Dx, cc=None: [100 * (sum(p * conflict(c, d / Dx, Dx, k) for p, d in _comps(f2)) + (1 - sum(p for p, _ in _comps(f2))) * conflict(c, 1.0, Dx, k))
                                      for c in (cs if cc is None else cc)]
    print("C. single-insertion reads: conflict probability (%); outcomes are never used in a fit")
    print("   c                                 " + " ".join(f"{c:6.2f}" for c in cs))
    if n1 == 0:
        notes.append("no single-insertion reads: no conflict test")
        print(f"   {notes[-1]}")
    else:
        if n1 < MIN_EV:
            notes.append(f"only {n1} single-insertion reads: conflict probabilities are very noisy")
        obs = C.obs_conflict(single); se = [100 * math.sqrt(max(o / 100 * (1 - o / 100), 0.0) / n1) for o in obs]
        rs = C.rho[single]
        if boot > 0:
            bo = np.stack([100 * np.mean(rs[rng.integers(n1, size=n1)][:, None] > np.array(cs)[None, :], axis=0) for _ in range(boot)])
            ci = [pctl(bo[:, i]) for i in range(len(cs))]
        else:
            ci = [[None, None] for _ in cs]
        P = {"exch": [100 * float(C.loo[single, i].mean()) for i in range(len(cs))], "closed": closed_at(D), "closed2": closed2_at(D),
             "semi1": C.pred_conflict(f1, single), "semi2": C.pred_conflict(f2, single)}
        res["single"] = {"n": n1, "obs": obs, "se": se, "ci": ci, **P, **{"err_" + a_: max_err(v, obs) for a_, v in P.items()}}
        print(f"   observed (n = {n1:5d})              {fmt(obs)}")
        print(f"   binomial standard error           {fmt(se)}")
        print(f"   bootstrap 95% CI, lower           {fmt([c_[0] for c_ in ci])}")
        print(f"   bootstrap 95% CI, upper           {fmt([c_[1] for c_ in ci])}")
        for a_, lab in (("exch", "exchangeable only (leave-one-out)"), ("closed", f"closed form, 1 scale, D = {D:4.1f}   "), ("closed2", f"closed form, 2 scales, D = {D:4.1f}  "),
                        ("semi1", "semi-parametric, 1 scale         "), ("semi2", "semi-parametric, 2 scales        ")):
            print(f"   {lab} {fmt(P[a_])}   max |err| {_s(res['single']['err_' + a_], '.1f')}")

    # ---- D. end to end
    obs_all = [100 * float(np.mean(C.rho > c)) for c in cs]
    res["e2e"] = {"obs": obs_all, "m1": simulate_reads(C, f1, rng, reps), "m2": simulate_reads(C, f2, rng, reps), "bg": simulate_reads(C, {"pi": [], "Ds": []}, rng, reps)}
    print(f"D. all reads: share (%) whose stale answer is not c-admissible, given the observed number of in-ball insertions of every read ({reps} simulations per read)")
    for a_, lab in (("obs", "observed                         "), ("m1", "simulated, 1 scale               "), ("m2", "simulated, 2 scales              "), ("bg", "simulated, background only       ")):
        print(f"   {lab} {fmt(res['e2e'][a_])}" + ("" if a_ == "obs" else f"   max |err| {_s(max_err(res['e2e'][a_], obs_all), '.1f')}"))

    # ---- E. tail exponent
    tl = {"obs": {}, "exch": {}, "model1": {}, "model2": {}, "Ds1": f1["Ds"][0], "Ds2": f2["Ds"]}
    if n1 > 0:
        rs = C.rho[single]; s0 = slopes(rs)
        sb = np.array([slopes(rs[rng.integers(n1, size=n1)]) for _ in range(boot)]).reshape(-1, len(TAIL))
        se_ = slopes(None, C.loo_suf[single])
        cc = (1.1, 1.2, 1.5, 2.0); sm1, sm2 = slope_of(C.pred_conflict(f1, single, cc)), slope_of(C.pred_conflict(f2, single, cc))
        for i, w in enumerate(TAIL):
            col = sb[:, i] if len(sb) else np.zeros(0)
            lo, hi = pctl(col) if (len(col) and np.isfinite(col).mean() >= 0.9 and math.isfinite(s0[i])) else (None, None)
            tl["obs"][tail_key(w)] = [s0[i], lo, hi]; tl["exch"][tail_key(w)] = se_[i]; tl["model1"][tail_key(w)] = sm1[i]; tl["model2"][tail_key(w)] = sm2[i]
        print("E. tail exponent: slope of log P(conflict) against log c (sign reversed), single-insertion reads; for large c it tends to the smallest sibling dimension that carries mass")
        for w in TAIL:
            o = tl["obs"][tail_key(w)]
            print(f"   c in [{w[0]:g}, {w[1]:g}]: observed {_s(o[0])} (95% CI {_s(o[1])}, {_s(o[2])}) | model 1 scale {_s(tl['model1'][tail_key(w)])}, 2 scales {_s(tl['model2'][tail_key(w)])}"
                  f" | exchangeable only {_s(tl['exch'][tail_key(w)])}")
        print(f"   fitted sibling dimensions: 1 scale {_l(f1['Ds'])}, 2 scales {_l(f2['Ds'])} (shares {_l(f2['pi'], '.2f')})")
    res["tail"] = tl

    # ---- F. sensitivity of the closed form to the estimate of the local dimension D
    if n1 > 0:
        ls = {}
        print("F. closed form (1 scale) under three estimates of the local dimension D")
        for a_, Dx in (("median", C.lid_median), ("pooled", C.lid_pool), ("twonn", C.lid_twonn)):
            pr = closed_at(Dx) if _ok(Dx) and Dx > 0 else [None] * len(cs)
            ls[a_] = {"D": Dx, "closed": pr, "err_closed": max_err(pr, res["single"]["obs"])}
            print(f"   {a_:6s} D = {_s(Dx, '5.1f')}: {fmt(pr)}   max |err| {_s(ls[a_]['err_closed'], '.1f')}")
        res["lid_sensitivity"] = ls
    return res


def cmd_fit(a):
    rng = np.random.default_rng(a.seed); out = {}
    for path in a.files:
        C = LawData(path, cs=a.cs)
        out[_unique(C.name, out)] = fit_one(C, rng, a.boot, a.boot_ks, a.reps)
    save(a.out, out)


def _unique(name, d):
    i, nm = 2, name
    while nm in d:
        nm = f"{name}#{i}"; i += 1
    return nm


# ----------------------------------------------------------------------------- command: report
def cmd_report(a):
    with open(a.files[0]) as f:
        R = json.load(f)
    for nm, r in R.items():
        if nm.startswith("_"):
            continue
        print(f"\n{nm}: k={r['k']} reads={r['n']} in-ball insertions={r['events']} ({r['mean_m']:.4f} per read) | LID median {_s(r['lid_median'], '.1f')} pooled {_s(r['lid_pooled'], '.1f')} TwoNN {_s(r['lid_twonn'], '.1f')}")
        print(f"  background from first principles {r['lam_bg']:.4f} per read (share {100 * r['w_bg']:.1f}%), sibling insertions per read {r['nu']:.4f}")
        for note in r.get("notes") or []:
            print(f"  note: {note}")
        if not r.get("f1"):
            continue
        f1, f2, fr = r["f1"], r["f2"], r["free1"]
        print(f"  1 scale : share {_l(f1['pi'], '.3f')} D_s {_l(f1['Ds'])} CI {_l(f1['Ds_ci'][0])} | BIC {f1['bic']:.1f} | KS {f1['ks']:.3f} (p = {_s(f1['ks_p'], '.3f')}) | rank chi2 p = {_s(f1['rank_chi2_p'], '.2g')}")
        print(f"  2 scales: share {_l(f2['pi'], '.3f')} D_s {_l(f2['Ds'])} | BIC {f2['bic']:.1f} | KS {f2['ks']:.3f} (p = {_s(f2['ks_p'], '.3f')}) | rank chi2 p = {_s(f2['rank_chi2_p'], '.2g')}")
        print(f"  free fit: share {_l(fr['pi'], '.3f')} D_s {_l(fr['Ds'])}")
        if r.get("rank"):
            k = r["k"]; q = r["rank"]
            print(f"  rank-1 share: observed {100 * q['obs'][0]:.1f}% | 1 scale {100 * q['pred1'][0]:.1f}% | 2 scales {100 * q['pred2'][0]:.1f}% | exchangeable {100 / k:.1f}%")
        s = r.get("single")
        print("  c                                " + " ".join(f"{c:6.2f}" for c in r["cs"]))
        if s:
            print(f"  single-insertion reads (n = {s['n']}), conflict %")
            print(f"    observed                       {fmt(s['obs'])}   (se {fmt(s['se'], 4)})")
            for a_, lab in (("exch", "exchangeable only     "), ("closed", "closed form           "), ("semi1", "semi-parametric 1 sc. "), ("semi2", "semi-parametric 2 sc. ")):
                print(f"    {lab}         {fmt(s[a_])}   max |err| {_s(s['err_' + a_], '.1f')}")
        e = r.get("e2e")
        if e:
            print("  all reads, not c-admissible %")
            for a_, lab in (("obs", "observed              "), ("m1", "simulated 1 scale     "), ("m2", "simulated 2 scales    "), ("bg", "background only       ")):
                print(f"    {lab}         {fmt(e[a_])}")
        t = r.get("tail")
        if t and t.get("obs"):
            print("  tail slope: " + ", ".join(f"[{w}] {_s(v[0])} ({_s(v[1])}, {_s(v[2])})" for w, v in t["obs"].items()) + f" | D_s 1 scale {_s(t['Ds1'])}, 2 scales {_l(t['Ds2'])}")
        ls = r.get("lid_sensitivity")
        if ls:
            print("  closed-form max |err| by choice of D: " + ", ".join(f"{a_} (D = {_s(v['D'], '.1f')}) {_s(v['err_closed'], '.1f')}" for a_, v in ls.items()))


# ----------------------------------------------------------------------------- command: lag
def _group(C):
    w = {a_: b for a_, b in C.workload.items() if a_ not in ("agents", "name")}
    lab = os.path.basename(str(w.get("data", w.get("kind", "workload")))).replace(".npz", "")
    extra = ",".join(f"{a_}={w[a_]}" for a_ in sorted(w) if a_ not in ("data", "kind", "base", "stream", "teams", "n_base", "n_stream", "dim"))
    return f"{lab} k={C.k}" + (f" [{extra}]" if extra else ""), json.dumps([w, C.k], sort_keys=True, default=str)


def cmd_lag(a):
    out = {}; groups = {}
    print("lag kernel: h = P(an overlapping commit at that stream lag falls inside the read's top-k ball), D = effective dimension 1 / mean(ln(r_k / t))")
    for path in a.files:
        C = LawData(path, cs=a.cs); T = C.lag_table(); hx = float(np.mean(C.k / (C.N + 1.0)))
        nm = _unique(C.name, out); out[nm] = {"buckets": T, "exchangeable_h": hx, "k": C.k, "n": C.n}
        lab, key = _group(C); groups.setdefault(key, [lab, []])[1].append(nm)
        print(f"\n{nm}: k={C.k}, reads {C.n}, in-ball insertions {len(C.y)}; exchangeable reference h = {100 * hx:.4f}%")
        print("   bucket     overlapping commits   in-ball        h %    (se)   h / exch.   eff. dimension (se)   share of in-ball %   new nearest %")
        for b, _ in BUCKETS:
            v = T[b]
            if v["overlap"] == 0 and v["inball"] == 0:
                continue
            print(f"   {b:8s} {v['overlap']:20d} {v['inball']:9d} {_s(100 * v['h'] if _ok(v['h']) else None, '10.4f'):>10s} {_s(100 * v['h_se'] if _ok(v['h_se']) else None, '7.4f'):>7s}"
                  f" {_s(v['h'] / hx if _ok(v['h']) else None, '9.1f'):>9s}   {_s(v['D'], '8.2f'):>8s} {('(' + _s(v['D_se'], '.2f') + ')'):>8s}"
                  f" {_s(100 * v['share'] if _ok(v['share']) else None, '18.1f'):>18s} {_s(100 * v['rank1'] if _ok(v['rank1']) else None, '15.1f'):>15s}")
    inv = {}
    print("\ninvariance across concurrency levels (same workload and k; buckets with at least 8 in-ball insertions in two or more files)")
    for key, (lab, names) in groups.items():
        if len(names) < 2:
            continue
        g = {"files": names, "buckets": {}}; print(f"   {lab}: {', '.join(names)}")
        for b, _ in BUCKETS:
            use = [nm for nm in names if out[nm]["buckets"][b]["D"] is not None]
            if len(use) < 2:
                continue
            hs = [out[nm]["buckets"][b]["h"] for nm in use]; ds = [out[nm]["buckets"][b]["D"] for nm in use]
            g["buckets"][b] = {"files": use, "h": hs, "D": ds, "h_ratio": max(hs) / min(hs), "D_min": min(ds), "D_max": max(ds)}
            print(f"      {b:6s} h % " + " / ".join(f"{100 * h:.3f}" for h in hs) + f" (max / min {max(hs) / min(hs):.2f}) | D " + " / ".join(f"{d:.2f}" for d in ds) + f" (range {min(ds):.2f} to {max(ds):.2f})")
        inv[_unique(lab, inv)] = g
    if not inv:
        print("   (needs two or more files of the same workload and k)")
    out["_invariance"] = inv
    save(a.out, out)


# ----------------------------------------------------------------------------- command: transfer
def lag_model(S, T, notes):
    """Kernel estimated on S, applied to the overlap counts of T.  Returns (model, predicted insertions per read)."""
    ks, kd = S.lag_table(), T.lag_table(); names = [b for b, _ in BUCKETS]
    has_h = [b for b in names if ks[b]["overlap"] >= MIN_OV]; has_d = [b for b in names if ks[b]["D"] is not None]

    def nearest(b, pool):                                   # nearest shorter lag of the source, else nearest longer
        i = names.index(b); cand = [x for x in pool if x != "other"]
        short = [x for x in cand if names.index(x) < i]; lng = [x for x in cand if names.index(x) > i]
        return short[-1] if short else (lng[0] if lng else None)
    nu, Ds, bgm = [], [], 0.0
    for b in names:
        ov = kd[b]["overlap"]
        if ov == 0:
            continue
        if b == "other" and b not in has_h:                 # source has no other teams: treat them as exchangeable
            bgm += ov / T.n * float(np.mean(T.k / (T.N + 1.0))); notes.append("other teams: no data in the source, treated as background"); continue
        bh = b if b in has_h else nearest(b, has_h); bd = b if b in has_d else (nearest(b, has_d) if b != "other" else None)
        if bh is None:
            notes.append(f"bucket {b}: no usable bucket in the source, left out"); continue
        v = ov / T.n * ks[bh]["h"]
        if bh != b and bd == bh:
            notes.append(f"bucket {b}: not observed in the source, h and D taken from bucket {bh}")
        elif bh != b:
            notes.append(f"bucket {b}: not observed in the source, h taken from bucket {bh}")
        if bd is None:                                      # no dimension available: shape of the background
            bgm += v; notes.append(f"bucket {b}: no effective dimension in the source, background shape used"); continue
        if bd != b and bd != bh:
            notes.append(f"bucket {b}: too few in-ball insertions in the source, D taken from bucket {bd}")
        nu.append(v); Ds.append(ks[bd]["D"])
    tot = sum(nu) + bgm
    return {"pi": [v / tot for v in nu] if tot > 0 else [], "Ds": Ds}, tot


def cmd_transfer(a):
    mode = a.mode; J = a.scales; notes = []
    S = LawData(a.src, cs=a.cs)
    if mode == "time":
        T = S
    elif not a.dst:
        raise SystemExit("transfer: --dst is required (except for --mode time)")
    else:
        T = LawData(a.dst, cs=a.cs)
    res = {"src": S.name, "dst": T.name, "mode": mode, "scales": None if mode == "lag" else J, "cs": list(T.cs), "notes": notes}
    rm = None; srcfit = None; pred_m = None; g = None
    if len(S.y) < MIN_EV:
        notes.append(f"source has only {len(S.y)} in-ball insertions (fewer than {MIN_EV}): no prediction")
    elif mode == "lag":
        if S.k != T.k:
            notes.append(f"k differs (source {S.k}, target {T.k}): the kernel depends on k, the prediction is not expected to hold")
        g, pred_m = lag_model(S, T, notes)
    elif mode == "time":
        order = np.argsort(S.task, kind="stable"); tr = np.zeros(S.n, dtype=bool); tr[order[:S.n // 2]] = True; rm = ~tr
        etr = tr[S.ej]; mtr = float(S.m[tr].mean())
        if etr.sum() < MIN_EV or (~etr).sum() < MIN_EV:
            notes.append("fewer than 30 in-ball insertions in one of the halves: no prediction")
        else:
            w = min(float(S.lam[tr].mean()) / mtr, 1.0); srcfit = fit_sib(S.y[etr], S.bg_pdf, w, J)
            nus = [p * mtr for p, _ in _comps(srcfit)]; lam = float(S.lam[rm].mean()); pred_m = lam + sum(nus)
            g = {"pi": [v / pred_m for v in nus], "Ds": [d for _, d in _comps(srcfit)]}
            notes.append("the background shape is the one of the whole run (static neighbour lists only, no outcomes)")
    else:
        srcfit = fit_sib(S.y, S.bg_pdf, S.w_bg, J); cp = _comps(srcfit)
        if mode == "params":
            g = {"pi": [p for p, _ in cp], "Ds": [d for _, d in cp]}; pred_m = S.mean_m
        else:
            if mode == "window":
                ratio = float(T.nc.mean() / S.nc.mean()); scale = lambda d: ratio; res["nc_ratio"] = ratio
                if S.k != T.k:
                    notes.append(f"k differs (source {S.k}, target {T.k}): window scaling alone is not expected to hold")
            else:                                           # k: growth of the sibling count with the ball
                big = T if T.k >= S.k else S; ka, kb = S.k, T.k
                scale = lambda d: float(np.mean((big.ro[:, kb - 1] / np.maximum(big.ro[:, ka - 1], 1e-9)) ** d))
                res["nc_ratio"] = float(T.nc.mean() / S.nc.mean()); res["growth"] = [scale(d) for _, d in cp]
                if abs(res["nc_ratio"] - 1) > 0.05:
                    notes.append(f"overlapping commits differ by a factor {res['nc_ratio']:.2f}: not corrected in mode k")
                if T.k < S.k:
                    notes.append("target k is smaller than source k: radii taken from the source file")
            nus = [p * S.mean_m * scale(d) for p, d in cp]; pred_m = T.lam_bg + sum(nus)
            g = {"pi": [v / pred_m for v in nus], "Ds": [d for _, d in cp]}
    print(f"transfer {S.name} -> {T.name}, mode {mode}" + (f", {J} scale{'s' if J > 1 else ''}" if mode != "lag" else "") + " (nothing is fitted on the target)")
    if srcfit is not None:
        res["src_fit"] = {"pi": srcfit["pi"], "Ds": srcfit["Ds"]}
        print(f"   source fit: sibling shares {_l(srcfit['pi'], '.3f')} D_s {_l(srcfit['Ds'])}")
    if g is None:
        res.update({"pred_m": None, "obs_m": float(T.m.mean()), "ks": None, "rank1_pred": None, "rank1_obs": None, "obs": None, "pred": None, "err": None})
    else:
        E = evaluate(T, g, rm); res.update(E); res["pred_m"] = pred_m; res["model"] = g
        single = np.where((np.ones(T.n, dtype=bool) if rm is None else rm) & (T.m == 1))[0]
        ex = [100 * float(T.loo[single, i].mean()) for i in range(len(T.cs))] if len(single) else [None] * len(T.cs)
        res["ref"] = {"exch": ex, "err_exch": max_err(ex, E["obs"]), "self": None, "err_self": None}
        em = np.ones(len(T.y), dtype=bool) if rm is None else rm[T.ej]
        if em.sum() >= MIN_EV and T.w_bg < 1:               # in-sample reference: the target's own fit (not a prediction)
            wt = T.w_bg if rm is None else min(float(T.lam[rm].mean()) / max(float(T.m[rm].mean()), 1e-12), 1.0)
            own = T.pred_conflict(fit_sib(T.y[em], T.bg_pdf, wt, 1 if mode == "lag" else J), single)
            res["ref"].update({"self": own, "err_self": max_err(own, E["obs"])})
        print(f"   transferred model: sibling shares {_l(g['pi'], '.3f')} D_s {_l(g['Ds'])}, background share {100 * (1 - sum(g['pi'])):.1f}%")
        print(f"   insertions per read: predicted {pred_m:.4f} observed {E['obs_m']:.4f} | KS(y) {_s(E['ks'], '.3f')} | rank-1 share predicted {_s(100 * E['rank1_pred'] if _ok(E['rank1_pred']) else None, '.1f')}%"
              f" observed {_s(100 * E['rank1_obs'] if _ok(E['rank1_obs']) else None, '.1f')}%")
        print(f"   single-insertion reads of the target (n = {E['n_single']}): conflict probability (%)")
        print("   c                        " + " ".join(f"{c:6.2f}" for c in T.cs))
        print(f"   observed                 {fmt(E['obs'])}   (se {fmt(E['se'], 4)})")
        print(f"   predicted (transfer)     {fmt(E['pred'])}   max |err| {_s(E['err'], '.1f')}")
        print(f"   exchangeable only        {fmt(ex)}   max |err| {_s(res['ref']['err_exch'], '.1f')}")
        if res["ref"]["self"]:
            print(f"   target's own fit         {fmt(res['ref']['self'])}   max |err| {_s(res['ref']['err_self'], '.1f')}   (in-sample reference)")
    for note in dict.fromkeys(notes):
        print(f"   note: {note}")
    res["notes"] = list(dict.fromkeys(notes))
    save(a.out, res)


# ----------------------------------------------------------------------------- command: planted
def planted_siblings(C, seeds):
    """Boolean mask over the in-ball insertions: the source is the planted parent of the reading task or its child.
    The workload is rebuilt from the law file; returns (mask or None, note)."""
    try:
        from .workloads import make_workload
    except Exception as e:
        return None, f"ariadne.workloads is not importable ({type(e).__name__}: {e})"
    why = "no seed to try"
    for seed in seeds:
        try:
            wl = make_workload(dict(C.workload), seed, C.k)
        except Exception as e:
            why = f"the workload could not be rebuilt ({type(e).__name__}: {e})"; continue
        par = None if wl.truth is None else np.asarray(wl.truth)
        if par is None or len(par) <= max(int(C.task.max()), int(C.esrc.max()) if len(C.esrc) else 0):
            why = "the rebuilt workload has no parent structure for these tasks"; continue
        R0, R1 = C.meta.get("R"), wl.meta.get("R")
        if R0 is not None and R1 is not None and abs(R0 - R1) > 1e-6 * max(abs(R0), 1.0):
            why = f"the workload rebuilt with seed {seed} does not match the run (radius {R1:.4f} against {R0:.4f})"; continue
        task = C.task[C.ej]; down = par[task] == C.esrc; up = par[C.esrc] == task; sib = down | up
        tp = wl.meta.get("t_plant")
        if tp is not None and sib.any():                    # the planted distance must be the distance seen by the read
            child = np.where(down, task, C.esrc)[sib]; dev = float(np.max(np.abs(np.asarray(tp, dtype=float)[child] - np.maximum(C.L["ev"][sib, 3], 0))))
            if dev > 5e-3:
                why = f"the workload rebuilt with seed {seed} does not match the run (planted distances differ by {dev:.3f})"; continue
        return sib, f"siblings identified from the planted parents (workload seed {seed})"
    return None, why


def workload_seeds(C, wl_seed=None):
    """Seeds to try when the workload is rebuilt: the one given, else the one stored in the law file, the suffix
    _s<seed> of the run name, and 1."""
    m_ = re.search(r"_s(\d+)$", C.name)
    stored = [int(C.workload_seed)] if getattr(C, "workload_seed", None) is not None else []
    return [wl_seed] if wl_seed is not None else list(dict.fromkeys(stored + ([int(m_.group(1))] if m_ else []) + [1]))


def planted_one(C, rng, boot, wl_seed=None):
    meta = C.meta; k, cs = C.k, C.cs; ne = len(C.y); notes = []
    res = {"k": k, "n": C.n, "events": ne, "mean_m": C.mean_m, "lam_bg": C.lam_bg, "w_bg": C.w_bg, "cs": list(cs), "notes": notes,
           "planted": None, "fit": None, "sib": None, "bg": None, "single": None}
    exch = bool(meta.get("exchangeable"))
    print(f"\n================ {C.name}: k={k}, reads {C.n}, in-ball insertions {ne} ({C.mean_m:.4f} per read)")
    if not exch and not ("Ds" in meta and "pi" in meta):
        notes.append("no planted parameters in the meta section of the law file: nothing to compare with"); print(f"   {notes[-1]}")
        return res
    Ds0 = None if exch else float(meta["Ds"]); Dl = meta.get("D"); D0 = float(Dl[0]) if isinstance(Dl, (list, tuple)) and len(Dl) == 1 else (float(Dl) if _ok(Dl) else None)
    res["planted"] = {"Ds": Ds0, "pi": 0.0 if exch else float(meta["pi"]), "D": Dl, "R": meta.get("R"), "lag_p": meta.get("lag_p"),
                      "sibling_share_realized": meta.get("sibling_share_realized"), "exchangeable": exch}
    if exch:
        print("   planted: exchangeable control (base and stream pooled and split at random): no siblings")
        sib = np.zeros(ne, dtype=bool)
    else:
        print(f"   planted: D_s = {Ds0:g}, sibling probability per task {meta['pi']} (realized {_s(meta.get('sibling_share_realized'), '.3f')}), manifold dimension {Dl}, radius {_s(meta.get('R'), '.3f')}, lag ~ Geometric({meta.get('lag_p')})")
        sib, why = planted_siblings(C, workload_seeds(C, wl_seed)); notes.append(why); print(f"   {why}")

    # ---- in-ball background rate against first principles
    if sib is not None:
        nb = int((~sib).sum()); exp = float(C.lam.sum()); z = (nb - exp) / math.sqrt(max(exp, 1e-12))
        res["bg"] = {"n": nb, "obs": nb / C.n, "lam_bg": C.lam_bg, "ratio": nb / exp if exp > 0 else None, "z": z}
        print(f"A. background rate: non-sibling in-ball insertions {nb} = {nb / C.n:.4f} per read | first principles n_c k / (N + 1) = {C.lam_bg:.4f} per read ({exp:.1f} expected)"
              f" | ratio {_s(res['bg']['ratio'])}, z = {z:+.2f} (Poisson)")
        yb = C.y[~sib]
        if len(yb) >= MIN_EV:
            cntb = np.bincount(C.ep[~sib], minlength=k + 1)[1:k + 1].astype(float); x2 = chisq(cntb, np.full(k, cntb.sum() / k), 0)
            res["bg"].update({"shape_ks": ks_stat(yb, C.bg_cdf), "rank1": float(cntb[0] / cntb.sum()), "rank_chi2_p": x2[2] if x2 else None})
            print(f"   their positions against the frozen background shape: KS {res['bg']['shape_ks']:.3f}; rank-1 share {100 * cntb[0] / cntb.sum():.1f}% (uniform {100 / k:.1f}%), chi2 against uniform ranks p = {_s(x2[2] if x2 else None, '.2g')}")
    else:
        print("A. background rate: siblings could not be identified, no check")

    # ---- sibling positions against Exp(D_s)
    if not exch and sib is not None:
        ysib = C.y[sib]; ns = len(ysib); task = C.task[C.ej][sib]; src = C.esrc[sib]
        _, first = np.unique(np.minimum(task, src).astype(np.int64) * (1 << 32) + np.maximum(task, src), return_index=True)
        npair = len(first)                                  # a concurrent pair can be seen from both sides (same t, two reads)
        res["sib"] = {"n": ns, "pairs": npair, "share": ns / ne if ne else None, "share_first_principles": 1 - C.w_bg,
                      "Ds_mle": None, "Ds_se": None, "z": None, "ks": None, "ks_p": None}
        if npair >= MIN_BUCKET and ysib.mean() > 0:
            d = 1.0 / float(ysib.mean()); se = d / math.sqrt(npair); kt = kstest(ysib[first], "expon", args=(0, 1.0 / Ds0))
            res["sib"].update({"Ds_mle": d, "Ds_se": se, "z": (d - Ds0) / se, "ks": float(kt.statistic), "ks_p": float(kt.pvalue)})
            print(f"B. sibling insertions (identified, n = {ns} from {npair} parent-child pairs, {100 * ns / ne:.1f}% of the in-ball insertions; first principles say {100 * (1 - C.w_bg):.1f}%): y = ln(r_k / t) against Exp({Ds0:g})")
            print(f"   maximum-likelihood D_s = 1 / mean(y) = {d:.2f} (se {se:.2f} counting every pair once, z = {(d - Ds0) / se:+.2f} against the planted value) | KS on one insertion per pair {kt.statistic:.3f}, p = {kt.pvalue:.3f}")
        else:
            notes.append(f"only {ns} identified sibling insertions: no test of their positions"); print(f"B. {notes[-1]}")

    # ---- the fit that does not know the siblings
    f1 = None
    if ne >= MIN_EV and C.w_bg < 1:
        f1 = fit_sib(C.y, C.bg_pdf, C.w_bg, 1); bs = boot_ds(C, boot, rng) if boot > 0 else np.zeros((0, 2))
        ci = pctl(bs[:, 1]); res["fit"] = {"Ds": f1["Ds"][0], "Ds_ci": ci, "pi": f1["pi"][0], "ks": ks_stat(C.y, C.model_cdf(f1))}
        inside = (ci[0] <= Ds0 <= ci[1]) if (Ds0 is not None and _ok(ci[0])) else None
        print(f"C. blind fit (1 scale, background share fixed at {100 * C.w_bg:.1f}%): D_s = {f1['Ds'][0]:.2f} (95% CI {_l(ci)}), sibling share {f1['pi'][0]:.3f}, KS {res['fit']['ks']:.3f}"
              + (f" | planted D_s = {Ds0:g} is {'inside' if inside else 'OUTSIDE'} the interval" if inside is not None else ""))
        res["fit"]["planted_inside_ci"] = inside
    elif ne < MIN_EV:
        notes.append(f"only {ne} in-ball insertions (fewer than {MIN_EV}): no fit"); print(f"C. {notes[-1]}")
    else:
        res["fit"] = {"Ds": None, "Ds_ci": [None, None], "pi": 0.0}
        print(f"C. blind fit: the first-principles background ({C.lam_bg:.4f} per read) accounts for all observed in-ball insertions ({C.mean_m:.4f} per read): sibling share 0, nothing to fit")

    # ---- single-insertion conflict predictions
    single = np.where(C.m == 1)[0]; n1 = len(single)
    if n1 > 0:
        obs = C.obs_conflict(single); se = [100 * math.sqrt(max(o / 100 * (1 - o / 100), 0.0) / n1) for o in obs]
        ps = 1 - C.w_bg
        P = {"exch": [100 * float(C.loo[single, i].mean()) for i in range(len(cs))],
             "pred_fit": C.pred_conflict(f1, single) if f1 is not None else ([100 * float(C.loo[single, i].mean()) for i in range(len(cs))] if C.w_bg >= 1 else [None] * len(cs)),
             "pred_planted": C.pred_conflict({"pi": [ps], "Ds": [Ds0]} if Ds0 else {"pi": [], "Ds": []}, single),
             "closed_planted": [100 * mix_conflict(c, ps if Ds0 else 0.0, (Ds0 / D0) if Ds0 else 1.0, D0, k) for c in cs] if D0 else [None] * len(cs)}
        res["single"] = {"n": n1, "obs": obs, "se": se, **P, "err_fit": max_err(P["pred_fit"], obs), "err_planted": max_err(P["pred_planted"], obs),
                         "err_closed": max_err(P["closed_planted"], obs), "err_exch": max_err(P["exch"], obs)}
        print(f"D. single-insertion reads (n = {n1}): conflict probability (%)" + ("" if n1 >= MIN_EV else "   [few reads: very noisy]"))
        print("   c                                        " + " ".join(f"{c:6.2f}" for c in cs))
        print(f"   observed                                 {fmt(obs)}   (se {fmt(se, 4)})")
        for a_, e_, lab in (("pred_fit", "err_fit", "semi-parametric, fitted D_s             "), ("pred_planted", "err_planted", "semi-parametric, planted D_s            "),
                            ("closed_planted", "err_closed", "closed form, planted D_s and D          "), ("exch", "err_exch", "exchangeable only                       ")):
            print(f"   {lab} {fmt(P[a_])}   max |err| {_s(res['single'][e_], '.1f')}")
    return res


def cmd_planted(a):
    rng = np.random.default_rng(a.seed); out = {}
    for path in a.files:
        C = LawData(path, cs=a.cs)
        out[_unique(C.name, out)] = planted_one(C, rng, a.boot, a.wl_seed)
    save(a.out, out)


# ----------------------------------------------------------------------------- command: strata
def stratify(C, by, bins, notes):
    """Strata of the reads: list of (label, boolean mask over reads)."""
    n = C.n; out = []; bins = max(int(bins), 1)

    def quant(v, lab, spec):                                # equal-count bins of a per-read value
        ok = np.isfinite(v)
        if not ok.all():
            notes.append(f"{int((~ok).sum())} reads without a finite {lab} are left out")
        order = np.where(ok)[0][np.argsort(v[ok], kind="stable")]
        for part in np.array_split(order, bins):
            if len(part):
                mk = np.zeros(n, dtype=bool); mk[part] = True
                out.append((f"{lab} {v[part].min():{spec}} to {v[part].max():{spec}}", mk))
    if by == "lid":
        quant(C.lid, "LID", ".1f")
    elif by == "rk":
        quant(C.ro[:, C.k - 1], "r_k", ".3f")
    elif by == "time":
        for part in np.array_split(np.arange(n), bins):
            if len(part):
                mk = np.zeros(n, dtype=bool); mk[part] = True; out.append((f"reads {part[0]} to {part[-1]}", mk))
    elif by == "team":
        team = np.asarray(C.L["team"]) if "team" in C.L else np.zeros(n, dtype=int)
        ids, cnt = np.unique(team, return_counts=True); keep = ids[np.argsort(-cnt, kind="stable")][:12]
        if len(ids) > 12:
            notes.append(f"{len(ids)} teams: only the 12 largest are shown")
        if len(ids) == 1:
            notes.append("only one team: a single stratum")
        out = [(f"team {t}", team == t) for t in keep]
    else:
        out = [("m = 1", C.m == 1), ("m = 2", C.m == 2), ("m >= 3", C.m >= 3)]
    return out


STRATA_PRED = (("exch", "exchangeable only                 "), ("semi_own", "semi-parametric, stratum fit      "), ("semi_global", "semi-parametric, file fit         "),
               ("semi_global_ds", "semi-par., file D_s, stratum share"), ("closed_pooled", "closed form, pooled D             "), ("closed_own", "closed form, stratum LID          "))


def strata_one(C, by, bins, rng, reps=30, nsim=20000):
    """Law inside strata of the reads of one file.  Prints the report, returns the JSON entry."""
    k, cs = C.k, C.cs; ne = len(C.y); notes = []; nul = [None] * len(cs)
    fg = fit_sib(C.y, C.bg_pdf, C.w_bg, 1) if ne >= MIN_EV else None
    cpg = _comps(fg); pig = sum(p for p, _ in cpg); dsg = cpg[0][1] if cpg else None
    if fg is None:
        notes.append(f"only {ne} in-ball insertions in the file (fewer than {MIN_EV}): no file-level fit")
    elif (1.0 - C.w_bg) * ne < MIN_BUCKET:
        notes.append(f"about {(1.0 - C.w_bg) * ne:.0f} sibling insertions in the file: the file-level D_s is not identified")
    res = {"by": by, "k": k, "n": C.n, "events": ne, "cs": list(cs), "strata": [], "summary": None, "notes": notes,
           "global": {"Ds": dsg, "pi": pig if fg is not None else None, "w_bg": C.w_bg, "lam_bg": C.lam_bg, "mean_m": C.mean_m,
                      "lid_pooled": C.lid_pool, "lid_median": C.lid_median}}

    def closed(D):                                          # file-level sibling share and D_s, local dimension D
        if fg is None or not _ok(D) or D <= 0:
            return nul
        return [100 * mix_conflict(c, pig, (dsg / D) if dsg else 1.0, D, k) for c in cs]

    def ideal(D, counts):                                   # the same with several insertions per read: Monte Carlo
        if fg is None or not _ok(D) or D <= 0 or len(counts) == 0:
            return nul
        rho, _ = simulate(D, k, nsim, rng, alpha=(dsg / D) if dsg else 1.0, counts=rng.choice(counts, size=nsim), pi=pig)
        return [100 * float(np.mean(rho > c)) for c in cs]
    what = "share of reads not c-admissible (%), all reads of the stratum, against end-to-end simulation" if by == "m" else "conflict probability (%) of the single-insertion reads"
    print(f"\n================ {C.name} by {by}: k={k}, reads {C.n}, in-ball insertions {ne} ({C.mean_m:.4f} per read); file-level fit: sibling share {_s(res['global']['pi'], '.3f')},"
          f" D_s {_s(dsg)}; LID pooled {_s(C.lid_pool, '.1f')}, median {_s(C.lid_median, '.1f')}")
    fits = []
    for label, mk in stratify(C, by, bins, notes):
        rows = np.where(mk)[0]; ns = len(rows); em = mk[C.ej]; nes = int(em.sum()); note = []
        ent = {"label": label, "n": ns, "events": nes, "n_single": int((C.m[rows] == 1).sum()), "n_test": 0, "mean_m": None, "lam_bg": None, "w_bg": None,
               "Ds": None, "Ds_se": None, "lid_median": None, "lid_pooled": None, "cs": list(cs), "obs": nul, "se": nul, "note": None,
               **{a_: nul for a_, _ in STRATA_PRED}, **{"err_" + a_: None for a_, _ in STRATA_PRED}}
        res["strata"].append(ent)
        if ns == 0:
            ent["note"] = "no reads"; print(f"-- {label}: no reads"); continue
        mm = float(C.m[rows].mean()); lam = float(C.lam[rows].mean())
        w = C.w_bg if by == "m" else (min(lam / mm, 1.0) if mm > 0 else 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ys = np.log(C.ro[rows, k:k + 1] / np.maximum(C.ro[rows, :k], 1e-6)).ravel()
        ys = ys[np.isfinite(ys) & (ys >= 0)]; lid = C.lid[rows]; lid = lid[np.isfinite(lid)]
        Dm = float(np.median(lid)) if len(lid) else float("nan"); Dp = float(1.0 / ys.mean()) if len(ys) and ys.mean() > 0 else float("nan")
        pdf = C.bg_pdf
        if len(ys) >= MIN_STATIC:                           # background shape of the stratum itself
            sub = ys if len(ys) <= BG_MAX else np.random.default_rng(0).choice(ys, size=BG_MAX, replace=False)
            _, bw, br = fit_mixexp(sub, C.bg_J); pdf = lambda t, bw=bw, br=br: mixexp_pdf(t, bw, br)
        else:
            note.append("few static distances, file-level background shape")
        fo = None
        if nes < MIN_EV:
            note.append(f"only {nes} in-ball insertions, no fit")
        elif w >= 1.0:
            fo = {"pi": [0.0], "Ds": [float("nan")]}; note.append("the background accounts for all in-ball insertions")
        else:
            fo = fit_sib(C.y[em], pdf, w, 1)
            if (1.0 - w) * nes < MIN_BUCKET:                # hardly any sibling mass: the value is not identified
                note.append(f"about {(1.0 - w) * nes:.0f} sibling insertions, D_s is not identified")
            else:
                fits.append((C.y[em], pdf(C.y[em]), w, fo["ll"]))
                if not DS_BOUNDS[0] * 1.01 < fo["Ds"][0] < DS_BOUNDS[1] * 0.99:
                    note.append(f"D_s sits at a bound of the search interval {DS_BOUNDS}")
        ds = _comps(fo)[0][1] if _comps(fo) else None
        fgs = None if fg is None else ({"pi": [1.0 - w], "Ds": [dsg]} if (dsg and w < 1.0) else {"pi": [], "Ds": []})      # file D_s, share of the stratum
        if by == "m":                                       # all reads of the stratum, any number of insertions
            sel = rows; counts = C.m[rows]
            obs = [100 * float(np.mean(C.rho[rows] > c)) for c in cs]
            P = {"exch": simulate_reads(C, {"pi": [], "Ds": []}, rng, reps, reads=rows),
                 "semi_own": simulate_reads(C, fo, rng, reps, reads=rows) if fo is not None else nul,
                 "semi_global": simulate_reads(C, fg, rng, reps, reads=rows) if fg is not None else nul,
                 "semi_global_ds": simulate_reads(C, fgs, rng, reps, reads=rows) if fgs is not None else nul,
                 "closed_pooled": closed(C.lid_pool) if counts.max() == 1 else ideal(C.lid_pool, counts),
                 "closed_own": closed(Dm) if counts.max() == 1 else ideal(Dm, counts)}
        else:
            sel = rows[C.m[rows] == 1]; obs = C.obs_conflict(sel)
            P = {"exch": [100 * float(C.loo[sel, i].mean()) for i in range(len(cs))] if len(sel) else nul,
                 "semi_own": C.pred_conflict(fo, sel) if fo is not None else nul, "semi_global": C.pred_conflict(fg, sel) if fg is not None else nul,
                 "semi_global_ds": C.pred_conflict(fgs, sel) if fgs is not None else nul, "closed_pooled": closed(C.lid_pool), "closed_own": closed(Dm)}
        n1 = len(sel)
        if 0 < n1 < MIN_EV:
            note.append(f"only {n1} reads in the test, very noisy")
        ent.update({"n_test": n1, "mean_m": mm, "lam_bg": lam, "w_bg": w, "Ds": ds, "Ds_se": ds / math.sqrt(max((1 - w) * nes, 1.0)) if ds else None,
                    "lid_median": Dm, "lid_pooled": Dp, "obs": obs, "se": [100 * math.sqrt(max(o / 100 * (1 - o / 100), 0.0) / n1) if _ok(o) else None for o in obs],
                    **P, **{"err_" + a_: max_err(P[a_], obs) for a_, _ in STRATA_PRED}, "note": "; ".join(note) or None})
        print(f"-- {label}: reads {ns}, in-ball insertions {nes} ({mm:.4f} per read; background {lam:.4f}, share {100 * w:.1f}%), D_s {_s(ds)} (se {_s(ent['Ds_se'])}),"
              f" LID median {_s(Dm, '.1f')} pooled {_s(Dp, '.1f')}" + (f"   [{ent['note']}]" if ent["note"] else ""))
        if n1 == 0:
            print("      no reads to test"); continue
        print(f"      c                                  {' '.join(f'{c:6.2f}' for c in cs)}")
        print(f"      observed (n = {n1:5d})               {fmt(obs)}   (se {fmt(ent['se'], 4)})")
        for a_, lab in STRATA_PRED:
            print(f"      {lab} {fmt(P[a_])}   max |err| {_s(ent['err_' + a_], '.1f')}")
    S = [e for e in res["strata"] if e["n_test"] >= MIN_EV]                         # strata large enough to be compared
    Sc = [e for e in S if all(_ok(e["err_" + a_]) for a_, _ in STRATA_PRED)]       # ... in which every predictor exists
    wmean = lambda E, a_: (float(np.average([e["err_" + a_] for e in E if _ok(e["err_" + a_])], weights=[e["n_test"] for e in E if _ok(e["err_" + a_])]))
                           if any(_ok(e["err_" + a_]) for e in E) else None)
    summ = {"Ds": [e["Ds"] for e in res["strata"]], "lr": None, "strata_compared": len(S), "strata_common": len(Sc),
            "mean_err": {a_: wmean(S, a_) for a_, _ in STRATA_PRED}, "mean_err_common": {a_: wmean(Sc, a_) for a_, _ in STRATA_PRED}}
    if len(fits) >= 2:                                      # one D_s for all strata against one D_s per stratum
        nll = lambda d: -sum(float(np.sum(np.log(w * fb + (1 - w) * d[0] * np.exp(-d[0] * y) + 1e-300))) for y, fb, w, _ in fits)
        best = min((minimize(nll, [s0], bounds=[DS_BOUNDS], method="L-BFGS-B") for s0 in (1.0, 3.0, 8.0)), key=lambda r: r.fun)
        stat = max(2.0 * (sum(f[3] for f in fits) + float(best.fun)), 0.0); dof = len(fits) - 1
        summ["lr"] = {"stat": stat, "dof": dof, "p": float(chi2.sf(stat, dof)), "Ds_common": float(best.x[0])}
    res["summary"] = summ
    print(f"   summary ({what}):")
    if Sc:
        print(f"      max |err| averaged over the {len(Sc)} strata with {MIN_EV} or more test reads and every predictor (weighted by reads):")
        print("         " + ", ".join(f"{lab.strip()} {_s(summ['mean_err_common'][a_], '.1f')}" for a_, lab in STRATA_PRED))
    elif S:
        print(f"      no stratum has every predictor; max |err| averaged over the strata with {MIN_EV} or more test reads where the predictor exists (not comparable across predictors):")
        print("         " + ", ".join(f"{lab.strip()} {_s(summ['mean_err'][a_], '.1f')}" for a_, lab in STRATA_PRED))
    else:
        print(f"      no stratum with {MIN_EV} or more test reads: nothing to average")
    if summ["lr"]:
        print(f"      one D_s for all strata ({summ['lr']['Ds_common']:.2f}) against one per stratum ({_l([d for d in summ['Ds'] if _ok(d)])}): likelihood ratio {summ['lr']['stat']:.1f} on {summ['lr']['dof']} dof, p = {summ['lr']['p']:.3g}")
    for nt in notes:
        print(f"   note: {nt}")
    return res


def cmd_strata(a):
    rng = np.random.default_rng(a.seed); out = {}
    for path in a.files:
        C = LawData(path, cs=a.cs)
        out[_unique(C.name, out)] = strata_one(C, a.by, a.bins, rng, a.reps)
    save(a.out, out)


# ----------------------------------------------------------------------------- command: bgrank
def wilson(x, n, z=1.959964):
    """Wilson score interval of a binomial share."""
    if n <= 0:
        return [None, None]
    p = x / n; den = 1.0 + z * z / n; mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / den
    return [max(mid - half, 0.0), min(mid + half, 1.0)]


def rank_test(ranks, k):
    """Old ranks of background in-ball insertions against the uniform distribution on 1..k."""
    ranks = np.asarray(ranks, dtype=int); n = len(ranks); cnt = np.bincount(ranks, minlength=k + 1)[1:k + 1]
    out = {"n": n, "rank_hist": cnt.tolist(), "rank1": None, "rank1_ci": [None, None], "rank1_p": None, "chi2": None, "chi2_p": None}
    if n:
        x2 = chisq(cnt.astype(float), np.full(k, n / k), 0)
        out.update({"rank1": float(cnt[0] / n), "rank1_ci": wilson(int(cnt[0]), n), "rank1_p": float(binomtest(int(cnt[0]), n, 1.0 / k).pvalue),
                    "chi2": list(x2[:2]) if x2 else None, "chi2_p": x2[2] if x2 else None})
    return out


def bg_tests(ranks, em, lag, k):
    ranks, em, lag = np.asarray(ranks, dtype=int), np.asarray(em, dtype=int), np.asarray(lag, dtype=int)
    t = rank_test(ranks, k)
    t["by_m"] = {"1": rank_test(ranks[em == 1], k), "2+": rank_test(ranks[em > 1], k)}
    t["by_lag"] = {f"<={LAG_NEAR}": rank_test(ranks[lag <= LAG_NEAR], k), f">{LAG_NEAR}": rank_test(ranks[lag > LAG_NEAR], k)}
    return t


def background_events(C, wl_seed=None):
    """In-ball insertions that are background by construction: (boolean mask or None, kind, note)."""
    meta = C.meta; ne = len(C.y)
    if meta.get("exchangeable"):
        return np.ones(ne, dtype=bool), "exchangeable control", None
    if "pi" in meta and "Ds" in meta:
        if float(meta["pi"]) == 0.0:
            return np.ones(ne, dtype=bool), "synthetic, pi = 0", None
        if meta.get("chain"):
            return None, None, "synthetic with chained siblings: the non-sibling insertions are not identified"
        sib, why = planted_siblings(C, workload_seeds(C, wl_seed))
        return (None, None, why) if sib is None else (~sib, "synthetic, non-sibling insertions", why)
    return None, None, "neither an exchangeable control nor a synthetic workload: the background insertions are not identified"


def _replicate(C):
    """Runs with the same value have the same geometry of unrelated (reader, source) pairs: the part of the workload
    that fixes the vectors (not the concurrency, and for synthetic workloads not the sibling parameters, which
    leave the base and all non-sibling items where they are), the workload seed and k."""
    drop = {"name", "agents", "teams", "ops", "arrival", "extra_reads", "extra_source", "expand_sigma"}
    if C.workload.get("kind") == "synthetic":
        drop |= {"Ds", "pi", "lag_p", "radius", "chain"}
    w = {a_: b for a_, b in C.workload.items() if a_ not in drop}
    m_ = re.search(r"_s(\d+)$", C.name)
    seed = C.workload_seed if getattr(C, "workload_seed", None) is not None else (int(m_.group(1)) if m_ else None)
    return json.dumps([w, seed, C.k], sort_keys=True, default=str)


def _rank_line(lab, t, k):
    if not t["n"]:
        return f"   {lab:26s} n =     0"
    return (f"   {lab:26s} n = {t['n']:5d}  rank-1 share {100 * t['rank1']:5.1f}% (95% CI {100 * t['rank1_ci'][0]:4.1f} to {100 * t['rank1_ci'][1]:4.1f}; uniform {100 / k:.1f}%; binomial p = {t['rank1_p']:.3g})"
            f"  chi2 {_s(t['chi2'][0] if t['chi2'] else None, '.1f')}/{t['chi2'][1] if t['chi2'] else '-'} p = {_s(t['chi2_p'], '.3g')}  ranks {t['rank_hist']}")


def cmd_bgrank(a):
    files = {}; pools = {}
    print("background in-ball insertions: old rank against the uniform distribution on 1..k, in-ball rate against n_c k / (N + 1)")
    for path in a.files:
        C = LawData(path, cs=a.cs); nm = _unique(C.name, files); k = C.k
        bg, kind, note = background_events(C, a.wl_seed)
        ent = {"k": k, "n_reads": C.n, "kind": kind, "note": note, "n": None, "rank_hist": None, "rank1": None, "rank1_ci": [None, None], "rank1_p": None,
               "chi2": None, "chi2_p": None, "rate_obs": None, "rate_exp": None, "rate_ratio": None, "rate_z": None, "by_m": None, "by_lag": None}
        files[nm] = ent
        if bg is None:
            print(f"\n{nm}: skipped ({note})"); continue
        nb = int(bg.sum()); exp = float(C.lam.sum())
        ent.update(bg_tests(C.ep[bg], C.em[bg], C.elag[bg], k))
        ent.update({"rate_obs": nb, "rate_exp": exp, "rate_ratio": nb / exp if exp > 0 else None, "rate_z": (nb - exp) / math.sqrt(exp) if exp > 0 else None})
        pools.setdefault(k, []).append((nm, _replicate(C), C.task[C.ej][bg], C.esrc[bg], C.ep[bg], C.em[bg], C.elag[bg], nb, exp))
        print(f"\n{nm}: k={k}, reads {C.n}, {kind}" + (f" ({note})" if note else ""))
        print(f"   in-ball background insertions {nb} against {exp:.1f} expected: ratio {_s(ent['rate_ratio'])}, z = {_s(ent['rate_z'], '+.2f')}")
        print(_rank_line("all", ent, k))
    pooled = {}
    for k in sorted(pools):
        items = pools[k]; seen = set(); rows = []; groups = {}
        for nm, sig, task, src, ranks, em, lag, nb, exp in items:
            g = groups.setdefault(sig, {"files": [], "n": 0, "x": 0}); g["files"].append(nm)
            for i in range(len(ranks)):                     # the same pair of the same geometry counts once
                key = (sig, int(task[i]), int(src[i]))
                if key not in seen:
                    seen.add(key); rows.append((ranks[i], em[i], lag[i])); g["n"] += 1; g["x"] += int(ranks[i] == 1)
        arr = np.array(rows, dtype=int).reshape(-1, 3); P = bg_tests(arr[:, 0], arr[:, 1], arr[:, 2], k)
        n_all = int(sum(it[7] for it in items)); exp = float(sum(it[8] for it in items)); infl = n_all / max(P["n"], 1)
        G = [g for g in groups.values() if g["n"]]; se_c = p_c = None
        if len(G) >= 3:                                     # insertions of one run are not independent: test across workloads
            dev = np.array([g["x"] - P["rank1"] * g["n"] for g in G])
            se_c = math.sqrt(len(G) / (len(G) - 1.0) * float((dev ** 2).sum())) / P["n"]
            p_c = float(2 * student_t.sf(abs(P["rank1"] - 1.0 / k) / se_c, len(G) - 1)) if se_c > 0 else None
        P.update({"k": k, "files": [it[0] for it in items], "n_all": n_all, "rate_obs": n_all, "rate_exp": exp, "rate_ratio": n_all / exp if exp > 0 else None,
                  "rate_z": (n_all - exp) / math.sqrt(exp * max(infl, 1.0)) if exp > 0 else None, "replicates": [g["files"] for g in groups.values() if len(g["files"]) > 1],
                  "workloads": [{"files": g["files"], "n": g["n"], "rank1": g["x"] / g["n"] if g["n"] else None} for g in groups.values()],
                  "rank1_se_cluster": se_c, "rank1_p_cluster": p_c})
        pooled[str(k)] = P
        print(f"\n================ pooled, k={k}: {len(items)} files, {len(groups)} independent workloads, {n_all} background insertions, {P['n']} distinct (reader, source) pairs")
        for g in groups.values():
            print(f"   workload: {', '.join(g['files'])}: {g['n']} pairs" + (f", rank-1 share {100 * g['x'] / g['n']:.1f}%" if g["n"] else "") + (" (same vectors, a pair counts once)" if len(g["files"]) > 1 else ""))
        print(f"   in-ball rate: observed {n_all} against {exp:.1f} expected, ratio {_s(P['rate_ratio'])}, z = {_s(P['rate_z'], '+.2f')}" + (f" (variance inflated by {infl:.2f} for the repeated pairs)" if infl > 1.001 else ""))
        print(_rank_line("all", P, k))
        if se_c is not None:
            print(f"   rank-1 share across the {len(G)} workloads (cluster-robust): {100 * P['rank1']:.1f}% with standard error {100 * se_c:.1f}, p = {_s(p_c, '.3g')} against {100 / k:.1f}%")
        else:
            print("   fewer than 3 independent workloads: no cluster-robust test (the binomial and chi-square p-values treat insertions of one run as independent)")
        for key, sub in (("by_m", "m of the read"), ("by_lag", "stream lag")):
            for lab, t in P[key].items():
                print(_rank_line(f"{sub} {lab}", t, k))
        if P["n"] < MIN_EV:
            print(f"   note: only {P['n']} insertions, the tests have almost no power")
    if not pooled:
        print("\nno file with identified background insertions: nothing to pool")
    main_k = max(pooled, key=lambda kk: pooled[kk]["n"]) if pooled else None
    save(a.out, {"files": files, "pooled": pooled[main_k] if main_k else None, "pooled_by_k": pooled})


# ----------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ariadne.law", description="Conflict law for approximate similarity reads: closed forms, fits, transfers.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, files=None):
        if files:
            p.add_argument("files", nargs=files, help="law files (<run>.law.pkl)")
        p.add_argument("--out", help="JSON file to write"); p.add_argument("--cs", help="c grid, e.g. 1.02,1.05,1.1 (default: the grid of the law file)")
        p.add_argument("--seed", type=int, default=0)
        return p
    p = common(sub.add_parser("mc", help="closed form against Monte Carlo")); p.add_argument("--n", type=int, default=60000)
    common(sub.add_parser("grid", help="closed-form conflict probability on a dense grid"))
    p = common(sub.add_parser("fit", help="fit and test the law on law files"), "+")
    p.add_argument("--boot", type=int, default=200, help="bootstrap replicates for confidence intervals")
    p.add_argument("--boot-ks", type=int, default=None, help="parametric-bootstrap replicates for the KS p-values (default: --boot)")
    p.add_argument("--reps", type=int, default=30, help="simulations per read in the end-to-end test")
    p = common(sub.add_parser("transfer", help="predict one configuration from another"))
    p.add_argument("--src", required=True); p.add_argument("--dst"); p.add_argument("--mode", required=True, choices=["window", "k", "lag", "time", "params"])
    p.add_argument("--scales", type=int, default=1, choices=[1, 2], help="sibling scales of the source fit (not used by mode lag)")
    common(sub.add_parser("lag", help="lag kernel table and invariance"), "+")
    p = common(sub.add_parser("planted", help="planted (synthetic) workloads: truth against fit"), "+")
    p.add_argument("--boot", type=int, default=200); p.add_argument("--wl-seed", type=int, default=None, help="seed of the workload (default: from the run name, else 1)")
    p = common(sub.add_parser("strata", help="the law inside strata of the reads"), "+")
    p.add_argument("--by", required=True, choices=["lid", "rk", "team", "time", "m"]); p.add_argument("--bins", type=int, default=3, help="number of strata for lid, rk and time")
    p.add_argument("--reps", type=int, default=30, help="simulations per read (only --by m)")
    p = common(sub.add_parser("bgrank", help="pooled rank test of the background insertions"), "+")
    p.add_argument("--wl-seed", type=int, default=None, help="seed of the workload (default: stored in the law file)")
    p = sub.add_parser("report", help="compact table from a fit JSON"); p.add_argument("files", nargs=1)
    a = ap.parse_args(argv)
    if getattr(a, "cs", None) and a.cmd != "mc":
        a.cs = [float(x) for x in a.cs.split(",")]
    {"mc": cmd_mc, "grid": cmd_grid, "fit": cmd_fit, "transfer": cmd_transfer, "lag": cmd_lag, "planted": cmd_planted, "strata": cmd_strata, "bgrank": cmd_bgrank, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
