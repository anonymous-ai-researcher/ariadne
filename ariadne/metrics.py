"""Distances, admissibility test and approximation ratio.

Convention used everywhere in the framework: an answer is a list of (item_id, dist) sorted by
dist, where dist is the METRIC distance (Euclidean, or the chord distance sqrt(2 * cosine distance)
for unit vectors).  Store adapters convert whatever their engine returns into this form.
"""
import math

EPS = 1e-9


def chord_from_cos(dcos):
    return math.sqrt(max(2.0 * dcos, 0.0))


def cos_from_chord(d):
    return 0.5 * d * d


def admissible(A, F, c):
    """A is still a c-approximate answer with respect to the reference list F (per rank)."""
    if len(A) != len(F):
        return False
    return all(a[1] <= c * f[1] + EPS for a, f in zip(A, F))


def rho(a_d, e_d):
    """Smallest c for which the distance list a_d is c-approximate given exact distances e_d."""
    m = min(len(a_d), len(e_d))
    if m == 0:
        return float("inf")
    return max((max(a_d[i], 0.0) + 1e-7) / (max(e_d[i], 0.0) + 1e-7) for i in range(m))


def insertion_conflicts(r, t, c):
    """Characterization theorem: does one insertion at distance t break c-admissibility of an exact
    answer with sorted distances r[0..k-1]?"""
    k = len(r)
    if t >= r[-1]:
        return False
    p = sum(1 for x in r if x < t)              # 0-based rank of the insertion
    if r[p] > c * t + EPS:
        return True
    return any(r[i] > c * r[i - 1] + EPS for i in range(p + 1, k))


def stale_conflicts(orig, cur, c):
    """General form for several insertions: orig = distances of the held answer, cur = current sorted
    distances of the top-k (after the insertions)."""
    return any(o > c * n + EPS for o, n in zip(orig, cur))
