"""Validation predicates (the thing every experiment varies) and the decision proxies.

A policy answers one question at commit time: given the answer A the transaction reasoned about and a
reference answer F for the current state, may the transaction commit the decision it derived from A?

Spec strings (used in the YAML configs):
    none            always commit (no isolation)
    exact           ordered identifiers identical            (re-execution validation)
    set             identifier sets identical
    overlap:0.9     at least 90% common identifiers          (recall-style tolerance)
    top1            nearest item unchanged
    topj:3          the j nearest items unchanged (as a set)
    radius:1.1      only the k-th distance is tested         (ball-radius test)
    rank1:1.1       only the nearest distance is tested
    sumdist:1.1     sum of distances within a factor c
    cadm:1.1        c-admissible, per rank                   (ours)
    cert:1.1        c-admissible against an EXACT validation read (certified variant)
    oracle:link     commit iff the named decision proxy is unchanged (lower bound when the decision
                    function is known); proxies: dup, link, top3, set, tag
    objects         commit iff every item of A still exists (object-granularity validation: what a
                    version check on the objects that were read gives; blind to insertions)
    redecide        always commit, but recompute the decision from F without thinking
                    (valid only for deciders that need no model call, e.g. threshold dedup)
A suffix "+exist" adds the existence check for the items of A (needed when the workload deletes).
"""
import math
from .metrics import admissible, EPS


class Policy:
    def __init__(self, spec, k):
        self.spec = spec
        self.k = k
        base = spec.replace("+exist", "")
        self.name, _, arg = base.partition(":")
        self.check_exist = spec.endswith("+exist") or self.name == "objects"
        self.arg = arg
        self.needs_exact = self.name == "cert"
        self.redecide = self.name == "redecide"
        self.needs_proxy = self.name == "oracle"
        if self.name in ("overlap",):
            self.frac = float(arg or 0.9)
        if self.name in ("radius", "rank1", "sumdist", "cadm", "cert"):
            self.c = float(arg or 1.0)
        if self.name == "topj":
            self.j = int(arg or 3)
        known = {"none", "exact", "set", "overlap", "top1", "topj", "radius", "rank1", "sumdist", "cadm", "cert",
                 "oracle", "redecide", "objects"}
        if self.name not in known:
            raise ValueError(f"unknown policy '{spec}'")

    def valid(self, A, F, proxy=None):
        n = self.name
        if n in ("none", "redecide", "objects"):
            return True
        ia, i_f = [i for i, _ in A], [i for i, _ in F]
        if n == "exact":
            return ia == i_f
        if n == "set":
            return set(ia) == set(i_f)
        if n == "overlap":
            return len(set(ia) & set(i_f)) >= math.ceil(self.frac * max(len(ia), 1))
        if n == "top1":
            return bool(ia) and bool(i_f) and ia[0] == i_f[0]
        if n == "topj":
            return set(ia[:self.j]) == set(i_f[:self.j])
        if n == "radius":
            return bool(A) and len(A) == len(F) and A[-1][1] <= self.c * F[-1][1] + EPS
        if n == "rank1":
            return bool(A) and bool(F) and A[0][1] <= self.c * F[0][1] + EPS
        if n == "sumdist":
            return len(A) == len(F) and sum(d for _, d in A) <= self.c * sum(d for _, d in F) + EPS
        if n in ("cadm", "cert"):
            return admissible(A, F, self.c)
        if n == "oracle":
            return proxy(A)[self.arg or "link"] == proxy(F)[self.arg or "link"]
        raise AssertionError(n)


PROXIES = ("dup", "tag", "link", "top3", "set")


def make_proxy(labels, tau, k):
    """Decision proxies of increasing sensitivity to the retrieved content.
    labels: mapping item_id -> label (array or dict); tau: duplicate threshold in metric distance."""
    def proxy(A):
        ids = [i for i, _ in A][:k]
        votes = {}
        for r, i in enumerate(ids):
            v = votes.setdefault(int(labels[i]) if labels is not None else 0, [0, r])
            v[0] += 1
        tag = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0] if votes else -1
        return {"dup": bool(A) and A[0][1] < tau, "tag": tag, "link": ids[0] if ids else -1,
                "top3": frozenset(ids[:3]), "set": frozenset(ids)}
    return proxy
