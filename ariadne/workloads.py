"""Workloads: who reads and writes what, in which order, with how much concurrency.

A workload fixes everything that does not depend on the protocol under test:

    base      [nb, d]  initial memory (item ids 0 .. nb-1)
    X         [n, d]   one vector per task; if the task writes, it writes item id nb + j with vector X[j]
    op        [n]      what the task wants: add | upsert | delete | read
    teams              closed loop: lists of task ids; `agents` client threads walk through each list in order
    arrival   [n]      open loop: arrival time of every task in seconds (teams is then ignored)
    Q                  optional extra read queries per task (multi-read transactions)
    labels             label of every item id (used by the `tag` decision proxy)
    texts              text of every item id (needed by the LLM decider only)
    truth     [n]      ground-truth identity of the fact a task reports (fan-out workloads), else None

YAML `workload:` section, by kind:

  kind: stream            replay a prepared data set in its natural order (LoCoMo, AG News, LongMemEval, ...)
    data: data/agnews__wordllama.npz
    base: 60000           items of the initial memory   (default: all items with role 0)
    stream: 6000          number of tasks               (default: all items with role 1)
    offset: 0             skip this many items first (role-less data only)
    teams: group|single|rr:<T>|block:<T>     one team per group id, one team, round robin, contiguous blocks
    agents: 16            client threads per team
    order: natural|shuffle|reverse|block:<B> stream order (shuffle removes temporal locality but keeps the items)
    exchangeable: false   true = pool base and stream and split at random (control: no sibling structure at all)
    ops: {add: 1.0}       operation mix, e.g. {add: 0.8, upsert: 0.1, delete: 0.05, read: 0.05}
    extra_reads: 0        additional similarity reads per task
    extra_source: prev|random|expand         previous items of the team / random base items / noisy copies of X[j]
    arrival: closed | {kind: poisson, rate: 50} | {kind: bursty, rate: 50, burst: 8}

  kind: fanout            synthetic paraphrases: every event is reported by f agents with slightly different vectors
    data: data/spacy__raw.npz
    n_base: 20000, n_events: 600, fanout: [[1, 0.5], [2, 0.3], [4, 0.2]]
    sigma: 0.20           paraphrase noise;   sep_cos: 0.25   events are at least this far from everything else
    conc: 16, think: 0.2  target number of in-flight tasks and mean think time (they fix the event rate)
    jitter: 1.0           copies of one event arrive within jitter * think seconds
    hot: 0                >0 concentrates events around this many anchors (topic locality)

  kind: clusters          real paraphrases: labelled duplicate clusters (Quora Question Pairs)
    data: data/qqp__wordllama.npz
    n_base: 50000, n_events: 2000, max_fan: 4, conc: 16, think: 0.2, jitter: 1.0

  kind: synthetic         planted law: background items exchangeable with the base, siblings at a planted dimension
    dim: 64, D: 16        ambient dimension and manifold dimension (a list of D gives heterogeneous dimensions)
    n_base: 50000, n_stream: 5000
    pi: 0.3, Ds: 4        sibling share and sibling dimension: d(parent, sibling) = R * U^(1/Ds)
    radius: auto          R; auto = 1.2 * 99th percentile of the k-th neighbour distance in the base
    lag_p: 0.5            the parent is l items earlier, l ~ Geometric(lag_p)
    chain: false          true lets siblings have siblings (clusters); false keeps the planted law exact
    teams / agents / ops / arrival as for `stream`
"""
import math
from dataclasses import dataclass, field
import numpy as np
from .data import load_prepared


@dataclass
class Workload:
    name: str
    base: np.ndarray
    X: np.ndarray
    op: list
    teams: list
    agents: int = 1
    team: np.ndarray = None
    arrival: np.ndarray = None
    Q: list = None
    labels: np.ndarray = None
    texts: list = None
    truth: np.ndarray = None
    meta: dict = field(default_factory=dict)

    @property
    def nb(self):
        return len(self.base)

    @property
    def n(self):
        return len(self.X)

    def queries(self, j):
        """All similarity reads of task j; the first one is the read for the task's own item."""
        if self.Q is None or self.Q[j] is None or len(self.Q[j]) == 0:
            return [self.X[j]]
        return [self.X[j]] + [q for q in self.Q[j]]

    def has_deletes(self):
        return any(o in ("upsert", "delete") for o in self.op)

    def n_clients(self):
        return len(self.teams) * self.agents


def _unit(X):
    return (X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)).astype("float32")


def _ops(cfg, n, rng):
    mix = cfg.get("ops") or {"add": 1.0}
    names = list(mix); p = np.array([float(mix[x]) for x in names]); p = p / p.sum()
    for x in names:
        if x not in ("add", "upsert", "delete", "read"):
            raise ValueError(f"unknown op '{x}'")
    if len(names) == 1:
        return [names[0]] * n
    return [names[i] for i in rng.choice(len(names), size=n, p=p)]


def _teams(spec, n, group):
    spec = str(spec or "single")
    if spec == "group" and group is not None and (group >= 0).any():
        ids = sorted(set(group.tolist()))
        return [[int(j) for j in np.where(group == g)[0]] for g in ids]
    if spec.startswith("rr:"):
        T = int(spec[3:]); return [list(range(t, n, T)) for t in range(T)]
    if spec.startswith("block:"):
        T = int(spec[6:]); cuts = np.linspace(0, n, T + 1).astype(int)
        return [list(range(cuts[t], cuts[t + 1])) for t in range(T)]
    return [list(range(n))]


def _team_array(teams, n):
    t = np.zeros(n, dtype=int)
    for ti, js in enumerate(teams):
        t[js] = ti
    return t


def _arrival(spec, n, rng):
    if spec in (None, "closed"):
        return None
    kind = spec.get("kind", "poisson"); rate = float(spec["rate"])
    if kind == "poisson":
        return np.cumsum(rng.exponential(1.0 / rate, size=n))
    if kind == "bursty":                                   # bursts of `burst` tasks arriving together, same mean rate
        b = int(spec.get("burst", 8)); nbur = math.ceil(n / b)
        starts = np.cumsum(rng.exponential(b / rate, size=nbur))
        return np.sort(np.repeat(starts, b)[:n] + rng.uniform(0, float(spec.get("spread", 0.0)), size=n))
    if kind == "uniform":
        return np.arange(n) / rate
    raise ValueError(f"unknown arrival kind '{kind}'")


def _extra_reads(cfg, X, base, teams, rng):
    r = int(cfg.get("extra_reads", 0))
    if r <= 0:
        return None
    src = cfg.get("extra_source", "prev"); n = len(X); Q = [None] * n
    if src == "prev":
        for js in teams:
            for i, j in enumerate(js):
                prev = js[max(0, i - r):i]
                Q[j] = X[prev] if prev else X[[j]]
    elif src == "random":
        for j in range(n):
            Q[j] = base[rng.choice(len(base), size=r, replace=False)]
    elif src == "expand":                                  # multi-query expansion: noisy copies of the task's own vector
        s = float(cfg.get("expand_sigma", 0.3)); d = X.shape[1]
        for j in range(n):
            Q[j] = _unit(X[j][None, :] + s * rng.normal(size=(r, d)) / math.sqrt(d))
    else:
        raise ValueError(f"unknown extra_source '{src}'")
    return Q


# ----------------------------------------------------------------------------- stream replay
def stream(cfg, seed, k=10):
    rng = np.random.default_rng(seed)
    X, role, group, label, texts = load_prepared(cfg["data"])
    n = len(X); idx = np.arange(n)
    has_roles = bool((role == 0).any() and (role == 1).any())
    if cfg.get("exchangeable"):
        nb = int(cfg.get("base", (role == 0).sum() if has_roles else n * 9 // 10))
        ns = int(cfg.get("stream", min(n - nb, (role == 1).sum() if has_roles else n - nb)))
        if nb <= 0 or ns <= 0 or nb + ns > n:
            raise ValueError("exchangeable split: base %d + stream %d does not fit the %d items of %s" % (nb, ns, n, cfg["data"]))
        perm = rng.permutation(n); bi, si = perm[:nb], perm[nb:nb + ns]
    elif has_roles:
        bi, si = idx[role == 0], idx[role == 1]
        if cfg.get("base"):
            bi = bi[:int(cfg["base"])]
        if cfg.get("stream"):
            si = si[:int(cfg["stream"])]
    else:
        off = int(cfg.get("offset", 0)); nb = int(cfg["base"]); ns = int(cfg["stream"])
        bi, si = idx[off:off + nb], idx[off + nb:off + nb + ns]
    order = str(cfg.get("order", "natural"))
    if order == "shuffle":
        si = si[rng.permutation(len(si))]
    elif order == "reverse":
        si = si[::-1]
    elif order.startswith("block:"):
        B = int(order[6:]); blocks = [si[i:i + B] for i in range(0, len(si), B)]
        si = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    ns = len(si)
    g = group[si] if not cfg.get("exchangeable") else None
    teams = _teams(cfg.get("teams", "group" if (g is not None and (g >= 0).any()) else "single"), ns, g)
    Xs = np.ascontiguousarray(X[si]); base = np.ascontiguousarray(X[bi])
    return Workload(name=cfg.get("name", "stream"), base=base, X=Xs, op=_ops(cfg, ns, rng), teams=teams,
                    agents=int(cfg.get("agents", 4)), team=_team_array(teams, ns), arrival=_arrival(cfg.get("arrival"), ns, rng),
                    Q=_extra_reads(cfg, Xs, base, teams, rng), labels=np.concatenate([label[bi], label[si]]),
                    texts=([texts[i] for i in bi] + [texts[i] for i in si]) if texts else None,
                    meta={"kind": "stream", "data": cfg["data"], "order": order, "exchangeable": bool(cfg.get("exchangeable"))})


# ----------------------------------------------------------------------------- fan-out (paraphrase copies)
def _fan_schedule(n_events, fan_of, conc, think, jitter, rng):
    """Arrival times: events arrive as a Poisson process whose rate keeps about `conc` tasks in flight;
    the copies of one event arrive within jitter * think seconds of each other."""
    mean_f = float(np.mean(fan_of))
    rate = conc / (mean_f * think)
    t = np.cumsum(rng.exponential(1.0 / rate, size=n_events))
    ev, at = [], []
    for e in range(n_events):
        for _ in range(int(fan_of[e])):
            ev.append(e); at.append(t[e] + rng.uniform(0, jitter * think))
    o = np.argsort(at, kind="stable")
    return np.array(ev)[o], np.array(at)[o]


def fanout(cfg, seed, k=10):
    rng = np.random.default_rng(seed)
    pool_all = load_prepared(cfg["data"])[0]
    nb, ne = int(cfg.get("n_base", 20000)), int(cfg.get("n_events", 600))
    sep = float(cfg.get("sep_cos", 0.25)); sigma = float(cfg.get("sigma", 0.20)); d = pool_all.shape[1]
    base, pool = pool_all[:nb], pool_all[nb:nb + int(cfg.get("pool", 200000))]
    import faiss
    ix = faiss.IndexFlatIP(d); ix.add(np.ascontiguousarray(base))
    d_base = 1.0 - ix.search(np.ascontiguousarray(pool), 1)[0][:, 0]
    cand = np.where(d_base > sep)[0]
    hot = int(cfg.get("hot", 0))
    if hot:
        anchors = pool[rng.choice(cand, size=hot, replace=False)]
        cand = cand[np.argsort(-(pool[cand] @ anchors.T).max(axis=1))]
    else:
        cand = rng.permutation(cand)
    chosen = np.zeros((0, d), dtype="float32"); pick = []
    for i in cand:                                          # greedy pairwise separation
        if len(pick) == 0 or float((chosen @ pool[i]).max()) < 1.0 - sep:
            pick.append(int(i)); chosen = np.vstack([chosen, pool[i][None, :]])
            if len(pick) == ne:
                break
    if len(pick) < ne:
        raise SystemExit(f"fanout: only {len(pick)} separable events available (asked for {ne}); lower n_events or sep_cos")
    fo = cfg.get("fanout", [[1, 0.5], [2, 0.3], [4, 0.2]])
    fs = np.array([f for f, _ in fo]); ps = np.array([p for _, p in fo], dtype=float); ps /= ps.sum()
    fan_of = rng.choice(fs, size=ne, p=ps)
    ev, at = _fan_schedule(ne, fan_of, float(cfg.get("conc", 16)), float(cfg.get("think", 0.2)), float(cfg.get("jitter", 1.0)), rng)
    X = _unit(chosen[ev] + sigma * rng.normal(size=(len(ev), d)).astype("float32") / math.sqrt(d))
    n = len(X)
    return Workload(name=cfg.get("name", "fanout"), base=np.ascontiguousarray(base), X=X, op=["add"] * n, teams=[list(range(n))], agents=1,
                    team=np.zeros(n, dtype=int), arrival=at - at[0], truth=ev,
                    meta={"kind": "fanout", "events": ne, "mean_fan": float(np.mean(fan_of)), "fan_of": fan_of.tolist()})


def clusters(cfg, seed, k=10):
    """Real paraphrases: every labelled duplicate cluster is one event, its members are the copies."""
    rng = np.random.default_rng(seed)
    X, role, group, label, texts = load_prepared(cfg["data"])
    nb, ne, mf = int(cfg.get("n_base", 50000)), int(cfg.get("n_events", 2000)), int(cfg.get("max_fan", 4))
    single = np.where(label < 0)[0]
    bi = single[:nb]
    cl = {}
    for i in np.where(label >= 0)[0]:
        cl.setdefault(int(label[i]), []).append(int(i))
    keys = [c for c in cl if len(cl[c]) >= int(cfg.get("min_fan", 1))]
    keys = [keys[i] for i in rng.permutation(len(keys))[:ne]]
    members = [cl[c][:mf] for c in keys]
    fan_of = np.array([len(m) for m in members])
    ev, at = _fan_schedule(len(keys), fan_of, float(cfg.get("conc", 16)), float(cfg.get("think", 0.2)), float(cfg.get("jitter", 1.0)), rng)
    nxt = [0] * len(keys); si = []
    for e in ev:
        si.append(members[e][nxt[e]]); nxt[e] += 1
    si = np.array(si); n = len(si)
    pair = [float(np.sqrt(max(2 - 2 * float(X[a] @ X[b]), 0))) for m in members for a in m for b in m if a < b]
    return Workload(name=cfg.get("name", "clusters"), base=np.ascontiguousarray(X[bi]), X=np.ascontiguousarray(X[si]), op=["add"] * n,
                    teams=[list(range(n))], agents=1, team=np.zeros(n, dtype=int), arrival=at - at[0], truth=ev,
                    labels=np.concatenate([label[bi], label[si]]), texts=([texts[i] for i in bi] + [texts[i] for i in si]) if texts else None,
                    meta={"kind": "clusters", "events": len(keys), "mean_fan": float(fan_of.mean()), "fan_of": fan_of.tolist(),
                          "within_cluster_chord_quantiles": np.percentile(pair, [10, 50, 90]).tolist() if pair else None})


# ----------------------------------------------------------------------------- planted sibling law
def _manifold(n, dim, D, rng, basis=None):
    """n points uniform on a D-dimensional unit sphere embedded in R^dim (a random (D+1)-dimensional subspace)."""
    if basis is None:
        basis = np.linalg.qr(rng.normal(size=(dim, D + 1)))[0]
    g = rng.normal(size=(n, D + 1))
    return _unit(g @ basis.T), basis


def synthetic(cfg, seed, k=10):
    rng = np.random.default_rng(seed)
    dim = int(cfg.get("dim", 64)); Ds = float(cfg.get("Ds", 4)); pi = float(cfg.get("pi", 0.3))
    nb, n = int(cfg.get("n_base", 50000)), int(cfg.get("n_stream", 5000))
    Dl = cfg.get("D", 16); Dl = [int(x) for x in (Dl if isinstance(Dl, (list, tuple)) else [Dl])]
    bases = [np.linalg.qr(rng.normal(size=(dim, D + 1)))[0] for D in Dl]
    mb = rng.integers(0, len(Dl), size=nb); ms = rng.integers(0, len(Dl), size=n)
    base = np.zeros((nb, dim), dtype="float32"); X = np.zeros((n, dim), dtype="float32")
    for m, B in enumerate(bases):
        base[mb == m] = _manifold(int((mb == m).sum()), dim, Dl[m], rng, B)[0]
        X[ms == m] = _manifold(int((ms == m).sum()), dim, Dl[m], rng, B)[0]
    R = cfg.get("radius", "auto")
    if R == "auto":
        import faiss
        ix = faiss.IndexFlatIP(dim); ix.add(base)
        q = base[rng.choice(nb, size=min(2000, nb), replace=False)]
        S = ix.search(q, k + 1)[0][:, -1]
        R = 1.2 * float(np.percentile(np.sqrt(np.maximum(2 - 2 * S, 0)), 99))
    R = min(float(R), 1.9)
    lag_p = float(cfg.get("lag_p", 0.5)); chain = bool(cfg.get("chain", False))
    parent = np.full(n, -1); has_child = np.zeros(n, dtype=bool); t_plant = np.full(n, np.nan)
    for j in range(1, n):
        if rng.uniform() >= pi:
            continue
        p = j - int(rng.geometric(lag_p))
        if p < 0 or (not chain and (parent[p] >= 0 or has_child[p])):
            continue
        B = bases[ms[p]]
        v = B @ rng.normal(size=B.shape[1]); v -= float(v @ X[p]) * X[p]; v /= np.linalg.norm(v)      # unit tangent direction on the manifold
        t = R * rng.uniform() ** (1.0 / Ds)                  # chord distance with density ~ t^(Ds-1) on [0, R]
        th = 2.0 * math.asin(min(t / 2.0, 1.0))
        X[j] = (math.cos(th) * X[p] + math.sin(th) * v).astype("float32"); ms[j] = ms[p]
        parent[j] = p; has_child[p] = True; t_plant[j] = t
    teams = _teams(cfg.get("teams", "single"), n, None)
    return Workload(name=cfg.get("name", "synthetic"), base=base, X=_unit(X), op=_ops(cfg, n, rng), teams=teams, agents=int(cfg.get("agents", 16)),
                    team=_team_array(teams, n), arrival=_arrival(cfg.get("arrival"), n, rng), Q=_extra_reads(cfg, X, base, teams, rng),
                    labels=np.concatenate([mb, ms]).astype("int32"), truth=parent,
                    meta={"kind": "synthetic", "D": Dl, "Ds": Ds, "pi": pi, "R": R, "lag_p": lag_p, "chain": chain,
                          "sibling_share_realized": float((parent >= 0).mean()), "t_plant": t_plant.tolist()})


KINDS = {"stream": stream, "fanout": fanout, "clusters": clusters, "synthetic": synthetic}


def make_workload(cfg, seed=1, k=10):
    kind = cfg.get("kind", "stream")
    if kind not in KINDS:
        raise ValueError(f"unknown workload kind '{kind}'")
    return KINDS[kind](cfg, seed, k)
