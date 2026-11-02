"""Execution engine: runs one workload under one concurrency-control protocol and logs every task.

A task is a read-think-write transaction of an agent:

    As  = similarity reads for the task                     (what the agent reasons about)
    dec = decider(task, As)                                 (think: one model call)
    write what `dec` says                                   (ADD / UPDATE / DELETE / NOOP)

Protocols (YAML `protocol:` section, field `kind`):

  validate    Algorithm AV: read without lock, think, take the commit lock, re-read (F), test the policy,
              apply or think again on F.
                policy: cadm:1.1            any spec from policies.py (none|exact|set|overlap:f|top1|topj:j|radius:c|
                                            rank1:c|sumdist:c|cadm:c|cert:c|oracle:<proxy>|objects|redecide)
                policy_by_op: {upsert: "cadm:1.02"}    per-operation override (per-decision tolerance)
                lock: global|group|none     commit-lock scope; none = re-read without mutual exclusion
                on_invalid: rethink|reread|abort       reuse F as the new answer / read again outside the lock / give up
                validation: ann|exact       how F is obtained;  ef_val: search budget of the validation read
                max_rounds: 30
                on_cap: abort|commit        at max_rounds failed validations: give up without writing (default; the
                                            safety theorem then holds for every commit) or commit unvalidated
                validate_readonly: false    read-only tasks are serialized at their read and skip validation
                exist_check: true           with deletions in the workload, also check that the items of A still exist
                read_isolation: null        pgvector only: isolation level of the read phase
                cs_isolation: null          pgvector only: isolation level of the critical section (default READ
                                            COMMITTED; REPEATABLE READ takes its snapshot before the lock is granted
                                            and is unsound, which is one of the ablations)
              `validate` with `policy: none` is the instrumented no-isolation run: it never rejects, but logs
              F for every task, which is what the counterfactual analysis and the conflict law are computed from.
  none        read, think, write; no re-read, no lock.   isolation: READ COMMITTED|REPEATABLE READ, txn: true|false
  lock_long   one global (or per-group) lock held across the think step (pessimistic, serial).
  ssi         PostgreSQL SERIALIZABLE.  variant: long (think inside the transaction, retry on failure)
              | short (think outside, then a short SERIALIZABLE transaction that re-reads and applies `policy`);
              retry: immediate|backoff;  max_aborts: 200
  ball        pessimistic predicate locking on the read ball, in the client library.  c: 1.0 is the strict
              ball lock (any insertion into the read's top-k ball waits); c > 1 lets through the insertions
              that keep the held answer c-admissible.  Deadlocks abort the requester (re-read, think again).
  compensate  insert first, re-read, delete the own item if a duplicate is found.  tiebreak: id|any
              (application-level "write then verify"; threshold dedup only).

Other top-level fields: k, tau_cos, ef, seed, clients: {processes: 1, max_open: 256}, log_answers: full|none,
faults: {crash_in_cs: 0.01} (a client fails inside the critical section with this probability, rolls back, starts over).

Every run returns (summary, recs, commit_log):
  commit_log   list of (pos, op, item, task) in commit order, op in {"i", "d"}
  recs[j]      j, team, op, rounds (model calls), R (answers the committed decision was computed from),
               F (answers at commit), s (commit-log size at the commit-time read), s_read (at the read of R),
               first = (R, F, s, s_read) at the first validation, dec, ops, lat, think, lock_wait, cs,
               aborts, rereads, capped, gave_up, t0, t1
"""
import itertools, json, math, multiprocessing as mp, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .metrics import chord_from_cos, stale_conflicts, EPS
from .policies import Policy, make_proxy
from .deciders import make_decider, Think
from .workloads import make_workload
from .stores import make_store


class CommitLog:
    """Positions come from a shared counter, so the log also works with client processes."""

    def __init__(self):
        self.n = mp.Value("q", 0)
        self.local = []

    def append(self, ops, j):
        if not ops:
            return self.n.value
        with self.n.get_lock():
            pos = self.n.value; self.n.value += len(ops)
        self.local.extend((pos + i, op, int(item), int(j)) for i, (op, item) in enumerate(ops))
        return pos

    def size(self):
        return self.n.value


# ----------------------------------------------------------------------------- ball predicate locks
class BallLocks:
    """Predicate locks on similarity reads, held from the read to the end of the transaction.

    A reader registers (query, answer).  A writer may insert v only if, for every registered read of
    another transaction, the held answer stays c-admissible with v added (c = 1: v must lie outside the
    read's top-k ball), and may delete an item only if no other registered answer contains it.
    Otherwise the writer waits for the readers in its way; a cycle of waiting writers is a deadlock and
    the transaction that closes the cycle aborts."""

    def __init__(self, c, k, metric):
        self.c, self.k, self.metric = float(c), k, metric
        self.cv = threading.Condition()
        self.active = {}                    # tid -> list of balls {q, orig: [dist], cur: [(dist, item)], ids}
        self.waits = {}                     # tid -> set of tids it waits for
        self.version = 0
        self.recent = deque(maxlen=4096)    # (version, op, item, vec)
        self.stats = {"waits": 0, "deadlocks": 0, "stale_reads": 0}

    def dist(self, q, v):
        if self.metric == "cosine":
            return float(np.sqrt(max(2.0 - 2.0 * float(q @ v), 0.0)))
        return float(np.linalg.norm(q - v))

    def _blocks(self, ball, op, item, vec):
        if op == "d":
            return item in ball["ids"]
        if item in ball["ids"] or any(i == item for _, i in ball["cur"]):
            return False
        if len(ball["orig"]) < self.k:
            return True
        t = self.dist(ball["q"], vec)
        if t >= ball["cur"][-1][0]:
            return False
        new = sorted([d for d, _ in ball["cur"]] + [t])[:len(ball["orig"])]
        return stale_conflicts(ball["orig"], new, self.c)

    def _note(self, op, item, vec):
        """Bookkeeping after a write went through: keep the current-distance lists of the other reads."""
        self.version += 1
        self.recent.append((self.version, op, item, vec))
        if op == "d":                       # a deleted item stays in the lists: they then under-estimate the current
            return                          # distances, which can only add conflicts (safe)
        for balls in self.active.values():
            for b in balls:
                self._absorb(b, item, vec)

    def _absorb(self, b, item, vec):
        t = self.dist(b["q"], vec)
        if b["cur"] and t < b["cur"][-1][0] and item not in b["ids"] and all(i != item for _, i in b["cur"]):
            b["cur"] = sorted(b["cur"] + [(t, item)])[:len(b["orig"])]

    def register(self, tid, Qs, As, v0):
        """Returns False if a write that committed while the read was running makes the answer stale."""
        balls = [{"q": q, "orig": [d for _, d in A], "cur": [(d, i) for i, d in A], "ids": {i for i, _ in A}} for q, A in zip(Qs, As)]
        with self.cv:
            for ver, op, item, vec in self.recent:
                if ver > v0 and any(self._blocks(b, op, item, vec) for b in balls):
                    self.stats["stale_reads"] += 1
                    return False
            if self.recent and self.recent[0][0] > v0 + 1 and len(self.recent) == self.recent.maxlen:
                self.stats["stale_reads"] += 1          # history too short to decide: read again
                return False
            for ver, op, item, vec in self.recent:          # tolerated insertions the read did not see yet
                if ver > v0 and op == "i":
                    for b in balls:
                        self._absorb(b, item, vec)
            self.active[tid] = balls
        return True

    def _cycle(self, tid):
        seen, stack = set(), list(self.waits.get(tid, ()))
        while stack:
            x = stack.pop()
            if x == tid:
                return True
            if x in seen:
                continue
            seen.add(x); stack.extend(self.waits.get(x, ()))
        return False

    def write(self, tid, writes, do):
        """writes: [(op, item, vec)].  Calls do() (the physical writes) once nothing is in the way.
        Returns ("ok", waited_seconds) or ("abort", waited_seconds)."""
        t0 = time.time(); waited = False
        with self.cv:
            while True:
                blockers = {t for t, balls in self.active.items() if t != tid
                            and any(self._blocks(b, op, item, vec) for b in balls for op, item, vec in writes)}
                if not blockers:
                    self.waits.pop(tid, None)
                    do()
                    for op, item, vec in writes:
                        self._note(op, item, vec)
                    self.cv.notify_all()
                    return "ok", time.time() - t0
                self.waits[tid] = blockers
                if self._cycle(tid):
                    self.stats["deadlocks"] += 1
                    self.waits.pop(tid, None); self.active.pop(tid, None)
                    self.cv.notify_all()
                    return "abort", time.time() - t0
                if not waited:
                    self.stats["waits"] += 1; waited = True
                self.cv.wait(timeout=1.0)

    def release(self, tid):
        with self.cv:
            self.active.pop(tid, None); self.waits.pop(tid, None)
            self.cv.notify_all()


# ----------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, cfg, wl, store):
        self.cfg, self.wl, self.store = cfg, wl, store
        self.k = int(cfg.get("k", 10)); self.seed = int(cfg.get("seed", 1))
        self.tau = chord_from_cos(float(cfg.get("tau_cos", 0.10))) if store.metric == "cosine" else float(cfg.get("tau", 0.45))
        self.ef = cfg.get("ef")
        P = dict(cfg.get("protocol") or {})
        self.P = P; self.kind = P.get("kind", "validate")
        self.policy = Policy(P.get("policy", "none" if self.kind != "ssi" else "redecide"), self.k)
        self.by_op = {op: Policy(sp, self.k) for op, sp in (P.get("policy_by_op") or {}).items()}
        self.scope = P.get("lock", "global")
        self.on_invalid = P.get("on_invalid", "rethink")
        self.max_rounds = int(P.get("max_rounds", 30)); self.max_aborts = int(P.get("max_aborts", 200))
        self.on_cap = P.get("on_cap", "abort")
        if self.on_cap not in ("abort", "commit"):
            raise ValueError("protocol.on_cap must be abort or commit, not %r" % self.on_cap)
        self.ef_val = P.get("ef_val", self.ef); self.val_exact = P.get("validation", "ann") == "exact"
        # deletions come from the workload (upsert / delete operations) or from a model that may answer UPDATE / DELETE
        can_delete = wl.has_deletes() or (cfg.get("decider") or {}).get("kind") == "llm"
        self.auto_exist = can_delete and P.get("exist_check", True)
        self.validate_ro = bool(P.get("validate_readonly", False))
        F = dict(cfg.get("faults") or {})
        self.p_crash = float(F.get("crash_in_cs", 0.0)); self.frng = np.random.default_rng(self.seed + 77); self.fmu = threading.Lock()
        self.decider = make_decider(cfg.get("decider"), wl, self.tau, self.seed)
        self.proxy = make_proxy(wl.labels, self.tau, self.k)
        self.log = CommitLog()
        self.balls = BallLocks(P.get("c", 1.0), self.k, store.metric) if self.kind == "ball" else None
        self.keep = cfg.get("log_answers", "full") != "none"
        self.tids = itertools.count(1)
        self.fn = {"validate": self.p_validate, "none": self.p_none, "lock_long": self.p_lock_long, "ssi": self.p_ssi,
                   "ball": self.p_ball, "compensate": self.p_compensate}[self.kind]
        if self.kind == "ssi" and not store.has_txn:
            raise SystemExit("protocol ssi needs a store with transactions (pgvector)")

    # -- helpers
    def pol(self, j):
        return self.by_op.get(self.wl.op[j], self.policy)

    def read(self, sess, j):
        return [sess.search(q, self.k, self.ef) for q in self.wl.queries(j)]

    def vread(self, sess, j, pol):
        if self.val_exact or pol.needs_exact:
            try:
                return [sess.exact_search(q, self.k) for q in self.wl.queries(j)]
            except NotImplementedError:
                raise SystemExit(f"store '{self.store.name}' has no exact search: exact validation (cert:c, validation: exact) "
                                 "needs pgvector, faissmem, qdrant or lancedb")
        return [sess.search(q, self.k, self.ef_val) for q in self.wl.queries(j)]

    def valid(self, sess, pol, As, Fs):
        for A, F in zip(As, Fs):
            if not pol.valid(A, F, self.proxy):
                return False
        if pol.check_exist or (self.auto_exist and pol.name not in ("none", "redecide")):
            ids = {i for A in As for i, _ in A}
            if len(sess.exists(ids)) != len(ids):
                return False
        return True

    def plan(self, j, dec):
        me = self.wl.nb + j
        if dec.action == "ADD":
            return [("i", me)]
        if dec.action == "UPDATE":
            return [("d", int(dec.target)), ("i", me)]
        if dec.action == "DELETE":
            return [("d", int(dec.target))]
        return []

    def apply(self, sess, j, ops):
        for op, item in ops:
            if op == "i":
                sess.insert(item, self.wl.X[j])
            else:
                sess.delete(item)

    def crash(self):
        if self.p_crash <= 0:
            return False
        with self.fmu:
            return bool(self.frng.uniform() < self.p_crash)

    def lock_key(self, j):
        return 0 if self.scope != "group" else 1 + int(self.wl.team[j])

    def finish(self, rec, As, Fs, s, s_read, dec, ops):
        rec["R"], rec["F"] = (As, Fs) if self.keep else (None, None)
        rec["s"], rec["s_read"] = s, s_read
        rec["dec"] = dec.key() if dec is not None else None
        rec["ops"] = ops

    # -- protocols
    def p_validate(self, sess, j, rec):
        pol = self.pol(j); key = self.lock_key(j)
        sess.begin(self.P.get("read_isolation"))
        s_read = self.log.size(); As = self.read(sess, j); sess.commit()
        while True:
            rec["rounds"] += 1
            dec, th = self.decider.decide(j, As); rec["think"] += th
            if self.wl.op[j] == "read" and not self.validate_ro and len(As) == 1:    # a single-read, read-only task is serialized at its read
                return self.finish(rec, As, None, s_read, s_read, dec, [])
            t1 = time.time()
            sess.begin(self.P.get("cs_isolation"))          # default READ COMMITTED: the snapshot must postdate the lock
            if self.scope != "none":
                sess.lock(key)
            t2 = time.time(); rec["lock_wait"] += t2 - t1
            Fs = self.vread(sess, j, pol); s = self.log.size()
            t3 = time.time(); rec["t_vread"] += t3 - t2
            ok = self.valid(sess, pol, As, Fs)
            rec["t_pred"] += time.time() - t3
            if self.crash():                                # injected client failure inside the critical section
                sess.rollback(); rec["crashes"] += 1; rec["cs"] += time.time() - t2
                s_read = self.log.size(); As = self.read(sess, j); sess.commit()
                continue
            if rec["first"] is None and self.keep:
                rec["first"] = (As, Fs, s, s_read)
            if rec["first_ok"] is None:
                rec["first_ok"] = bool(ok)
            if pol.redecide:
                dec, _ = self.decider.decide(j, Fs, quick=True); As, s_read, ok = Fs, s, True
            capped = (not ok) and rec["rounds"] >= self.max_rounds
            commit_capped = capped and self.on_cap == "commit"          # unvalidated commit: outside the safety theorem
            if ok or capped or self.on_invalid == "abort":
                ops = self.plan(j, dec) if (ok or commit_capped) else []
                t4 = time.time()
                self.apply(sess, j, ops); self.log.append(ops, j)
                sess.commit(); rec["t_write"] += time.time() - t4; rec["cs"] += time.time() - t2
                rec["capped"] = capped; rec["gave_up"] = not (ok or commit_capped)
                return self.finish(rec, As, Fs, s, s_read, dec, ops)
            sess.commit(); rec["cs"] += time.time() - t2
            if self.on_invalid == "reread":
                s_read = self.log.size(); As = self.read(sess, j); sess.commit()
            else:
                As, s_read = Fs, s

    def p_none(self, sess, j, rec):
        txn = bool(self.P.get("txn", False))
        sess.begin(self.P.get("isolation"))
        s_read = self.log.size(); As = self.read(sess, j)
        if not txn:
            sess.commit()
        rec["rounds"] = 1
        dec, th = self.decider.decide(j, As); rec["think"] += th
        s = self.log.size(); ops = self.plan(j, dec)
        try:
            self.apply(sess, j, ops); sess.commit(); self.log.append(ops, j)
        except Exception as e:                              # snapshot isolation can refuse a concurrent delete/update
            if not self._retryable(e):
                raise
            sess.rollback(); rec["aborts"] += 1; rec["gave_up"] = True; ops = []
        self.finish(rec, As, None, s, s_read, dec, ops)

    def p_lock_long(self, sess, j, rec):
        sess.begin(None)
        t1 = time.time(); sess.lock(self.lock_key(j)); t2 = time.time(); rec["lock_wait"] += t2 - t1
        s = self.log.size(); As = self.read(sess, j)
        rec["rounds"] = 1
        dec, th = self.decider.decide(j, As); rec["think"] += th
        ops = self.plan(j, dec); self.apply(sess, j, ops); self.log.append(ops, j)
        sess.commit(); rec["cs"] += time.time() - t2
        self.finish(rec, As, As, s, s, dec, ops)

    def _retryable(self, e):
        E = getattr(self.store, "errors", None)
        return E is not None and isinstance(e, (E.SerializationFailure, E.DeadlockDetected))

    def _backoff(self, rec):
        if self.P.get("retry", "immediate") == "backoff":   # capped exponential backoff with full jitter
            time.sleep(np.random.uniform(0, min(1.0, 0.01 * 2 ** min(rec["aborts"], 10))))

    def p_ssi(self, sess, j, rec):
        pol = self.pol(j); short = self.P.get("variant", "long") == "short"
        As = dec = None
        if short:
            sess.begin("READ COMMITTED")
            s_read = self.log.size(); As = self.read(sess, j); sess.commit()
            rec["rounds"] += 1
            dec, th = self.decider.decide(j, As); rec["think"] += th
        sess.begin("SERIALIZABLE")
        while True:
            try:
                t2 = time.time()
                if short:
                    Fs = self.vread(sess, j, pol); s = self.log.size()
                    ok = self.valid(sess, pol, As, Fs)
                    if rec["first"] is None and self.keep:
                        rec["first"] = (As, Fs, s, s_read)
                    if rec["first_ok"] is None:
                        rec["first_ok"] = bool(ok)
                    if pol.redecide:
                        dec, _ = self.decider.decide(j, Fs, quick=True); As, s_read, ok = Fs, s, True
                    if not ok and rec["rounds"] < self.max_rounds:
                        sess.rollback()
                        As, s_read = Fs, s; rec["rounds"] += 1
                        dec, th = self.decider.decide(j, As); rec["think"] += th
                        continue
                else:
                    s_read = s = self.log.size(); As = self.read(sess, j); Fs = As
                    rec["rounds"] += 1
                    dec, th = self.decider.decide(j, As); rec["think"] += th
                ops = self.plan(j, dec); self.apply(sess, j, ops)
                sess.commit(); self.log.append(ops, j); rec["cs"] += time.time() - t2
                return self.finish(rec, As, Fs, s, s_read, dec, ops)
            except Exception as e:
                if not self._retryable(e):
                    raise
                sess.rollback(); rec["aborts"] += 1
                if rec["aborts"] > self.max_aborts:
                    rec["gave_up"] = True
                    return self.finish(rec, As, None, self.log.size(), self.log.size(), dec, [])
                self._backoff(rec)

    def p_ball(self, sess, j, rec):
        B = self.balls; tid = next(self.tids); Qs = self.wl.queries(j)
        sess.begin(None)
        while True:
            while True:
                v0 = B.version; s_read = self.log.size()
                As = self.read(sess, j); sess.commit()
                if B.register(tid, Qs, As, v0):
                    break
                rec["rereads"] += 1
            rec["rounds"] += 1
            dec, th = self.decider.decide(j, As); rec["think"] += th
            ops = self.plan(j, dec); out = {}

            def do():
                out["s"] = self.log.size()
                self.apply(sess, j, ops); sess.commit(); self.log.append(ops, j)
            if not ops:
                B.release(tid)
                return self.finish(rec, As, None, self.log.size(), s_read, dec, ops)
            vec = self.wl.X[j]
            res, waited = B.write(tid, [(op, item, vec if op == "i" else None) for op, item in ops], do)
            rec["lock_wait"] += waited
            if res == "ok":
                B.release(tid)
                return self.finish(rec, As, None, out["s"], s_read, dec, ops)
            rec["aborts"] += 1
            if rec["aborts"] > self.max_aborts:
                rec["gave_up"] = True
                return self.finish(rec, As, None, self.log.size(), s_read, dec, [])

    def p_compensate(self, sess, j, rec):
        me = self.wl.nb + j
        sess.begin(None)
        s_read = self.log.size(); As = self.read(sess, j); sess.commit()
        rec["rounds"] = 1
        dec, th = self.decider.decide(j, As); rec["think"] += th
        ops = self.plan(j, dec); s = self.log.size(); Fs = None
        self.apply(sess, j, ops); sess.commit(); self.log.append(ops, j)
        if dec.action == "ADD":
            t2 = time.time()
            Fs = [sess.search(self.wl.X[j], self.k, self.ef_val)]; sess.commit()
            rivals = [i for i, d in Fs[0] if i != me and d < self.tau]
            if rivals and (self.P.get("tiebreak", "id") == "any" or min(rivals) < me):
                sess.delete(me); sess.commit(); self.log.append([("d", me)], j); ops = ops + [("d", me)]
                rec["compensated"] = True
            rec["cs"] += time.time() - t2
        self.finish(rec, As, Fs, s, s_read, dec, ops)

    # -- one task
    def run_task(self, sess, j, t_sched=None):
        rec = {"j": int(j), "team": int(self.wl.team[j]) if self.wl.team is not None else 0, "op": self.wl.op[j], "rounds": 0,
               "think": 0.0, "lock_wait": 0.0, "cs": 0.0, "aborts": 0, "rereads": 0, "capped": False, "gave_up": False,
               "first": None, "first_ok": None, "t_vread": 0.0, "t_pred": 0.0, "t_write": 0.0, "crashes": 0}
        t0 = time.time() if t_sched is None else t_sched
        rec["t0"] = t0
        try:
            self.fn(sess, j, rec)
        except BaseException:                               # never leave the commit lock or a ball behind
            try:
                sess.rollback()
            except Exception:
                pass
            if self.balls is not None:
                with self.balls.cv:
                    self.balls.active.clear(); self.balls.waits.clear(); self.balls.cv.notify_all()
            raise
        rec["t1"] = time.time(); rec["lat"] = rec["t1"] - t0
        return rec


# ----------------------------------------------------------------------------- client drivers
def _closed_threads(R, recs):
    wl = R.wl; ptr = [0] * len(wl.teams); mu = threading.Lock(); err = []

    def worker(t):
        try:
            sess = R.store.session()
            while True:
                with mu:
                    if ptr[t] >= len(wl.teams[t]):
                        break
                    j = wl.teams[t][ptr[t]]; ptr[t] += 1
                recs[j] = R.run_task(sess, j)
            sess.close()
        except BaseException as e:                          # surface worker failures instead of hanging the run
            err.append(e)
            with mu:
                for i in range(len(ptr)):
                    ptr[i] = 10 ** 12
    ths = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(len(wl.teams)) for _ in range(wl.agents)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if err:
        raise err[0]


def _open_threads(R, recs):
    wl = R.wl; tl = threading.local(); sessions = []; mu = threading.Lock()
    order = np.argsort(wl.arrival, kind="stable")
    scale = float(R.cfg.get("arrival_scale", 1.0))

    def one(j, t_sched):
        if not hasattr(tl, "s"):
            tl.s = R.store.session()
            with mu:
                sessions.append(tl.s)
        recs[j] = R.run_task(tl.s, j, t_sched)
    with ThreadPoolExecutor(max_workers=int((R.cfg.get("clients") or {}).get("max_open", 256))) as ex:
        t0 = time.time(); futs = []
        for j in order:
            due = t0 + float(wl.arrival[j]) * scale
            d = due - time.time()
            if d > 0:
                time.sleep(d)
            futs.append(ex.submit(one, int(j), due))
        err = None
        for f in futs:
            try:
                f.result()
            except BaseException as e:
                err = err or e
        if err is not None:
            raise err
    for s in sessions:
        s.close()


def _proc_main(R, idx, team, ptr, q):
    try:
        R.decider.think = Think(R.decider.think.spec, R.decider.think.scale, R.seed + 1000 * (idx + 1))
        np.random.seed(R.seed + idx)
        sess = R.store.session(); out = []
        while True:
            with ptr.get_lock():
                i = ptr[team]
                if i >= len(R.wl.teams[team]):
                    break
                ptr[team] = i + 1
            out.append(R.run_task(sess, R.wl.teams[team][i]))
        sess.close()
        q.put((out, R.log.local, None))
    except BaseException as e:
        q.put(([], R.log.local, repr(e)))


def _closed_procs(R, recs):
    wl = R.wl
    if R.kind == "ball" or R.store.name == "faissmem":
        raise SystemExit("client processes need a server store and a protocol without in-process lock state")
    if not R.store.has_txn and not R.store.locks.file_dir:
        raise SystemExit("client processes on a store without transactions need `lock: file:<dir>` in the store config")
    ctx = mp.get_context("fork"); ptr = mp.Array("q", len(wl.teams)); q = ctx.Queue()
    ps = [ctx.Process(target=_proc_main, args=(R, i, t, ptr, q)) for i, t in
          enumerate(t for t in range(len(wl.teams)) for _ in range(wl.agents))]
    for p in ps:
        p.start()
    err = None
    for _ in ps:
        out, local, e = q.get()
        err = err or e
        for r in out:
            recs[r["j"]] = r
        R.log.local.extend(local)
    for p in ps:
        p.join()
    if err:
        raise RuntimeError(err)


# ----------------------------------------------------------------------------- one run
def pcts(x, qs=(50, 95, 99)):
    x = np.asarray(x, dtype=float)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} if len(x) else {}


def summarize(R, recs, clog, wall):
    wl = R.wl; n = len(recs)
    calls = sum(r["rounds"] for r in recs)
    lat = [r["lat"] for r in recs]
    live = set()
    for _, op, item, _ in clog:
        (live.add if op == "i" else live.discard)(item)
    S = {"tasks": n, "clients": wl.n_clients() if wl.arrival is None else None, "open_loop": wl.arrival is not None,
         "wall_s": round(wall, 2), "tps": round(n / wall, 2),
         "llm_calls": calls, "llm_calls_per_task": round(calls / n, 4), "extra_calls_per_100": round(100 * (calls - n) / n, 2),
         "tasks_with_redo_%": round(100 * sum(r["rounds"] > 1 for r in recs) / n, 2),
         "first_validation_failed_%": round(100 * sum(r["first_ok"] is False for r in recs) / n, 2),
         "max_rounds_seen": max(r["rounds"] for r in recs), "capped": sum(r["capped"] for r in recs),
         "gave_up": sum(r["gave_up"] for r in recs), "aborts_per_task": round(sum(r["aborts"] for r in recs) / n, 4),
         "rereads_per_task": round(sum(r["rereads"] for r in recs) / n, 4),
         "lat_mean_s": round(float(np.mean(lat)), 4), **{"lat_" + k + "_s": v for k, v in pcts(lat).items()},
         "think_mean_s": round(float(np.mean([r["think"] for r in recs])), 4),
         "lock_wait_mean_ms": round(1000 * float(np.mean([r["lock_wait"] for r in recs])), 3),
         "lock_wait_p99_ms": round(1000 * float(np.percentile([r["lock_wait"] for r in recs], 99)), 3),
         "cs_mean_ms": round(1000 * float(np.mean([r["cs"] for r in recs])), 3),
         "cs_p99_ms": round(1000 * float(np.percentile([r["cs"] for r in recs], 99)), 3),
         "cs_vread_mean_ms": round(1000 * float(np.mean([r["t_vread"] for r in recs])), 3),
         "cs_predicate_mean_ms": round(1000 * float(np.mean([r["t_pred"] for r in recs])), 4),
         "cs_write_mean_ms": round(1000 * float(np.mean([r["t_write"] for r in recs])), 3),
         "crashes": sum(r["crashes"] for r in recs),
         "rounds_p99": int(np.percentile([r["rounds"] for r in recs], 99)),
         "lock_held_fraction": (round(float(sum(r["cs"] for r in recs)) / wall, 4)       # share of the run during which the commit
                                if (R.kind == "lock_long" or (R.kind == "validate" and R.scope == "global")) else None),   # lock was held
         "actions": {a: sum(1 for r in recs if r["dec"] and r["dec"][0] == a) for a in ("ADD", "NOOP", "UPDATE", "DELETE")},
         "inserted": sum(1 for _, op, _, _ in clog if op == "i"), "deleted": sum(1 for _, op, _, _ in clog if op == "d"),
         "stored_stream_items": len([i for i in live if i >= wl.nb])}
    try:
        S["stored_in_store"] = R.store.count_extra()
    except Exception:
        S["stored_in_store"] = None
    if wl.truth is not None and wl.meta.get("kind") in ("fanout", "clusters"):
        ne = int(wl.meta["events"]); cnt = np.zeros(ne, dtype=int)
        for i in live:
            if i >= wl.nb:
                cnt[wl.truth[i - wl.nb]] += 1
        S["events"] = ne
        S["dup_rows"] = int(np.maximum(cnt - 1, 0).sum()); S["dup_per_100_events"] = round(100 * S["dup_rows"] / ne, 2)
        S["events_with_dup_%"] = round(100 * float(np.mean(cnt > 1)), 2); S["lost_events"] = int(np.sum(cnt == 0))
        fan = np.array(wl.meta["fan_of"])
        S["dup_per_100_events_by_fan"] = {int(f): round(100 * float(np.maximum(cnt[fan == f] - 1, 0).sum()) / max(int((fan == f).sum()), 1), 2)
                                          for f in sorted(set(fan.tolist()))}
    if R.balls is not None:
        S["ball"] = dict(R.balls.stats)
    S["decider"] = R.decider.stats()
    if hasattr(R.store, "versions"):
        try:
            S["versions"] = R.store.versions()
        except Exception:
            pass
    try:                                                   # the machine (checked against paper/env.yaml)
        import os
        S["machine"] = {"cpus": os.cpu_count(),
                        "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30)}
    except Exception:
        pass
    return S


def execute(cfg, wl=None, store=None):
    """Run one configuration.  Pass `wl` and a set-up `store` to reuse them across runs."""
    seed = int(cfg.get("seed", 1)); k = int(cfg.get("k", 10))
    if wl is None:
        wl = make_workload(cfg["workload"], cfg.get("workload_seed", seed), k)
    if store is None:
        store = make_store(cfg.get("store") or {}); store.setup(wl.base)
    store.reset()
    R = Runner(cfg, wl, store)
    recs = [None] * wl.n
    procs = int((cfg.get("clients") or {}).get("processes", 1))
    t0 = time.time()
    if wl.arrival is not None:
        _open_threads(R, recs)
    elif procs > 1:
        _closed_procs(R, recs)
    else:
        _closed_threads(R, recs)
    wall = time.time() - t0
    assert all(r is not None for r in recs), "some tasks did not finish"
    clog = sorted(R.log.local)
    assert [c[0] for c in clog] == list(range(len(clog))), "commit log has gaps"
    S = summarize(R, recs, clog, wall)
    return S, recs, clog
