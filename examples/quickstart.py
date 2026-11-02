#!/usr/bin/env python3
"""Admissibility validation in about a hundred lines: the duplicate race and how Algorithm AV removes it.

Several agents share one in-process vector store (FAISS HNSW, no server, no data download).  Each agent receives
paraphrases of the same facts and runs the usual check-then-insert transaction:

    read the k nearest memories  ->  think (a sleep stands in for the model call)  ->  insert if nothing is within tau

Run without isolation, agents that think about the same fact at the same time all insert it.  Run with admissibility
validation (policy cadm:c), each transaction takes the commit lock, repeats its read, tests the held answer against
the fresh one per rank, and only then writes; otherwise it thinks again on the fresh answer.

    python examples/quickstart.py                     # compares none, exact and cadm:1.1
    python examples/quickstart.py --agents 8 --facts 60 --think 0.05
"""
import argparse, os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ariadne.policies import Policy
from ariadne.stores.base import make_store


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_data(n_base, n_facts, fanout, dim, noise, seed):
    rng = np.random.default_rng(seed)
    base = unit(rng.standard_normal((n_base, dim))).astype("float32")
    facts = unit(rng.standard_normal((n_facts, dim)))
    # every fact is reported `fanout` times as a light paraphrase (small perturbation of the same vector)
    reports = [(f, unit(facts[f] + noise * rng.standard_normal(dim)).astype("float32"))
               for f in range(n_facts) for _ in range(fanout)]
    order = rng.permutation(len(reports))
    return base, [reports[i] for i in order]


def run(policy_spec, base, reports, agents, k, tau, think, max_rounds=30):
    store = make_store({"kind": "faissmem", "index": "hnsw", "metric": "cosine"})
    store.setup(base)
    pol = Policy(policy_spec, k)
    next_id = [len(base)]
    id_lock = threading.Lock()
    stored = {}                       # item id -> fact id, for counting duplicates afterwards
    calls = [0]
    queue = list(reports)
    q_lock = threading.Lock()

    def think_once(A):
        """The 'model call': decide from the held answer A whether the report is new."""
        with id_lock:
            calls[0] += 1
        time.sleep(think)
        return not A or A[0][1] > tau             # insert iff no memory within tau

    def agent():
        s = store.session()
        while True:
            with q_lock:
                if not queue:
                    return
                fact, v = queue.pop()
            A = s.search(v, k)                    # read phase, no lock
            for _ in range(max_rounds):
                insert = think_once(A)
                if pol.name == "none":            # no isolation: write what was decided
                    break
                s.lock()                          # commit lock
                F = s.search(v, k)                # validation read
                if pol.valid(A, F):
                    break                         # commit under the lock
                s.unlock()
                A = F                             # think again on the fresh answer
            else:
                continue                          # aborted after max_rounds failed validations
            if insert:
                with id_lock:
                    item = next_id[0]; next_id[0] += 1
                s.insert(item, v)
                stored[item] = fact
            if pol.name != "none":
                s.unlock()

    ts = [threading.Thread(target=agent) for _ in range(agents)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    facts = set(stored.values())
    return {"policy": policy_spec, "stored": len(stored), "facts": len(facts),
            "duplicates": len(stored) - len(facts), "model_calls": calls[0], "tasks": len(reports),
            "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", default="none,exact,cadm:1.1")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--facts", type=int, default=40)
    ap.add_argument("--fanout", type=int, default=3, help="reports per fact")
    ap.add_argument("--base", type=int, default=5000, help="initial memory size")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.3, help="duplicate threshold (metric distance)")
    ap.add_argument("--noise", type=float, default=0.01, help="paraphrase noise")
    ap.add_argument("--think", type=float, default=0.05, help="seconds per model call")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    base, reports = make_data(a.base, a.facts, a.fanout, a.dim, a.noise, a.seed)
    print(f"{a.agents} agents, {a.facts} facts x {a.fanout} reports, k={a.k}, tau={a.tau}, think={a.think}s")
    print(f"{'policy':<10} {'stored':>7} {'duplicates':>11} {'model calls':>12} {'extra calls':>12} {'wall s':>7}")
    for p in a.policies.split(","):
        r = run(p, base, reports, a.agents, a.k, a.tau, a.think)
        print(f"{r['policy']:<10} {r['stored']:>7} {r['duplicates']:>11} {r['model_calls']:>12} "
              f"{r['model_calls'] - r['tasks']:>12} {r['wall_s']:>7}")


if __name__ == "__main__":
    main()
