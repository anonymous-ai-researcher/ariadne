#!/usr/bin/env python3
"""Writes EXPERIMENTS.md (the experiment master plan) from the experiment files, so that the catalogue cannot drift
away from what is runnable.

    python experiments/_plan.py            # -> EXPERIMENTS.md
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import yaml
import run as R

QUESTION = {
    "A1": "Does the check-then-insert duplicate race occur on every vector store, and does the store's own isolation level remove it?",
    "A2": "Which workload parameters set the size of the race (arrival gap, think time, fan-out, threshold, k, hot spots, store size, concurrency)?",
    "A3": "Is the race an artifact of synthetic paraphrases? Real duplicate-question clusters and real text streams.",
    "A4": "Which anomalies occur besides duplicate inserts (forked updates, decisions on deleted items, stale read-only answers), and which protocol removes which?",
    "A5": "What does PostgreSQL SERIALIZABLE cost on transactions with similarity reads (aborts, recomputation, tail latency)?",
    "A6": "Is SERIALIZABLE sound for pgvector index reads? Predicate-lock granularity, deterministic reproduction with a debugger, stress dose response, patch.",
    "A7": "Does GiST, which implements predicate locks, protect similarity-style searches (missed conflicts after pruning at an internal page)?",
    "A8": "Is the abort storm a property of SERIALIZABLE or of the similarity predicate? B-tree key control.",
    "A9": "Does the race reach users through the public API of an agent-memory framework (Mem0)?",
    "A10": "Do the findings of A6 and A7 hold on PostgreSQL 14 to 18 (Docker version matrix)?",
    "A11": "How far do application-level remedies go and where do they fail (re-read, locked re-read, write-then-check, object-level validation)?",
    "B1": "Does the safety theorem hold on every execution (the auditor rebuilds the exact state at every commit)?",
    "B2": "Separation: is the final store (tau/c)-separated, and is the bound tight?",
    "B3": "Transactions with several similarity reads: how does validation cost grow, and does safety hold?",
    "B4": "Update- and delete-heavy workloads: role and cost of the existence check.",
    "B5a": "Does the auditor catch protocols that look correct but are not (no lock, group lock, REPEATABLE READ critical section)?",
    "B5b": "Fault injection into the logs of correct runs: detection rate of the auditor.",
    "B6": "How many recomputations are caused by the index rather than by the data (physical versus logical change)?",
    "B7": "Does the characterization theorem agree with every observed single-insertion case?",
    "C1": "Extra model calls beyond one per task for every validation predicate (concurrent runs).",
    "C2": "Trade-off frontier: calls saved and deviating decisions admitted by every predicate (one instrumented run, all predicates replayed offline).",
    "C3": "A language model makes ADD/UPDATE/DELETE/NOOP decisions: calls, decision deviations and token cost.",
    "C4": "Is memory built under tolerant validation as good as memory built under exact validation for downstream question answering?",
    "C5": "End-to-end latency and throughput of every protocol as the number of agents grows.",
    "C6": "Interaction of index quality and tolerance: read budget, validation budget, exact validation (certified variant).",
    "C7": "How to choose c: fine sweep, knee, comparison with the index's own approximation ratio.",
    "C8": "Robustness to the think-time distribution.",
    "C8b": "Open-loop arrivals (Poisson, bursts).",
    "C9": "Scalability in store size, run length and number of agents.",
    "C10": "Optimistic validation against pessimistic predicate locks on the read ball (with and without tolerance).",
    "C11": "Starvation: tail of rounds per task for slow thinkers and hot topics.",
    "C12": "Sensitivity to the retrieval depth k and the duplicate threshold tau.",
    "C13": "Effect of the share of read-only tasks; do read-only tasks need validation?",
    "D1": "Is the closed form of the conflict law correct? Monte Carlo grid and (c, D_s) heat-map grid.",
    "D2a": "Does the estimation procedure recover a planted law (synthetic streams with planted D_s and sibling share)?",
    "D2b": "Analysis of D2a: planted against recovered, background rate, background-rank uniformity, resampling calibration.",
    "D3a": "What do concurrent writes around a read look like on real text streams (instrumented runs, all data sets and concurrency levels)?",
    "D3b": "Analysis of D3a: fits, lag kernel, transfers without refitting, stratified tests, prediction from the static stream kernel.",
    "D4a": "Are the conflicts beyond the i.i.d. prediction caused by the temporal locality of the stream (controls that remove structure step by step)?",
    "D4b": "Analysis of D4a.",
    "D5": "Does the closed form fail when the data mixes regions of different intrinsic dimension?",
    "D5b": "Analysis of D5: fits and stratification by local intrinsic dimension.",
    "D6": "Is the sibling parameter a property of the text stream or of the embedding model? k from 5 to 100.",
    "D6b": "Analysis of D6.",
    "E1": "Which part of the c-admissibility predicate matters (per rank, radius only, nearest only, distance sum, identity tests)?",
    "E2": "Cost of the single commit lock, and what a narrower lock scope does.",
    "E3": "What to do after a failed validation (re-think with F, re-read outside the lock, give up) and the cap on rounds.",
    "E4": "How much search budget the validation read should get.",
    "E5": "Which isolation level the read phase and the critical section should use on PostgreSQL (a REPEATABLE READ critical section is unsound).",
    "E6": "Can the tolerance be chosen per operation type (strict for destructive operations, loose for appends)?",
    "F1": "Does the protocol behave the same on every store (portability)?",
    "F2": "Do the main results hold for other embedding models and data sets?",
    "F3": "How large is the run-to-run variance (main cells repeated ten times)?",
    "F4": "Are the results an artifact of Python threads (one process per agent)?",
    "F5": "Is a long run stable?",
    "F6": "Do safety and liveness hold when a client fails inside the critical section?",
    "F7": "Metric self-check: L2 and cosine on unit vectors must give the same results.",
    "G1": "Where does time go inside the critical section (lock wait, validation read, predicate, write)?",
    "G2": "Overhead of validation without any contention.",
    "G3": "Store-free microbenchmarks: predicate cost against k, offline audit cost against log size.",
}
GROUPS = [("A", "Group A: anomalies on real systems"), ("B", "Group B: semantics and protocol correctness"),
          ("C", "Group C: savings and cost"), ("D", "Group D: conflict theory"), ("E", "Group E: protocol ablations"),
          ("F", "Group F: robustness"), ("G", "Group G: microbenchmarks")]


def think_mean(cfg):
    d = cfg.get("decider") or {}
    if d.get("kind") == "llm":
        return 1.0
    spec = str(d.get("think", "lognormal:0.2,0.3")); kind, _, arg = spec.partition(":")
    try:
        return float(arg.split(",")[0]) * float(d.get("think_scale", 1.0))
    except Exception:
        return 0.2


def est_seconds(cfg):
    """Rough wall-clock of one cell: think time only, plus a fixed overhead for set-up and audits."""
    w = cfg.get("workload") or {}; kind = w.get("kind", "stream"); th = think_mean(cfg); P = cfg.get("protocol") or {}
    if kind in ("fanout", "clusters"):
        fo = w.get("fanout", [[1, 0.5], [2, 0.3], [4, 0.2]]); mf = sum(f * p for f, p in fo) / sum(p for _, p in fo) if kind == "fanout" else 2.5
        n = w.get("n_events", 600) * mf; clients = float(w.get("conc", 16)); th = float(w.get("think", th))
    else:
        if kind == "synthetic":
            n = w.get("n_stream", 5000); teams = 1
        else:
            loc = "locomo" in str(w.get("data", "")); n = w.get("stream") or (2541 if loc else 6000); teams = 10 if (loc and w.get("teams", "group") == "group") else 1
        t = str(w.get("teams", ""))
        if t.startswith("rr:") or t.startswith("block:"):
            teams = int(t.split(":")[1])
        clients = teams * float(w.get("agents", 4))
        arr = w.get("arrival")
        if isinstance(arr, dict):
            return n / float(arr["rate"]) + 30
    if P.get("kind") == "lock_long":
        clients = 1 if P.get("lock", "global") != "group" else min(clients, 10)
    mult = 6.0 if (P.get("kind") == "ssi" and P.get("variant", "long") == "long") else 1.4
    return n * th * mult / max(clients, 1) + 30


def hours(x):
    return f"{x / 3600:.1f} h" if x >= 3600 else f"{x / 60:.0f} min"


PREFACE = r"""# Ariadne experiment catalogue

This file is generated by `python experiments/_plan.py` from `experiments/*.yaml`, so the catalogue always matches
the runnable configurations. Every experiment runs through one entry point:

    python experiments/run.py experiments/<ID>.yaml            # full scale
    python experiments/run.py experiments/<ID>.yaml --smoke    # small version (minutes)
    python experiments/run.py --all --smoke                    # smoke run of every experiment
    python -m ariadne.plots.all --results results --out figures   # every figure

The smoke version runs a small grid in minutes and checks an installation.

## 1. Protocols and baselines

| Class | Protocol in the framework | Configuration | Practice it stands for |
|---|---|---|---|
| No or weak isolation | read, think, write | `protocol: {kind: none}`; on pgvector also `isolation: REPEATABLE READ, txn: true` | default of vector stores; READ COMMITTED, snapshot isolation |
| Application-level remedies | re-read before writing (no lock), locked re-read with re-decision by rule, write then check and delete own row | `validate + redecide + lock: none`, `validate + redecide`, `compensate (tiebreak: id/any)` | common deduplication patches; for threshold deduplication the locked re-read is a simple, strong baseline |
| Pessimistic: global lock | lock held through the whole think phase | `kind: lock_long` | serializes all agents |
| Pessimistic: predicate lock | predicate lock on the read ball, strict and tolerant, with deadlock detection | `kind: ball, c: 1.0 / 1.1` | precision locks, granular locks and phantom protection of R-trees and GiST, applied to similarity reads |
| Serializable snapshot isolation | PostgreSQL SERIALIZABLE, long transactions (immediate retry or backoff) and short transactions | `kind: ssi, variant: long/short, retry: backoff` | PostgreSQL SSI with pgvector (stock and patched) |
| Optimistic: re-execution | ordered identifiers identical | `policy: exact` | Kung-Robinson validation |
| Optimistic: identity tolerance | same set, overlap, top-1, top-j | `set`, `overlap:0.9`, `top1`, `topj:3` | recall-style tolerance |
| Optimistic: single-rank distance tests | k-th distance only, nearest only, distance sum | `radius:c`, `rank1:c`, `sumdist:c` | ablations of the predicate |
| Optimistic: c-admissible (this work) | per-rank distance test; certified variant with an exact validation read | `cadm:c`, `cert:c` | Algorithm AV |
| Object-level validation | checks only that the objects read still exist | `policy: objects` | object version checks; what agent-level concurrency control on shared objects can see (not phantoms from inserts) |
| Oracle lower bound | with the decision function known, recompute only if the decision changes | `oracle:dup/tag/link/top3/set` | lower bound for any validation predicate |
| Memory frameworks | the race through a public API | `ariadne.ext.mem0_race` | Mem0; adapter notes for LangMem, Letta, Zep, A-MEM and MemoryOS in `ariadne/ext/README.md` |

## 2. Data sets and embedding models

| Data set | Role |
|---|---|
| LoCoMo | conversational memory stream: dialogue turns are the initial memory, extracted facts the write stream, one team per conversation |
| AG News | news stream in natural file order; neighboring items are often reports of the same event |
| LongMemEval | haystack sessions of a long-term memory benchmark |
| Quora Question Pairs | real clusters of duplicate questions (the real counterpart of the fan-out workload), also usable as a stream |
| MS MARCO passages | initial memory at the scale of 10^6 items |
| spaCy word vectors | point cloud for the synthetic paraphrase fan-out workload |
| ANN-Benchmarks HDF5 | vector-only control |
| Synthetic manifold | planted law: exchangeable background, siblings distributed with planted dimension D_s |
| Generic JSONL | any data with text / label / group / role fields |

The embedding model is chosen by `ariadne.data prep --embedder`: `wordllama` (static, 256 dimensions),
`st:<sentence-transformers model>`, `openai:<model>`. Experiment files select prepared data through the environment
variable `EMB`. After changing the model, choose `tau_cos` again with `python -m ariadne.data stats`.

## 3. Main metrics

| Metric | Meaning | Source |
|---|---|---|
| `extra_calls_per_100` | model calls per 100 tasks beyond one call per task | summary |
| `dup_per_100_events` | extra stored rows per 100 facts in the fan-out workload | summary |
| `dup_written_per_100`, `redundant_items` | duplicate writes judged against the exact state | audit run / sep |
| `div_committed_vs_exact_%` | divergence between the committed answer and the exact answer under five decision proxies | audit run |
| `rho_committed`, `rho_fresh` | approximation ratio of the committed and the validation answer against the exact state; `>c` is the contract violation rate | audit run |
| `safety_theorem.violations` | violations of rho(committed) <= c * rho(fresh); must be 0 | audit run |
| `separation_proposition.violations` | items of the final store closer than tau/c | audit sep |
| `different_action_per_100` | tasks whose action differs from a serial exact execution in commit order | audit serial |
| `aborts_per_task`, `ball.deadlocks`, `lock_wait_mean_ms`, `cs_*_ms`, `lock_held_fraction` | cost of the concurrency control itself | summary |
| `characterization` | cases in which the characterization theorem agrees with the observation | audit law |
| conflict-law quantities | background share, D_s, lag kernel, prediction error | `ariadne.law` |

## 4. Scale

At full scale there are {n_cells} engine cells and {n_cmds} script commands. The durations in the tables below are rough
wall-clock estimates (tasks x think time / concurrency, one machine, sequential, without index builds and real model
latency), about {n_hours} hours in total. Experiments can run on different machines in parallel (one PostgreSQL
instance per machine, or separate run databases with `--set store.rundb=...`). Cells that already have results are
skipped, so an interrupted run can simply be restarted.

## 5. Notes on the framework

- Protocols without a commit lock (none, re-read without lock, compensate, ssi) have an approximate commit-log order:
  use final-state metrics for them (dup, sep, serial), not per-commit rho.
- The predicate-lock manager of the `ball` protocol runs inside one client process; it shows the behavior of the
  design, not the performance of a server-side implementation.
- The group lock is unsound on a shared store by construction and appears only as an ablation.
- With two seeds the figures show the range of the two runs instead of a 95% interval.
- For transactions with several reads the auditor computes decision proxies on the first read; rho is the maximum
  over all reads.

## 6. Experiments
"""


def main():
    files = sorted(glob.glob(os.path.join(HERE, "*.yaml")), key=lambda f: (os.path.basename(f)[0], int(re.match(r"[A-G](\d+)", os.path.basename(f)).group(1)), os.path.basename(f)))
    exps = [yaml.safe_load(R.expand_env(open(f).read())) for f in files]
    info = []; n_cells = n_cmds = 0
    for e in exps:
        eid = e["name"].split("_")[0]
        if e.get("kind") == "script":
            nc = len(e.get("commands", [])); n_cmds += nc
            info.append((eid, e, "script", f"{nc} command{'s' if nc != 1 else ''}", "", ""))
        else:
            full = R.cells(e, False); sm = R.cells(e, True); n_cells += len(full)
            est = sum(est_seconds(c) for c in full)
            info.append((eid, e, "engine", f"{len(full)}", f"{len(sm)}", hours(est)))
    tot_h = sum(sum(est_seconds(c) for c in R.cells(e, False)) for e in exps if e.get("kind") != "script") / 3600
    out = [PREFACE.replace("{n_exp}", str(len(exps))).replace("{n_cells}", str(n_cells)).replace("{n_cmds}", str(n_cmds)).replace("{n_hours}", str(int(round(tot_h, -1))))]
    for g, title in GROUPS:
        out.append(f"\n### {title}\n\n| ID | Question | Type | Full cells | Smoke cells | Rough duration |\n|---|---|---|---|---|---|")
        for eid, e, typ, nfull, nsm, est in info:
            if not eid.startswith(g):
                continue
            out.append(f"| `{e['name']}` | {QUESTION.get(eid, '')} | {typ} | {nfull} | {nsm} | {est} |")
    out.append("\n## 7. Figures and experiments\n\nSystem figures are listed in `ariadne/plots/README.md`, theory figures in "
               "`ariadne/plots/README_theory.md`; both list the experiment each figure reads and what it shows. Tables are written by "
               "`python -m ariadne.plots.tables` to `figures/tables/*.tex`.\n")
    out.append("\n## 8. Experiment details\n\nThe text below is the `doc` field of each experiment file.\n")
    for eid, e, typ, nfull, nsm, est in info:
        out.append(f"\n### {e['name']}\n\n{QUESTION.get(eid, '')}\n\n```\n{e.get('doc', '').rstrip()}\n```\n")
        if typ == "engine":
            grid = e.get("grid") or {}
            axes = []
            for k, v in grid.items():
                vals = [x["_name"] if isinstance(x, dict) and "_name" in x else x for x in (v if isinstance(v, list) else [v])]
                axes.append(f"`{k}` = {', '.join(str(x) for x in vals)}")
            cases = [c.get("_name", "") for c in (e.get("cases") or [])]
            out.append((f"- Workload cases: {', '.join(cases)}\n" if cases else "") + "".join(f"- Grid: {a}\n" for a in axes) +
                       f"- Repeats: {e.get('repeat', 1)} seed(s); audits: {', '.join(a if isinstance(a, str) else a['name'] for a in (e.get('audits') or [])) or 'none'}\n"
                       f"- Run: `python experiments/run.py experiments/{e['name']}.yaml`\n")
        else:
            out.append("- Commands:\n" + "".join(f"  - `{c}`\n" for c in e.get("commands", [])) + f"- Run: `python experiments/run.py experiments/{e['name']}.yaml`\n")
    open(os.path.join(ROOT, "EXPERIMENTS.md"), "w", encoding="utf-8").write("\n".join(out))
    print(f"EXPERIMENTS.md written: {len(exps)} experiments, {n_cells} cells, {n_cmds} commands")


if __name__ == "__main__":
    main()
