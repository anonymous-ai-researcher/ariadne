"""Downstream question answering on the memory that a run produced.

Question: does the memory built under a given concurrency-control protocol answer benchmark questions
as well as the memory built by a serial execution?  For every run file the final memory is rebuilt from
the workload and the commit log (initial items, plus inserted items, minus deleted items).  Each
benchmark question is embedded with the SAME embedder that produced the data set, the top-k memories
are retrieved by exact search (restricted to the question's conversation when the workload has groups),
a model answers from the retrieved memories only, and the answer is scored against the gold answer.

Invocation (from the repository root)
    python -m ariadne.qa --runs results/X/serial.pkl results/X/none.pkl results/X/cadm.pkl \\
        --dataset locomo --raw /path/locomo10.json --embedder wordllama --model gpt-4o-mini \\
        --k 10 --judge --compare --workers 8 --out results/X/qa.jsonl
    python -m ariadne.qa --runs R1.pkl R2.pkl --dataset locomo --raw /path/locomo10.json --mock --limit 200 --compare

Options
    --runs R.pkl ...        run files written by experiments/run.py; with --compare the first one is the reference
    --dataset locomo|longmemeval   --raw PATH   benchmark questions (the raw file is only parsed as JSON)
    --embedder SPEC         wordllama | st:<model> | openai:<model> (ariadne.data.embedder); default: inferred from
                            the data file name when that is unambiguous.  Only the dimension is checked: a different
                            embedder with the same dimension is not detected
    --model M               answering model; endpoint and key from OPENAI_BASE_URL / OPENAI_API_KEY; temperature 0
    --cache FILE.jsonl      answers and verdicts cached by hash of (endpoint, model, prompt)
    --k 10                  memories retrieved per question
    --limit N --seed 0      fixed random sample of the questions (the same sample for every run)
    --touched               only questions whose group receives stream items in the workload (the same questions for
                            every run and seed of a workload, whatever the protocol committed)
    --adversarial skip|abstain   LoCoMo category 5 has no gold answer: skip it (default), or expect an abstention
                            ("no information available" / "not mentioned" in the answer scores 1, else 0)
    --no-dates              do not prefix a memory with the date of its session (LoCoMo; on by default)
    --judge                 second model call per question: CORRECT / WRONG against the gold answer (cached)
    --mock                  no network: the answer is the retrieved line with the largest word overlap with the
                            question, the verdict a word-overlap rule (smoke test of the pipeline, not a result)
    --workers 4             concurrent model calls
    --compare               paired comparison of every run with the first: mean per-question F1 difference with a
                            bootstrap 95% interval
    --details FILE.jsonl    one line per (run, question): gold, answer, scores, retrieved item ids
    --tau-cos X             near-duplicate threshold for the retrieval statistics (default: the run's tau_cos)

Output: one JSON line per run (stdout and --out)
    run, name, cell, protocol, dataset, embedder, model, mock, k, questions
    memory        {items, initial, stream, deleted_initial, groups_with_stream}
    f1, em        mean token-level F1 and exact match after normalization (lower case, no punctuation, no articles)
    judge_acc     share of CORRECT verdicts (with --judge)
    by_category   {category: {n, f1, em, judge_acc}}
    retrieval     {stream_per_q, initial_per_q, stream_share, q_with_stream_item_%, dup_pairs_per_q,
                   q_with_dup_pair_%, evidence_recall, evidence_hit_%, tau}
                  dup pairs: pairs of retrieved items closer than tau.  evidence: gold evidence turns found among
                  the retrieved items, by exact text (LoCoMo; initial items only)
    llm           {calls, cache_hits, errors, prompt_tokens, completion_tokens}
and with --compare one more line per non-reference run
    compare, reference, run, n, f1_ref, f1, diff_f1, ci95, diff_em, diff_judge, ci95_judge,
    answers_changed_%, retrieved_set_changed_%,
    diff_dup_pairs_per_q, ci95_dup_pairs_per_q   near-duplicate pairs among the retrieved memories, per question,
                                                 this run minus the reference (paired bootstrap 95% interval)
    q_more_dup_pairs_%, q_fewer_dup_pairs_%      questions with more / fewer such pairs than in the reference
    diff_evidence_recall, ci95_evidence_recall   evidence turns found, this run minus the reference, divided by the
                                                 evidence turns of all questions (LoCoMo; bootstrap over questions)

LoCoMo categories are reported as <number>_<name> with the usual reading of the numbers (1 multi_hop,
2 temporal, 3 open_domain, 4 single_hop, 5 adversarial).  Scores are plain token F1 for every category,
which is not the per-category scoring of the LoCoMo evaluation script.
"""
import argparse, hashlib, json, logging, os, re, string, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .audit import load_run, workload_of
from .data import embedder as make_embedder, embed_all, load_prepared
from .metrics import chord_from_cos
from .ext.mock_openai import short_answer, judge_rule

LOCOMO_CATS = {"1": "multi_hop", "2": "temporal", "3": "open_domain", "4": "single_hop", "5": "adversarial"}
ABSTAIN = ("no information available", "not mentioned")

QA_SYSTEM = """You answer questions about the people in a long conversation, using a shared memory.
Use only the MEMORIES given in the message; do not use outside knowledge.
Answer with a short phrase (a name, a date, a place, a few words), not a full sentence.
If the memories do not contain the answer, reply exactly: No information available."""

JUDGE_SYSTEM = """You grade an answer to a question against the gold answer.
The answer is CORRECT if it conveys the same information as the gold answer; wording, extra detail and date format do not matter.
Otherwise it is WRONG. Reply with one word: CORRECT or WRONG."""


# ----------------------------------------------------------------------------- benchmark loaders
def load_locomo_qa(path, adversarial="skip"):
    """LoCoMo locomo10.json.  Returns (questions, dates): a question is {qid, group, question, answer, category,
    evidence, adversarial}; the group is the index of the conversation, as in ariadne.data.load_locomo, and
    `evidence` lists the evidence turns as texts in the form of the initial memory items ("speaker: text").
    `dates` maps the item label (conversation * 100 + session) to the date string of that session."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    qs, dates = [], {}
    for ci, s in enumerate(d):
        conv = s.get("conversation", {}); turn = {}
        for key, val in conv.items():
            m = re.fullmatch(r"session_(\d+)(_date_time)?", key)
            if m and m.group(2):
                dates[ci * 100 + int(m.group(1))] = str(val)
            elif m and isinstance(val, list):
                for t in val:
                    if isinstance(t, dict) and "dia_id" in t:
                        turn[str(t["dia_id"]).strip()] = f'{t.get("speaker", "")}: {t.get("text", "")}'
        for qi, q in enumerate(s.get("qa", [])):
            cat = str(q.get("category")); adv = cat == "5" or q.get("answer") is None
            if adv and adversarial == "skip":
                continue
            ev = [turn[e] for e in re.findall(r"D\d+:\d+", str(q.get("evidence", ""))) if e in turn]
            qs.append({"qid": f"{ci}:{qi}", "group": ci, "question": str(q["question"]),
                       "answer": "No information available" if adv else str(q["answer"]),
                       "category": f'{cat}_{LOCOMO_CATS.get(cat, "other")}', "evidence": ev, "adversarial": adv})
    return qs, dates


def load_longmemeval_qa(path, adversarial="skip"):
    """LongMemEval JSON: a list of objects with question, answer, question_type and question_id.  The group of
    the question at position gi is gi, as in ariadne.data.load_longmemeval (one haystack per question).
    Question ids ending in "_abs" are taken to be abstention questions and are handled like LoCoMo's
    adversarial category."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    qs = []
    for gi, q in enumerate(d):
        qid = str(q.get("question_id", gi)); adv = qid.endswith("_abs")
        if adv and adversarial == "skip":
            continue
        qs.append({"qid": qid, "group": gi, "question": str(q["question"]),
                   "answer": "No information available" if adv else str(q.get("answer", "")),
                   "category": str(q.get("question_type", "all")), "evidence": [], "adversarial": adv})
    return qs, {}


QA_LOADERS = {"locomo": load_locomo_qa, "longmemeval": load_longmemeval_qa}


# ----------------------------------------------------------------------------- scoring
def normalize(s):
    s = "".join(ch for ch in str(s).lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    return 0.0 if same == 0 else 2.0 * same / (len(p) + len(g))


def score(pred, q):
    """(f1, em) of an answer."""
    if q["adversarial"]:
        ok = float(any(a in str(pred).lower() for a in ABSTAIN))
        return ok, ok
    return f1_score(pred, q["answer"]), float(normalize(pred) == normalize(q["answer"]))


def bootstrap_ci(d, B=2000, seed=0):
    d = np.asarray(d, dtype=float); n = len(d)
    if n == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(B)])
    return [round(float(x), 4) for x in np.percentile(means, [2.5, 97.5])]


# ----------------------------------------------------------------------------- memory of a run
def final_memory(run, wl):
    """Item ids that are in the store after the whole commit log."""
    live = np.zeros(wl.nb + wl.n, dtype=bool); live[:wl.nb] = True
    for _, op, item, _ in sorted(run["clog"]):
        live[item] = op == "i"
    return np.where(live)[0]


def item_groups(wl, cfg):
    """Group (conversation) of every item id, recovered from the prepared data set by matching vectors
    (the workload does not keep the row numbers).  None if the data set has no groups."""
    path = (cfg.get("workload") or {}).get("data")
    if not path or not os.path.exists(path):
        return None
    X, _, group, _, _ = load_prepared(path)
    if not (group >= 0).any():
        return None
    rows = {}
    for r in range(len(X)):
        rows.setdefault(X[r].tobytes(), []).append(r)
    out = np.full(wl.nb + wl.n, -1, dtype=np.int64); used = {}
    for i, v in enumerate(np.concatenate([wl.base, wl.X])):
        key = v.tobytes(); cand = rows.get(key)
        if cand:                                            # identical vectors: assign the rows in file order
            c = used.get(key, 0); used[key] = c + 1
            out[i] = group[cand[min(c, len(cand) - 1)]]
    if (out < 0).any():
        print(f"qa: warning: {int((out < 0).sum())} items could not be matched to a group; no question retrieves them", file=sys.stderr)
    return out


def _dist(Q, V, metric):
    if metric == "cosine":
        return np.sqrt(np.maximum(2.0 - 2.0 * (Q @ V.T), 0.0))
    return np.sqrt(np.maximum((Q ** 2).sum(1)[:, None] + (V ** 2).sum(1)[None, :] - 2.0 * (Q @ V.T), 0.0))


def retrieve(Qv, qgroup, ids, V, groups, k, metric):
    """Exact top-k item ids per question, restricted to the question's group when there are groups."""
    out = [None] * len(Qv); g_of = groups[ids] if groups is not None else None
    for g in (sorted(set(qgroup)) if g_of is not None else [None]):
        qi = [i for i in range(len(Qv)) if g is None or qgroup[i] == g]
        cand = np.arange(len(ids)) if g is None else np.where(g_of == g)[0]
        if len(cand) == 0:
            for i in qi:
                out[i] = []
            continue
        D = _dist(Qv[qi], V[cand], metric)
        top = np.argsort(D, axis=1, kind="stable")[:, :k]
        for a, i in enumerate(qi):
            out[i] = [int(ids[cand[c]]) for c in top[a]]
    return out


# ----------------------------------------------------------------------------- model calls
class LLM:
    """Chat calls with a JSONL cache keyed by hash of (endpoint, model, system prompt, user prompt)."""

    def __init__(self, model, cache_path, mock=False, max_tokens=64, retries=3):
        self.model, self.mock, self.max_tokens, self.retries = model, mock, max_tokens, retries
        self.endpoint = "mock" if mock else (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        self.cache_path, self.cache, self.mu = cache_path, {}, threading.Lock()
        self.n = {"calls": 0, "cache_hits": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.client = None
        if mock:
            return
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                try:
                    o = json.loads(line); self.cache[o["h"]] = o
                except Exception:
                    pass
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        from openai import OpenAI
        for name in ("httpx", "httpx2", "openai"):          # one log line per request otherwise, once anything configures logging
            logging.getLogger(name).setLevel(logging.WARNING)
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "none"))

    def ask(self, system, user, mock_answer=None):
        with self.mu:
            self.n["calls"] += 1
        if self.mock:
            return mock_answer()
        h = hashlib.sha1("\x00".join([self.endpoint, self.model, system, user]).encode()).hexdigest()
        with self.mu:
            o = self.cache.get(h)
            if o is not None:
                self.n["cache_hits"] += 1
                return o["ans"]
        err = None
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=self.max_tokens,
                                                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                break
            except Exception as e:
                err = e; time.sleep(min(2.0 ** attempt, 8.0))
        else:
            with self.mu:
                self.n["errors"] += 1
            print(f"qa: model call failed: {err!r}", file=sys.stderr)
            return ""
        u = getattr(r, "usage", None)
        o = {"h": h, "ans": (r.choices[0].message.content or "").strip(), "model": self.model, "endpoint": self.endpoint,
             "pt": u.prompt_tokens if u else 0, "ct": u.completion_tokens if u else 0}
        with self.mu:
            self.cache[h] = o; self.n["prompt_tokens"] += o["pt"]; self.n["completion_tokens"] += o["ct"]
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        return o["ans"]


def qa_prompt(question, lines):
    return "MEMORIES:\n" + "\n".join(f"[{r}] {t}" for r, t in enumerate(lines, 1)) + f"\n\nQUESTION: {question}\nANSWER:"


def judge_prompt(q, pred):
    return f'QUESTION: {q["question"]}\nGOLD ANSWER: {q["answer"]}\nMODEL ANSWER: {pred}\nVERDICT:'


# ----------------------------------------------------------------------------- one run
def evaluate(path, run, wl, qs, Qv, llm, a, dates):
    cfg = run["cfg"]; metric = (cfg.get("store") or {}).get("metric", "cosine")
    if wl.texts is None:
        raise SystemExit(f"{path}: the workload has no texts; question answering needs a data set prepared with a text loader")
    if Qv.shape[1] != wl.base.shape[1]:
        raise SystemExit(f"{path}: question vectors have dimension {Qv.shape[1]}, the data set {wl.base.shape[1]}: wrong --embedder")
    tau = chord_from_cos(a.tau_cos if a.tau_cos else float(cfg.get("tau_cos", 0.10))) if metric == "cosine" else float(cfg.get("tau", 0.45))
    nb = wl.nb; ids = final_memory(run, wl); groups = item_groups(wl, cfg)
    allv = np.concatenate([wl.base, wl.X]); V = allv[ids]
    use_dates = bool(dates) and not a.no_dates and wl.labels is not None and groups is not None \
        and bool((wl.labels // 100 == groups).all())
    got = retrieve(Qv, [q["group"] for q in qs], ids, V, groups, a.k, metric)
    n0 = dict(llm.n)

    def line(i):
        d = dates.get(int(wl.labels[i])) if use_dates else None
        return f"[{d}] {wl.texts[i]}" if d else wl.texts[i]

    def one(x):
        q = qs[x]; lines = [line(i) for i in got[x]]
        pred = llm.ask(QA_SYSTEM, qa_prompt(q["question"], lines), lambda: short_answer(q["question"], lines))
        verdict = None
        if a.judge:
            v = llm.ask(JUDGE_SYSTEM, judge_prompt(q, pred), lambda: judge_rule(q["answer"], pred)).upper()
            verdict = float("CORRECT" in v and "INCORRECT" not in v and "WRONG" not in v)
        return pred, verdict
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        res = list(ex.map(one, range(len(qs))))
    preds = [p for p, _ in res]
    sc = np.array([score(p, q) for p, q in zip(preds, qs)], dtype=float).reshape(-1, 2)
    jv = np.array([v for _, v in res], dtype=float) if a.judge else None
    n_stream = np.array([sum(i >= nb for i in g) for g in got]); n_init = np.array([sum(i < nb for i in g) for g in got])
    dup = np.zeros(len(qs), dtype=int); ev_found = ev_total = ev_hit = ev_q = 0
    evf = np.zeros(len(qs), dtype=int); evt = np.zeros(len(qs), dtype=int)       # per question: evidence found / evidence turns
    for x, g in enumerate(got):
        if len(g) > 1:
            D = _dist(allv[g], allv[g], metric)
            dup[x] = int((D[np.triu_indices(len(g), 1)] < tau).sum())
        ev = set(qs[x]["evidence"])
        if ev:
            found = len(ev & {wl.texts[i] for i in g})
            ev_found += found; ev_total += len(ev); ev_hit += found > 0; ev_q += 1
            evf[x] = found; evt[x] = len(ev)
    cats = sorted({q["category"] for q in qs}); cat = np.array([q["category"] for q in qs])
    rnd = lambda v: round(float(v), 4)
    by = {c: {"n": int((cat == c).sum()), "f1": rnd(sc[cat == c, 0].mean()), "em": rnd(sc[cat == c, 1].mean()),
              "judge_acc": rnd(jv[cat == c].mean()) if jv is not None else None} for c in cats}
    n = max(len(qs), 1); tot = max(int(n_stream.sum() + n_init.sum()), 1)
    out = {"run": path, "name": cfg.get("name"), "cell": cfg.get("cell"), "protocol": cfg.get("protocol"), "dataset": a.dataset,
           "embedder": a.embedder, "model": None if a.mock else a.model, "mock": bool(a.mock), "k": a.k, "questions": len(qs),
           "memory": {"items": int(len(ids)), "initial": int((ids < nb).sum()), "stream": int((ids >= nb).sum()),
                      "deleted_initial": int(nb - (ids < nb).sum()),
                      "groups_with_stream": sorted({int(g) for g in groups[ids[ids >= nb]]}) if groups is not None else None},
           "f1": rnd(sc[:, 0].mean()) if len(qs) else None, "em": rnd(sc[:, 1].mean()) if len(qs) else None,
           "judge_acc": rnd(jv.mean()) if jv is not None and len(qs) else None, "by_category": by,
           "retrieval": {"stream_per_q": rnd(n_stream.sum() / n), "initial_per_q": rnd(n_init.sum() / n),
                         "stream_share": rnd(n_stream.sum() / tot), "q_with_stream_item_%": rnd(100 * (n_stream > 0).sum() / n),
                         "dup_pairs_per_q": rnd(dup.sum() / n), "q_with_dup_pair_%": rnd(100 * (dup > 0).sum() / n),
                         "evidence_recall": rnd(ev_found / ev_total) if ev_total else None,
                         "evidence_hit_%": rnd(100 * ev_hit / ev_q) if ev_q else None, "tau": rnd(tau), "dates_in_prompt": use_dates},
           "llm": {key: llm.n[key] - n0[key] for key in llm.n}}
    return out, {"f1": sc[:, 0], "em": sc[:, 1], "judge": jv, "pred": preds, "got": got, "dup": dup, "ev_found": evf, "ev_total": evt}


def bootstrap_ratio_ci(num, den, B=2000, seed=0):
    """95% bootstrap interval (over questions) of sum(num) / sum(den); [None, None] if sum(den) is 0."""
    num = np.asarray(num, dtype=float); den = np.asarray(den, dtype=float); n = len(num)
    if n == 0 or den.sum() <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, size=n); s = den[i].sum()
        if s > 0:
            vals.append(num[i].sum() / s)
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])] if vals else [None, None]


def compare(ref_path, ref, path, cur, seed=0):
    d = cur["f1"] - ref["f1"]; n = len(d)
    out = {"compare": True, "reference": ref_path, "run": path, "n": n,
           "f1_ref": round(float(ref["f1"].mean()), 4), "f1": round(float(cur["f1"].mean()), 4),
           "diff_f1": round(float(d.mean()), 4), "ci95": bootstrap_ci(d, seed=seed),
           "diff_em": round(float((cur["em"] - ref["em"]).mean()), 4), "diff_judge": None, "ci95_judge": None,
           "answers_changed_%": round(100 * sum(x != y for x, y in zip(cur["pred"], ref["pred"])) / max(n, 1), 2),
           "retrieved_set_changed_%": round(100 * sum(set(x) != set(y) for x, y in zip(cur["got"], ref["got"])) / max(n, 1), 2)}
    if cur["judge"] is not None and ref["judge"] is not None:
        dj = cur["judge"] - ref["judge"]
        out["diff_judge"] = round(float(dj.mean()), 4); out["ci95_judge"] = bootstrap_ci(dj, seed=seed)
    # per-question retrieval statistics, paired (added fields; the fields above keep their meaning):
    # near-duplicate pairs among the k retrieved memories (this run minus the reference), and the evidence turns found
    if cur.get("dup") is not None and ref.get("dup") is not None:
        dd = np.asarray(cur["dup"], dtype=float) - np.asarray(ref["dup"], dtype=float)
        out["diff_dup_pairs_per_q"] = round(float(dd.mean()), 4); out["ci95_dup_pairs_per_q"] = bootstrap_ci(dd, seed=seed)
        out["q_more_dup_pairs_%"] = round(100 * float((dd > 0).sum()) / max(n, 1), 2)
        out["q_fewer_dup_pairs_%"] = round(100 * float((dd < 0).sum()) / max(n, 1), 2)
    if cur.get("ev_found") is not None and ref.get("ev_found") is not None and int(np.sum(ref["ev_total"])) > 0:
        de = np.asarray(cur["ev_found"], dtype=float) - np.asarray(ref["ev_found"], dtype=float)
        out["diff_evidence_recall"] = round(float(de.sum() / np.sum(ref["ev_total"])), 4)
        out["ci95_evidence_recall"] = bootstrap_ratio_ci(de, ref["ev_total"], seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser(description="Question answering on the memory a run produced")
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--dataset", choices=list(QA_LOADERS), default="locomo")
    ap.add_argument("--raw", required=True); ap.add_argument("--embedder"); ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--cache", default="results/qa_cache.jsonl"); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--touched", action="store_true")
    ap.add_argument("--adversarial", choices=["skip", "abstain"], default="skip"); ap.add_argument("--no-dates", action="store_true")
    ap.add_argument("--judge", action="store_true"); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--compare", action="store_true"); ap.add_argument("--details"); ap.add_argument("--out")
    ap.add_argument("--tau-cos", type=float)
    a = ap.parse_args()
    runs = [(p, load_run(p)) for p in a.runs]
    if not a.embedder:                                      # the file name only identifies embedders without ':' or '/'
        data = str((runs[0][1]["cfg"].get("workload") or {}).get("data", ""))
        m = re.search(r"__([^/]*)\.npz$", data)
        if not m or m.group(1) != "wordllama":
            raise SystemExit(f"cannot infer the embedder of '{data}': pass --embedder (the one used by `ariadne.data prep`)")
        a.embedder = m.group(1)
    qs, dates = QA_LOADERS[a.dataset](a.raw, a.adversarial)
    wl0 = workload_of(runs[0][1]["cfg"])
    if a.touched:
        g0 = item_groups(wl0, runs[0][1]["cfg"])
        if g0 is not None:
            touched = {int(g) for g in g0[wl0.nb:] if g >= 0}         # groups of the stream items of the workload
            qs = [q for q in qs if q["group"] in touched]
    if a.limit and a.limit < len(qs):
        keep = np.sort(np.random.default_rng(a.seed).permutation(len(qs))[:a.limit])
        qs = [qs[i] for i in keep]
    if not qs:
        raise SystemExit("no questions left after filtering")
    Qv = embed_all(make_embedder(a.embedder), [q["question"] for q in qs])
    llm = LLM(a.model, a.cache, a.mock, a.max_tokens)

    def emit(line):
        print(json.dumps(line, default=float), flush=True)
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=float) + "\n")
    ref = None
    for path, run in runs:
        wl = workload_of(run["cfg"])
        if a.compare and run["cfg"].get("workload") != runs[0][1]["cfg"].get("workload"):
            print(f"qa: warning: {path} has a different workload than the reference; the paired comparison is not meaningful", file=sys.stderr)
        line, per = evaluate(path, run, wl, qs, Qv, llm, a, dates)
        emit(line)
        if a.details:
            with open(a.details, "a", encoding="utf-8") as f:
                for x, q in enumerate(qs):
                    f.write(json.dumps({"run": path, "qid": q["qid"], "category": q["category"], "question": q["question"],
                                        "gold": q["answer"], "answer": per["pred"][x], "f1": float(per["f1"][x]), "em": float(per["em"][x]),
                                        "judge": None if per["judge"] is None else float(per["judge"][x]), "retrieved": per["got"][x]},
                                       ensure_ascii=False) + "\n")
        if ref is None:
            ref = (path, per)
        elif a.compare:
            emit(compare(ref[0], ref[1], path, per, a.seed))


if __name__ == "__main__":
    main()
