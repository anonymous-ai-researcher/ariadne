"""Does the duplicate race appear through a real agent-memory framework?  (Mem0)

Question: several agents report the same new fact at about the same time.  Each one retrieves similar
memories, sees none, and adds the fact, so a shared memory ends up with duplicates.  Does this happen
through the public API of Mem0 (`Memory.add` with one shared `user_id`), and does the simple remedy,
one global lock around every add, remove it and at what cost in wall time?

Workload: E events (facts), each reported by f agents (fan-out distribution) as light paraphrases
(template rewording, no model needed).  Events arrive as a Poisson process whose rate keeps about
`--conc` adds in flight when one add takes `--think` seconds; the copies of one event arrive within
`--jitter` * think seconds (the same parametrization as the `fanout` workload of ariadne.workloads).
Without --think, one think time is the mean duration of an add measured in the serial mode.

Modes (each on a fresh, empty memory, with the same reports in the same arrival order)
    serial      one thread, one add after the other: the reference
    concurrent  the framework as it is: threads call add at the scheduled arrival times
    lock        like concurrent, with one global lock around each add

Invocation (from the repository root)
    python -m ariadne.ext.mem0_race --mock --latency 0.2 --events 100 --out results/T_ext/mem0_race.jsonl
    OPENAI_API_KEY=... python -m ariadne.ext.mem0_race --base-url https://api.openai.com/v1 --model gpt-4o-mini \\
        --embed-model text-embedding-3-small --events 200 --think 1.5 --no-marker --out results/mem0_race.jsonl

Options
    --mock                    start ariadne.ext.mock_openai in this process as the model and embedding endpoint
      --latency 0.2 --latency-jitter 0.3      its chat latency (mean seconds, lognormal sigma)
    --base-url URL --model M --embed-model M  a real OpenAI-compatible endpoint (default base URL: OPENAI_BASE_URL);
                              the key is read from OPENAI_API_KEY
    --embed-dims N            embedding dimension; default: asked from the endpoint with one probe call.  If given,
                              Mem0 also sends it as `dimensions` with every embedding request
    --store qdrant-local|qdrant-server|chroma|faiss   vector store behind Mem0 (default qdrant-local: embedded
                              Qdrant in a local directory, no server);  qdrant-server needs --qdrant-url HOST:PORT
                              or --qdrant-bin PATH (starts a private server on a free port and stops it at the end)
    --handles shared|per-thread   one Memory object shared by all threads, or one per thread on the same store
                              (per-thread needs a server store: an embedded store allows one client); the objects
                              are created before the clock starts
    --events 100 --fanout 1:0.5,2:0.3,4:0.2 --conc 8 --think S --jitter 1.0 --threads 32 --seed 1
                              --think defaults to the mean add duration of the serial mode when that mode runs
                              first, else to the mock latency (--mock) or 1.0
    --facts auto|synthetic|locomo|FILE.jsonl   event texts: built-in synthetic sentences, the stream part of
                              data/locomo.texts.jsonl, or a JSONL file with a "text" field (auto: locomo if present)
    --distinct 0.5            drop a fact that shares this share of its content words with an earlier one, so that
                              events are distinct by construction (0 disables)
    --no-marker               do not append a marker "(ref EV00012)" to every report; stored memories are then
                              matched to events by embedding similarity only (--match-sim 0.5)
    --modes serial concurrent lock      --user-id shared_team      --workdir DIR (default: a temporary directory)

Output: one JSON line per mode (stdout and --out)
    mode, framework, framework_version, store, handles, events, tasks, mean_fan, conc, think, jitter, threads, seed
                              think: seconds used as one think time for the arrival schedule (null in serial mode,
                              which ignores the schedule)
    stored                    memories in the store at the end (listed through the framework)
    stored_per_event_mean, dup_rows, dup_per_100_events, events_with_dup_%, lost_events, dup_per_100_events_by_fan
                              from matching every stored memory back to its event (marker token, else nearest
                              event by embedding): dup_rows = sum over events of (stored - 1), lost = events with
                              no stored memory
    unmatched_memories, matched_by_marker, matched_by_embedding
    events_with_multiple_adders   events for which more than one add call reported an ADD (from the return values
                              of add, independent of the text matching)
    add_calls, add_errors, first_error, llm_calls (calls made by the framework), llm_calls_per_add
    wall_s, add_lat_mean_s, add_lat_p95_s, max_in_flight
                              add latency includes the wait for the lock.  max_in_flight far above --conc means that
                              this machine is saturated (with --mock the model server runs inside this process):
                              raise --latency or lower --conc
    endpoint {base_url, model, embed_model, mock, latency}, server (mock only: its request counters for the mode)
    nlp_extras                whether spaCy is importable (Mem0 2.x links entities only then)

With a real model the stored text is the model's rewording: compare duplicates against the serial mode,
which shows how many memories per event the model writes without any concurrency.
Mem0 telemetry is switched off (MEM0_TELEMETRY=False) and MEM0_DIR points into the work directory.

Used with mem0ai 2.2.1, against the mock server (offline tests) and against GPT-4o mini through the OpenAI API (experiment
A9_memory_frameworks, the numbers of the paper).  The fallbacks for the get_all signature and return shapes of mem0ai 0.1 / 1.x
follow the earlier API and were not exercised.
"""
import argparse, copy, glob, importlib.util, json, os, re, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .mock_openai import overlap

MARK = re.compile(r"\bEV(\d{5})\b")
WRAPS = ["{f}", "Note: {f}", "FYI, {f}", "Update from the team: {f}", "For the record, {f}", "Just learned this: {f}",
         "Reminder: {f}", "Heads up, {f}", "{f} Please remember this.", "As reported earlier today: {f}"]
NAMES = "Avery Bianca Carlos Dalia Emre Farah Gustav Hana Ilya Jonas Keiko Lamar Mirela Nikhil Oksana Pablo Quinn Rosa Soren Tariq".split()
VERBS = "repaired painted borrowed sold photographed inherited restored misplaced donated packed weighed polished shipped rented sketched".split()
ADJS = "copper wooden striped antique foldable woolen ceramic rusty velvet glass bamboo leather marble plastic silver".split()
NOUNS = "kettle bicycle telescope typewriter canoe lantern accordion suitcase compass easel hammock sewing-machine globe umbrella ladder".split()
PLACES = "harbor market mountain cabin riverside library rooftop garden railway depot village bakery desert outpost island ferry".split()
TIMES = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
PARTS = "morning noon afternoon evening night".split()


# ----------------------------------------------------------------------------- workload
def synthetic_facts(n, rng):
    """Slot-filled sentences; two of them rarely share more than one or two content words."""
    out, seen = [], set()
    while len(out) < n:
        p = rng.integers(0, len(PLACES) // 2) * 2
        t = (f"{rng.choice(NAMES)} {rng.choice(VERBS)} the {rng.choice(ADJS)} {rng.choice(NOUNS)} at the {PLACES[p]} {PLACES[p + 1]} "
             f"on {rng.choice(TIMES)} {rng.choice(PARTS)}.")
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def load_facts(spec, n, rng, distinct):
    if spec == "auto":
        spec = "locomo" if glob.glob("data/locomo__*.npz") and os.path.exists("data/locomo.texts.jsonl") else "synthetic"
    if spec == "synthetic":
        return synthetic_facts(n, rng), "synthetic"
    if spec == "locomo":                                    # stream part (role 1) of the prepared LoCoMo data set
        role = np.load(sorted(glob.glob("data/locomo__*.npz"))[0])["role"]
        texts = [json.loads(l)["text"] for l in open("data/locomo.texts.jsonl", encoding="utf-8")]
        pool = [texts[i] for i in np.where(role == 1)[0]]
    else:
        pool = [json.loads(l)["text"] for l in open(spec, encoding="utf-8") if l.strip()]
    out = []
    for t in pool:
        t = " ".join(str(t).split())
        if t and not (distinct > 0 and any(overlap(t, u) >= distinct for u in out)):
            out.append(t)
            if len(out) == n:
                break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} distinct facts in '{spec}' (asked for {n}); lower --events or --distinct")
    return out, spec


def report_text(fact, e, c, marker):
    """Copy c of event e: the fact, a marker token if wanted, and one of the wrappers (different for every copy)."""
    f = fact.strip()
    if marker:
        f = f.rstrip(".") + f" (ref EV{e:05d})."
    return WRAPS[(7 * e + 3 * c) % len(WRAPS)].format(f=f)


def build_tasks(a, rng):
    facts, source = load_facts(a.facts, a.events, rng, a.distinct)
    fo = [x.split(":") for x in a.fanout.split(",")]
    fs = np.array([int(f) for f, _ in fo]); ps = np.array([float(p) for _, p in fo]); ps = ps / ps.sum()
    if fs.max() > len(WRAPS):
        raise SystemExit(f"fan-out above {len(WRAPS)} needs more paraphrase templates")
    fan_of = rng.choice(fs, size=a.events, p=ps)
    rate = a.conc / float(fan_of.mean())                    # in units of one think time; run_mode scales by --think
    t_ev = np.cumsum(rng.exponential(1.0 / rate, size=a.events))
    tasks = [(float(t_ev[e] + rng.uniform(0, a.jitter)), e, c, report_text(facts[e], e, c, not a.no_marker))
             for e in range(a.events) for c in range(int(fan_of[e]))]
    tasks.sort(key=lambda t: t[0])
    t0 = tasks[0][0]
    return facts, source, fan_of, [(due - t0, e, c, text) for due, e, c, text in tasks]


# ----------------------------------------------------------------------------- framework adapter
class Mem0Adapter:
    """What the test needs from a framework: a shared-memory handle, an add call, a list of what is stored."""
    name = "mem0"

    def __init__(self, a, tag, workdir, endpoint, dims, qdrant_addr, n_handles=1):
        os.environ["MEM0_TELEMETRY"] = "False"              # both are read when mem0 is imported
        os.environ.setdefault("MEM0_DIR", os.path.join(workdir, "mem0_home"))
        from importlib.metadata import version
        from mem0 import Memory
        self.Memory, self.version = Memory, version("mem0ai")
        self.user, self.per_thread, self.cap = a.user_id, a.handles == "per-thread", 100000
        self.workdir, self.tag = workdir, tag
        llm = {"model": endpoint["model"], "openai_base_url": endpoint["base_url"], "api_key": endpoint["api_key"], "temperature": 0.0}
        emb = {"model": endpoint["embed_model"], "openai_base_url": endpoint["base_url"], "api_key": endpoint["api_key"]}
        if a.embed_dims:
            emb["embedding_dims"] = a.embed_dims
        if a.store == "qdrant-local":
            vs = {"provider": "qdrant", "config": {"collection_name": tag, "path": os.path.join(workdir, "qdrant_" + tag), "embedding_model_dims": dims}}
        elif a.store == "qdrant-server":
            host, _, port = qdrant_addr.rpartition(":")
            vs = {"provider": "qdrant", "config": {"collection_name": tag, "host": host, "port": int(port), "embedding_model_dims": dims}}
        elif a.store == "chroma":
            vs = {"provider": "chroma", "config": {"collection_name": tag, "path": os.path.join(workdir, "chroma_" + tag)}}
        else:
            vs = {"provider": "faiss", "config": {"collection_name": tag, "path": os.path.join(workdir, "faiss_" + tag),
                                                  "embedding_model_dims": dims, "distance_strategy": "cosine"}}
        self.cfg = {"llm": {"provider": "openai", "config": llm}, "embedder": {"provider": "openai", "config": emb}, "vector_store": vs}
        self.mu, self.local, self.handles, self.calls = threading.Lock(), threading.local(), [], 0
        self.main = self._new()                             # also creates the collection before any thread starts
        self.pool = [self._new() for _ in range(n_handles)] if self.per_thread else []      # built before the clock starts

    def _new(self):
        with self.mu:
            n = len(self.handles); self.handles.append(None)
        cfg = copy.deepcopy(self.cfg); cfg["history_db_path"] = os.path.join(self.workdir, f"history_{self.tag}_{n}.db")
        m = self.Memory.from_config(cfg)
        inner = m.llm.generate_response

        def counted(*args, **kw):                           # every model call of the framework goes through here
            with self.mu:
                self.calls += 1
            return inner(*args, **kw)
        m.llm.generate_response = counted
        self.handles[n] = m
        return m

    def handle(self):
        if not self.per_thread:
            return self.main
        if not hasattr(self.local, "m"):
            with self.mu:
                m = self.pool.pop() if self.pool else None
            self.local.m = m or self._new()
        return self.local.m

    @staticmethod
    def _items(r):
        return r.get("results", []) if isinstance(r, dict) else (r or [])

    def add(self, text, agent):
        """Returns the number of memories that the call reports as added."""
        r = self.handle().add(text, user_id=self.user)
        return sum(1 for it in self._items(r) if isinstance(it, dict) and it.get("event", "ADD") == "ADD")

    def list(self):
        try:
            r = self.main.get_all(filters={"user_id": self.user}, top_k=self.cap)        # mem0ai 2.x
        except (TypeError, ValueError):
            r = self.main.get_all(user_id=self.user, limit=self.cap)                     # mem0ai 0.1 / 1.x
        return [str(it.get("memory", "")) for it in self._items(r) if isinstance(it, dict)]

    def close(self):
        for m in self.handles:
            try:
                if m is not None and hasattr(m, "close"):
                    m.close()
                if m is not None and hasattr(getattr(m.vector_store, "client", None), "close"):
                    m.vector_store.client.close()
            except Exception:
                pass


ADAPTERS = {"mem0": Mem0Adapter}


# ----------------------------------------------------------------------------- helpers
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_qdrant(binary, workdir):
    """Private Qdrant server on a free port, storage inside the work directory."""
    d = os.path.join(workdir, "qdrant_server"); os.makedirs(d, exist_ok=True)
    port = free_port()
    env = dict(os.environ, QDRANT__SERVICE__HOST="127.0.0.1", QDRANT__SERVICE__HTTP_PORT=str(port),
               QDRANT__SERVICE__GRPC_PORT=str(free_port()), QDRANT__STORAGE__STORAGE_PATH=os.path.join(d, "storage"),
               QDRANT__STORAGE__SNAPSHOTS_PATH=os.path.join(d, "snapshots"), QDRANT__TELEMETRY_DISABLED="true")
    proc = subprocess.Popen([binary], cwd=d, env=env, stdout=open(os.path.join(d, "qdrant.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(150):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=1).read()
            return proc, f"127.0.0.1:{port}"
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    proc.kill()
    raise SystemExit(f"the Qdrant server did not start; see {d}/qdrant.log")


def make_embed(endpoint):
    from openai import OpenAI
    cl = OpenAI(base_url=endpoint["base_url"], api_key=endpoint["api_key"])

    def f(texts):
        out = []
        for i in range(0, len(texts), 128):
            r = cl.embeddings.create(model=endpoint["embed_model"], input=texts[i:i + 128])
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        X = np.asarray(out, dtype="float32")
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return f


def match_events(texts, facts, embed, min_sim):
    """Event of every stored memory: the marker token if it survived, else the nearest event by embedding."""
    ev = np.full(len(texts), -1, dtype=int); todo = []
    for i, t in enumerate(texts):
        m = MARK.search(t)
        if m and int(m.group(1)) < len(facts):
            ev[i] = int(m.group(1))
        else:
            todo.append(i)
    by_marker = len(texts) - len(todo)
    if todo:
        S = embed([texts[i] for i in todo]) @ embed(facts).T
        for x, i in enumerate(todo):
            j = int(np.argmax(S[x]))
            if S[x, j] >= min_sim:
                ev[i] = j
    return ev, by_marker, int((ev >= 0).sum()) - by_marker


# ----------------------------------------------------------------------------- one mode
def run_mode(a, mode, adapter, tasks, facts, fan_of, embed, think):
    n = len(tasks); res = [None] * n
    lock, mu, fl = threading.Lock(), threading.Lock(), {"now": 0, "max": 0}

    def call(x):
        with mu:
            fl["now"] += 1; fl["max"] = max(fl["max"], fl["now"])
        try:
            return adapter.add(tasks[x][3], tasks[x][2]), None
        except Exception as e:
            return 0, repr(e)[:300]
        finally:
            with mu:
                fl["now"] -= 1

    def one(x):
        t0 = time.time()
        if mode == "lock":
            with lock:
                added, err = call(x)
        else:
            added, err = call(x)
        res[x] = (added, time.time() - t0, err)
    c0 = adapter.calls; t_start = time.time()
    if mode == "serial":
        for x in range(n):
            one(x)
    else:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            futs = []
            for x in range(n):
                d = t_start + tasks[x][0] * think - time.time()
                if d > 0:
                    time.sleep(d)
                futs.append(ex.submit(one, x))
            for f in futs:
                f.result()
    wall = time.time() - t_start
    texts = adapter.list()
    ev, by_marker, by_emb = match_events(texts, facts, embed, a.match_sim)
    E = len(facts); cnt = np.bincount(ev[ev >= 0], minlength=E)
    adders = np.zeros(E, dtype=int)
    for x, (added, _, _) in enumerate(res):
        adders[tasks[x][1]] += added > 0
    lat = np.array([r[1] for r in res]); errs = [r[2] for r in res if r[2]]
    dup = np.maximum(cnt - 1, 0); calls = adapter.calls - c0
    return {"mode": mode, "framework": adapter.name, "framework_version": adapter.version, "store": a.store, "handles": a.handles,
            "events": E, "tasks": n, "mean_fan": round(float(fan_of.mean()), 3), "conc": a.conc, "think": None if mode == "serial" else round(think, 4), "jitter": a.jitter,
            "threads": 1 if mode == "serial" else a.threads, "seed": a.seed,
            "stored": len(texts), "stored_per_event_mean": round(float(cnt.mean()), 4), "dup_rows": int(dup.sum()),
            "dup_per_100_events": round(100 * float(dup.sum()) / E, 2), "events_with_dup_%": round(100 * float((cnt > 1).mean()), 2),
            "lost_events": int((cnt == 0).sum()),
            "dup_per_100_events_by_fan": {int(f): round(100 * float(dup[fan_of == f].sum()) / max(int((fan_of == f).sum()), 1), 2)
                                          for f in sorted(set(fan_of.tolist()))},
            "unmatched_memories": int((ev < 0).sum()), "matched_by_marker": by_marker, "matched_by_embedding": by_emb,
            "events_with_multiple_adders": int((adders > 1).sum()),
            "add_calls": n, "add_errors": len(errs), "first_error": errs[0] if errs else None,
            "llm_calls": calls, "llm_calls_per_add": round(calls / n, 3),
            "wall_s": round(wall, 2), "add_lat_mean_s": round(float(lat.mean()), 4), "add_lat_p95_s": round(float(np.percentile(lat, 95)), 4),
            "max_in_flight": fl["max"]}


def main():
    ap = argparse.ArgumentParser(description="Duplicate race through an agent-memory framework (Mem0)")
    ap.add_argument("--framework", choices=list(ADAPTERS), default="mem0")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--latency", type=float, default=0.2)
    ap.add_argument("--latency-jitter", type=float, default=0.3)
    ap.add_argument("--base-url"); ap.add_argument("--model", default="gpt-4o-mini"); ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--embed-dims", type=int)
    ap.add_argument("--store", choices=["qdrant-local", "qdrant-server", "chroma", "faiss"], default="qdrant-local")
    ap.add_argument("--qdrant-url"); ap.add_argument("--qdrant-bin")
    ap.add_argument("--handles", choices=["shared", "per-thread"], default="shared")
    ap.add_argument("--events", type=int, default=100); ap.add_argument("--fanout", default="1:0.5,2:0.3,4:0.2")
    ap.add_argument("--conc", type=float, default=8.0); ap.add_argument("--think", type=float); ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=32); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--facts", default="auto"); ap.add_argument("--distinct", type=float, default=0.5)
    ap.add_argument("--no-marker", action="store_true"); ap.add_argument("--match-sim", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", choices=["serial", "concurrent", "lock"], default=["serial", "concurrent", "lock"])
    ap.add_argument("--user-id", default="shared_team"); ap.add_argument("--workdir"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.handles == "per-thread" and a.store != "qdrant-server":
        raise SystemExit("--handles per-thread needs --store qdrant-server (an embedded store allows one client)")
    if a.store == "qdrant-server" and not (a.qdrant_url or a.qdrant_bin):
        raise SystemExit("--store qdrant-server needs --qdrant-url HOST:PORT or --qdrant-bin PATH")
    workdir = a.workdir or tempfile.mkdtemp(prefix="mem0_race_"); os.makedirs(workdir, exist_ok=True)
    srv = qproc = None
    try:
        if a.mock:
            from . import mock_openai
            srv = mock_openai.start(port=0, latency=a.latency, jitter=a.latency_jitter, dim=a.embed_dims or 256)
            endpoint = {"base_url": srv.url, "api_key": "mock", "model": "mock-model", "embed_model": "mock-embedding"}
        else:
            base = a.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("set OPENAI_API_KEY (any string for an endpoint without authentication), or use --mock")
            endpoint = {"base_url": base, "api_key": os.environ["OPENAI_API_KEY"], "model": a.model, "embed_model": a.embed_model}
        embed = make_embed(endpoint)
        dims = a.embed_dims or int(embed(["dimension probe"]).shape[1])
        qaddr = a.qdrant_url
        if a.store == "qdrant-server" and not qaddr:
            qproc, qaddr = start_qdrant(a.qdrant_bin, workdir)
        rng = np.random.default_rng(a.seed)
        facts, source, fan_of, tasks = build_tasks(a, rng)
        think = a.think or (a.latency if a.mock else 1.0)
        for mode in a.modes:
            tag = f"race_{mode}_{a.seed}_{os.getpid()}_{int(time.time())}"
            if mode != "serial" and a.store != "qdrant-server":
                print(f"mem0_race: note: the embedded store '{a.store}' may fail under concurrent writers; see add_errors, "
                      "or use --store qdrant-server", file=sys.stderr)
            adapter = ADAPTERS[a.framework](a, tag, workdir, endpoint, dims, qaddr, 1 if mode == "serial" else a.threads)
            if srv:
                srv.reset_stats()
            try:
                line = run_mode(a, mode, adapter, tasks, facts, fan_of, embed, think)
            finally:
                adapter.close()
            if mode == "serial" and not a.think:            # calibrate: one think time = one add without contention
                think = line["add_lat_mean_s"]
            line["facts"] = source; line["marker"] = not a.no_marker
            line["endpoint"] = {"base_url": endpoint["base_url"], "model": endpoint["model"], "embed_model": endpoint["embed_model"],
                                "mock": bool(a.mock), "latency": a.latency if a.mock else None}
            line["server"] = {k: v for k, v in srv.stats().items() if k in ("requests", "chat_kinds", "max_in_flight")} if srv else None
            line["nlp_extras"] = importlib.util.find_spec("spacy") is not None
            print(json.dumps(line), flush=True)
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
    finally:
        if qproc:
            qproc.terminate()
            try:
                qproc.wait(timeout=10)
            except Exception:
                qproc.kill()
        if srv:
            srv.shutdown()
        if not a.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
