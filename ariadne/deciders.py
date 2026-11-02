"""Deciders: what an agent does with a retrieved answer, and how long it thinks.

A decider maps (task, answers) to one memory operation, in the vocabulary used by agent-memory systems:

    ADD            store the task's item
    NOOP           the item is already covered
    UPDATE target  replace memory `target` by the task's item
    DELETE target  remove memory `target`

YAML `decider:` section:

  kind: threshold        rule on the nearest distance (no model call; think time is simulated)
    add     -> NOOP if the nearest memory is within tau, else ADD
    upsert  -> UPDATE the nearest memory if it is within tau, else ADD
    delete  -> DELETE the nearest memory if it is within tau, else NOOP
    read    -> NOOP (read-only task)

  kind: llm              one chat-completion call per decision (OpenAI-compatible endpoint)
    model: gpt-4o-mini   base_url / api key from OPENAI_BASE_URL / OPENAI_API_KEY unless given
    temperature: 0, max_tokens: 200
    cache: results/llm_cache.jsonl     answers are cached by (endpoint, model, temperature, prompt), so replays cost nothing
    mock: false          true = no network; a deterministic content-dependent stand-in (for smoke tests)
    show_distance: false include similarity scores in the prompt

  think: "lognormal:0.2,0.3"   simulated think time per decision
         none | fixed:<s> | exp:<mean> | lognormal:<mean>,<sigma> | uniform:<lo>,<hi> | pareto:<mean>,<shape>
         | trace:<file with one latency per line> | tokens:<base>,<per_retrieved_item>
  think_scale: 1.0       multiply every think time (e.g. 0.1 to run a 2 s model at 0.2 s)

With kind: llm and think: none the think time is the real latency of the call.
"""
import hashlib, json, math, os, re, threading, time
from dataclasses import dataclass
import numpy as np


@dataclass
class Decision:
    action: str
    target: int = -1

    def key(self):
        return (self.action, int(self.target))


class Think:
    def __init__(self, spec, scale=1.0, seed=1):
        self.spec = str(spec or "none"); self.scale = float(scale)
        kind, _, arg = self.spec.partition(":")
        a = [x for x in arg.split(",") if x != ""]
        rng = np.random.default_rng(seed); n = 1 << 18
        self.kind = kind; self.per_item = 0.0
        if kind == "none":
            self.lat = None
        elif kind == "fixed":
            self.lat = np.full(1, float(a[0]))
        elif kind == "exp":
            self.lat = rng.exponential(float(a[0]), size=n)
        elif kind == "lognormal":
            m, s = float(a[0]), float(a[1]) if len(a) > 1 else 0.3
            self.lat = rng.lognormal(mean=math.log(m) - s * s / 2, sigma=s, size=n)
        elif kind == "uniform":
            self.lat = rng.uniform(float(a[0]), float(a[1]), size=n)
        elif kind == "pareto":                              # heavy tail with the given mean; shape > 1
            m, sh = float(a[0]), float(a[1]) if len(a) > 1 else 2.5
            self.lat = (rng.pareto(sh, size=n) + 1.0) * m * (sh - 1.0) / sh
        elif kind == "trace":
            self.lat = np.array([float(x) for x in open(arg).read().split()])
            self.lat = self.lat[rng.permutation(len(self.lat))]
        elif kind == "tokens":
            self.lat = np.full(1, float(a[0])); self.per_item = float(a[1]) if len(a) > 1 else 0.0
        else:
            raise ValueError(f"unknown think model '{spec}'")
        self.i = 0; self.mu = threading.Lock()

    def sample(self, n_items=0):
        if self.lat is None:
            return 0.0
        with self.mu:
            d = float(self.lat[self.i % len(self.lat)]); self.i += 1
        return self.scale * (d + self.per_item * n_items)

    def sleep(self, n_items=0):
        d = self.sample(n_items)
        if d > 0:
            time.sleep(d)
        return d


def threshold_rule(op, A, tau):
    near = bool(A) and A[0][1] < tau
    if op == "add":
        return Decision("NOOP") if near else Decision("ADD")
    if op == "upsert":
        return Decision("UPDATE", A[0][0]) if near else Decision("ADD")
    if op == "delete":
        return Decision("DELETE", A[0][0]) if near else Decision("NOOP")
    return Decision("NOOP")


class ThresholdDecider:
    kind = "threshold"

    def __init__(self, cfg, wl, tau, seed=1):
        self.wl, self.tau = wl, tau
        self.think = Think(cfg.get("think", "lognormal:0.2,0.3"), cfg.get("think_scale", 1.0), seed)
        self.calls = 0; self.mu = threading.Lock()

    def decide(self, j, As, quick=False):
        """quick=True recomputes the decision without thinking (only sound for rules that need no model)."""
        slept = 0.0
        if not quick:
            slept = self.think.sleep(sum(len(a) for a in As))
            with self.mu:
                self.calls += 1
        return threshold_rule(self.wl.op[j], As[0], self.tau), slept

    def stats(self):
        return {"decider": "threshold", "calls": self.calls}


SYSTEM = """You maintain the shared long-term memory of a team of assistants.
You are given a NEW FACT and the EXISTING MEMORIES that are most similar to it.
Choose exactly one operation:
  ADD     the fact is new information and must be stored
  UPDATE  the fact refines or supersedes one existing memory (give its number)
  DELETE  the fact contradicts one existing memory, which must be removed (give its number)
  NOOP    the fact is already covered by the existing memories
Answer with one JSON object and nothing else: {"op": "ADD|UPDATE|DELETE|NOOP", "target": <number or null>}"""


class LLMDecider:
    kind = "llm"

    def __init__(self, cfg, wl, tau, seed=1):
        if wl.texts is None and not cfg.get("mock"):
            raise SystemExit("the LLM decider needs a workload with texts (prepare the data set with a text loader)")
        self.cfg, self.wl, self.tau = cfg, wl, tau
        self.model = cfg.get("model", "gpt-4o-mini"); self.mock = bool(cfg.get("mock", False))
        self.think = Think(cfg.get("think", "none"), cfg.get("think_scale", 1.0), seed)
        self.cache_path = cfg.get("cache", "results/llm_cache.jsonl"); self.cache = {}
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                try:
                    o = json.loads(line); self.cache[o["h"]] = o
                except Exception:
                    pass
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        self.mu = threading.Lock()
        self.calls = self.hits = self.ptok = self.ctok = self.ptok_new = self.ctok_new = self.bad = 0
        self.lat = []
        # A call answered from the cache behaves like the original call: it takes the recorded latency (so the window
        # between read and commit, and with it the validation failures, do not depend on which cells ran before) and
        # its tokens are counted.  The replay audit switches the sleep off.
        self.replay_latency = bool(cfg.get("replay_latency", True))
        self.client = None
        if not self.mock:
            from openai import OpenAI
            self.client = OpenAI(base_url=cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
                                 api_key=cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "none"))

    def text(self, item):
        if self.wl.texts is None:
            return f"<item {item}>"
        return self.wl.texts[item]

    def prompt(self, j, As):
        A = As[0]
        lines = [f"NEW FACT: {self.text(self.wl.nb + j)}", "", "EXISTING MEMORIES:"]
        for r, (i, d) in enumerate(A, 1):
            sc = f" (similarity {1 - 0.5 * d * d:.3f})" if self.cfg.get("show_distance") else ""
            lines.append(f"[{r}] {self.text(i)}{sc}")
        if len(As) > 1:
            lines += ["", "RELATED CONTEXT:"]
            seen = {i for i, _ in A}
            for B in As[1:]:
                for i, _ in B:
                    if i not in seen:
                        seen.add(i); lines.append(f"- {self.text(i)}")
        return "\n".join(lines)

    def _mock(self, j, As, user):
        """Deterministic stand-in whose answer depends on the retrieved identifiers, like a model's would."""
        A = As[0]; h = int(hashlib.sha1(user.encode()).hexdigest()[:8], 16)
        if A and A[0][1] < self.tau:
            return {"op": "UPDATE" if h % 4 == 0 else "NOOP", "target": 1}
        if A and h % 10 == 0:
            return {"op": "UPDATE", "target": 1 + h % min(3, len(A))}
        return {"op": "ADD", "target": None}

    def _call(self, user):
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.model, temperature=float(self.cfg.get("temperature", 0)),
                                                max_tokens=int(self.cfg.get("max_tokens", 200)),
                                                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        txt = r.choices[0].message.content or ""
        u = getattr(r, "usage", None)
        return txt, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0), time.time() - t0

    def decide(self, j, As, quick=False):
        if quick or self.wl.op[j] == "read":
            return threshold_rule(self.wl.op[j], As[0], self.tau), 0.0
        user = self.prompt(j, As)
        src = "mock" if self.mock else str(self.cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "default")
        h = hashlib.sha1("\x00".join([src, self.model, str(self.cfg.get("temperature", 0)), SYSTEM, user]).encode()).hexdigest()
        slept = self.think.sleep(sum(len(a) for a in As))
        with self.mu:
            self.calls += 1
            o = self.cache.get(h)
            if o is not None:
                self.hits += 1
        if o is None:
            if self.mock:
                ans, pt, ct, lat = self._mock(j, As, user), len(user) // 4, 12, 0.0
            else:
                txt, pt, ct, lat = self._call(user)
                m = re.search(r"\{.*\}", txt, re.S)
                try:
                    ans = json.loads(m.group(0)) if m else None
                except Exception:
                    ans = None
                if not isinstance(ans, dict) or str(ans.get("op", "")).upper() not in ("ADD", "UPDATE", "DELETE", "NOOP"):
                    ans = {"op": "BAD", "raw": txt[:200]}
            o = {"h": h, "ans": ans, "pt": pt, "ct": ct, "lat": lat}
            with self.mu:
                self.ptok += pt; self.ctok += ct; self.ptok_new += pt; self.ctok_new += ct; self.lat.append(lat)
                if ans.get("op") != "BAD":                  # an unparsable reply is not cached, so it is asked again next time
                    self.cache[h] = o
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(o) + "\n")
            slept += lat
        else:
            lat = float(o.get("lat") or 0.0)
            if self.replay_latency and lat > 0:
                time.sleep(lat); slept += lat
            with self.mu:
                self.ptok += int(o.get("pt") or 0); self.ctok += int(o.get("ct") or 0); self.lat.append(lat)
        ans = o["ans"]; op = str(ans.get("op", "BAD")).upper(); A = As[0]
        if op == "BAD":
            with self.mu:
                self.bad += 1
            return threshold_rule("add", A, self.tau), slept
        if op in ("UPDATE", "DELETE"):
            try:
                t = int(ans.get("target")) - 1
            except Exception:
                t = 0
            if not A:
                return Decision("ADD" if op == "UPDATE" else "NOOP"), slept
            return Decision(op, A[min(max(t, 0), len(A) - 1)][0]), slept
        return Decision(op), slept

    def stats(self):
        return {"decider": "llm", "model": self.model, "mock": self.mock, "calls": self.calls, "cache_hits": self.hits,
                # tokens of every call of the run; a call answered from the cache counts with the tokens of the original call
                "prompt_tokens": self.ptok, "completion_tokens": self.ctok, "tokens_include_cached": True,
                "prompt_tokens_uncached": self.ptok_new, "completion_tokens_uncached": self.ctok_new,
                "cached_latency_replayed": self.replay_latency, "unparsable": self.bad,
                "llm_latency_mean_s": float(np.mean(self.lat)) if self.lat else None}


def make_decider(cfg, wl, tau, seed=1):
    cfg = dict(cfg or {})
    kind = cfg.get("kind", "threshold")
    if kind == "threshold":
        return ThresholdDecider(cfg, wl, tau, seed)
    if kind == "llm":
        return LLMDecider(cfg, wl, tau, seed)
    raise ValueError(f"unknown decider kind '{kind}'")
