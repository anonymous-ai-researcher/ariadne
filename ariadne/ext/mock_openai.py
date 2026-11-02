"""Mock OpenAI-compatible server for offline tests (standard library only).

Purpose: stand in for a model endpoint so that the LLM decider (ariadne.deciders), the question
answering pipeline (ariadne.qa) and the framework race test (ariadne.ext.mem0_race) can be exercised
without network access.  Replies are rule-based and deterministic; the artificial latency plays the
role of the model's think time, which is what opens the window for concurrent writers.
Nothing measured against this server says anything about the quality of a real model.

Invocation
    python -m ariadne.ext.mock_openai --port 8765 --latency 0.05 [--jitter 0.3] [--dim 256] [--dup 0.6]
    from ariadne.ext import mock_openai
    srv = mock_openai.start(port=0, latency=0.05)      # daemon thread; port 0 picks a free port
    srv.url                                            # http://127.0.0.1:<port>/v1  (use as OPENAI_BASE_URL)
    srv.stats(); srv.reset_stats(); srv.shutdown()

Endpoints (the POST and model paths are also accepted without the /v1 prefix)
    POST /v1/embeddings          hashed bag-of-words vectors, unit norm, `--dim` components (the request field
                                 `dimensions` overrides it); texts that share words are close.  Honours
                                 encoding_format float | base64.
    POST /v1/chat/completions    one rule-based reply, chosen from the prompt (field `kind` in /stats):
        decision      prompt of ariadne.deciders.LLMDecider -> {"op": "NOOP", "target": r} if listed memory r shares
                      at least `--dup` of its content words with the new fact, else {"op": "ADD", "target": null}
        mem0_extract  Mem0 additive extraction (mem0ai 2.x: sections "## Existing Memories" / "## New Messages")
                      -> {"memory": [{"id", "text", "attributed_to"}]}: one memory per new message, skipped when an
                      existing (or recently extracted) memory already shares `--dup` of its words
        mem0_facts    Mem0 fact extraction (mem0ai 0.1/1.x: "Input:\\n...") -> {"facts": [...]}
        mem0_update   Mem0 memory update (mem0ai 0.1/1.x: "smart memory manager")
                      -> {"memory": [{"id", "text", "event": "ADD" | "NONE"}]}
        qa            prompt of ariadne.qa (MEMORIES: / QUESTION:) -> the memory line with the largest word overlap
        judge         prompt of ariadne.qa (GOLD ANSWER: / MODEL ANSWER:) -> CORRECT or WRONG
        other         "OK."
      `response_format` json_object / json_schema: a reply that is not a JSON object is wrapped as {"answer": ...}.
      `tools`: a tool call is returned only if `tool_choice` forces one ("required" or a named function); its
      arguments are the JSON reply.  `stream` is not supported (HTTP 400).
      Latency: `--latency` seconds per chat call; with `--jitter s` it is lognormal with that mean and sigma s,
      derived from a hash of the prompt (same prompt, same latency).  `--embed-latency` for embedding calls.
    GET  /v1/models              one model entry
    GET  /stats                  {"requests": {chat, embeddings, models}, "chat_kinds": {...}, "prompt_tokens",
                                  "completion_tokens", "embedded_texts", "errors", "in_flight", "max_in_flight",
                                  "uptime_s"};  POST /stats/reset zeroes the counters

Token counts in `usage` are rough (characters / 4).
"""
import argparse, ast, base64, hashlib, json, math, re, struct, threading, time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from statistics import NormalDist

STOP = frozenset("""a an the and or of to in on at for with by is are was were be been it its this that these those as from
has have had he she they them his her their i you we my your our me do does did not no so but if then than there here about
into over after before up out just very also what when where who whom which why how""".split())


# ----------------------------------------------------------------------------- text helpers (also used by ariadne.qa --mock)
def stem(w):
    """Crude suffix stripping, enough for 'attends' / 'attended' / 'attending' to meet."""
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def words(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


def content(text):
    ws = words(text)
    return {w for w in ws if w not in STOP} or set(ws)


def overlap(a, b):
    """Share of the smaller content-word set that also occurs in the other text (0..1)."""
    A, B = content(a), content(b)
    return len(A & B) / max(1, min(len(A), len(B)))


def best_line(question, lines):
    """Index of the line that shares the most content words with the question (first one on ties), or -1."""
    q = content(question); best, arg = 0, -1
    for i, line in enumerate(lines):
        s = len(q & content(line))
        if s > best:
            best, arg = s, i
    return arg


def short_answer(question, lines, max_words=30):
    """The mock 'reader': copy the most relevant line."""
    i = best_line(question, lines)
    return "No information available" if i < 0 else " ".join(lines[i].split()[:max_words])


def judge_rule(gold, answer):
    """The mock 'judge': CORRECT if at least half of the gold content words occur in the answer."""
    g = content(gold)
    return "CORRECT" if len(g & content(answer)) >= 0.5 * max(1, len(g)) else "WRONG"


@lru_cache(maxsize=1 << 16)
def _slots(word, dim):
    d = hashlib.blake2b(word.encode(), digest_size=8).digest()
    return ((int.from_bytes(d[0:3], "big") % dim, 1.0 if d[3] & 1 else -1.0),
            (int.from_bytes(d[4:7], "big") % dim, 1.0 if d[7] & 1 else -1.0))


def embed(text, dim=256):
    """Feature-hashed bag of words (two signed slots per word, stop words down-weighted), unit norm."""
    v = [0.0] * dim
    for w in words(text):
        wt = 0.25 if w in STOP else 1.0
        for i, s in _slots(w, dim):
            v[i] += s * wt
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        v[0] = n = 1.0
    return [x / n for x in v]


# ----------------------------------------------------------------------------- chat rules
def _text(content_field):
    if isinstance(content_field, list):                    # content given as parts
        return "\n".join(p.get("text", "") for p in content_field if isinstance(p, dict))
    return "" if content_field is None else str(content_field)


def classify(system, user):
    both = system + "\n" + user
    if "NEW FACT:" in user and "EXISTING MEMORIES" in user:
        return "decision"
    if "## New Messages" in user and "## Existing Memories" in user:
        return "mem0_extract"
    if "smart memory manager" in both and "retrieved facts" in both:
        return "mem0_update"
    if "Information Organizer" in both and "Input:" in user:
        return "mem0_facts"
    if "GOLD ANSWER:" in user and "MODEL ANSWER:" in user:
        return "judge"
    if "QUESTION:" in user and "MEMORIES:" in user:
        return "qa"
    return "other"


def _decision(user, dup):
    m = re.search(r"^NEW FACT:\s*(.*)$", user, re.M)
    fact = m.group(1) if m else ""
    body = user.split("EXISTING MEMORIES:", 1)[-1].split("RELATED CONTEXT:", 1)[0]
    for r, text in re.findall(r"^\[(\d+)\]\s*(.*)$", body, re.M):
        if overlap(fact, re.sub(r"\s*\(similarity [0-9.]+\)\s*$", "", text)) >= dup:
            return {"op": "NOOP", "target": int(r)}
    return {"op": "ADD", "target": None}


def _messages_of(body):
    """Messages of a Mem0 prompt body: 'role: content' lines (continuation lines join the previous one) or a JSON list."""
    body = body.strip()
    if body.startswith("["):
        try:
            return [(str(m.get("role", "user")), _text(m.get("content"))) for m in json.loads(body) if isinstance(m, dict)]
        except Exception:
            pass
    out = []
    for line in body.splitlines():
        m = re.match(r"^(user|assistant|system):\s?(.*)$", line)
        if m:
            out.append([m.group(1), m.group(2)])
        elif out and line.strip():
            out[-1][1] += " " + line.strip()
        elif line.strip():
            out.append(["user", line.strip()])
    return [(r, t.strip()) for r, t in out if r != "system" and t.strip()]


def _memory_texts(blob):
    """Texts of a serialized list of memories (JSON, or the Python repr that older Mem0 versions interpolate)."""
    blob = blob.strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            items = parse(blob)
        except Exception:
            continue
        if isinstance(items, dict):
            items = items.get("memory", items.get("results", []))
        out = []
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict):
                out.append((str(it.get("id", len(out))), str(it.get("text", it.get("memory", "")))))
            else:
                out.append((str(len(out)), str(it)))
        return out
    return []


def _mem0_extract(user, dup):
    sec = {}
    for name, body in re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |^# Output:|\Z)", user, re.M | re.S):
        sec[name.strip()] = body
    known = [t for _, t in _memory_texts(sec.get("Existing Memories", "[]")) + _memory_texts(sec.get("Recently Extracted Memories", "[]"))]
    out = []
    for role, text in _messages_of(sec.get("New Messages", "")):
        if any(overlap(text, t) >= dup for t in known):
            continue
        known.append(text)
        out.append({"id": str(len(out)), "text": text, "attributed_to": "assistant" if role == "assistant" else "user"})
    return {"memory": out}


def _mem0_facts(user):
    return {"facts": [t for _, t in _messages_of(user.split("Input:", 1)[-1])]}


def _mem0_update(prompt, dup):
    def block(anchor):
        m = re.search(re.escape(anchor) + r".*?```(.*?)```", prompt, re.S)
        return m.group(1) if m else "[]"
    old = _memory_texts(block("Below is the current content of my memory"))
    new = [t for _, t in _memory_texts(block("The new retrieved facts are mentioned in the triple backticks"))]
    out = [{"id": i, "text": t, "event": "NONE"} for i, t in old]
    known = [t for _, t in old]; nxt = len(old)
    for fact in new:
        if any(overlap(fact, t) >= dup for t in known):
            continue
        known.append(fact)
        out.append({"id": str(nxt), "text": fact, "event": "ADD"}); nxt += 1
    return {"memory": out}


def _qa(user):
    body, _, rest = user.partition("QUESTION:")
    question = rest.split("\n", 1)[0].strip()
    lines = [re.sub(r"^\[\d+\]\s*", "", l).strip() for l in body.split("MEMORIES:", 1)[-1].splitlines() if l.strip()]
    return short_answer(question, lines)


def _judge(user):
    g = re.search(r"^GOLD ANSWER:\s*(.*)$", user, re.M); a = re.search(r"^MODEL ANSWER:\s*(.*)$", user, re.M)
    return judge_rule(g.group(1) if g else "", a.group(1) if a else "")


def respond(messages, dup=0.6):
    """(kind, reply) for a list of chat messages; the reply is a dict (JSON reply) or a string."""
    system = "\n".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    user = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "user")
    kind = classify(system, user)
    if kind == "decision":
        return kind, _decision(user, dup)
    if kind == "mem0_extract":
        return kind, _mem0_extract(user, dup)
    if kind == "mem0_update":
        return kind, _mem0_update(system + "\n" + user, dup)
    if kind == "mem0_facts":
        return kind, _mem0_facts(user)
    if kind == "judge":
        return kind, _judge(user)
    if kind == "qa":
        return kind, _qa(user)
    return kind, "OK."


# ----------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def _error(self, code, msg):
        self.server.count(errors=1)
        self._send(code, {"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": None}})

    def _path(self):
        p = self.path.split("?", 1)[0].rstrip("/")
        return p[3:] if p.startswith("/v1/") or p == "/v1" else p

    def do_GET(self):
        S = self.server; p = self._path()
        if p == "/models":
            S.count(models=1)
            return self._send(200, {"object": "list", "data": [{"id": S.model, "object": "model", "created": int(S.t0), "owned_by": "mock"}]})
        if p == "/stats":
            return self._send(200, S.stats())
        if p in ("", "/health"):
            return self._send(200, {"status": "ok"})
        self._error(404, f"unknown path {self.path}")

    def do_POST(self):
        S = self.server; p = self._path()
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            req = json.loads(raw or b"{}")
            if not isinstance(req, dict):
                raise ValueError("the request body must be a JSON object")
        except Exception as e:
            return self._error(400, f"bad request body: {e}")
        if p == "/stats/reset":
            S.reset_stats(); return self._send(200, {"status": "reset"})
        if p not in ("/chat/completions", "/embeddings"):
            return self._error(404, f"unknown path {self.path}")
        S.enter()
        try:
            if p == "/embeddings":
                self._embeddings(req)
            else:
                self._chat(req)
        except Exception as e:                              # a bug in a rule must not hang the client
            self._error(500, f"mock server error: {e!r}")
        finally:
            S.leave()

    def _embeddings(self, req):
        S = self.server; inp = req.get("input")
        texts = [inp] if isinstance(inp, str) else list(inp or [])
        if not texts or not all(isinstance(t, str) for t in texts):
            return self._error(400, "`input` must be a string or a list of strings (token arrays are not supported)")
        dim = int(req.get("dimensions") or S.dim)
        if S.embed_latency > 0:
            time.sleep(S.embed_latency)
        vecs = [embed(t, dim) for t in texts]
        if req.get("encoding_format") == "base64":
            vecs = [base64.b64encode(struct.pack(f"<{dim}f", *v)).decode() for v in vecs]
        ntok = sum(len(t) for t in texts) // 4 + len(texts)
        S.count(embeddings=1, embedded_texts=len(texts), prompt_tokens=ntok)
        self._send(200, {"object": "list", "model": req.get("model") or S.model,
                         "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                         "usage": {"prompt_tokens": ntok, "total_tokens": ntok}})

    def _chat(self, req):
        S = self.server; msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
            return self._error(400, "`messages` must be a non-empty list of message objects")
        if req.get("stream"):
            return self._error(400, "the mock server does not support `stream`")
        prompt = "\n".join(_text(m.get("content")) for m in msgs)
        kind, reply = respond(msgs, S.dup)
        rf = (req.get("response_format") or {}).get("type") if isinstance(req.get("response_format"), dict) else None
        if rf in ("json_object", "json_schema") and not isinstance(reply, dict):
            reply = {"answer": reply}
        text = json.dumps(reply, ensure_ascii=False) if isinstance(reply, dict) else reply
        time.sleep(S.delay(prompt))
        message = {"role": "assistant", "content": text}; finish = "stop"
        tools, choice = req.get("tools"), req.get("tool_choice")
        if tools and (choice == "required" or isinstance(choice, dict)):
            name = (choice.get("function") or {}).get("name") if isinstance(choice, dict) else None
            name = name or (tools[0].get("function") or {}).get("name", "tool")
            args = text if isinstance(reply, dict) else json.dumps({"text": text})
            message = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "call_" + hashlib.sha1(prompt.encode()).hexdigest()[:12], "type": "function",
                                       "function": {"name": name, "arguments": args}}]}
            finish = "tool_calls"
        pt = len(prompt) // 4 + 4 * len(msgs); ct = max(1, len(text) // 4)
        S.count(chat=1, prompt_tokens=pt, completion_tokens=ct, kind=kind)
        self._send(200, {"id": "chatcmpl-" + hashlib.sha1(prompt.encode()).hexdigest()[:24], "object": "chat.completion",
                         "created": int(time.time()), "model": req.get("model") or S.model,
                         "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
                         "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256

    def __init__(self, port=0, latency=0.0, jitter=0.0, embed_latency=0.0, dim=256, dup=0.6, host="127.0.0.1",
                 model="mock-model", verbose=False):
        super().__init__((host, port), Handler)
        self.latency, self.jitter, self.embed_latency = float(latency), float(jitter), float(embed_latency)
        self.dim, self.dup, self.model, self.verbose = int(dim), float(dup), model, verbose
        self.mu = threading.Lock(); self.t0 = time.time(); self.in_flight = 0
        self.reset_stats()

    @property
    def port(self):
        return self.server_address[1]

    @property
    def url(self):
        return f"http://{self.server_address[0]}:{self.port}/v1"

    def delay(self, prompt):
        if self.latency <= 0 or self.jitter <= 0:
            return max(self.latency, 0.0)
        u = (int(hashlib.sha1(prompt.encode()).hexdigest()[:12], 16) + 0.5) / 16 ** 12
        return self.latency * math.exp(self.jitter * NormalDist().inv_cdf(u) - self.jitter ** 2 / 2)

    def reset_stats(self):
        with self.mu:
            self.c = {"chat": 0, "embeddings": 0, "models": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "embedded_texts": 0, "errors": 0}
            self.kinds = {}; self.max_in_flight = self.in_flight

    def count(self, kind=None, **inc):
        with self.mu:
            for k, v in inc.items():
                self.c[k] += v
            if kind:
                self.kinds[kind] = self.kinds.get(kind, 0) + 1

    def enter(self):
        with self.mu:
            self.in_flight += 1; self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def leave(self):
        with self.mu:
            self.in_flight -= 1

    def stats(self):
        with self.mu:
            c = dict(self.c)
            return {"requests": {k: c[k] for k in ("chat", "embeddings", "models")}, "chat_kinds": dict(self.kinds),
                    "prompt_tokens": c["prompt_tokens"], "completion_tokens": c["completion_tokens"],
                    "embedded_texts": c["embedded_texts"], "errors": c["errors"], "in_flight": self.in_flight,
                    "max_in_flight": self.max_in_flight, "uptime_s": round(time.time() - self.t0, 3),
                    "config": {"latency": self.latency, "jitter": self.jitter, "embed_latency": self.embed_latency,
                               "dim": self.dim, "dup": self.dup}}

    def shutdown(self):
        super().shutdown(); self.server_close()


def start(port=0, latency=0.0, **kw):
    """Start the server in a daemon thread and return it (see MockServer for the other options)."""
    srv = MockServer(port=port, latency=latency, **kw)
    threading.Thread(target=srv.serve_forever, name="mock-openai", daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description="Mock OpenAI-compatible server (offline tests)")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--latency", type=float, default=0.0, help="mean seconds per chat call")
    ap.add_argument("--jitter", type=float, default=0.0, help="lognormal sigma of the chat latency (0 = fixed)")
    ap.add_argument("--embed-latency", type=float, default=0.0, help="seconds per embedding call")
    ap.add_argument("--dim", type=int, default=256, help="embedding dimension")
    ap.add_argument("--dup", type=float, default=0.6, help="word-overlap share above which two texts count as the same fact")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    a = ap.parse_args()
    srv = MockServer(a.port, a.latency, a.jitter, a.embed_latency, a.dim, a.dup, a.host, verbose=a.verbose)
    print(f"mock OpenAI server on {srv.url}  (latency {a.latency}s, jitter {a.jitter}, dim {a.dim}); stats at /stats", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
