# ariadne.ext: tests through agent-memory frameworks

Files

| file | purpose |
|---|---|
| `mock_openai.py` | OpenAI-compatible mock server (chat, embeddings), standard library only, for offline tests |
| `mem0_race.py` | the duplicate race through the public API of Mem0: modes `serial`, `concurrent`, `lock` |

Quick start (from the repository root, no network needed):

    python -m ariadne.ext.mem0_race --mock --latency 0.5 --events 40 --facts synthetic \
        --store qdrant-server --qdrant-bin /path/to/qdrant --out results/T_ext/mem0_race_mock.jsonl

Against a real endpoint (as in experiment `A9_memory_frameworks`, which `experiments/vldb/run_llm.sh` runs with GPT-4o mini):

    OPENAI_API_KEY=... python -m ariadne.ext.mem0_race --base-url https://api.openai.com/v1 \
        --model gpt-4o-mini --embed-model text-embedding-3-small --events 200 \
        --store qdrant-server --qdrant-url 127.0.0.1:6333 --out results/mem0_race.jsonl

The module docstrings list every option and every output field.

## The test

E events (facts), each reported by f agents as light paraphrases, arrive with jitter. Every report is one
call of the framework's add operation on one shared memory. Afterwards the stored memories are listed
through the framework and matched back to the events. The outcome is the number of stored memories per
event: duplicates per 100 events, lost events, wall time, model calls.

- `serial`: one thread, the reference. It shows how many memories per event the framework writes without
  any concurrency (with a real model this need not be exactly one).
- `concurrent`: the framework as it is.
- `lock`: one global lock around each add, the simple remedy; its cost is the wall time.

The test is meaningful for a framework whose add operation reads similar memories, lets a model decide,
and then writes, without holding anything that makes the read and the write one atomic step. Whether a
given framework behaves like that has to be established by running the test, not assumed.

## What an adapter needs

An adapter is a small class like `Mem0Adapter` in `mem0_race.py`, registered in `ADAPTERS`. Three things:

1. **A shared-memory handle.** `__init__(args, tag, workdir, endpoint, dims, qdrant_addr, n_handles)` builds a
   fresh, empty memory named `tag` that all agents write to (one scope: the same user id, namespace,
   group id or collection for every caller), configured to use `endpoint` (`base_url`, `api_key`, `model`,
   `embed_model`) for the model and the embedder, and a store that needs no external service where
   possible. Provide `handle()` so that the test can use one object for all threads or one per thread.
2. **An add call.** `add(text, agent) -> int` sends one report through the framework's public write path
   (the one that retrieves and decides, not a raw insert) and returns how many memories the call says
   it added. Exceptions propagate; the test counts them as `add_errors`.
3. **A way to list what is stored.** `list() -> [str]` returns the text of every memory in the shared
   scope at the end.

Also expected: an attribute `calls` (model calls made by the framework, counted by wrapping its model
client), `name`, `version`, and `close()`.

Things to check for a new framework before trusting numbers:

- the write path must be synchronous from the caller's point of view, or the adapter must wait until
  background consolidation has finished before `list()`;
- the unit that can be duplicated (a memory row, a note, a graph node or edge) decides what `list()` returns;
- with a real model the stored text is reworded, so the marker token may not survive: check
  `unmatched_memories` and `matched_by_embedding`, or run with `--no-marker`;
- the store behind the framework must accept concurrent writers, otherwise `add_errors` is the result.

## Frameworks for which the same test can be set up

Only Mem0 was run. The other lines say what an adapter would have to map; the API names follow the
public documentation and were not checked against installed packages. Nothing below states how these
systems behave under the test.

| framework | what would have to be adapted |
|---|---|
| Mem0 (`mem0ai`) | done: handle = `Memory.from_config`, add = `Memory.add(text, user_id=shared)`, list = `Memory.get_all(filters={"user_id": shared})` |
| LangMem / LangGraph store | handle = one `BaseStore` with one shared namespace; add = the LangMem memory manager invoked on the report (the path that searches the store before writing), not a bare `store.put`; list = `store.search` over the namespace |
| Letta | handle = a memory that several agents share (a shared block or a shared archival store); add = a message to an agent that makes it store the fact, or the archival insert call; list = the passages or the block content, which for a block means splitting one text into facts |
| Zep / Graphiti | handle = one graph with one group id; add = one episode per report; list = the fact edges (and entity nodes) of the group, so duplicates are counted on edges or nodes, not on rows |
| A-MEM | handle = one memory system object shared by the threads; add = its note creation call; list = all notes; if the add path also rewrites existing notes, check that the marker matching still works |
| MemoryOS | handle = one instance for one shared user id; add = its add-memory call (if it takes a dialogue pair, the report goes into one side of it); list = the entries of all memory tiers, after any deferred consolidation, if there is one, has run |

## What was run

- The paper's numbers (Section 3.1: duplicates per 100 facts through the add operation of Mem0, concurrent
  against serial) come from `experiments/A9_memory_frameworks.yaml` with `mem0ai` 2.2.1 and GPT-4o mini
  (`gpt-4o-mini-2024-07-18`) as model, run by `experiments/vldb/run_llm.sh gpt4omini`.
- The mock server is for offline tests of the harness (no network, no cost); its numbers say nothing about
  a real model.

What `Memory.add` does in mem0ai 2.2.1 (from the installed source, `mem0/memory/main.py`): embed the
message, search the 10 most similar memories of the scope, one model call (additive extraction: the
model returns the memories to add and is told to skip what the retrieved memories already cover), an
exact-text hash check against the retrieved memories, embed, insert. There is no UPDATE or DELETE on
this path in this version, and nothing makes the search and the insert one atomic step.

Mem0 notes: `MEM0_TELEMETRY=False` and `MEM0_DIR` are set by the script before `mem0` is imported.
Importing `mem0` without `MEM0_DIR` creates `~/.mem0/config.json`.
