"""Datasets and embedders.

Prepared datasets live in  data/<name>__<embedder>.npz  with arrays
    X      float32 [n, d]   unit-normalized embeddings, in the dataset's natural order
    role   int8    [n]      0 = initial memory, 1 = write stream, -1 = unspecified (split by position)
    group  int32   [n]      team / conversation / story id (-1 if none)
    label  int32   [n]      class label used by the `tag` decision proxy (-1 if none)
and  data/<name>.texts.jsonl  with one {"text": ...} object per item (needed by the LLM decider).

Prepare with, for example:
    python -m ariadne.data prep --dataset agnews --raw /path/train.csv --embedder wordllama
    python -m ariadne.data prep --dataset locomo --raw /path/locomo10.json --embedder st:BAAI/bge-small-en-v1.5
    python -m ariadne.data prep --dataset spacy  --raw en_core_web_md-*.whl --name spacy      (vectors only)
    python -m ariadne.data prep --dataset hdf5   --raw glove-100-angular.hdf5 --name glove100   (vectors only)
    python -m ariadne.data prep --dataset jsonl  --raw my.jsonl --name mydata --embedder openai:text-embedding-3-small

Check the first few parsed items of a new file (`--peek 5`) before a long run.
"""
import argparse, csv, hashlib, json, os, re, sys
import numpy as np


# ----------------------------------------------------------------------------- embedders
def embedder(spec, cache_dir="data/.cache"):
    """spec: wordllama | st:<sentence-transformers model> | openai:<model> | hash:<dim> (deterministic, tests only)"""
    kind, _, arg = spec.partition(":")
    if kind == "wordllama":
        import shutil
        from pathlib import Path
        import wordllama
        from wordllama import WordLlama
        cache = Path(cache_dir) / "wordllama"; pkg = Path(wordllama.__file__).parent
        dst = WordLlama.get_file_path("tokenizer", cache); dst.mkdir(parents=True, exist_ok=True)
        for f in (pkg / "tokenizers").glob("*.json"):       # the wheel ships the tokenizer where the loader does not look
            if not (dst / f.name).exists():
                shutil.copy(f, dst / f.name)
        wl = WordLlama.load(cache_dir=str(cache), disable_download=True)
        return lambda texts: np.asarray(wl.embed(texts), dtype="float32")
    if kind == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(arg)
        return lambda texts: np.asarray(m.encode(texts, batch_size=128, show_progress_bar=False), dtype="float32")
    if kind == "openai":
        from openai import OpenAI
        cl = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "none"))

        def f(texts):
            out = []
            for i in range(0, len(texts), 256):
                r = cl.embeddings.create(model=arg, input=texts[i:i + 256])
                out.extend(d.embedding for d in r.data)
            return np.asarray(out, dtype="float32")
        return f
    if kind == "hash":
        dim = int(arg or 64)

        def f(texts):
            X = np.zeros((len(texts), dim), dtype="float32")
            for i, t in enumerate(texts):
                for w in re.findall(r"\w+", t.lower()):
                    r = np.random.default_rng(int(hashlib.sha1(w.encode()).hexdigest()[:8], 16))
                    X[i] += r.normal(size=dim)
            return X
        return f
    raise ValueError(f"unknown embedder '{spec}'")


def embed_all(fn, texts, bs=2048):
    out = [fn(texts[i:i + bs]) for i in range(0, len(texts), bs)]
    X = np.vstack(out).astype("float32")
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------- raw loaders
def load_agnews(path):
    """AG News train.csv (class, title, description), kept in file order."""
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.reader(f)):
            if i == 0 and not r[0].strip().isdigit():          # copies with a header row ("Class Index,Title,Description")
                continue
            items.append({"text": (r[1] + ". " + r[2]).replace("\\", " "), "label": int(r[0]), "group": -1, "role": -1})
    return items


def load_locomo(path):
    """LoCoMo locomo10.json: dialogue turns become the initial memory, extracted observations the stream."""
    d = json.load(open(path))
    base, stream = [], []
    for ci, s in enumerate(d):
        conv = s["conversation"]
        sess = sorted([k for k in conv if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        for k in sess:
            si = int(k.split("_")[1])
            for t in conv[k]:
                base.append({"text": f'{t["speaker"]}: {t.get("text", "")}', "label": ci * 100 + si, "group": ci, "role": 0})
        for k in sorted(s["observation"], key=lambda k: int(k.split("_")[1])):
            si = int(k.split("_")[1]); facts = []
            for sp, fs in s["observation"][k].items():
                for f in fs:
                    src = f[1] if isinstance(f[1], str) else f[1][0]
                    m = re.match(r"D(\d+):(\d+)", src.strip())
                    facts.append(((int(m.group(2)) if m else 10 ** 6), f[0]))
            for _, txt in sorted(facts, key=lambda x: x[0]):
                stream.append({"text": txt, "label": ci * 100 + si, "group": ci, "role": 1})
    return base + stream


def load_qqp(path):
    """Quora Question Pairs TSV (id, qid1, qid2, question1, question2, is_duplicate).
    Each distinct question is one item in order of first appearance; `label` is a duplicate-cluster id
    obtained by union-find over the pairs marked as duplicates (-1 for questions without a duplicate)."""
    parent, first, text = {}, [], {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    with open(path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t"); next(rd)
        for r in rd:
            if len(r) < 6:
                continue
            for q, t in ((r[1], r[3]), (r[2], r[4])):
                if q not in text:
                    text[q] = t; first.append(q)
            if r[5] == "1":
                parent[find(r[1])] = find(r[2])
    size = {}
    for q in first:
        size[find(q)] = size.get(find(q), 0) + 1
    cid = {}
    return [{"text": text[q], "label": (cid.setdefault(find(q), len(cid)) if size[find(q)] > 1 else -1), "group": -1, "role": -1}
            for q in first]


def load_longmemeval(path):
    """LongMemEval JSON: a list of questions, each with `haystack_sessions`
    (a list of sessions, each a list of {role, content} turns).  Turns become stream items in order;
    the question id is the group, so one team works on one haystack."""
    d = json.load(open(path)); items = []
    for gi, q in enumerate(d):
        for si, sess in enumerate(q.get("haystack_sessions", [])):
            for t in sess:
                txt = t.get("content", "") if isinstance(t, dict) else str(t)
                if txt.strip():
                    items.append({"text": f'{t.get("role", "user") if isinstance(t, dict) else "user"}: {txt}',
                                  "label": gi * 1000 + si, "group": gi, "role": 1})
    return items


def load_msmarco(path, limit=None):
    """MS MARCO passage collection.tsv (pid <tab> passage)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            items.append({"text": line.rstrip("\n").split("\t", 1)[-1], "label": -1, "group": -1, "role": -1})
    return items


def load_jsonl(path):
    """Generic: one object per line with `text` and optional `label`, `group`, `role` (base|stream)."""
    items = []
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        items.append({"text": o["text"], "label": int(o.get("label", -1)), "group": int(o.get("group", -1)),
                      "role": {"base": 0, "stream": 1}.get(o.get("role"), -1)})
    return items


def load_hdf5(path):
    """ANN-Benchmarks HDF5 (datasets `train`, `test`): vectors only, used as an exchangeable control."""
    import h5py
    with h5py.File(path, "r") as f:
        X = np.asarray(f["train"], dtype="float32")
    return X


def load_spacy_whl(path):
    """Word vectors shipped inside a spaCy model wheel (en_core_web_md / _lg): vectors only, deduplicated.
    Used by the synthetic fan-out workload, which only needs a realistic point cloud."""
    import io, zipfile
    z = zipfile.ZipFile(path)
    name = [n for n in z.namelist() if n.endswith("vocab/vectors")][0]
    X = np.load(io.BytesIO(z.read(name))).astype("float32")
    X = X[np.linalg.norm(X, axis=1) > 0]
    return np.unique(X, axis=0)


VECTOR_LOADERS = {"hdf5": load_hdf5, "spacy": load_spacy_whl}
LOADERS = {"agnews": load_agnews, "locomo": load_locomo, "qqp": load_qqp, "longmemeval": load_longmemeval,
           "msmarco": load_msmarco, "jsonl": load_jsonl}


def prep(dataset, raw, emb, out="data", name=None, limit=None):
    name = name or dataset
    os.makedirs(out, exist_ok=True)
    tag = emb.replace(":", "-").replace("/", "_")
    if dataset in VECTOR_LOADERS:
        X = VECTOR_LOADERS[dataset](raw)
        X = X[np.random.default_rng(12345).permutation(len(X))][:limit]      # fixed shuffle: these sets have no meaningful order
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        n = len(X); role = np.full(n, -1, "int8"); group = np.full(n, -1, "int32"); label = np.full(n, -1, "int32")
        np.savez(f"{out}/{name}__raw.npz", X=X, role=role, group=group, label=label)
        return f"{out}/{name}__raw.npz"
    items = LOADERS[dataset](raw)
    if limit:
        items = items[:limit]
    X = embed_all(embedder(emb), [it["text"] for it in items])
    np.savez(f"{out}/{name}__{tag}.npz", X=X, role=np.array([it["role"] for it in items], "int8"),
             group=np.array([it["group"] for it in items], "int32"), label=np.array([it["label"] for it in items], "int32"))
    with open(f"{out}/{name}.texts.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"text": it["text"]}, ensure_ascii=False) + "\n")
    return f"{out}/{name}__{tag}.npz"


def stats(npz, k=10, sample=5000, target_share=None):
    """Locality statistics that decide whether a stream has sibling structure worth testing.

    share_with_neighbour_within_cos_<t>: share of items whose nearest other item is within cosine distance t.  To use another
    embedding model with a comparable duplicate threshold, pass --target-share with the share that the main model has at its
    tau_cos (for example the value of share_with_neighbour_within_cos_0.10 of the main model's file): tau_cos_for_target_share
    is the threshold at which this file has the same share."""
    import faiss
    z = np.load(npz); X = z["X"]; n = len(X)
    rng = np.random.default_rng(0); idx = np.sort(rng.choice(n, size=min(sample, n), replace=False))
    ix = faiss.IndexFlatIP(X.shape[1]); ix.add(X)
    S, I = ix.search(X[idx], k + 1)
    d = np.sqrt(np.maximum(2 - 2 * S[:, 1:], 0))
    lid = -1.0 / np.mean(np.log(np.maximum(d[:, :-1], 1e-9) / d[:, -1:]), axis=1)
    adj = np.sqrt(np.maximum(2 - 2 * (X[1:] * X[:-1]).sum(1), 0))
    rnd = np.sqrt(np.maximum(2 - 2 * (X[rng.permutation(n)] * X).sum(1), 0))
    gap = np.abs(I[:, 1] - idx)
    near = d[:, 0] < np.sqrt(2 * 0.10)
    shares = {"share_with_neighbour_within_cos_%.2f" % t: float(np.mean(d[:, 0] < np.sqrt(2 * t))) for t in (0.05, 0.15, 0.20)}
    if target_share is not None:
        q = float(np.quantile(d[:, 0], min(max(float(target_share), 0.0), 1.0)))
        shares["tau_cos_for_target_share"] = round(q * q / 2, 4)          # chord c <-> cosine distance c^2 / 2
    return {**shares, "n": n, "dim": int(X.shape[1]), "median_nn_chord": float(np.median(d[:, 0])), "median_kth_chord": float(np.median(d[:, -1])),
            "median_lid": float(np.median(lid[np.isfinite(lid)])), "adjacent_median_chord": float(np.median(adj)),
            "random_median_chord": float(np.median(rnd)), "share_with_neighbour_within_cos_0.10": float(near.mean()),
            "near_dup_row_gap_median": float(np.median(gap[near])) if near.any() else None,
            "near_dup_within_100_rows": float(np.mean(gap[near] <= 100)) if near.any() else None}


def load_prepared(path):
    z = np.load(path)
    texts = None
    tpath = re.sub(r"__[^/]*\.npz$", ".texts.jsonl", path)
    if os.path.exists(tpath):
        texts = [json.loads(l)["text"] for l in open(tpath, encoding="utf-8")]
    return z["X"].astype("float32"), z["role"], z["group"], z["label"], texts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "stats"])
    ap.add_argument("--dataset"); ap.add_argument("--raw"); ap.add_argument("--embedder", default="wordllama")
    ap.add_argument("--out", default="data"); ap.add_argument("--name"); ap.add_argument("--limit", type=int)
    ap.add_argument("--npz"); ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--target-share", type=float, help="stats: also print the tau_cos at which this share of items has a neighbour within tau")
    a = ap.parse_args()
    if a.cmd == "prep":
        if a.peek:
            for it in LOADERS[a.dataset](a.raw)[:a.peek]:
                print(it)
            sys.exit(0)
        p = prep(a.dataset, a.raw, a.embedder, a.out, a.name, a.limit)
        print(p); print(json.dumps(stats(p), indent=1))
    else:
        print(json.dumps(stats(a.npz, target_share=a.target_share), indent=1))
