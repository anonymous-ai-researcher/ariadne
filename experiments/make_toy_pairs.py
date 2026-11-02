#!/usr/bin/env python3
"""Toy stand-in for a labelled duplicate data set (used by the smoke test of A3 when Quora Question Pairs is not at hand).

Takes a texts file written by `ariadne.data prep` and writes a JSONL with 3000 singleton texts (label -1) and 100 clusters of 2 to 4
light rewordings of one text each (label = cluster id), in the format of the generic loader:
    python experiments/make_toy_pairs.py data/agnews.texts.jsonl data/toyqqp.jsonl
    python -m ariadne.data prep --dataset jsonl --raw data/toyqqp.jsonl --name toyqqp --embedder wordllama
"""
import json, random, sys

src, dst = sys.argv[1], sys.argv[2]
rnd = random.Random(7)
texts = [json.loads(l)["text"] for _, l in zip(range(4000), open(src, encoding="utf-8"))]
with open(dst, "w", encoding="utf-8") as f:
    for t in texts[:3000]:
        f.write(json.dumps({"text": t, "label": -1}) + "\n")
    for c, t in enumerate(texts[3000:3100]):
        w = t.split()
        for _ in range(rnd.choice((2, 3, 4))):
            v = [x for x in w if rnd.random() > 0.08]              # drop a few words
            if len(v) > 4:
                i = rnd.randrange(len(v) - 1); v[i], v[i + 1] = v[i + 1], v[i]
            f.write(json.dumps({"text": " ".join(v), "label": c}) + "\n")
print("wrote", dst)
