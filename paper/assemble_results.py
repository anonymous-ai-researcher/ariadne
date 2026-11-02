#!/usr/bin/env python
"""Builds the results root that paper/build.sh reads, from the result directories of the separate machines
(symbolic links, plus merged copies where rows of two machines are combined case by case).

    python paper/assemble_results.py --vm RESULTS_BASE --perf results_vldb_perf \\
        --gpt results_vldb_gpt4omini --qwen results_vldb_qwen7b --out results_vldb_merged
    RESULTS=results_vldb_merged bash paper/build.sh

Inputs:
    --vm      base results root with every experiment (the 192-core server; its C1 is the cross-machine comparison)
    --perf    the timing server (experiments/vldb/run_perf.sh): its experiments replace those of the base root
    --gpt, --qwen   the two model directories (experiments/vldb/run_llm.sh gpt4omini | qwen7b)
    --counts  optional count runs merged into the base root case by case (see below)

Layout of --out:
    <experiment>   the run of the timing server when it has one, otherwise the run of the base root
    vm/            the base root, all experiments (C1 for the cross-machine comparison)
    perf/          the timing server (its ENV/ is read by the environment checks)
    gpt/, qwen/    the two model directories (C3_llm_decisions, C4_downstream_qa, ENV/)
    C3_llm_decisions, C4_downstream_qa   -> gpt/... (generated tables and figures that read the top level)
A model directory or the perf directory may be missing (the corresponding keys then have no value).

--counts COUNTS is merged into the base root case by case: the rows of a case that COUNTS has replace the base rows of
that case, the other rows stay; a file X_<CASE>.jsonl replaces the rows of X.jsonl whose cell starts with <CASE>_.
Experiments the base root does not have are taken whole. A1_race_stores of the timing server adds only the cases the
base root did not run (Qdrant, Weaviate) to the top level; vm/ keeps the base root's own rows. Merged experiments are
written to <out>.cells/ (real files); everything else stays a link.
"""
import argparse, glob, json, os, re, shutil, sys

BYCASE_PERF = ("A1_race_stores",)


def _rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def merge_bycase(base, over, dst, only_new=False):
    """rows of `over` replace the rows of `base` with the same case; X_<CASE>.jsonl of `over` replaces cells <CASE>_* of X.jsonl"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    names = lambda d: {os.path.basename(f) for f in glob.glob(os.path.join(d, "*"))} if d and os.path.isdir(d) else set()
    nb, no = names(base), names(over)
    cases = {r.get("case") for f in glob.glob(os.path.join(over, "*.jsonl")) if os.path.basename(f) != "journal.jsonl"
             for r in _rows(f) if r.get("case") is not None}
    if only_new:                                  # keep the base's rows of every case it has; add only the other cases
        have = {r.get("case") for f in glob.glob(os.path.join(base, "*.jsonl")) if os.path.basename(f) != "journal.jsonl" for r in _rows(f)}
        cases = cases - have
    special = {}                                  # X.jsonl -> (CASE, file)
    for f in no:
        stem = f[:-6] if f.endswith(".jsonl") else None
        if stem and "_" in stem and re.fullmatch(r"[A-Z][A-Z0-9]+", stem.rsplit("_", 1)[1]) and stem.rsplit("_", 1)[0] + ".jsonl" in nb and stem.rsplit("_", 1)[0] + ".jsonl" not in no:
            special[stem.rsplit("_", 1)[0] + ".jsonl"] = (stem.rsplit("_", 1)[1], f)
    for f in sorted(nb | no):
        q = os.path.join(dst, f)
        if f in {v[1] for v in special.values()}:
            continue
        if f == "journal.jsonl":
            R = (_rows(os.path.join(base, f)) if f in nb else []) + (_rows(os.path.join(over, f)) if f in no else [])
        elif f in special:
            c, g = special[f]
            R = [r for r in _rows(os.path.join(base, f)) if not str(r.get("cell", "")).startswith(c + "_")] + _rows(os.path.join(over, g))
        elif f.endswith(".jsonl") and f in nb and f in no:
            R = [r for r in _rows(os.path.join(base, f)) if r.get("case") not in cases] + [r for r in _rows(os.path.join(over, f)) if r.get("case") in cases]
        else:
            src = os.path.join(over if f in no else base, f)
            if os.path.isdir(src):
                os.symlink(os.path.abspath(src), q)
            else:
                os.symlink(os.path.abspath(src), q)
            continue
        with open(q, "w") as w:
            for r in R:
                w.write(json.dumps(r) + "\n")


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vm", required=True); ap.add_argument("--perf"); ap.add_argument("--gpt"); ap.add_argument("--qwen")
    ap.add_argument("--counts", help="count runs merged into the base root case by case")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):                     # start clean: only links live here
        p = os.path.join(a.out, f)
        if os.path.islink(p):
            os.remove(p)
        else:
            sys.exit("assemble_results: %s is not a link; --out must be a directory of links" % p)
    exps = lambda d: sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f)) and f != "ENV") if d else []
    cells = os.path.abspath(a.out).rstrip("/") + ".cells"
    if os.path.exists(cells):
        shutil.rmtree(cells)
    vmd = {e: os.path.join(a.vm, e) for e in exps(a.vm)}
    if os.path.isdir(os.path.join(a.vm, "ENV")):
        vmd["ENV"] = os.path.join(a.vm, "ENV")
    merged = []
    if a.counts:                                  # count runs into the base root, case by case
        for e in exps(a.counts):
            if e in vmd:
                merge_bycase(vmd[e], os.path.join(a.counts, e), os.path.join(cells, "vm", e)); vmd[e] = os.path.join(cells, "vm", e)
            else:
                vmd[e] = os.path.join(a.counts, e)
            merged.append(e)
    perf = set(exps(a.perf))
    top = dict(vmd)
    for e in BYCASE_PERF:                         # cases only the timing server ran (A1: Qdrant, Weaviate) join the base root's
        if e in perf and e in vmd:
            merge_bycase(vmd[e], os.path.join(a.perf, e), os.path.join(cells, "top", e), only_new=True)
            top[e] = os.path.join(cells, "top", e); perf.discard(e); merged.append(e + " (perf, new cases)")
    if any(not m.endswith(")") for m in merged):                                    # vm/ is then a directory of links with the merged experiments
        vroot = os.path.join(cells, "vm")
        os.makedirs(vroot, exist_ok=True)
        for e, d in vmd.items():
            if os.path.abspath(d) != os.path.abspath(os.path.join(vroot, e)):
                link(d, os.path.join(vroot, e))
    else:
        vroot = a.vm
    n = {"vm": 0, "perf": 0}
    for e, d in sorted(top.items()):
        if e not in perf and e != "ENV":
            link(d, os.path.join(a.out, e)); n["vm"] += 1
    for e in perf:
        link(os.path.join(a.perf, e), os.path.join(a.out, e)); n["perf"] += 1
    link(vroot, os.path.join(a.out, "vm"))
    for tag, d in (("perf", a.perf), ("gpt", a.gpt), ("qwen", a.qwen)):
        if d:
            link(d, os.path.join(a.out, tag))
    # results of other embedding models next to the base root (results_<EMB>/, F2 and D6): next to --out
    vmroot = os.path.abspath(a.vm).rstrip("/")
    for d in sorted(os.listdir(os.path.dirname(vmroot))):
        p = os.path.join(os.path.dirname(vmroot), d)
        if d.startswith(os.path.basename(vmroot) + "_") and os.path.isdir(p):
            dst = os.path.abspath(a.out).rstrip("/") + d[len(os.path.basename(vmroot)):]
            if os.path.lexists(dst) and not os.path.islink(dst):
                sys.exit("assemble_results: %s exists and is not a link" % dst)
            link(p, dst)
    # F2 and D6 of the timing server with the other embedding models (<perf>_<EMB>/) replace those of the base root
    if a.perf:
        proot = os.path.abspath(a.perf).rstrip("/")
        for d in sorted(os.listdir(os.path.dirname(proot))):
            p = os.path.join(os.path.dirname(proot), d)
            if d.startswith(os.path.basename(proot) + "_") and os.path.isdir(p):
                dst = os.path.abspath(a.out).rstrip("/") + d[len(os.path.basename(proot)):]
                if os.path.lexists(dst) and not os.path.islink(dst):
                    sys.exit("assemble_results: %s exists and is not a link" % dst)
                link(p, dst)
    if a.gpt:                                   # C3, C4 and A9 (Mem0 with GPT-4o mini) come from the GPT-4o mini directory
        for e in exps(a.gpt):
            link(os.path.join(a.gpt, e), os.path.join(a.out, e))
    print("assemble_results: %d experiments from the base root, %d from the timing server (%s); merged by case: %s; models: %s -> %s"
          % (n["vm"], n["perf"], ", ".join(sorted(perf)) or "none", ", ".join(merged) or "none", ", ".join(t for t, d in (("gpt", a.gpt), ("qwen", a.qwen)) if d) or "none", a.out))


if __name__ == "__main__":
    main()
