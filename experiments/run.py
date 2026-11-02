#!/usr/bin/env python3
"""Single entry point for every experiment.

    python experiments/run.py experiments/C1_calls_vs_policy.yaml            full run
    python experiments/run.py experiments/C1_calls_vs_policy.yaml --smoke    tiny version (minutes)
    python experiments/run.py --all --smoke                                  smoke-run every experiment
    python experiments/run.py X.yaml --dry-run                               list the cells without running
    python experiments/run.py X.yaml --only protocol.policy=cadm:1.1         run a subset of the grid
    python experiments/run.py X.yaml --set store.dsn="dbname=bench host=..." override any field
    python experiments/run.py X.yaml --force                                 re-run cells that already have results
    python experiments/run.py X.yaml --repeat 8 --only case=A64              seeds 1..8 for one case (existing seeds are kept)

Experiment files may contain ${VAR:-default}; the variables used throughout are
    EMB (embedder tag of the prepared data, default wordllama), EMB_SPEC (embedder spec for C4's question embeddings), PG_DSN
    (default dbname=root; the pgtests also read it), QDRANT_BIN, WORK (scratch directory for embedded stores, default work),
    LLM_MODEL, OPENAI_BASE_URL / OPENAI_API_KEY (language-model endpoint), EMBED_MODEL (A9), LOCOMO_RAW, QDRANT_URL,
    WEAVIATE_HOST.  Environment read by the code itself: ARIADNE_PG_DSN (pgtests, overrides PG_DSN), ARIADNE_PG_TMP
    (handshake directory of the gdb tests, default <tmp>/ariadne_pgtmp).

An experiment file has one of two forms.

1. Engine experiment (the default):

    name: C1_calls_vs_policy
    doc: free text (question, hypothesis, what to report)
    base:   {k: 10, tau_cos: 0.10, store: {...}, workload: {...}, decider: {...}, protocol: {...}}
    grid:   {protocol.policy: [none, exact, "cadm:1.1"], workload.agents: [4, 8]}     # cartesian product
    cases:  [{_name: L4, workload.data: ..., workload.agents: 4}, {_name: A64, ...}]  # named joint settings
    repeat: 3                 # seeds 1..3 (workload_seed stays 1 unless vary_workload: true)
    audits: [run, sep, {name: first, only: {protocol.policy: none}}, {name: law, only: {protocol.policy: none}}]
    smoke:  {base: {...}, grid: {...}, cases: [...], repeat: 1}                       # overrides for --smoke

   Every cell writes results/<name>/<cell>.pkl (config, per-task records, commit log) and appends its
   summary to results/<name>/summary.jsonl; audits append to results/<name>/audit_<audit>.jsonl.

2. Script experiment:

    name: A6_ssi_soundness
    kind: script
    commands: ["python -m ariadne.pgtests.ssi_hole --trials {trials} --out {out}/hole.jsonl"]
    vars: {trials: 8}
    smoke: {vars: {trials: 2}}
"""
import argparse, copy, glob, itertools, json, os, pickle, re, subprocess, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml


def expand_env(text):
    """${VAR:-default} and ${VAR} in experiment files are replaced from the environment."""
    def sub(m):
        return os.environ.get(m.group(1), m.group(3) if m.group(3) is not None else "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}", sub, text)


def set_path(d, path, val):
    ks = path.split(".")
    for k in ks[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[ks[-1]] = val


def get_path(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def deep_update(d, u):
    for k, v in (u or {}).items():
        if "." in k:
            set_path(d, k, copy.deepcopy(v))
        elif isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def short(v):
    s = json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
    return re.sub(r"[^A-Za-z0-9.+-]+", "", s.replace(":", "-"))[:28]


def cells(exp, smoke=False, repeat=None):
    base = copy.deepcopy(exp.get("base") or {}); grid = dict(exp.get("grid") or {}); cases = exp.get("cases"); rep = int(exp.get("repeat", 1))
    if smoke and exp.get("smoke"):
        sm = exp["smoke"]
        deep_update(base, sm.get("base"))
        for k, v in (sm.get("grid") or {}).items():
            grid[k] = v
        cases = sm.get("cases", cases); rep = int(sm.get("repeat", 1))
    if repeat:                                              # --repeat N: seeds 1..N (cells that already exist are skipped)
        rep = int(repeat)
    cases = cases or [{}]
    keys = list(grid)
    out = []
    for case in cases:
        for combo in itertools.product(*[grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]):
            for seed in range(1, rep + 1):
                cfg = copy.deepcopy(base); par = {}
                for a, b in case.items():
                    if a != "_name":
                        set_path(cfg, a, copy.deepcopy(b))
                for a, b in zip(keys, combo):
                    if isinstance(b, dict) and "_name" in b:            # a grid value may itself be a named bundle of settings
                        for x, y in b.items():
                            if x != "_name":
                                set_path(cfg, x, copy.deepcopy(y))
                        par[a] = b["_name"]
                    else:
                        set_path(cfg, a, copy.deepcopy(b)); par[a] = b
                cfg["seed"] = seed
                cfg.setdefault("workload_seed", seed if exp.get("vary_workload") else 1)
                parts = ([case["_name"]] if "_name" in case else []) + [short(v) for v in par.values()] + [f"s{seed}"]
                cfg["name"] = exp["name"]; cfg["cell"] = "_".join(parts)
                cfg["params"] = {"case": case.get("_name"), **{a: (b if not isinstance(b, (dict, list)) else json.dumps(b)) for a, b in par.items()}, "seed": seed}
                out.append(cfg)
    return out


def match(cfg, only):
    for cond in only or []:
        k, _, v = cond.partition("=")
        cur = cfg["params"].get(k, get_path(cfg, k))
        if k == "case":
            cur = cfg["params"].get("case")
        if str(cur) != v:
            return False
    return True


def audit_wanted(a, cfg):
    if isinstance(a, str):
        return a, True
    name = a["name"]
    for k, v in (a.get("only") or {}).items():
        cur = get_path(cfg, k)
        if (cur not in v) if isinstance(v, list) else (cur != v):
            return name, False
    return name, True


# Experiments whose per-task logs (.pkl) are read by a later step (C4 qa, B5b selftest, the C2 dense audit, the D2b redraw,
# figures fig_s_19 / fig_s_26 / fig_t_01 / fig_t_11, G3): --no-logs keeps their logs.
NEEDS_LOGS = ("B5a_", "C2_", "C3_", "C11_", "D2a_", "D3a_", "F5_", "G1_", "G3_")


def write_summary(exp, cfg, S, out):
    line = {"name": exp["name"], "cell": cfg["cell"], **cfg["params"], **S}
    with open(os.path.join(out, "summary.jsonl"), "a") as f:
        f.write(json.dumps(line, default=float) + "\n")


def cells_in(path):
    """Cell names that have a line in a .jsonl file."""
    got = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                got.add(json.loads(line).get("cell"))
            except ValueError:
                pass
    return got


def run_audits(exp, cfg, run, path, out, names, AU):
    """Run the named audits of one cell; a failure is printed and recorded in failed.jsonl.  Returns {name: ok}."""
    res_ok = {}
    for name in names:
        try:
            if name == "law":
                L = AU.law_extract(run, name=cfg["cell"])
                with open(path[:-4] + ".law.pkl", "wb") as f:
                    pickle.dump(L, f, protocol=4)
                res = AU.law_brief(L)
            else:
                res = AU.CMDS[name](run)
                arr = res.pop("_arrays", None)
                if arr is not None:
                    import numpy as np
                    np.savez_compressed(path[:-4] + ".audit.npz", **arr)
            with open(os.path.join(out, f"audit_{name}.jsonl"), "a") as f:
                f.write(json.dumps({"name": exp["name"], "cell": cfg["cell"], **cfg["params"], **res}, default=float) + "\n")
            res_ok[name] = True
        except Exception as e:
            res_ok[name] = False
            print(f"      audit {name} FAILED: {e!r}", flush=True); traceback.print_exc()
            with open(os.path.join(out, "failed.jsonl"), "a") as f:
                f.write(json.dumps({"cell": cfg["cell"], "audit": name, "error": repr(e)}) + "\n")
    return res_ok


def complete_cell(exp, cfg, path, out, AU):
    """For a cell whose log exists: write its summary line and run its audits if an interruption left them out."""
    names = [n for n, want in (audit_wanted(a, cfg) for a in exp.get("audits") or []) if want]
    need_sum = cfg["cell"] not in cells_in(os.path.join(out, "summary.jsonl"))
    missing = [n for n in names if (not os.path.exists(path[:-4] + ".law.pkl")) if n == "law"] + \
              [n for n in names if n != "law" and cfg["cell"] not in cells_in(os.path.join(out, f"audit_{n}.jsonl"))]
    if not need_sum and not missing:
        return []
    run = AU.load_run(path)
    done = []
    if need_sum:
        write_summary(exp, cfg, run["summary"], out); done.append("summary")
    if missing:
        AU._wl_cache.clear()
        for n, ok in run_audits(exp, cfg, run, path, out, missing, AU).items():
            done.append("audit %s%s" % (n, "" if ok else " FAILED"))
    return done


def run_engine(exp, args):
    from ariadne import audit as AU
    from ariadne import provenance as PV
    from ariadne.engine import execute
    from ariadne.workloads import make_workload
    from ariadne.stores import make_store
    out = os.path.join(args.out, exp["name"] + ("_smoke" if args.smoke else ""))
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)
    cs = [c for c in cells(exp, args.smoke, args.repeat) if match(c, args.only)]
    for c in cs:
        for kv in args.set or []:
            k, _, v = kv.partition("=")
            set_path(c, k, yaml.safe_load(v))
    print(f"== {exp['name']}: {len(cs)} cells -> {out}", flush=True)
    if args.dry_run:
        for c in cs:
            print("  ", c["cell"], json.dumps(c["params"]))
        return 0
    cache = {"wk": None, "wl": None, "sk": None, "store": None}
    failed = 0
    for i, cfg in enumerate(cs):
        path = os.path.join(out, cfg["cell"] + ".pkl")
        done_marker = path[:-4] + ".done"                   # written by --no-logs, which deletes the .pkl
        if not args.force and (os.path.exists(path) or os.path.exists(done_marker)):
            # Resume: a cell whose run finished is not run again, but a run that was interrupted after its log was saved
            # gets its missing summary line and its missing audits now.
            try:
                fixed = complete_cell(exp, cfg, path, out, AU) if os.path.exists(path) else []
            except Exception as e:                             # an unreadable log: report it; --force re-runs the cell
                fixed = ["log unreadable (%r): run again with --force --only cell=... FAILED" % e]
            print(f"  [{i + 1}/{len(cs)}] {cfg['cell']}: exists, skipped" + (" (completed: %s)" % ", ".join(fixed) if fixed else ""), flush=True)
            failed += sum(1 for x in fixed if x.endswith("FAILED"))
            continue
        try:
            wk = json.dumps([cfg["workload"], cfg["workload_seed"], cfg.get("k", 10)], sort_keys=True, default=str)
            if wk != cache["wk"]:
                cache.update(wk=wk, wl=make_workload(cfg["workload"], cfg["workload_seed"], int(cfg.get("k", 10))), sk=None)
            sk = json.dumps([cfg.get("store"), wk], sort_keys=True, default=str)
            if sk != cache["sk"]:
                if cache["store"] is not None:
                    try:
                        cache["store"].close()
                    except Exception:
                        pass
                st = make_store(cfg.get("store") or {}); st.setup(cache["wl"].base)
                cache.update(sk=sk, store=st)
            t0 = time.time(); started_at = PV.now()
            S, recs, clog = execute(cfg, cache["wl"], cache["store"])
            S["provenance"] = PV.cell_provenance(cfg, started_at, PV.now())
            AU.save_run(path, cfg, S, recs, clog)
            with open(os.path.join(out, "journal.jsonl"), "a") as f:
                f.write(json.dumps({"cell": cfg["cell"], "tasks": S.get("tasks"), "wall_s": S.get("wall_s"), **S["provenance"]}, default=str) + "\n")
            write_summary(exp, cfg, S, out)
            show = {k: S.get(k) for k in ("tps", "extra_calls_per_100", "aborts_per_task", "lat_p50_s", "lat_p99_s", "dup_per_100_events", "stored_stream_items") if S.get(k) is not None}
            print(f"  [{i + 1}/{len(cs)}] {cfg['cell']}: {time.time() - t0:.1f}s {json.dumps(show)}", flush=True)
            run = {"cfg": cfg, "summary": S, "recs": recs, "clog": clog}
            AU._wl_cache.clear(); AU._wl_cache[json.dumps([cfg["workload"], cfg.get("workload_seed", cfg.get("seed", 1)), cfg.get("k", 10)], sort_keys=True, default=str)] = cache["wl"]
            names = [n for n, want in (audit_wanted(a, cfg) for a in exp.get("audits") or []) if want]
            failed += sum(1 for ok in run_audits(exp, cfg, run, path, out, names, AU).values() if not ok)
            if args.no_logs:
                if any(n == "law" for n in names) or exp["name"].startswith(NEEDS_LOGS):
                    print(f"      --no-logs ignored for {cfg['cell']}: its log is needed by a later step", flush=True)
                else:
                    open(done_marker, "w").close()
                    os.remove(path)
        except (Exception, SystemExit) as e:
            failed += 1
            print(f"  [{i + 1}/{len(cs)}] {cfg['cell']}: FAILED {e!r}", flush=True); traceback.print_exc()
            with open(os.path.join(out, "failed.jsonl"), "a") as f:
                f.write(json.dumps({"cell": cfg["cell"], "error": repr(e)}) + "\n")
    return failed


def run_script(exp, args):
    out = os.path.join(args.out, exp["name"] + ("_smoke" if args.smoke else ""))
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)
    vars_ = dict(exp.get("vars") or {})
    if args.smoke:
        vars_.update((exp.get("smoke") or {}).get("vars") or {})
    for kv in args.set or []:
        k, _, v = kv.partition("="); vars_[k] = v
    vars_.update(out=out, results=args.out, smoke="_smoke" if args.smoke else "", py=sys.executable)
    cmds = (exp.get("smoke") or {}).get("commands") if args.smoke and (exp.get("smoke") or {}).get("commands") else exp.get("commands", [])
    print(f"== {exp['name']}: {len(cmds)} commands -> {out}", flush=True)
    failed = 0
    for c in cmds:
        cmd = c.format(**vars_)
        print("  $", cmd, flush=True)
        if args.dry_run:
            continue
        from ariadne import provenance as PV
        started_at = PV.now()
        r = subprocess.run(cmd, shell=True, cwd=ROOT)
        with open(os.path.join(out, "journal.jsonl"), "a") as f:
            f.write(json.dumps({"cmd": cmd, "exit": r.returncode, "code_sha256": PV.code_sha256(), "started_at": started_at,
                                "vars": {k: v for k, v in vars_.items() if k not in ("out", "results", "smoke", "py")},
                                "finished_at": PV.now(), "versions": PV.versions()}, default=str) + "\n")
        if r.returncode != 0:
            failed += 1
            print(f"    exit code {r.returncode}", flush=True)
            with open(os.path.join(out, "failed.jsonl"), "a") as f:
                f.write(json.dumps({"cmd": cmd, "exit": r.returncode}) + "\n")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*"); ap.add_argument("--all", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--only", action="append")
    ap.add_argument("--set", action="append"); ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--no-logs", action="store_true", help="delete the per-task log after the audits (keeps summaries only)")
    ap.add_argument("--skip-scripts", action="store_true")
    ap.add_argument("--repeat", type=int, help="number of seeds (1..N) instead of the experiment file's repeat")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(ROOT, "experiments", "*.yaml"))) if a.all else a.files
    os.chdir(ROOT)
    total = 0
    for f in files:
        exp = yaml.safe_load(expand_env(open(f).read()))
        if exp.get("kind") == "script":
            if not a.skip_scripts:
                total += run_script(exp, a)
        else:
            total += run_engine(exp, a)
    print(f"done, {total} failures", flush=True)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
