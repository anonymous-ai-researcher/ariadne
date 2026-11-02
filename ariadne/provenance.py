"""Provenance of a run: which code, which data, which machine, when.

experiments/run.py stores the result of `cell_provenance` in the summary of every cell (summary["provenance"]) and
appends one line per cell to results/<experiment>/journal.jsonl, so that every result can be traced to the code, the
data and the software that produced it; nothing here influences what a run computes.

    code_sha256   sha256 over the framework files listed by code_files() (ariadne/**/*.py, experiments/*.yaml,
                  experiments/run.py, experiments/prepare_data.sh, patches/*), in sorted order, path + content.
                  `python -m ariadne.provenance code` prints it; compare it with summary["provenance"]["code_sha256"] of a result.
    data          {path: {"x_sha256": sha256 of the float32 embedding array X, "rows": n, "dim": d}} for every data file
                  the cell reads (the array, not the .npz bytes: the zip container stores timestamps).
"""
import datetime, glob, hashlib, json, os, platform, socket, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = None
_DATA = {}
_PGLIB = {}


def code_files(root=ROOT):
    pats = ["ariadne/**/*.py", "experiments/*.yaml", "experiments/vldb/*.yaml", "experiments/vldb/*.sh", "experiments/run.py", "experiments/prepare_data.sh", "patches/*"]
    out = set()
    for p in pats:
        out.update(f for f in glob.glob(os.path.join(root, p), recursive=True) if os.path.isfile(f))
    return sorted(os.path.relpath(f, root).replace(os.sep, "/") for f in out if "__pycache__" not in f)


def code_sha256(root=ROOT):
    global _CODE
    if _CODE is not None and root == ROOT:
        return _CODE
    h = hashlib.sha256()
    for rel in code_files(root):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(root, rel), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    if root == ROOT:
        _CODE = h.hexdigest()
    return h.hexdigest()


def array_sha256(path):
    """sha256 of the embedding array X of a prepared data file (float32, C order)."""
    import numpy as np
    key = os.path.abspath(path)
    if key in _DATA:
        return _DATA[key]
    with np.load(path, allow_pickle=False) as a:
        X = np.ascontiguousarray(a["X"], dtype="float32")
    info = {"x_sha256": hashlib.sha256(X.tobytes()).hexdigest(), "rows": int(X.shape[0]), "dim": int(X.shape[1])}
    _DATA[key] = info
    return info


def data_paths(cfg):
    out = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x.endswith(".npz") and os.path.exists(x):
            out.append(x)
    walk(cfg.get("workload") or {})
    return sorted(set(out))


def pg_library(dsn):
    """sha256 of the pgvector shared library that the server loads (vector.so in the server's PKGLIBDIR)."""
    if dsn in _PGLIB:
        return _PGLIB[dsn]
    res = {"pkglibdir": None, "vector_so_sha256": None, "error": None}
    try:
        import psycopg2
        con = psycopg2.connect(dsn); cur = con.cursor()
        try:
            cur.execute("SELECT setting FROM pg_config WHERE name = 'PKGLIBDIR'"); res["pkglibdir"] = cur.fetchone()[0]
        finally:
            con.close()
    except Exception as e:                                   # pg_config view needs a superuser; fall back to the binary
        try:
            res["pkglibdir"] = subprocess.run(["pg_config", "--pkglibdir"], capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception as e2:
            res["error"] = "%r / %r" % (e, e2)
    if res["pkglibdir"]:
        so = os.path.join(res["pkglibdir"], "vector.so")
        try:
            with open(so, "rb") as f:
                res["vector_so_sha256"] = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            res["error"] = "cannot read %s: %r" % (so, e)
    _PGLIB[dsn] = res
    return res


def versions():
    v = {"python": platform.python_version()}
    for m in ("numpy", "scipy", "faiss", "psycopg2", "wordllama", "openai"):
        try:
            mod = __import__(m)
            v[m] = getattr(mod, "__version__", "?")
        except Exception:
            v[m] = None
    return v


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def cell_provenance(cfg, started_at, finished_at):
    p = {"code_sha256": code_sha256(), "started_at": started_at, "finished_at": finished_at,
         "host": socket.gethostname(), "platform": platform.platform(), "pid": os.getpid(), "argv": sys.argv,
         "versions": versions(), "data": {d: array_sha256(d) for d in data_paths(cfg)}}
    st = cfg.get("store") or {}
    if st.get("kind") == "pgvector":
        p["pgvector_library"] = pg_library(st.get("dsn") or "dbname=root")
    return p


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "code":
        print(code_sha256())
    elif len(sys.argv) > 2 and sys.argv[1] == "data":
        for f in sys.argv[2:]:
            print(f, json.dumps(array_sha256(f)))
    else:
        print("usage: python -m ariadne.provenance code | data FILE.npz ...")
