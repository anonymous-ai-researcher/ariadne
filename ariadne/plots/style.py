"""Shared look of every figure of the paper (system side and theory side).

    from ariadne.plots import style as st
    st.use()                                   # rcParams; call once per figure module
    fig, ax = st.subplots(cols=1)              # 3.33 in wide (single column); cols=2 -> 7.0 in
    ax.plot(x, y, **st.policy_line("cadm:1.1"))
    st.save(fig, "fig_s_05_calls_vs_c", out)   # <out>/<name>.pdf and .png (300 dpi)

One fixed mapping, used by every figure:

    policy_style(spec)   colour, marker, line style, hatch of a protocol or policy spec.  Accepts the policy
                         spec strings of ariadne/policies.py ("cadm:1.1", "overlap:0.9", "oracle:link", ...)
                         and the protocol names used as case names in the experiment files ("av-1.1",
                         "ssi-short", "lock-long", "ball-1.0", "write-verify-id", ...).
    dataset_style(case)  colour, marker, line style of a data set, from a case name ("L4", "A64", "LME4",
                         "QQP64", "MSM64", "ag-2000", ...) or a data path ("data/agnews__wordllama.npz").
    pretty(spec)         human label ("cadm:1.1" -> "c-adm. (c=1.1)").
    dataset_label(case)  human label ("A64" -> "AG News, 64 agents").

Colours are the Okabe-Ito palette plus grey.  The validation family is cool (blue ramp = c-admissible, ours;
bluish green = certified), exact validation is black, no isolation is grey, the weaker predicates are orange
(identity tests) and reddish purple (single-rank tests), oracles are yellow.  There are more protocol families
than Okabe-Ito hues, so the three remaining families are dark warm shades next to SERIALIZABLE's vermillion
(ball predicate locks: burnt orange; application-level remedies: dark red; lock held across think: dark brown).
These shades were chosen by maximising the separation, under simulated protanopia, deuteranopia and tritanopia,
between the families that share a figure; they differ mainly in lightness.  Every family also has its own marker,
line style and hatch, so that no figure depends on colour alone (greyscale printing, colour-vision deficiency).

No em dash characters anywhere in figure text.
"""
import colorsys, math, os, re

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap

# ------------------------------------------------------------------------------------------------ sizes
COL1 = 3.33           # single column of acmart sigconf, inches
COL2 = 7.0            # both columns
FONT = 7.5            # base font size, points
SMALL = 6.5           # tick labels, legends, annotations
TINY = 5.8            # dense annotations
LW = 1.1              # data lines
MS = 3.6              # markers

# ------------------------------------------------------------------------------------------------ palette
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
GREY = "#777777"
LIGHT = "#D9D9D9"      # grid, reference fills
INK = "#222222"        # text
MUTED = "#555555"      # secondary text
ACCENT = OKABE_ITO["vermillion"]   # the one colour used to flag "unsound / violation" annotations

# sequential ramp of the c-admissible family: tolerance c from strict (c = 1, light) to loose (dark)
_CADM_ANCHORS = ["#A9D8F5", "#56B4E9", "#0072B2", "#003A5C"]
CADM_CMAP = LinearSegmentedColormap.from_list("cadm", _CADM_ANCHORS)
_CERT_ANCHORS = ["#8FD9C2", "#009E73", "#00503B"]
CERT_CMAP = LinearSegmentedColormap.from_list("cert", _CERT_ANCHORS)
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#F4F9FD", "#A9D8F5", "#56B4E9", "#0072B2", "#003A5C"])
WARM_CMAP = LinearSegmentedColormap.from_list("warm", ["#FFFFFF", "#FBE3C8", "#F0A868", "#D55E00"])


def c_position(c):
    """Position of a tolerance c on the sequential ramp, in [0, 1]: log scale in (c - 1), c = 1 at 0."""
    try:
        c = float(c)
    except (TypeError, ValueError):
        return 0.6
    if not c > 1.0:
        return 0.0
    t = (math.log10(c - 1.0) + 3.0) / 3.5          # c = 1.001 -> 0, c = 1.1 -> 0.57, c = 2 -> 0.86, c = 4.2 -> 1
    return min(max(0.12 + 0.88 * t, 0.12), 1.0)


def c_color(c, family="cadm"):
    return to_hex((CERT_CMAP if family == "cert" else CADM_CMAP)(c_position(c)))


# family -> (colour, marker, line style, hatch, order in legends and on axes, label of the family)
FAMILIES = {
    "none":     dict(color=GREY,                    marker="x", ls=(0, (1, 1.2)),          hatch="",       order=0,  label="no isolation"),
    "exact":    dict(color=OKABE_ITO["black"],      marker="s", ls="-",                    hatch="",       order=1,  label="exact (re-execution)"),
    "identity": dict(color=OKABE_ITO["orange"],     marker="^", ls=(0, (4, 1.2, 1, 1.2)),  hatch="///",    order=2,  label="identity tests (overlap, top-j)"),
    "rank":     dict(color=OKABE_ITO["purple"],     marker="P", ls=(0, (3, 1, 1, 1, 1, 1)), hatch="xxx",    order=3,  label="single-rank distance tests"),
    "cadm":     dict(color=OKABE_ITO["blue"],       marker="o", ls="-",                    hatch="",       order=4,  label="c-admissible (ours)"),
    "cert":     dict(color=OKABE_ITO["green"],      marker="D", ls=(0, (4, 1.5)),          hatch="...",    order=5,  label="certified c-admissible"),
    "oracle":   dict(color="#B8A200",               marker="*", ls=(0, (1, 1)),            hatch="",       order=6,  label="oracle"),
    "lock":     dict(color="#4A2C2A",               marker="P", ls=(0, (6, 2)),            hatch="||", order=7, label="lock held across think"),
    "ssi":      dict(color=OKABE_ITO["vermillion"], marker="v", ls=(0, (5, 1, 1, 1)),      hatch="\\\\\\", order=8,  label="SERIALIZABLE"),
    "ball":     dict(color="#A04000",               marker="h", ls=(0, (2, 1)),            hatch="oo",     order=9,  label="ball predicate locks"),
    "app":      dict(color="#7E3030",               marker="d", ls=(0, (3, 1.5, 1, 1.5, 1, 1.5)), hatch="--", order=10, label="application-level remedies"),
    "other":    dict(color="#444444",               marker=".", ls="-",                    hatch="",       order=11, label="other"),
}
_MARKER_VARIANT = {"overlap": "^", "top1": ">", "topj": "<", "set": "s", "radius": "P", "rank1": "X",
                   "sumdist": "p", "objects": "d", "redecide": "8", "compensate": "d", "reread": "1"}

_NUM = r"(\d+(?:\.\d+)?)"


def parse(spec):
    """(family, name, argument) of a policy spec or protocol name.  Never raises: unknown -> ("other", spec, "")."""
    s = str(spec).strip()
    low = s.lower().replace("_", "-")
    base = low.replace("+exist", "")
    name, _, arg = base.partition(":")
    if name in ("none", "raw", "no-isolation", "rc", "read-committed", "rr", "repeatable-read", "no-lock", "nolock"):
        return "none", name, arg
    if name in ("exact", "set", "reexec", "av-exact", "re-execution"):
        return "exact", "set" if name == "set" else "exact", arg
    if name in ("overlap", "top1", "topj"):
        return "identity", name, arg
    if name in ("radius", "rank1", "sumdist"):
        return "rank", name, arg
    if name in ("cadm", "cert"):
        return name, name, arg or "1.0"
    if name == "oracle":
        return "oracle", name, arg or "link"
    if name in ("objects", "redecide"):
        return "app", name, arg
    # protocol names used as case names
    m = re.match(r"^(?:av-)?cadm[-:]?" + _NUM + r"(.*)$", low) or re.match(r"^av[-:]" + _NUM + r"(.*)$", low) \
        or re.match(r"^all[-:]" + _NUM + r"(.*)$", low)
    if m:
        return "cadm", "cadm", m.group(1)
    m = re.match(r"^(?:av-)?cert[-:]?" + _NUM + r"(.*)$", low)
    if m:
        return "cert", "cert", m.group(1)
    if low.startswith("ssi") or low.startswith("serializable"):
        return "ssi", low, ""
    if low.startswith("ball"):
        m = re.search(_NUM, low)
        return "ball", "ball", m.group(1) if m else "1.0"
    if low.startswith("lock-long") or low.startswith("locklong") or low in ("group-lock", "lock", "pessimistic"):
        return "lock", low, ""
    if low.startswith("lock-short"):
        return "app", "reread+lock", ""
    if low.startswith("write-verify") or low.startswith("compensate") or low.startswith("reread") \
            or low.startswith("objects") or low.startswith("redecide"):
        return "app", low, ""
    if low.startswith("oracle"):
        return "oracle", "oracle", low.partition("-")[2] or "link"
    if low.startswith("exact"):
        return "exact", "exact", ""
    if low.startswith("none"):
        return "none", "none", ""
    if low.startswith("overlap") or low.startswith("top"):
        return "identity", low, ""
    return "other", s, ""


def family(spec):
    return parse(spec)[0]


def policy_style(spec):
    """Style of a protocol or policy spec: dict with color, marker, ls, hatch, mfc, mec, order, family.

    Use policy_line / policy_marker / policy_bar to get keyword arguments for plot / scatter / bar."""
    fam, name, arg = parse(spec)
    st = dict(FAMILIES[fam]); st["family"] = fam
    if fam in ("cadm", "cert"):
        st["color"] = c_color(arg, fam)
    if fam == "ball":
        try:
            if float(arg) > 1.0:
                st["color"] = "#CC9466"; st["ls"] = (0, (2, 1, 0.6, 1))
        except ValueError:
            pass
    if fam == "rank":                                  # three single-rank tests: one hue, three shades and dashes
        st["color"], st["ls"] = {"radius": ("#CC79A7", (0, (4, 1.2))), "rank1": ("#8E3B74", (0, (1, 1))),
                                 "sumdist": ("#E0A3C8", (0, (6, 1.5, 1, 1.5)))}.get(name, (st["color"], st["ls"]))
    key = name.split("-")[0].split(":")[0]
    if key in _MARKER_VARIANT:
        st["marker"] = _MARKER_VARIANT[key]
    if name.startswith("write-verify") or name.startswith("compensate"):
        st["marker"] = "d"
    if name.startswith("reread"):
        st["marker"] = "1" if "lock" not in name else "2"
        if "lock" not in name:
            st["color"] = "#C29A9A"; st["hatch"] = "////"
        else:
            st["hatch"] = ""
    if name.startswith("write-verify") or name.startswith("compensate"):
        st["hatch"] = "xxxx" if name.endswith("any") else "...."
    if name.startswith("objects"):
        st["hatch"] = "----"
    st["mfc"] = st["color"]; st["mec"] = st["color"]
    if name == "overlap":
        try:
            if float(arg or 0.9) < 0.9:
                st["mfc"] = "white"
        except ValueError:
            pass
    if fam == "oracle":
        st["mfc"] = OKABE_ITO["yellow"]; st["mec"] = "#000000"
    if fam == "ssi" and "short" in name:
        st["mfc"] = "white"; st["ls"] = (0, (5, 1, 1, 1, 1, 1))
    if fam == "ssi" and "backoff" in name:
        st["marker"] = "<"
    if fam == "lock" and name == "group-lock":
        st["mfc"] = "white"
    try:                                               # order: by family, then by tolerance, then by name
        sub = float(arg)
    except (TypeError, ValueError):
        sub = 0.0
    st["order"] = (st["order"], sub, name)
    return st


def policy_line(spec, **kw):
    """Keyword arguments for ax.plot / ax.errorbar."""
    s = policy_style(spec)
    out = dict(color=s["color"], marker=s["marker"], ls=s["ls"], lw=LW, ms=MS + (1.6 if s["marker"] == "*" else 0),
               mfc=s["mfc"], mec=s["mec"], mew=0.7)
    out.update(kw)
    return out


_C_VARIANT = {1.02: ("v", (0, (1, 1))), 1.05: ("^", (0, (4, 1.5))), 1.1: ("o", "-"), 1.2: ("D", (0, (1.5, 1))), 1.5: ("p", (0, (6, 1.5, 1, 1.5)))}


def policy_line_distinct(spec, **kw):
    """policy_line, but tolerances of one family also differ in marker and dash (for figures with several c lines,
    whose colours alone are too close on the ramp): c = 1.1 keeps the circle and the solid line of the main text."""
    out = policy_line(spec)
    fam, _, arg = parse(spec)
    if fam in ("cadm", "cert"):
        try:
            c = round(float(arg), 3)
        except (TypeError, ValueError):
            c = None
        if c in _C_VARIANT:
            out["marker"], out["ls"] = _C_VARIANT[c]
    out.update(kw)
    return out


def policy_marker(spec, **kw):
    """Keyword arguments for a single mark (ax.plot with no line)."""
    kw.setdefault("ls", "none")
    return policy_line(spec, **kw)


def policy_bar(spec, **kw):
    """Keyword arguments for ax.bar / ax.barh."""
    s = policy_style(spec)
    out = dict(color=s["color"], edgecolor="white", linewidth=0.5, hatch=s["hatch"])
    if s["family"] == "oracle":
        out.update(color=OKABE_ITO["yellow"], edgecolor="#000000")
    if s["mfc"] == "white":
        out.update(color="white", edgecolor=s["color"], hatch=s["hatch"] or "////")
    out.update(kw)
    return out


def policy_order(specs):
    """Specs sorted in the one order used across figures."""
    return sorted(specs, key=lambda s: policy_style(s)["order"])


_PROXY_NAMES = {"dup": "duplicate", "tag": "tag", "link": "link", "top3": "top-3", "set": "set"}
PROXY_LABEL = {"dup": "duplicate test", "tag": "majority tag", "link": "link target", "top3": "top-3 set", "set": "full set"}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return ("%g" % v)


def pretty(spec, short=False):
    """Human label of a policy spec or protocol name.  short=True drops the family words where possible."""
    fam, name, arg = parse(spec)
    low = str(spec).lower()
    ex = " +exist" if "+exist" in low else ""
    if fam == "none":
        return {"rc": "READ COMMITTED", "rr": "REPEATABLE READ", "no-lock": "validation, no lock",
                "raw": "no isolation, uninstrumented"}.get(name, "no isolation")
    if fam == "exact":
        return ("set-exact" if name == "set" else "exact") + ex
    if fam == "identity":
        if name == "overlap":
            return "overlap %s%%" % _num(100 * float(arg or 0.9)) + ex
        if name == "top1":
            return "top-1" + ex
        if name == "topj":
            return "top-%s" % (arg or "3") + ex
        return str(spec)
    if fam == "rank":
        lab = {"radius": "radius", "rank1": "rank-1", "sumdist": "sum-dist."}[name]
        return "%s (c=%s)" % (lab, _num(arg or 1)) + ex
    if fam in ("cadm", "cert"):
        head = "c-adm." if fam == "cadm" else "certified"
        tail = ""
        if "noexist" in low:
            tail = ", no exist. check"
        elif low.endswith("-ro"):
            tail = ", validates reads"
        if short:
            return ("c=%s" % _num(arg) if fam == "cadm" else "cert. c=%s" % _num(arg)) + ((" (" + tail[2:] + ")") if tail else "")
        return "%s (c=%s%s)" % (head, _num(arg), tail) + ex
    if fam == "oracle":
        return "oracle (%s)" % _PROXY_NAMES.get(arg, arg)
    if fam == "lock":
        return "group lock" if name == "group-lock" else "lock across think"
    if fam == "ssi":
        head = "SSI" if short else "SERIALIZABLE"
        if "short" in name:
            return head + " (short)"
        if "backoff" in name:
            return head + " (backoff)"
        return head
    if fam == "ball":
        try:
            c = float(arg)
        except ValueError:
            c = 1.0
        return "ball lock (c=1)" if c <= 1.0 else "ball lock (c=%s)" % _num(c)
    if fam == "app":
        if name.startswith("write-verify") or name.startswith("compensate"):
            tb = name.rpartition("-")[2]
            return "write, verify" + (" (%s)" % tb if tb in ("id", "any") else "")
        if name.startswith("reread"):
            return "re-read + lock" if "lock" in name else "re-read"
        if name.startswith("objects"):
            return "object versions"
        if name.startswith("redecide"):
            return "re-decide"
    return str(spec)


def pretty_math(spec, short=False):
    """pretty() with the tolerance set in mathtext italics, for axis and legend text."""
    return re.sub(r"\bc(?=[-=])", r"$c$", pretty(spec, short))


# ------------------------------------------------------------------------------------------------ data sets
DATASETS = {
    "locomo":      dict(color=OKABE_ITO["blue"],       marker="o", ls="-",                   label="LoCoMo",      order=0),
    "agnews":      dict(color=OKABE_ITO["vermillion"], marker="s", ls=(0, (4, 1.5)),         label="AG News",     order=1),
    "longmemeval": dict(color=OKABE_ITO["green"],      marker="^", ls=(0, (4, 1.2, 1, 1.2)), label="LongMemEval", order=2),
    "qqp":         dict(color=OKABE_ITO["purple"],     marker="D", ls=(0, (1, 1)),           label="QQP",         order=3),
    "msmarco":     dict(color=OKABE_ITO["orange"],     marker="v", ls=(0, (3, 1, 1, 1, 1, 1)), label="MS MARCO",  order=4),
    "synthetic":   dict(color="#666666",               marker="x", ls=(0, (6, 2)),           label="synthetic",   order=5),
}


def dataset_key(case):
    """Data set of a case name, a data path, or a summary row (dict with 'workload.data' or 'case')."""
    if isinstance(case, dict):
        for k in ("workload.data", "data", "case", "cell"):
            if case.get(k) not in (None, "None", ""):
                return dataset_key(case[k])
        return "synthetic"
    s = str(case)
    low = s.lower()
    for key, pat in (("longmemeval", r"longmemeval|^lme"), ("locomo", r"locomo"), ("agnews", r"agnews|ag[-_ ]?news|^ag[-_]"),
                     ("msmarco", r"msmarco|ms[-_ ]marco|^msm"), ("qqp", r"qqp|quora|toy-?pairs")):
        if re.search(pat, low):
            return key
    if re.match(r"^L\d", s):
        return "locomo"
    if re.match(r"^A\d", s):
        return "agnews"
    return "synthetic"


def dataset_style(case):
    """dict with color, marker, ls, label, order for the data set behind a case name / data path / row."""
    return dict(DATASETS[dataset_key(case)])


def dataset_line(case, **kw):
    s = dataset_style(case)
    out = dict(color=s["color"], marker=s["marker"], ls=s["ls"], lw=LW, ms=MS, mfc=s["color"], mec=s["color"], mew=0.7)
    out.update(kw)
    return out


def dataset_label(case, agents=True, short=False):
    """ "A64" -> "AG News, 64 agents" (short=True: "AG News 64"); names without a count give the data set name alone."""
    s = str(case)
    lab = DATASETS[dataset_key(s)]["label"]
    m = re.match(r"^(?:L|A|LME|QQP|MSM)(\d+)(.*)$", s)
    if m and agents:
        extra = {"mix": ", mixed ops", "k50": ", k=50", "-llm": ", LLM", "multi": ", multi-read", "upsert": ", upserts", "add": ", adds"}.get(m.group(2))
        if extra is None:
            mm = re.match(r"^rr(\d+)$", m.group(2))
            extra = ", %s teams" % mm.group(1) if mm else (", " + m.group(2).lstrip("-_") if m.group(2) else "")
        if short:
            return "%s %s%s" % (lab, m.group(1), extra)
        return "%s, %s agents%s" % (lab, m.group(1), extra)
    if dataset_key(s) == "synthetic" and s not in ("None", ""):
        return {"fan": "fan-out", "tight": "planted (tight)", "A": "AG News"}.get(s, s)
    return lab


def case_lines(cases, **kw):
    """{case: plot keyword arguments} for several workloads in one plot: colour and marker of the data set; cases of
    the same data set get lighter shades (the last one, in case_order, is the full colour) and alternate marker fills."""
    out = {}
    cases = case_order(cases)
    by = {}
    for c in cases:
        by.setdefault(dataset_key(c), []).append(c)
    for key, cs in by.items():
        n = len(cs)
        for i, c in enumerate(cs):
            d = dataset_line(c)
            col = lighten(d["color"], 0.5 * (1 - (i + 1) / n)) if n > 1 else d["color"]
            d.update(color=col, mec=col, mfc=col if (n - 1 - i) % 2 == 0 else "white")
            if key == "synthetic" and n > 1:
                d.update(marker=["x", "+", "1", "2", "3", "4"][i % 6], color=darken(col, 0.25), mec=darken(col, 0.25))
            d.update(kw)
            out[c] = d
    return out


STORES = {
    "pgvector-rc": "pgvector (RC)", "pgvector-rr": "pgvector (RR)", "pgvector": "pgvector", "pgvector-hnsw": "pgvector HNSW",
    "pgvector-ivfflat": "pgvector IVFFlat", "qdrant": "Qdrant", "chroma": "Chroma", "milvus": "Milvus", "lancedb": "LanceDB",
    "weaviate": "Weaviate", "faiss": "FAISS", "faiss-hnsw": "FAISS HNSW", "faiss-ivf": "FAISS IVF", "faiss-flat": "FAISS flat",
    "faiss-cos": "FAISS (cosine)", "faiss-l2": "FAISS (L2)", "pg-cos": "pgvector (cosine)", "pg-l2": "pgvector (L2)",
    "exact": "exact scan", "hnsw-ef10": "HNSW, ef=10",
}


def store_label(name):
    return STORES.get(str(name), str(name))


def case_label(case):
    """Label of any case name: a store, a data set with its number of agents, or the name itself."""
    s = str(case)
    if s in STORES:
        return STORES[s]
    return dataset_label(s)


def case_order(cases):
    """Cases ordered by data set, then by the number in the name."""
    def key(c):
        m = re.search(r"(\d+(?:\.\d+)?)", str(c))
        return (DATASETS[dataset_key(c)]["order"], float(m.group(1)) if m else 0.0, str(c))
    return sorted(cases, key=key)


# ------------------------------------------------------------------------------------------------ rcParams
def use():
    """Apply the shared rcParams.  Idempotent."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.default": "it",
        "mathtext.cal": "Liberation Sans:italic", "mathtext.sf": "Liberation Sans", "mathtext.tt": "Liberation Mono",
        "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#333333",
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35, "ytick.minor.width": 0.35,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": LIGHT, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": 0.7,
        "patch.linewidth": 0.5, "hatch.linewidth": 0.35, "hatch.color": "white",
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25, "legend.columnspacing": 1.0,
        "legend.borderpad": 0.2,
        "errorbar.capsize": 1.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03,
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    })


def figsize(cols=1, aspect=0.62, width=None):
    """(width, height) in inches.  cols=1: single column (3.33 in), cols=2: both columns (7.0 in);
    fractional values scale the single column (cols=1.5 -> 5.0 in).  height = aspect * width."""
    if width is None:
        width = COL1 if cols == 1 else COL2 if cols == 2 else COL1 * cols
        width = min(width, COL2)
    return (width, width * aspect)


def subplots(nrows=1, ncols=1, cols=1, aspect=0.62, **kw):
    """plt.subplots with the shared rcParams and a paper-sized figure."""
    use()
    kw.setdefault("figsize", figsize(cols, aspect))
    return plt.subplots(nrows, ncols, **kw)


def grid(ax, axis="y", minor=False):
    """Light grid on one axis (use only where reading values off the axis matters)."""
    ax.grid(True, axis=axis, which="major", color=LIGHT, lw=0.4)
    if minor:
        ax.grid(True, axis=axis, which="minor", color="#ECECEC", lw=0.3)


def panel_label(ax, letter, text=None, loc="left", dy=1.03):
    """Panel label "(a)" above the top-left corner of the axes, optionally followed by a short description."""
    s = "(%s)" % letter + (" " + text if text else "")
    x, ha = (0.0, "left") if loc == "left" else (0.5, "center")
    return ax.text(x, dy, s, transform=ax.transAxes, ha=ha, va="bottom", fontsize=FONT, fontweight="bold" if not text else None)


def panel_labels(axes, texts=None, **kw):
    for i, ax in enumerate(list(axes)):
        panel_label(ax, "abcdefghijklmnopqrstuvwxyz"[i], None if texts is None else texts[i], **kw)


def direct_label(ax, x, y, text, color=INK, dx=3, dy=0, ha="left", va="center", **kw):
    """Label a mark or the end of a line (offset in points), instead of a legend entry."""
    kw.setdefault("fontsize", SMALL)
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color,
                       annotation_clip=False, **kw)


def spread_labels(ys, min_gap):
    """Move label positions apart (1-d) so that neighbours are at least min_gap apart; keeps the order."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(idx, idx[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def lighten(color, amount=0.5):
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def darken(color, amount=0.3):
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    return to_hex(colorsys.hls_to_rgb(h, max(0.0, l * (1 - amount)), s))


def text_on(color):
    """Black or white, whichever is readable on a fill of this colour."""
    r, g, b = to_rgb(color)
    return "#FFFFFF" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def legend_handles(specs, kind="line", label=pretty_math):
    """Proxy artists for a legend of policies (kind: line | marker | bar)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    for s in specs:
        if kind == "bar":
            hs.append(Patch(label=label(s), **{("facecolor" if k == "color" else k): v for k, v in policy_bar(s).items()}))
        else:
            kw = policy_line(s) if kind == "line" else policy_line_distinct(s) if kind == "line_distinct" else policy_marker(s)
            hs.append(Line2D([], [], label=label(s), **kw))
    return hs


def save(fig, name, out, close=True):
    """Write <out>/<name>.pdf and <out>/<name>.png (300 dpi); returns the two paths.

    The figure keeps its exact size (so fonts have their true size in the paper) unless an artist sticks out
    of the canvas, in which case it is saved with a tight bounding box instead of being clipped."""
    os.makedirs(out, exist_ok=True)
    kw = {}
    try:
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        if tb.x0 < -0.015 or tb.y0 < -0.015 or tb.x1 > w + 0.015 or tb.y1 > h + 0.015:
            kw = dict(bbox_inches="tight", pad_inches=0.02)
            print("  note: %s does not fit its %.2f x %.2f in canvas (content %.2f x %.2f in); saved with a tight box"
                  % (name, w, h, tb.width, tb.height))
    except Exception:
        kw = dict(bbox_inches="tight", pad_inches=0.02)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(p, dpi=300, **kw)
        paths.append(p)
    if close:
        plt.close(fig)
    print("  wrote %s (.pdf, .png)" % os.path.join(out, name))
    return paths


# ------------------------------------------------------------------------------------------------ plot helpers
def _mlh(S):
    import numpy as np
    return (np.array([s.mean for s in S], dtype=float), np.array([s.lo for s in S], dtype=float),
            np.array([s.hi for s in S], dtype=float))


def line_ci(ax, xs, S, band=True, **kw):
    """Mean line over the seeds with the 95% interval as a band (S: list of load.Stats, one per x).
    With a single seed per point only the line is drawn.  Returns the line."""
    m, lo, hi = _mlh(S)
    (ln,) = ax.plot(xs, m, **kw)
    if band and any(s.n > 1 for s in S) and len(xs) > 1:
        ax.fill_between(xs, lo, hi, color=ln.get_color(), alpha=0.12, lw=0, zorder=ln.get_zorder() - 0.5)
    elif band and any(s.n > 1 for s in S):
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor=ln.get_color(), elinewidth=0.6, capsize=1.5)
    return ln


def point_ci(ax, x, s, xs=None, **kw):
    """One mark with its 95% interval as caps (s: load.Stats for y; xs: optional load.Stats for x)."""
    color = kw.get("color", INK)
    yerr = [[s.mean - s.lo], [s.hi - s.mean]] if s.n > 1 else None
    xerr = None
    if xs is not None:
        xerr = [[xs.mean - xs.lo], [xs.hi - xs.mean]] if xs.n > 1 else None
        x = xs.mean
    if yerr is not None or xerr is not None:
        ax.errorbar([x], [s.mean], yerr=yerr, xerr=xerr, fmt="none", ecolor=color, elinewidth=0.6, capsize=1.5, zorder=2.5)
    kw.setdefault("zorder", 3)                     # the mark covers its own caps (hollow marks would show them through)
    return ax.plot([x], [s.mean], **kw)[0]


def grouped_bars(ax, groups, series, stat, kw_of, width=0.82, zero_label=True, value_labels=False, fmt="%.3g",
                 horizontal=False, missing="", label_size=None):
    """Grouped bars: one group per entry of `groups` on the category axis, one bar per entry of `series`.

    stat(group, series) -> load.Stats or None (no data);  kw_of(series) -> bar keyword arguments (style.policy_bar).
    Intervals are drawn as caps; bars of height 0 get a small "0" so that they do not read as missing data.
    Returns the centres of the groups."""
    import numpy as np
    n = max(len(series), 1)
    bw = width / n
    centres = np.arange(len(groups), dtype=float)
    fs = label_size or TINY
    for j, sname in enumerate(series):
        for i, g in enumerate(groups):
            s = stat(g, sname)
            pos = centres[i] - width / 2 + (j + 0.5) * bw
            if s is None or s.n == 0 or s.mean != s.mean:
                if missing:
                    (ax.text(0, pos, missing, fontsize=fs, color=MUTED, ha="left", va="center") if horizontal else
                     ax.text(pos, 0, missing, fontsize=fs, color=MUTED, ha="center", va="bottom"))
                continue
            kw = dict(kw_of(sname))
            err = None
            if s.n > 1 and s.hi > s.lo:
                err = [[s.mean - s.lo], [s.hi - s.mean]]
            ekw = dict(ecolor="#333333", elinewidth=0.6, capsize=1.3, capthick=0.6)
            if horizontal:
                ax.barh(pos, s.mean, height=bw * 0.94, xerr=err, error_kw=ekw, **kw)
            else:
                ax.bar(pos, s.mean, width=bw * 0.94, yerr=err, error_kw=ekw, **kw)
            top = s.hi if err else s.mean
            if (zero_label and s.mean == 0) or value_labels:
                txt = "0" if s.mean == 0 else fmt % s.mean
                if horizontal:
                    ax.annotate(txt, (top, pos), xytext=(1.5, 0), textcoords="offset points", ha="left", va="center",
                                fontsize=fs, color=MUTED)
                else:
                    ax.annotate(txt, (pos, top), xytext=(0, 1.0), textcoords="offset points", ha="center", va="bottom",
                                fontsize=fs, color=MUTED)
    return centres


def end_labels(ax, items, pad=3.0, fontsize=None, min_gap=None, side="right", leader=True):
    """Direct labels at the ends of lines instead of a legend.

    items: list of (x_last, y_last, text, color).  Labels are placed just outside the right (or left) edge of the
    axes at the height of the last point and moved apart vertically when they would overlap.  Call it after the
    axis limits and scales are final."""
    if not items:
        return []
    fs = fontsize or SMALL
    fig = ax.figure
    try:
        h_pt = ax.get_position().height * fig.get_size_inches()[1] * 72.0
    except Exception:
        h_pt = 100.0
    gap = min_gap if min_gap is not None else (fs * 1.12) / max(h_pt, 1.0)
    tr = ax.transData + ax.transAxes.inverted()
    ys = []
    for x, y, _, _ in items:
        try:
            ys.append(float(tr.transform((x, y))[1]))
        except Exception:
            ys.append(0.5)
    ys = [min(max(v, 0.0), 1.0) if v == v else 0.0 for v in ys]
    new = spread_labels(ys, gap)
    over = max(new) - 1.0                       # keep the stack inside the axes height when possible
    if over > 0:
        new = [v - over for v in new]
        order = sorted(range(len(new)), key=lambda i: -new[i])
        for a, b in zip(order, order[1:]):
            if new[a] - new[b] < gap:
                new[b] = new[a] - gap
    out = []
    xa = 1.0 if side == "right" else 0.0
    for (x, y, text, color), y0, y1 in zip(items, ys, new):
        if leader and abs(y1 - y0) > gap * 0.35:
            ax.annotate("", xy=(xa, y0), xycoords="axes fraction", xytext=(pad - 0.5 if side == "right" else -pad + 0.5, (y1 - y0) * h_pt),
                        textcoords="offset points", arrowprops=dict(arrowstyle="-", lw=0.35, color=color, shrinkA=0, shrinkB=0),
                        annotation_clip=False)
        out.append(ax.annotate(text, xy=(xa, y1), xycoords="axes fraction", xytext=(pad if side == "right" else -pad, 0),
                               textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                               fontsize=fs, color=INK, annotation_clip=False))
    return out


def label_points(ax, items, fontsize=None, pad=3.0, obstacles=(), color=None, prefer=("ne", "se", "nw", "sw", "n", "s", "e", "w"), segments=()):
    """Label scattered marks without overlaps (greedy): items = [(x, y, text)] or [(x, y, text, colour)].

    Every label tries the positions in `prefer` around its mark (compass points) and takes the first one that does
    not cover another mark, another label or the outside of the axes; obstacles = extra (x, y) data points to avoid,
    segments = ((x0, y0), (x1, y1)) data segments (lines between marks) to keep clear of.
    Call it after the limits are final; the figure is drawn once to know the geometry."""
    if not items:
        return []
    fs = fontsize or TINY
    fig = ax.figure
    try:
        fig.canvas.draw()
    except Exception:
        pass
    tr = ax.transData
    px = 72.0 / fig.dpi                                   # display pixels -> points
    P = [tuple(tr.transform((it[0], it[1])) * px) for it in items]
    O = list(P) + [tuple(tr.transform(o) * px) for o in obstacles]
    L = []                                                  # points along the segments, weaker obstacles
    for a, b in segments:
        pa, pb = tr.transform(a) * px, tr.transform(b) * px
        n = max(int(((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 / 3.0), 2)
        L += [(pa[0] + (pb[0] - pa[0]) * t / n, pa[1] + (pb[1] - pa[1]) * t / n) for t in range(1, n)]
    bb = ax.get_window_extent()
    x0, y0, x1, y1 = bb.x0 * px, bb.y0 * px, bb.x1 * px, bb.y1 * px
    placed = []
    out = []
    off = {"ne": (1, 1), "se": (1, -1), "nw": (-1, 1), "sw": (-1, -1), "n": (0, 1.3), "s": (0, -1.3), "e": (1.3, 0), "w": (-1.3, 0)}

    def box(pt, text, d):
        lines = text.split("\n")
        w = max(len(l.replace("$", "")) for l in lines) * fs * 0.50 + 1.0
        h = len(lines) * fs * 1.15
        dx, dy = off[d]
        cx = pt[0] + dx * (pad + w / 2) if dx else pt[0]
        cy = pt[1] + dy * (pad + h / 2) if dy else pt[1]
        if dx and not dy:
            cx = pt[0] + (1 if dx > 0 else -1) * (pad + 1.5 + w / 2)
        if dy and not dx:
            cy = pt[1] + (1 if dy > 0 else -1) * (pad + 1.0 + h / 2)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def cost(b, i):
        c = 0.0
        if b[0] < x0 or b[2] > x1 or b[1] < y0 or b[3] > y1:
            c += 50
        for j, o in enumerate(O):
            if j != i and b[0] - 2.5 < o[0] < b[2] + 2.5 and b[1] - 2.5 < o[1] < b[3] + 2.5:
                c += 10
        for q in placed:
            if not (b[2] < q[0] or q[2] < b[0] or b[3] < q[1] or q[3] < b[1]):
                c += 20
        c += min(sum(1.0 for o in L if b[0] - 1 < o[0] < b[2] + 1 and b[1] - 1 < o[1] < b[3] + 1), 8.0)
        return c

    order = sorted(range(len(items)), key=lambda i: -sum(1 for o in O if abs(o[0] - P[i][0]) < 40 and abs(o[1] - P[i][1]) < 15))
    res = {}
    for i in order:
        text = items[i][2]
        best = min(prefer, key=lambda d: (cost(box(P[i], text, d), i), prefer.index(d)))
        b = box(P[i], text, best)
        placed.append(b); res[i] = best
    for i, it in enumerate(items):
        d = res[i]; dx, dy = off[d]
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        ox = (pad if dx > 0 else -pad if dx < 0 else 0) + (1.5 * (1 if dx > 0 else -1) if dx and not dy else 0)
        oy = (pad if dy > 0 else -pad if dy < 0 else 0) + (1.0 * (1 if dy > 0 else -1) if dy and not dx else 0)
        col = it[3] if len(it) > 3 and it[3] else (color or MUTED)
        out.append(ax.annotate(it[2], (it[0], it[1]), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=fs,
                               color=col, annotation_clip=False, zorder=7, linespacing=1.0))
    return out


def _break_mark(ax, x):
    """Axis break mark on the x axis at data position x."""
    ax.plot([x / 1.07, x * 1.07], [0, 0], transform=ax.get_xaxis_transform(), color="white", lw=2.4, clip_on=False, zorder=5,
            solid_capstyle="butt")
    for f in (1 / 1.07, 1.07):
        for d in (1, -1):
            ax.annotate("", xy=(x * f, 0), xycoords=ax.get_xaxis_transform(), xytext=(1.2 * d, 2.6 * d), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333", shrinkA=0, shrinkB=0), annotation_clip=False, zorder=6)


def c_axis(ax, cs, label=True, min_sep=None, left=(), right=(), step=2.6):
    """Log x axis in (c - 1) for a tolerance sweep, labelled with the values of c.

    Returns pos(c): the x coordinate of a tolerance.  c = 1 (c - 1 = 0 has no place on a log axis) sits at a pseudo
    position to the left of the smallest c > 1, behind an axis break mark.  `left` and `right` are names of extra
    categorical columns before and after the sweep (reference marks such as "exact" or "overlap 90%"); their
    coordinates are pos.at[name]."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter
    cs = sorted({float(c) for c in cs})
    pos_cs = [c for c in cs if c > 1.0 + 1e-12]
    has_one = any(abs(c - 1.0) <= 1e-12 for c in cs)
    lo = min(pos_cs) - 1.0 if pos_cs else 0.01
    hi = max(pos_cs) - 1.0 if pos_cs else 1.0
    if hi / lo < 3 and step <= 2.6:
        step = 1.8
    x_one = lo / step
    at = {}
    x = x_one if has_one else lo
    for name in list(left)[::-1]:
        x = x / step; at[name] = x
    x_min = x
    x = hi
    for name in right:
        x = x * step; at[name] = x
    x_max = x

    def pos(c):
        if isinstance(c, str) and c in at:
            return at[c]
        c = float(c)
        return x_one if c <= 1.0 + 1e-12 else c - 1.0

    pos.at = at
    pos.one = x_one
    ax.set_xscale("log")
    pad = step ** 0.5
    # label only the tolerances whose labels have room: keep the round values first
    try:
        width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    except Exception:
        width_in = 2.5
    decades = math.log10((x_max * pad) / (x_min / pad))
    need = min_sep if min_sep is not None else (0.036 * plt.rcParams["xtick.labelsize"]) / max(width_in / decades, 1e-6)
    fixed = [math.log10(x_one)] if has_one else []
    fixed += [math.log10(at[n]) for n in at]
    ticks = _thin(pos_cs, need, fixed)
    tpos = [c - 1.0 for c in ticks] + ([x_one] if has_one else []) + [at[n] for n in at]
    tlab = [_num(c) for c in ticks] + (["1"] if has_one else []) + [n for n in at]
    ax.xaxis.set_major_locator(FixedLocator(tpos)); ax.xaxis.set_major_formatter(FixedFormatter(tlab))
    minor = [c - 1.0 for c in pos_cs if c not in ticks]
    ax.xaxis.set_minor_locator(FixedLocator(minor) if minor else NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(x_min / pad, x_max * pad)
    if pos_cs:
        if has_one:
            _break_mark(ax, math.sqrt(x_one * lo))
        if left:
            first = x_one if has_one else lo
            _break_mark(ax, math.sqrt(at[list(left)[-1]] * first))
        if right:
            _break_mark(ax, math.sqrt(hi * at[list(right)[0]]))
    if label:
        ax.set_xlabel("tolerance $c$  (log scale in $c-1$)")
    return pos


def _thin(cands, need, fixed=()):
    """Greedy choice of tick values on a log axis in (value - 1): round values first, at least `need` decades apart."""
    prio = [1.1, 1.01, 2.0, 1.001, 1.03, 1.3, 1.003, 1.5, 1.2, 1.05, 1.02, 3.0, 1.005, 5.0, 1.002, 1.15, 1.07, 10.0]
    order = sorted(cands, key=lambda c: (min([i for i, r in enumerate(prio) if abs(c - r) < 1e-9] or [len(prio)]), c))
    keep = []
    for c in order:
        lc = math.log10(c - 1.0)
        if all(abs(lc - math.log10(t - 1.0)) >= need for t in keep) and all(abs(lc - f) >= need * 0.9 for f in fixed):
            keep.append(c)
    return sorted(keep)


def ratio_axis(ax, lo=1.005, hi=2.0, label="approximation ratio", min_sep=None):
    """Log x axis in (ratio - 1), labelled with the ratio itself (1.01, 1.03, 1.1, ...); labels are thinned to fit."""
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    ax.set_xscale("log")
    ax.set_xlim(lo - 1.0, hi - 1.0)
    cands = [t for t in (1.001, 1.002, 1.003, 1.005, 1.01, 1.02, 1.03, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0) if lo - 1e-12 <= t <= hi + 1e-12]
    try:
        width_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    except Exception:
        width_in = 2.5
    decades = math.log10((hi - 1.0) / (lo - 1.0))
    need = min_sep if min_sep is not None else (0.036 * plt.rcParams["xtick.labelsize"]) / max(width_in / decades, 1e-6)
    ticks = _thin(cands, need)
    ax.xaxis.set_major_locator(FixedLocator([t - 1.0 for t in ticks])); ax.xaxis.set_major_formatter(FixedFormatter([_num(t) for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([t - 1.0 for t in cands if t not in ticks])); ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel(label)


def spec_c(spec):
    """Tolerance c of a spec ("cadm:1.1" -> 1.1, "av-1.05" -> 1.05), or None."""
    fam, _, arg = parse(spec)
    if fam in ("cadm", "cert", "ball", "rank"):
        try:
            return float(arg)
        except (TypeError, ValueError):
            return None
    return None


def hide_spines(ax, which=("top", "right")):
    for s in which:
        ax.spines[s].set_visible(False)


def nice_log_ticks(ax, axis="both"):
    """Plain numbers (1, 10, 100; 0.1) instead of powers of ten on log axes with a short range."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: ("%g" % v))
    if axis in ("x", "both") and ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both") and ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(f)


def log_ticks(ax, axis="x", steps=(1, 2, 5)):
    """Ticks at 1, 2, 5, 10, 20, 50, ... written as plain numbers (for log axes that span about two decades or less)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    if lo <= 0 or hi <= 0:
        return
    if hi / lo > 3000:
        steps = (1,)
    elif hi / lo > 150:
        steps = (1, 3) if steps == (1, 2, 5) else steps
    ticks = [m * 10.0 ** e for e in range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1) for m in steps]
    ticks = [t for t in ticks if lo * 0.999 <= t <= hi * 1.001]
    a = ax.xaxis if axis == "x" else ax.yaxis
    if len(ticks) < 2:
        return
    a.set_major_locator(FixedLocator(ticks)); a.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    a.set_minor_locator(NullLocator()); a.set_minor_formatter(NullFormatter())


def floor0(ax, axis="y", headroom=0.05):
    """Start a value axis at zero and leave a little room above the largest mark.  Call it after plotting; with
    shared axes call it once, after every panel is drawn."""
    if axis == "y":
        ax.autoscale(enable=True, axis="y"); ax.autoscale_view(scalex=False, scaley=True)
        hi = ax.get_ylim()[1]
        ax.set_ylim(0, hi * (1 + headroom) if hi > 0 else 1.0)
    else:
        ax.autoscale(enable=True, axis="x"); ax.autoscale_view(scalex=True, scaley=False)
        hi = ax.get_xlim()[1]
        ax.set_xlim(0, hi * (1 + headroom) if hi > 0 else 1.0)


def no_data(ax, text="no data"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=SMALL, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

use()
