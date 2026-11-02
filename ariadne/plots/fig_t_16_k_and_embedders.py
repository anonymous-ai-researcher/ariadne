"""Figure T16: are the sibling parameters properties of the text stream or of k and the embedding model?

Reads     D6_embedders_and_k/fits.json when it exists, else the law files D6_embedders_and_k/*.law.pkl fitted in
          process (_theory.quick_fit: same library calls as python -m ariadne.law fit, no bootstrap).  Cells are
          "<case>_<k>_s<seed>".  Other embedding models are read from sibling results directories
          "<results>_<EMB>/D6_embedders_and_k<suffix>/" (the layout that the experiment file prescribes); the model
          of a run is taken from the data path of its workload.
Shows     against k (log scale), one line per data set (colour) and embedding model (line style and marker: the first
          model keeps the data set's filled marker, the others have hollow markers of their own):
          (a) in-ball concurrent insertions per read, observed (lines) and the first-principles background
              n_c k / (N + 1) (thin dashed, same colour): the background grows like k, the sibling part does not
          (b) the sibling share of the in-ball insertions that follows from (a)
          (c) the fitted sibling dimension D_s (95% interval: over seeds with three or more, else the interval
              of the fit)
          (d) conflict probability after one in-ball insertion at c = --c: observed (markers, 95% interval) and
              predicted by the semi-parametric form with one sibling scale (lines); the markers of the embedding
              models are dodged slightly along k (factor 1.07 per model) so that they do not cover each other
Paper     appendix table or figure (robustness of the law).

Options   --c 1.1   --exp D6_embedders_and_k
"""
import glob, os, re

import numpy as np

from . import _theory as T

NAME = "fig_t_16_k_and_embedders"
LS = ["-", (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.2, 1, 1.2))]
# readable names of the embedding models (ids as they appear in the data paths)
EMB_LABEL = {"st-BAAI_bge-small-en-v1.5": "BGE-small", "st-sentence-transformers_all-MiniLM-L6-v2": "MiniLM-L6",
             "openai-text-embedding-3-small": "OpenAI 3-small", "wordllama": "WordLlama"}
EMB_MARK = [None, "^", "D", "v", "P"]   # marker per embedding model: the first keeps the data set's marker (filled), the others hollow
DODGE = 1.07                                                            # panel (d): horizontal offset factor between embedders


def emb_label(e):
    return EMB_LABEL.get(e, e)


def embedder_of(entry, path):
    wl = entry.get("workload") or {}
    m = re.search(r"__([^/]+?)\.npz$", str(wl.get("data", "")))
    if m:
        return m.group(1)
    if path and os.path.exists(path):
        m = re.search(r"__([^/]+?)\.npz$", str((T.law_load(path).get("workload") or {}).get("data", "")))
        if m:
            return m.group(1)
    return None


def collect(results, exp, suffix):
    """{(embedder, case): {k: [entries]}} over the results directory and its "<results>_<EMB>" siblings."""
    roots = [results] + sorted(d for d in glob.glob(results.rstrip("/") + "_*") if os.path.isdir(os.path.join(d, exp + suffix)))
    out = {}
    for root in roots:
        F, _ = T.fits_or_quick(root, exp, suffix, exp, NAME)
        laws = T.law_paths(root, exp, suffix)
        for name, e in F.items():
            if not isinstance(e, dict) or str(name).startswith("_"):
                continue
            case_k, _ = T.split_cell(name)
            m = re.match(r"^(.*)_(\d+)$", case_k)
            case = m.group(1) if m else case_k
            emb = embedder_of(e, laws.get(name)) or (root[len(results.rstrip("/")) + 1:] if root != results else "default")
            out.setdefault((emb, case), {}).setdefault(int(e["k"]), []).append(e)
    return out


def at_c(e, key, c):
    sg = e.get("single") or {}
    if not sg.get(key):
        return None
    j = [i for i, cc in enumerate(e["cs"]) if abs(cc - c) < 1e-9]
    if not j:
        return None
    v = sg[key][j[0]]
    return v if not isinstance(v, (list, tuple)) else v


def main(results="results", suffix="", out="figures", exp="D6_embedders_and_k", c=1.1):
    plt = T.setup()
    A = collect(results, exp, suffix)
    if not A:
        return T.skip(NAME, "no fits.json and no law files in %s" % T.exp_dir(results, exp, suffix))
    embs = []
    for emb, _ in A:
        if emb not in embs:
            embs.append(emb)
    fig, axes = plt.subplots(1, 4, figsize=(T.COL2, 2.05), sharex=True)
    ks_all = sorted({k for v in A.values() for k in v})
    for (emb, case), byk in sorted(A.items(), key=lambda kv: (T.ds_style(kv[0][1]).get("order", 9), embs.index(kv[0][0]))):
        st = T.ds_style(case); col = st["color"]; ei = embs.index(emb); ls = LS[ei % len(LS)]
        mfc = col if ei == 0 else "white"
        mk = EMB_MARK[ei % len(EMB_MARK)] or st["marker"]
        mks = 3.4 if mk != "D" else 3.0
        ks = sorted(byk)
        mm = [float(np.mean([e["mean_m"] for e in byk[k]])) for k in ks]
        lam = [float(np.mean([e["lam_bg"] for e in byk[k]])) for k in ks]
        share = [float(np.mean([1.0 - e["w_bg"] for e in byk[k]])) for k in ks]
        axes[0].plot(ks, mm, color=col, ls=ls, lw=T.LW, marker=mk, ms=mks, mfc=mfc, mec=col, mew=0.7, zorder=3)
        axes[0].plot(ks, lam, color=col, ls=(0, (2, 1.5)), lw=0.6, zorder=2)
        axes[1].plot(ks, [100 * s for s in share], color=col, ls=ls, lw=T.LW, marker=mk, ms=mks, mfc=mfc, mec=col, mew=0.7)
        dk, dm, dl, dh = [], [], [], []
        for k in ks:
            es = [e for e in byk[k] if e.get("f1") and e["f1"].get("Ds") and e["f1"]["Ds"][0] is not None and np.isfinite(e["f1"]["Ds"][0])]
            if es:
                ci = [(e["f1"].get("Ds_ci") or [[None, None]])[0] for e in es]
                m, lo, hi, _ = T.pooled_ci([e["f1"]["Ds"][0] for e in es], [x[0] for x in ci], [x[1] for x in ci])
                dk.append(k); dm.append(m); dl.append(lo); dh.append(hi)
        if dk:
            axes[2].plot(dk, dm, color=col, ls=ls, lw=T.LW, zorder=2)
            T.caps(axes[2], dk, dm, dl, dh, color=col, marker=mk, mfc=mfc, ms=mks, capsize=1.4)
        ck, co, cl, ch, cp = [], [], [], [], []
        for k in ks:
            es = [e for e in byk[k] if at_c(e, "obs", c) is not None]
            if es:
                ob = [at_c(e, "obs", c) for e in es]; ci = [at_c(e, "ci", c) or [None, None] for e in es]
                m, lo, hi, _ = T.pooled_ci(ob, [x[0] for x in ci], [x[1] for x in ci])
                pr = [at_c(e, "semi1", c) for e in es]
                ck.append(k); co.append(m); cl.append(lo); ch.append(hi); cp.append(float(np.mean([p for p in pr if p is not None])) if any(p is not None for p in pr) else np.nan)
        if ck:
            axes[3].plot(ck, cp, color=col, ls=ls, lw=T.LW, zorder=2)
            dx = DODGE ** (ei - (len(embs) - 1) / 2.0)                  # observed markers dodged per embedder (lines stay at k)
            T.caps(axes[3], [k * dx for k in ck], co, cl, ch, color=col, marker=mk, mfc=mfc, ms=mks, capsize=1.4)
        print("  %s: %-8s %-12s k = %s | D_s %s | in-ball per read %s" % (NAME, case, emb, ks, " ".join("%.2f" % v for v in dm), " ".join("%.3f" % v for v in mm)))
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks(ks_all); ax.set_xticklabels(["%d" % k for k in ks_all]); ax.minorticks_off()
        ax.set_xlim(min(ks_all) / 1.3, max(ks_all) * 1.3)
        ax.set_xlabel("$k$"); ax.grid(True, axis="y", color=T.LIGHT, lw=0.4)
    axes[0].set_yscale("log"); T.log_plain(axes[0], "y")
    axes[1].set_ylim(0, 100); axes[2].set_ylim(0, None); axes[3].set_ylim(0, None)
    for ax, letter, txt in zip(axes, "abcd", ["in-ball insertions per read", "sibling share (%)", "sibling dimension $D_s$", "conflict at $c$ = %g (%%)" % c]):
        T.panel(ax, letter, txt)
    from matplotlib.lines import Line2D
    cases = []
    for _, case in A:
        if T.ds_key(case) not in cases:
            cases.append(T.ds_key(case))
    from matplotlib.legend_handler import HandlerLine2D
    hs = [Line2D([], [], color=T.ds_style(d)["color"], marker=T.ds_style(d)["marker"], ms=3.4, lw=T.LW, label=T.ds_style(d)["label"]) for d in cases]
    hm = {}
    if len(embs) > 1:                                                   # embedders: long handles, markers at both ends, dashes visible between
        for i, e in enumerate(embs):
            h = Line2D([], [], color=T.MUTED, ls=LS[i % len(LS)], lw=T.LW, marker=EMB_MARK[i % len(EMB_MARK)] or "o", ms=2.8 if i else 2.8,
                       mfc=T.MUTED if i == 0 else "white", mec=T.MUTED, label=emb_label(e) + (" (marker of the data set)" if i == 0 else ""))
            hs.append(h); hm[h] = HandlerLine2D(numpoints=2)
    hs += [Line2D([], [], color=T.MUTED, ls=(0, (2, 1.5)), lw=0.6, label="first-principles background (a)"),
           Line2D([], [], color=T.MUTED, lw=T.LW, label="predicted, 1 scale (d)")]
    fig.legend(handles=hs, handler_map=hm, loc="outside upper center", ncol=min(len(hs), 4), handlelength=3.4, columnspacing=1.4,
               fontsize=T.SMALL)
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__, [("--exp", dict(default="D6_embedders_and_k")), ("--c", dict(type=float, default=1.1))])
