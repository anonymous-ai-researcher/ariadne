"""Figure S15: contract violations against the search budget of the index.

Reads     C6_index_operating_points/summary.jsonl joined with audit_run.jsonl  (rho_committed[">c"], rho_fresh[">c"]:
          share of committed answers, and of validation reads, whose approximation ratio against exact k-NN exceeds
          the tolerance c of the run; grid: index, budget ef<n> of the reads, validation variant).
Shows     share of committed answers that break the c-approximation contract against the search budget, one panel per
          index (columns) and workload (rows).  Lines: c-admissible validation with the read budget, with a large
          validation budget, with exact validation (certified, 0 by the theorem), exact re-execution, no isolation.
          The grey dashed line is what the index alone delivers (rho_fresh of the same-budget runs): approximate
          validation follows it, c * rho_F being the guarantee.  Bands are intervals over seeds.
Paper     main text or appendix, figure "contract violations vs search budget".
"""
from . import style as st, load as ld, panels as pn

EXP = "C6_index_operating_points"
NAME = "fig_s_15_contract_vs_budget"
VARIANTS = {   # val name -> (spec for the style, label, extra style)
    "none":   ("none", "no isolation", {}),
    "reexec": ("exact", "exact (re-execution)", {}),
    "same":   ("cadm:1.1", "c-adm., validation with the read budget", {}),
    "ef400":  ("cadm:1.1", "c-adm., validation with ef = 400", dict(mfc="white", ls=(0, (4, 1.5)))),
    "exact":  ("cert:1.1", "certified (exact validation)", dict(ms=3.0, zorder=7, clip_on=False)),   # at 0: on top of the axis and of ef = 400
}
INDEX = {"hnsw-M8": "HNSW, M = 8", "hnsw-M16": "HNSW, M = 16", "hnsw-M32": "HNSW, M = 32", "ivfflat": "IVFFlat (probes scale with ef)"}


def tol(r):
    c = ld.get(r, "safety_theorem.c")
    return float(c) if c else 1.1


def viol(field):
    def f(r):
        d = r.get(field) or {}
        key = ">%s" % tol(r)
        if key not in d:
            key = ">1.1"
        return d.get(key)
    return f


def kw_of(v):
    spec, _, extra = VARIANTS.get(v, (v, v, {}))
    return st.policy_line(spec, **extra)


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    R = [r for r in R if isinstance(r.get("rho_committed"), dict)]
    if not R:
        return ld.skip(NAME, "no rows with the run audit")
    cases = st.case_order(ld.values(R, "case"))
    idx = [i for i in INDEX if ld.where(R, index=i)] + [i for i in ld.values(R, "index") if i not in INDEX]
    budgets = ld.values(R, "budget")
    vals = [v for v in VARIANTS if ld.where(R, val=v)] + [v for v in ld.values(R, "val") if v not in VARIANTS]
    nr, nc = len(cases), len(idx)
    fig, axes = st.subplots(nr, nc, cols=1 if nc <= 2 else 2, aspect=(0.62 if nc == 1 else 0.40 if nc == 2 else 0.25) * nr + 0.05,
                            sharex=True, sharey="row", squeeze=False)
    for j, c in enumerate(cases):
        for i, ix in enumerate(idx):
            ax = axes[j][i]
            Rp = ld.where(R, case=c, index=ix)
            pos = pn.ordinal(ax, budgets, lambda b: "%g" % ld.number_in(b))
            F = ld.where(Rp, val="same") or Rp
            A = ld.agg(F, "budget", viol("rho_fresh"))
            pts = [(pos[b], s) for b, s in A.items() if b in pos and s.n]
            if pts:
                # on top, thin and dashed, cased in white: it runs on the c-admissible curve that follows it
                import matplotlib.patheffects as pe
                ax.plot([p[0] for p in pts], [p[1].mean for p in pts], color="#5E5E5E", lw=0.7, ls=(0, (2, 1.2)), zorder=6.5,
                        path_effects=[pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()])
            n0 = len(ax.lines)
            pn.lines(ax, Rp, "budget", viol("rho_committed"), by="val", xpos=pos, kw_of=kw_of, order=lambda vs: [v for v in vals if v in vs])
            for ln in list(ax.lines[n0:]):       # the marks once more above the index line, which runs between them
                if ln.get_marker() not in ("None", "none", None, ""):
                    ax.plot(ln.get_xdata(), ln.get_ydata(), ls="none", marker=ln.get_marker(), ms=ln.get_markersize(),
                            mfc=ln.get_markerfacecolor(), mec=ln.get_markeredgecolor(), mew=ln.get_markeredgewidth(),
                            zorder=max(ln.get_zorder(), 6.6), clip_on=ln.get_clip_on())
            st.grid(ax, "y")
            if j == 0 and nc > 1:
                st.panel_label(ax, "abcdefgh"[i], INDEX.get(ix, ix))
            if i == 0:
                ax.set_ylabel(("%s\n" % st.dataset_label(c, short=True) if nr > 1 else "") + "answers above $c$ (%)")
    pn.zero_floor([a for row in axes for a in row])
    for ax in axes[-1]:
        ax.set_xlabel("search budget of reads (ef)")
    from matplotlib.lines import Line2D
    H = [Line2D([], [], label=VARIANTS.get(v, (v, v, {}))[1].replace("c-adm.", "$c$-adm."), **kw_of(v)) for v in vals]
    import matplotlib.patheffects as pe
    H.append(Line2D([], [], color="#5E5E5E", lw=0.7, ls=(0, (2, 1.2)), label="the index alone (validation reads)",
                    path_effects=[pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()]))
    fig.legend(handles=H, loc="outside upper center", ncol=1 if nc == 1 else 2 if nc == 2 else 3, handlelength=2.2)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
