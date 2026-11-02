"""Figure S21: sensitivity to the retrieval depth k and to the duplicate threshold tau.

Reads     C12_k_tau_sensitivity/summary.jsonl  (grid kt = k<k> | tau<cosine threshold>, policy; extra_calls_per_100).
Shows     extra model calls per 100 tasks against k (left) and against tau (right), one row per workload, one line per
          policy; bands are intervals over seeds.  Values are placed at equal steps.  The default configuration
          (k = 10, tau = 0.1) is marked by a dotted vertical line in both sweeps; each sweep varies one parameter and
          keeps the other at its default.
Paper     appendix figure.
"""
import re

from . import style as st, load as ld, panels as pn

EXP = "C12_k_tau_sensitivity"
NAME = "fig_s_21_k_tau"
KEY = "protocol.policy"
DEFAULT = {"k": 10.0, "tau": 0.1}           # base configuration of the experiment (k10 and tau0.1 are the same setting)



def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    for r in R:
        m = re.match(r"^(k|tau)(\d+(?:\.\d+)?)$", str(r.get("kt")))
        r["_sweep"], r["_v"] = (m.group(1), float(m.group(2))) if m else (None, None)
    K = ld.where(R, _sweep="k")
    T = ld.where(R, _sweep="tau")
    sweeps = [(n, X, lab) for n, X, lab in (("k", K, "retrieval depth $k$"), ("tau", T, r"threshold $\tau$ (cosine)")) if X]
    cases = st.case_order(ld.values(R, "case"))
    pols = st.policy_order(ld.values(R, KEY))
    fig, axes = st.subplots(len(cases), len(sweeps), cols=1, aspect=0.50 * len(cases) + 0.12, sharey="row", squeeze=False,
                            gridspec_kw=dict(width_ratios=[max(len(ld.values(X, "_v")), 2) for _, X, _ in sweeps]))
    for j, c in enumerate(cases):
        for i, (n, X, lab) in enumerate(sweeps):
            ax = axes[j][i]
            vals = sorted(ld.values(X, "_v"))
            pos = pn.ordinal(ax, vals, lambda v: "%g" % v)
            d = DEFAULT.get(n)
            if d in pos:
                ax.axvline(pos[d], color=st.MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=0.8)
                if j == 0:
                    ax.text(pos[d], 1.0, "default", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=st.TINY, color=st.MUTED)
            pn.lines(ax, ld.where(X, case=c), "_v", "extra_calls_per_100", by=KEY, xpos=pos, kw_of=st.policy_line_distinct)
            st.grid(ax, "y")
            if j == len(cases) - 1:
                ax.set_xlabel(lab)
            else:
                ax.tick_params(labelbottom=False)
            if i == 0:
                ax.set_ylabel(("%s\n" % st.dataset_label(c, short=True) if len(cases) > 1 else "") + "extra calls per 100 tasks")
    pn.zero_floor([a for row in axes for a in row])
    pn.letters(axes[0])
    pn.policy_legend(fig, pols, kind="line_distinct")
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
