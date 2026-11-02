"""Figure S13: robustness to the think-time distribution.

Reads     C8_think_and_arrival/summary.jsonl  (cases think = <distribution>-<mean seconds>: fixed, exp, logn0.3, logn1.0,
          pareto; extra_calls_per_100 per policy, workload, seed).
Shows     extra model calls per 100 tasks against the mean think time (log-log), one panel per workload.  Colour and
          line style give the policy, the marker gives the distribution: each policy is a bundle of one line per
          distribution, and the bundles stay apart (the saving does not depend on how variably agents think).
Paper     appendix figure, left half of "think time and arrivals" (the open-loop half is figure S12).
"""
import re

from . import style as st, load as ld, panels as pn

EXP = "C8_think_and_arrival"
NAME = "fig_s_13_think_time"
KEY = "protocol.policy"
DIST = {"fixed": ("fixed", "o"), "exp": ("exponential", "s"), "logn0.3": (r"lognormal, $\sigma$=0.3", "^"),
        "logn1.0": (r"lognormal, $\sigma$=1", "v"), "pareto": ("Pareto (heavy tail)", "D")}


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    for r in R:
        d, _, m = str(r.get("think")).rpartition("-")
        r["_dist"], r["_mean"] = d, ld.num(m)
    cases = st.case_order(ld.values(R, "case"))
    pols = st.policy_order(ld.values(R, KEY))
    dists = [d for d in DIST if ld.where(R, _dist=d)] + [d for d in ld.values(R, "_dist") if d not in DIST]
    fig, axes = st.subplots(1, len(cases), cols=1 if len(cases) <= 2 else 2, aspect=0.70 if len(cases) <= 2 else 0.36, sharey=True, squeeze=False)
    means = sorted(ld.values(R, "_mean"))
    for i, (ax, c) in enumerate(zip(axes[0], cases)):
        for d in dists:
            mk = DIST.get(d, (d, "x"))[1]
            pn.lines(ax, ld.where(R, case=c, _dist=d), "_mean", "extra_calls_per_100", by=KEY, band=False,
                     kw_of=lambda p, mk=mk: st.policy_line(p, marker=mk, ms=2.8, lw=0.8, mfc="white", mew=0.7))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(means[0] / 1.6, means[-1] * 1.6) if len(means) > 1 else ax.set_xlim(means[0] / 3, means[0] * 3)
        ax.set_xticks(means); ax.set_xticklabels(["%g" % m for m in means]); ax.xaxis.set_minor_locator(st.plt.NullLocator())
        st.grid(ax, "y")
        st.panel_label(ax, "abcdef"[i], st.dataset_label(c))
        ax.set_xlabel("mean think time (s)")
    st.log_ticks(axes[0][0], "y")
    axes[0][0].set_ylabel("extra model calls per 100 tasks")
    from matplotlib.lines import Line2D
    H = [Line2D([], [], label=st.pretty_math(p), **st.policy_line(p, marker="none")) for p in pols]
    H += [Line2D([], [], ls="none", marker=DIST.get(d, (d, "x"))[1], mfc="white", mec=st.INK, ms=3.2, label=DIST.get(d, (d, "x"))[0]) for d in dists]
    fig.legend(handles=H, loc="outside upper center", ncol=3 if len(cases) <= 2 else 8, handlelength=2.0, columnspacing=1.0)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
