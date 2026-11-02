"""Figure S24: choosing the tolerance c.

Reads     C7_choosing_c/summary.jsonl joined with audit_run.jsonl  (live sweep of c on a fine grid, plus exact:
          extra_calls_per_100; div_committed_vs_exact_% per decision proxy; rho_fresh quantiles of the index).
Shows     per workload (rows): (left) extra model calls per 100 tasks against c, (right) share of committed decisions
          that differ from the decision on the exact answer, one line per decision proxy, on a symmetric-log axis.
          Vertical lines: the 90th and 99th percentile of the index's own approximation ratio (rho_fresh: per task the
          worst ratio of its validation reads against exact k-NN; percentile over all audited tasks of a run, tasks at
          ratio 1 included; mean over the runs of the workload, all tolerances and seeds of C7), drawn when they fall inside the swept range and always printed in the corner of the left panel:
          a tolerance below them buys no safety that the index does not already give away.  The marks in the
          "exact" column are exact (re-execution) validation.  Bands are intervals over seeds.
Paper     appendix figure with the guidance paragraph on choosing c (the main-text figure S9 shows the coarse grid).
"""
from . import style as st, load as ld, panels as pn

EXP = "C7_choosing_c"
NAME = "fig_s_24_choosing_c"
KEY = "protocol.policy"
PROXIES = ("set", "top3", "tag", "link", "dup")
PSTYLE = {"set": ("#1A1A1A", "s", "-"), "top3": ("#4D4D4D", "^", (0, (4, 1.5))), "tag": ("#777777", "D", (0, (4, 1.2, 1, 1.2))),
          "link": ("#999999", "o", (0, (1, 1))), "dup": ("#B8B8B8", "v", (0, (3, 1, 1, 1, 1, 1)))}
INDEX_COLOR = st.MUTED     # the index percentiles: neutral (orange is the colour of AG News)
PNAME = {"set": "full set", "top3": "top-3 set", "tag": "majority tag", "link": "link target", "dup": "duplicate test"}


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    R = [r for r in R if r.get(KEY) is not None]
    cad = [r for r in R if st.family(r[KEY]) == "cadm"]
    if not cad:
        return ld.skip(NAME, "no c-admissible rows")
    cases = st.case_order(ld.values(R, "case"))
    cs = sorted({st.spec_c(r[KEY]) for r in cad})
    has_exact = bool(ld.where(R, **{KEY: "exact"}))
    has_div = any(isinstance(r.get("div_committed_vs_exact_%"), dict) for r in cad)
    ncol = 2 if has_div else 1
    fig, axes = st.subplots(len(cases), ncol, cols=1, aspect=0.50 * len(cases) + 0.16, sharex=True, squeeze=False)
    for j, c in enumerate(cases):
        Rc, Cc = ld.where(R, case=c), ld.where(cad, case=c)
        q = {}
        for name in ("p90", "p99"):
            s = ld.stats([ld.num(ld.get(r, "rho_fresh." + name)) for r in Rc])
            if s.n and s.mean > 1.0:
                q[name] = s.mean
        for i in range(ncol):
            ax = axes[j][i]
            pos = st.c_axis(ax, cs, left=["exact"] if has_exact else [], label=False, min_sep=0.62 if ncol > 1 else None, step=5.5 if ncol > 1 else 2.6)
            real = [x - 1.0 for x in cs if x > 1.0]
            lo, hi = (min(real) / 1.3, max(real) * 1.3) if real else (1.0, 0.0)
            if i == 0 and q:
                # what the numbers are: per run, the 90th / 99th percentile over all audited tasks (unconditional:
                # tasks at ratio 1 included) of rho_fresh, the ratio of the task's validation read(s) (worst read of the
                # task) against exact k-NN; then the mean over the runs of the workload (all tolerances and seeds)
                ax.text(0.04, 0.03, "index alone,\nall tasks:\n" + "\n".join("%s = %.3f" % (n_, v_) for n_, v_ in q.items()), transform=ax.transAxes, ha="left", va="bottom", linespacing=1.1,
                        fontsize=st.TINY, color=INDEX_COLOR)
            for name, v in q.items():
                if lo < v - 1.0 < hi:
                    ax.axvline(v - 1.0, color=INDEX_COLOR, lw=0.6, ls=(0, (1, 1.3)), zorder=1)
                    if i == 0:
                        ax.annotate("index " + name, (v - 1.0, 1.0), xycoords=("data", "axes fraction"), xytext=(1.5, -1), textcoords="offset points",
                                    rotation=90, ha="left", va="top", fontsize=st.TINY, color=INDEX_COLOR)
            if i == 0:
                xs, S = ld.series(Cc, KEY, "extra_calls_per_100", xnum=st.spec_c)
                kw = st.dataset_line(c)
                st.line_ci(ax, [pos(x) for x in xs], S, **kw)
                if has_exact:
                    s = ld.stats([ld.num(r.get("extra_calls_per_100")) for r in ld.where(Rc, **{KEY: "exact"})])
                    if s.n:
                        st.point_ci(ax, pos("exact"), s, **st.policy_marker("exact"))
                st.floor0(ax)
                ax.set_ylabel(("%s\n" % st.dataset_label(c, short=True) if len(cases) > 1 else "") + "extra calls per 100 tasks")
            else:
                ends = []
                for p in PROXIES:
                    col, mk, ls = PSTYLE[p]
                    val = lambda r, p=p: (r.get("div_committed_vs_exact_%") or {}).get(p)
                    xs, S = ld.series(Cc, KEY, val, xnum=st.spec_c)
                    if not xs:
                        continue
                    st.line_ci(ax, [pos(x) for x in xs], S, color=col, marker=mk, ls=ls, ms=2.8, lw=0.9, mfc=col, mec=col)
                    if has_exact:
                        s = ld.stats([ld.num(val(r)) for r in ld.where(Rc, **{KEY: "exact"})])
                        if s.n:
                            st.point_ci(ax, pos("exact"), s, color=col, marker=mk, ms=2.8, mfc="white", mec=col, ls="none")
                    ends.append((pos(xs[-1]), S[-1].mean, PNAME[p], col))
                ax.set_yscale("symlog", linthresh=0.5, linscale=0.5)
                top = max([e[1] for e in ends] + [1.0])
                ax.set_ylim(-0.08, top * 1.6)
                from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
                ticks = [t for t in (0, 0.5, 1, 2, 5, 10, 20, 50, 100) if t <= top * 1.6]
                ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v)); ax.yaxis.set_minor_locator(NullLocator())
                ax.set_ylabel("decisions that differ (%)")
                st.end_labels(ax, ends, fontsize=st.TINY)
            st.grid(ax, "y")
    pn.letters(axes[0])
    fig.supxlabel("tolerance $c$  (log scale in $c-1$)", fontsize=st.FONT)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
