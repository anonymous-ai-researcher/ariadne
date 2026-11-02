"""Figure S16: where the time inside the critical section goes.

Reads     G1_critical_section/summary.jsonl  (cs_vread_mean_ms, cs_predicate_mean_ms, cs_write_mean_ms, cs_mean_ms,
          lock_wait_mean_ms per configuration k<k>-ef<ef> | k<k>-exact and number of agents).
Shows     (a) stacked bars of the mean critical-section time: validation read, predicate test, write, and the rest
          (lock round trips, bookkeeping), one bar per configuration and number of agents; the predicate test is
          microseconds and its range is printed because it is too thin to see;  (b) mean wait for the commit lock.
          Caps are intervals over seeds (on the total in (a)).
Paper     prototype or evaluation section (small stacked-bar figure), one column.
"""
import re

from . import style as st, load as ld

EXP = "G1_critical_section"
NAME = "fig_s_16_critical_section"
PARTS = [("cs_vread_mean_ms", "validation read", st.OKABE_ITO["blue"], ""), ("cs_predicate_mean_ms", "predicate test", st.OKABE_ITO["orange"], ""),
         ("cs_write_mean_ms", "write", "#8C8C8C", "///"), ("_rest", "other", "#D9D9D9", "")]


def cfg_label(kb):
    m = re.match(r"^k(\d+)-(ef(\d+)|exact)$", str(kb))
    if not m:
        return str(kb)
    return "$k$=%s, %s" % (m.group(1), "exact" if m.group(2) == "exact" else "ef=%s" % m.group(3))


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    for r in R:
        r["_rest"] = max(ld.num(r.get("cs_mean_ms"), 0) - sum(ld.num(r.get(k), 0) for k, _, _, _ in PARTS[:3]), 0.0)
    cfgs = ld.values(R, "kb")
    agents = sorted(ld.values(R, "workload.agents"))
    n = len(agents)
    fig, (ax, ax_w) = st.subplots(1, 2, cols=1, aspect=min(0.30 + 0.065 * len(cfgs) * max(n, 1.5), 1.2), sharey=True,
                                  gridspec_kw=dict(width_ratios=[2.2, 1.0]))
    h = 0.80 / n
    for i, cfg in enumerate(cfgs):
        for j, a in enumerate(agents):
            Rc = ld.where(R, kb=cfg, **{"workload.agents": a})
            if not Rc:
                continue
            y = i - 0.40 + (j + 0.5) * h
            left = 0.0
            for key, lab, col, hatch in PARTS:
                s = ld.stats([ld.num(r.get(key)) for r in Rc])
                ax.barh(y, s.mean, left=left, height=h * 0.90, color=col, edgecolor="white", linewidth=0.4, hatch=hatch)
                left += s.mean
            tot = ld.stats([ld.num(r.get("cs_mean_ms")) for r in Rc])
            if tot.n > 1 and tot.hi > tot.lo:
                ax.errorbar([tot.mean], [y], xerr=[[tot.mean - tot.lo], [tot.hi - tot.mean]], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.2)
            ax.annotate(("%g agents" % a) if (n > 1 and i == 0) else "", (max(left, tot.hi), y), xytext=(2.5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=st.TINY, color=st.MUTED)
            w = ld.stats([ld.num(r.get("lock_wait_mean_ms")) for r in Rc])
            ax_w.barh(y, w.mean, height=h * 0.90, color=st.lighten("#444444", 0.5 * (1 - (j + 1) / n)), edgecolor="white", linewidth=0.4,
                      xerr=[[w.mean - w.lo], [w.hi - w.mean]] if w.n > 1 and w.hi > w.lo else None,
                      error_kw=dict(ecolor="#333333", elinewidth=0.6, capsize=1.2))
    ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([cfg_label(c) for c in cfgs])
    ax.set_ylim(len(cfgs) - 0.45, -0.55)
    for a_ in (ax, ax_w):
        a_.tick_params(axis="y", length=0); st.grid(a_, "x")
    ax.margins(x=0.05)
    ax.set_xlabel("time in the critical section (ms)")
    if max(ld.num(r.get("lock_wait_mean_ms"), 0) for r in R) > 30 * max(min(ld.num(r.get("lock_wait_mean_ms"), 0) for r in R), 1e-3):
        ax_w.set_xscale("log")
        # a narrow panel over many decades: a label every second decade, as 10^k, the other decades as minor ticks
        import math
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        lo_, hi_ = ax_w.get_xlim()
        ks = list(range(math.ceil(math.log10(lo_) - 1e-9), math.floor(math.log10(hi_) + 1e-9) + 1))
        major = [k for k in ks if (k - ks[-1]) % 2 == 0] if len(ks) > 3 else ks
        ax_w.xaxis.set_major_locator(FixedLocator([10.0 ** k for k in major]))
        ax_w.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "$\\mathregular{10^{%d}}$" % round(math.log10(v))))
        ax_w.xaxis.set_minor_locator(FixedLocator([10.0 ** k for k in ks if k not in major])); ax_w.xaxis.set_minor_formatter(NullFormatter())
    ax_w.set_xlabel("lock wait (ms)")
    st.panel_label(ax, "a"); st.panel_label(ax_w, "b")
    pr = [ld.num(r.get("cs_predicate_mean_ms")) for r in R]
    pr = [1000 * p for p in pr if p == p]
    from matplotlib.patches import Patch
    H = [Patch(facecolor=col, edgecolor="white", hatch=hatch, label=lab + (" (%.0f to %.0f µs)" % (min(pr), max(pr)) if key == "cs_predicate_mean_ms" and pr else ""))
         for key, lab, col, hatch in PARTS]
    if n > 1:                            # key of the grey shades of (b)
        H += [Patch(facecolor=st.lighten("#444444", 0.5 * (1 - (j + 1) / n)), edgecolor="white", label="(b) %g agents" % a)
              for j, a in enumerate(agents)]
    fig.legend(handles=H, loc="outside upper center", ncol=2, handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
