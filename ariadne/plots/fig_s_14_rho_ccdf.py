"""Figure S14: distribution of the approximation ratio of committed answers.

Reads     C1_calls_vs_policy/summary.jsonl and the per-task arrays C1_calls_vs_policy/<cell>.audit.npz
          (rho_committed: ratio of the committed answer against exact k-NN in the state at its commit position;
          rho_fresh: the same for the validation read, i.e. what the index delivers by itself).
Shows     complementary CDF (share of committed tasks whose ratio exceeds x) per policy, tasks pooled over seeds, on a
          log axis in (ratio - 1).  Vertical lines mark each tolerance c in the colour of its curve: a c-admissible
          curve may pass its line only as far as the index floor does.  The grey area is the floor, the CCDF of
          rho_fresh (pooled over the policies shown).  One panel per workload.
Paper     main text or appendix, next to the main cost table (the table reports the p99 of these curves).
"""
import numpy as np

from . import style as st, load as ld

EXP = "C1_calls_vs_policy"
NAME = "fig_s_14_rho_ccdf"
KEY = "protocol.policy"
PREFERRED = ("L4", "A64", "L8", "A16")


def chosen(policies):
    """The policies drawn: no isolation, exact, overlap 90%, every c-admissible tolerance, one certified variant."""
    out = [p for p in policies if st.family(p) in ("none", "exact", "cadm") and p != "set"]
    out += [p for p in policies if p == "overlap:0.9"]
    cert = sorted([p for p in policies if st.family(p) == "cert"], key=lambda p: abs(st.spec_c(p) - 1.1))
    return st.policy_order(out + cert[:1])


def draw(ax, results, suffix, R, case, lo=1.002, hi=None, n_axes=1):
    Rc = ld.where(R, case=case)
    pols = chosen(ld.values(Rc, KEY))
    fresh, top, n_min = [], lo, None
    curves = []
    vlines = {}                                            # c -> colour of its first line (cert. c=1.1 sits on c-adm. c=1.1)
    for p in pols:
        rows = ld.where(Rc, **{KEY: p})
        v = ld.pooled(results, EXP, rows, "rho_committed", suffix)
        f = ld.pooled(results, EXP, rows, "rho_fresh", suffix)
        v = v[np.isfinite(v)]
        if f.size:
            fresh.append(f[np.isfinite(f)])
        if v.size:
            curves.append((p, v)); top = max(top, float(v.max()))
            n_min = v.size if n_min is None else min(n_min, v.size)
    if not curves:
        return False
    hi = hi or max(1.05, min(1.0 + (top - 1.0) * 1.25, 5.0))
    grid = 1.0 + np.logspace(np.log10(lo - 1.0), np.log10(hi - 1.0), 400)
    floor_y = 100.0 / max(n_min, 1)
    if fresh:
        f = np.concatenate(fresh)
        y = np.array([100.0 * np.mean(f > g) for g in grid])
        ax.fill_between(grid - 1.0, np.maximum(y, floor_y / 3), floor_y / 3, color="#E3E3E3", lw=0, zorder=1, step="post")
        ax.plot(grid - 1.0, np.where(y > 0, y, np.nan), color="#A0A0A0", lw=0.6, zorder=1.2, drawstyle="steps-post")
    for p, v in curves:
        y = np.array([100.0 * np.mean(v > g) for g in grid])
        kw = st.policy_line(p, marker="none")
        ax.plot(grid - 1.0, np.where(y > 0, y, np.nan), drawstyle="steps-post", zorder=3, **kw)
        c = st.spec_c(p)
        if st.family(p) in ("cadm", "cert") and c and lo < c < hi and round(c, 6) not in vlines:
            # one line per value of c, in the colour of its first curve (the label above the axes has that colour too)
            ax.axvline(c - 1.0, color=kw["color"], lw=0.6, ls=(0, (1, 1.2)), zorder=2)
            vlines[round(c, 6)] = kw["color"]
    st.ratio_axis(ax, lo, hi, label=None, min_sep=0.6 if n_axes > 1 else None)
    ax.set_yscale("log")
    ax.set_ylim(floor_y / 1.6, 130)
    st.log_ticks(ax, "y", steps=(1,))
    st.grid(ax, "y")
    if vlines:                                             # the value of c of every vertical line, on top of the axes
        top = ax.secondary_xaxis("top")
        cs = sorted(vlines)
        top.set_xticks([c - 1.0 for c in cs], ["%g" % c for c in cs], minor=False)
        top.set_xticks([], minor=True)
        top.tick_params(axis="x", which="major", length=1.5, pad=1.0, labelsize=st.TINY, labelrotation=90)
        for t, c in zip(top.get_xticklabels(), cs):
            t.set_color(vlines[c])
        top.spines["top"].set_visible(False)
        ax.text(0.0, 1.0, "$c$:", transform=ax.transAxes, ha="right", va="bottom", fontsize=st.TINY, color=st.MUTED)
    return pols


def main(results="results", suffix="", out="figures"):
    R = [r for r in ld.rows(results, EXP, suffix=suffix) if r.get(KEY) is not None]
    if not R or not ld.npz(results, EXP, suffix):
        return ld.skip(NAME, "no summary or no <cell>.audit.npz files")
    cases = ld.values(R, "case")
    pick = [c for c in PREFERRED if c in cases][:2] or st.case_order(cases)[:2]
    pick = st.case_order(pick)
    fig, axes = st.subplots(1, len(pick), cols=1, aspect=0.66, sharey=True, sharex=True, squeeze=False)
    top = 1.05
    for c in pick:                                         # one x range for all panels
        for r in ld.where(R, case=c):
            if r.get(KEY) in chosen(ld.values(ld.where(R, case=c), KEY)):
                a = ld.cell_arrays(results, EXP, r, suffix)
                if a is not None and len(a.get("rho_committed", [])):
                    v = np.asarray(a["rho_committed"], dtype=float)
                    top = max(top, float(v[np.isfinite(v)].max()))
    hi = max(1.05, min(1.0 + (top - 1.0) * 1.25, 5.0))
    pols = []
    for i, (ax, c) in enumerate(zip(axes[0], pick)):
        got = draw(ax, results, suffix, R, c, hi=hi, n_axes=len(pick))
        if got is False:
            st.no_data(ax)
            continue
        pols = pols if len(pols) >= len(got) else got
        st.panel_label(ax, "ab"[i], st.dataset_label(c), dy=1.17)
    if not pols:
        st.plt.close(fig)
        return ld.skip(NAME, "no per-task arrays for the chosen workloads")
    axes[0][0].set_ylabel("answers above the ratio (%)")
    fig.supxlabel("approximation ratio of the held answer (log scale in ratio $-$ 1)", fontsize=st.FONT)
    from matplotlib.patches import Patch
    H = st.legend_handles(pols, "line", label=lambda p: st.pretty_math(p, short=True))
    for h in H:
        h.set_marker("none")
    H.append(Patch(facecolor="#E3E3E3", edgecolor="#A0A0A0", linewidth=0.6, label="fresh answers (index floor)"))
    fig.legend(handles=H, loc="outside upper center", ncol=4, handlelength=2.0, columnspacing=1.0)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
