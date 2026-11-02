"""Figure S9: extra model calls against the tolerance c.

Reads     C1_calls_vs_policy/summary.jsonl (extra_calls_per_100 per workload and policy): the rows behind the cost table
          and behind the numbers quoted in the text, so that figure, table and text agree.  (C7_choosing_c has a finer
          grid of c for two workloads; it is drawn in fig_s_24.)
Shows     extra model calls per 100 tasks of c-admissible validation against c (log axis in c - 1), one line per
          workload (colour and marker of the data set; lighter shades for fewer agents), bands = 95% intervals over
          seeds.  Reference marks in their own columns: exact (re-execution) validation on the left, the overlap
          (recall-style) test on the right.
Paper     main text, figure "extra calls vs c" (one column).
"""
import math

from . import style as st, load as ld

EXPS = ("C1_calls_vs_policy",)
NAME = "fig_s_09_calls_vs_c"
KEY = "protocol.policy"


def load(results, suffix):
    R = []
    for e in EXPS:
        R += ld.rows(results, e, suffix=suffix, quiet=(e != EXPS[0]))
    return [r for r in R if r.get("extra_calls_per_100") is not None and r.get(KEY) is not None]


def draw(ax, R, value="extra_calls_per_100", refs=("exact", "overlap:0.9"), labels=True):
    cases = st.case_order(ld.values(R, "case"))
    cad = [r for r in R if st.family(r[KEY]) == "cadm"]
    if not cad:
        return False
    cs = sorted({st.spec_c(r[KEY]) for r in cad})
    left = ["exact"] if "exact" in refs and ld.where(R, **{KEY: "exact"}) else []
    over = [p for p in refs if st.family(p) == "identity" and ld.where(R, **{KEY: p})]
    right = [st.pretty(p).replace("overlap ", "overlap\n") for p in over]
    pos = st.c_axis(ax, cs, left=left, right=right)
    lines = st.case_lines(cases)
    data = []
    for c in cases:
        xs, S = ld.series(ld.where(cad, case=c), KEY, value, xnum=st.spec_c)
        if not xs:
            continue
        ex = ld.stats([ld.num(r.get(value)) for r in ld.where(R, case=c, **{KEY: "exact"})]) if left else None
        ov = [(name, ld.stats([ld.num(r.get(value)) for r in ld.where(R, case=c, **{KEY: p})])) for p, name in zip(over, right)]
        data.append((c, xs, S, ex, ov))
    # marks that would hide each other in one column are moved sideways by whole marker widths (greedy, in the order of
    # the data sets): the reference columns and the first tolerance, where the workloads bunch together
    ys = [s.hi for _, _, S, ex, ov in data for s in S + ([ex] if ex is not None and ex.n else []) + [o for _, o in ov if o.n] if s.n]
    fig = ax.figure
    bb = ax.get_position()
    w_pt, h_pt = bb.width * fig.get_size_inches()[0] * 72, bb.height * fig.get_size_inches()[1] * 72
    x0, x1 = ax.get_xlim()
    dec_per_pt = math.log10(x1 / x0) / w_pt
    y_per_pt = (max(ys) * 1.1 if ys else 1.0) / h_pt
    gap_pt = st.MS + 0.9

    def dodge(col_x, marks):
        placed, out = [], {}
        for key, y in marks:
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                if all(pk != k or abs(py - y) >= gap_pt * y_per_pt for pk, py in placed):
                    break
            placed.append((k, y))
            out[key] = col_x * 10 ** (k * gap_pt * dec_per_pt)
        return out

    x_ex = dodge(pos("exact"), [(c, ex.mean) for c, _, _, ex, _ in data if ex is not None and ex.n]) if left else {}
    x_first = dodge(pos(min(cs)), [(c, S[0].mean) for c, xs, S, _, _ in data if abs(xs[0] - min(cs)) < 1e-9])
    x_ov = {name: dodge(pos(name), [(c, o.mean) for c, _, _, _, ov in data for n, o in ov if n == name and o.n]) for name in right}
    ends = []
    for c, xs, S, ex, ov in data:
        kw = lines[c]
        px = [x_first.get(c, pos(x)) if i == 0 else pos(x) for i, x in enumerate(xs)]
        st.line_ci(ax, px, S, **kw)
        y_end = S[-1].mean
        mk = dict(kw, ls="none")
        if ex is not None and ex.n:
            st.point_ci(ax, x_ex[c], ex, **mk)
            ax.plot([x_ex[c], px[0]], [ex.mean, S[0].mean], color=kw["color"], lw=0.5, ls=(0, (1, 1.5)), zorder=1)
        for name, s in ov:
            if s.n:
                st.point_ci(ax, x_ov[name][c], s, **dict(mk, zorder=3, clip_on=False))      # the series' own marker and fill; the column says which test
                y_end = s.mean
        ends.append((0, y_end, st.dataset_label(c, short=True), kw["color"]))
    st.floor0(ax)
    ax.set_ylabel("extra model calls per 100 tasks")
    st.grid(ax, "y")
    if labels and len(ends) <= 4:
        st.end_labels(ax, ends)
    elif labels:
        hs = [st.plt.Line2D([], [], label=st.dataset_label(c, short=True), **lines[c]) for c in cases]
        ax.figure.legend(handles=hs, loc="outside upper center", ncol=3, handlelength=2.4, columnspacing=1.2)
    return True


def main(results="results", suffix="", out="figures"):
    R = load(results, suffix)
    fig, ax = st.subplots(cols=1, aspect=0.66)
    if not R or not draw(ax, R):
        st.plt.close(fig)
        return ld.skip(NAME, "no c-admissible rows in C1 / C7")
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
