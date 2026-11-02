"""Figure S10: the trade-off frontier of validation predicates (counterfactual replay).

Reads     C2_counterfactual_pareto/first_dense.jsonl when it exists (dense grid of c, written by
              python -m ariadne.audit first results/C2_counterfactual_pareto/*_s[0-9].pkl --cs 1.005,1.01,... \
                     --out results/C2_counterfactual_pareto/first_dense.jsonl
          (the glob *_s[0-9].pkl leaves out the <cell>.law.pkl files of the same directory, which are not run files),
          else C2_counterfactual_pareto/audit_first.jsonl.  One line per instrumented run: for every predicate replayed
          on the logged (read answer, validation answer) pairs, redo_% and accepted_div_vs_exact_% per decision proxy;
          floor_fresh_vs_exact_% is the index floor.  --dense FILE reads another file of the same format.
Shows     x = model calls saved relative to exact validation, 100 * (redo(exact) - redo(p)) / redo(exact);
          y = share of accepted decisions that differ from the decision on the exact answer.  One panel per decision
          proxy (dup, tag, link, top3, set).  The c-admissible predicate traces a curve over c (colour ramp, strict to
          loose); the single-rank distance tests trace thin curves; overlap and top-j are points; the star is the
          oracle that knows the panel's decision function.  The thick grey line is the lower-right frontier (the
          non-dominated predicates, oracles excluded); the dotted horizontal line is the index floor.
          Values are means over the seeds of one workload.
Writes    fig_s_10_pareto (the primary workload) and, with several workloads, fig_s_10_pareto_all (one row per workload).
Paper     main text, figure "Pareto frontier" (two columns); the all-workloads grid goes to the appendix.
"""
import argparse, os

from . import style as st, load as ld

EXP = "C2_counterfactual_pareto"
NAME = "fig_s_10_pareto"
PROXIES = ("dup", "tag", "link", "top3", "set")
PRIMARY = ("A64", "L4", "L8", "A16")            # the main-text figure is captioned A64
CURVES = ("cadm", "radius", "rank1", "sumdist")
DEGENERATE = {"tag": "no tag labels"}           # text of a panel whose values are all 0


def read(results, suffix, dense=None):
    if dense:
        if not os.path.exists(dense):
            ld.note("dense file %s not found" % dense)
            return []
        return [r for r in ld.read_jsonl(dense) if isinstance(r.get("policies"), dict)]
    for fn in ("first_dense.jsonl", "audit_first.jsonl"):
        R = [r for r in ld.rows(results, EXP, fn, suffix, quiet=True) if isinstance(r.get("policies"), dict)]
        if R:
            return R
    ld.note("%s: no first_dense.jsonl or audit_first.jsonl" % EXP)
    return []


def case_of(r):
    if r.get("case"):
        return r["case"]
    cell = str(r.get("cell") or r.get("name") or r.get("run") or "run")
    return os.path.basename(cell).split("_")[0]


def points(rows):
    """{policy: {"x": saved %, proxy: divergence %}} averaged over the rows (seeds) of one workload, and the floor."""
    acc, floor = {}, {p: [] for p in PROXIES}
    for r in rows:
        P = r["policies"]
        base = ld.num((P.get("exact") or {}).get("redo_%"))
        if not base > 0:
            continue
        for pol, d in P.items():
            a = acc.setdefault(pol, {"x": [], **{p: [] for p in PROXIES}})
            a["x"].append(100.0 * (base - ld.num(d.get("redo_%"))) / base)
            for p in PROXIES:
                a[p].append(ld.num((d.get("accepted_div_vs_exact_%") or {}).get(p)))
        for p in PROXIES:
            floor[p].append(ld.num((r.get("floor_fresh_vs_exact_%") or {}).get(p)))
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return ({pol: {k: mean(v) for k, v in a.items()} for pol, a in acc.items()}, {p: mean(v) for p, v in floor.items()})


def frontier(pts):
    """Non-dominated points for (maximise x, minimise y), sorted by x."""
    out = []
    for x, y in sorted(set(pts), key=lambda t: (-t[0], t[1])):
        if not out or y < out[-1][1] - 1e-12:
            out.append((x, y))
    return out[::-1]


def draw(ax, P, floor, proxy, annotate=False):
    cand = [(d["x"], d[proxy]) for pol, d in P.items() if st.family(pol) != "oracle" and d[proxy] == d[proxy]]
    if not max([y for _, y in cand] + [0.0]) > 0:
        # degenerate panel (every value 0, e.g. a data set without tag labels): only the note, on a unit axis
        ax.set_xlim(-6, 106)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(-0.04, 1.10)
        ax.set_yticks([])
        ax.text(0.5, 0.5, DEGENERATE.get(proxy, "all values are 0"), transform=ax.transAxes, ha="center", va="center",
                fontsize=st.SMALL, color=st.MUTED, zorder=7)
        return True
    F = frontier(cand)
    if len(F) > 1:
        ax.plot([p[0] for p in F], [p[1] for p in F], color="#D0D0D0", lw=3.2, solid_capstyle="round", solid_joinstyle="round", zorder=1)
    if floor.get(proxy) == floor.get(proxy):
        ax.axhline(floor[proxy], color=st.MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1.5)
    top = max([y for _, y in cand] + [1e-9])
    labels = []
    for name in CURVES:                                     # curves over c
        pols = sorted([p for p in P if st.parse(p)[1] == name], key=st.spec_c)
        if not pols:
            continue
        xs = [P[p]["x"] for p in pols]; ys = [P[p][proxy] for p in pols]
        if name == "cadm":
            ax.plot(xs, ys, color=st.c_color(1.15), lw=1.0, zorder=3)
            c_all = [st.spec_c(p) for p in pols]
            pick = ({c_all[0], c_all[-1]} | ({1.1} if 1.1 in c_all else set())) if annotate else set()
            for p, x, y in zip(pols, xs, ys):
                if st.spec_c(p) in pick:      # a labelled point is drawn above the single marks (a little smaller, so that
                    ax.plot([x], [y], zorder=5.2, **st.policy_marker(p, ms=2.9, mec="white", mew=0.45))   # a mark under it still shows)
                else:
                    ax.plot([x], [y], zorder=4, **st.policy_marker(p, ms=3.4, mec="white", mew=0.3))
            labels += [(x, y, "c=%g" % st.spec_c(p), st.INK) for p, x, y in zip(pols, xs, ys) if st.spec_c(p) in pick]
        else:
            kw = st.policy_line(pols[0], lw=0.6, ms=2.2, mew=0.4, alpha=0.9)
            ax.plot(xs, ys, zorder=2, **kw)
    for pol in P:                                            # single points
        fam, name, arg = st.parse(pol)
        if name in CURVES or pol in ("set",):
            continue
        if fam == "oracle" and arg != proxy:
            continue
        if fam == "app":
            continue
        # the oracle star is large and drawn behind the other marks (above the curves): where it coincides with another
        # mark (exact validation on the full set, top-1 on link, top-3 on the top-3 set) that mark sits inside the star
        if fam == "oracle":
            ax.plot([P[pol]["x"]], [P[pol][proxy]], zorder=3.5, **st.policy_marker(pol, ms=9.5, mew=0.6))
        else:
            on_star = any(abs(P[o]["x"] - P[pol]["x"]) < 1.5 and abs(P[o][proxy] - P[pol][proxy]) < 0.01 * max(1e-9, top)
                          for o in P if st.parse(o)[0] == "oracle" and st.parse(o)[2] == proxy)
            ax.plot([P[pol]["x"]], [P[pol][proxy]], zorder=5, **st.policy_marker(pol, ms=4.0, **(dict(mec="white", mew=0.5) if on_star else {})))
    ax.set_xlim(-6, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    top = max([y for _, y in cand] + [1e-9])
    ax.set_ylim(-0.04 * top, top * 1.10)
    if labels:
        # labels of marks that sit on the floor of the panel (all near the x axis, as in the duplicate test) would
        # crowd each other: they go up on staggered leader lines; the others are placed around their marks
        low = sorted([l for l in labels if l[1] < 0.25 * top], key=lambda l: l[0])
        for i, (x, y, text, color) in enumerate(low):
            ha = "left" if x < 12 else ("right" if x > 88 else "center")     # keep clear of the axis spines
            ax.annotate(text, (x, y), xytext=(-2 if ha == "left" else 0, 12 + 9 * i), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=st.TINY, color=color, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=st.MUTED, lw=0.4, shrinkA=0, shrinkB=2.5))
        rest = [l for l in labels if l[1] >= 0.25 * top]
        if rest:
            st.label_points(ax, rest, obstacles=cand, prefer=("nw", "w", "sw", "ne", "n", "s"))
    return True


def legend(fig, P, **kw):
    from matplotlib.lines import Line2D
    H = []
    pols = list(P)
    cad = sorted([p for p in pols if st.family(p) == "cadm"], key=st.spec_c)
    if cad:
        mid = cad[len(cad) // 2]
        H.append(Line2D([], [], label=r"$c$-admissible (ours), $c$ from %g to %g" % (st.spec_c(cad[0]), st.spec_c(cad[-1])),
                        **st.policy_line(mid, color=st.c_color(1.15))))
    for p in st.policy_order([p for p in pols if st.family(p) in ("exact", "none") and p != "set"]):
        H.append(Line2D([], [], label=st.pretty(p) + (" validation" if p == "exact" else ""), **st.policy_marker(p, ms=4.0)))
    for p in sorted([p for p in pols if st.family(p) == "identity"], key=lambda p: (st.parse(p)[1] != "overlap", st.parse(p)[1], -ld.num(st.parse(p)[2], 0))):
        H.append(Line2D([], [], label=st.pretty(p), **st.policy_marker(p, ms=4.0)))
    for name, lab in (("radius", "radius only"), ("rank1", "rank 1 only"), ("sumdist", "sum of distances")):
        ps = [p for p in pols if st.parse(p)[1] == name]
        if ps:
            H.append(Line2D([], [], label=lab + r" (over $c$)", **st.policy_line(ps[0], lw=0.6, ms=2.6)))
    if any(st.family(p) == "oracle" for p in pols):
        H.append(Line2D([], [], label="oracle (knows the decision)", **st.policy_marker("oracle:link", ms=7.5, mew=0.6)))
    H.append(Line2D([], [], color="#D0D0D0", lw=3.2, label="frontier"))
    H.append(Line2D([], [], color=st.MUTED, lw=0.6, ls=(0, (1, 1.5)), label="index floor"))
    return fig.legend(handles=H, loc="outside upper center", ncol=kw.pop("ncol", 5), handlelength=1.8, columnspacing=1.1, **kw)


def main(results="results", suffix="", out="figures", dense=None):
    R = read(results, suffix, dense)
    if not R:
        return ld.skip(NAME, "no first-validation audit")
    by = {}
    for r in R:
        by.setdefault(case_of(r), []).append(r)
    data = {c: points(rows) for c, rows in by.items()}
    data = {c: d for c, d in data.items() if d[0]}
    if not data:
        return ld.skip(NAME, "no run with a positive redo rate under exact validation")
    cases = st.case_order(data)
    primary = next((c for c in PRIMARY if c in data), cases[0])
    written = []
    # ---- the primary workload: one row of five panels
    P, floor = data[primary]
    fig, axes = st.subplots(1, len(PROXIES), cols=2, aspect=0.285)
    for i, (ax, proxy) in enumerate(zip(axes, PROXIES)):
        draw(ax, P, floor, proxy, annotate=(i == 0))
        st.panel_label(ax, "abcde"[i], st.PROXY_LABEL[proxy])
    axes[0].set_ylabel("accepted decisions that differ\nfrom the exact answer's (%)")
    fig.supxlabel("model calls saved relative to exact validation (%)", fontsize=st.FONT)
    print("  %s: primary workload %s (%s)" % (NAME, primary, st.dataset_label(primary)))
    legend(fig, P)
    written += st.save(fig, NAME, out)
    # ---- all workloads: one row each
    if len(cases) > 1:
        n = len(cases)
        fig, axes = st.subplots(n, len(PROXIES), cols=2, aspect=min(0.06 + 0.185 * n, 1.32), sharex=True, squeeze=False)
        for j, c in enumerate(cases):
            P, floor = data[c]
            for i, proxy in enumerate(PROXIES):
                ax = axes[j][i]
                draw(ax, P, floor, proxy)
                if j == 0:
                    st.panel_label(ax, "abcde"[i], st.PROXY_LABEL[proxy])
            axes[j][0].set_ylabel(st.dataset_label(c, short=True), fontsize=st.SMALL)
        fig.supxlabel("model calls saved relative to exact validation (%)", fontsize=st.FONT)
        fig.supylabel("accepted decisions that differ (%)", fontsize=st.FONT)
        legend(fig, data[primary][0])
        written += st.save(fig, NAME + "_all", out)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results"); ap.add_argument("--suffix", default=""); ap.add_argument("--out", default="figures")
    ap.add_argument("--dense", default=None, help="a first-validation audit file with a dense grid of c (see the docstring)")
    a = ap.parse_args()
    main(a.results, a.suffix, a.out, a.dense)
