"""Figure S3: the race against every workload factor, one at a time (small multiples).

Reads     A2_race_sensitivity/summary.jsonl  (cases <factor><value>: jitter, think, fan, tau, k, hot, base, conc).
Shows     duplicate rows per 100 facts under no isolation against each factor around the default point; shared
          y axis except the fan-out panel (its own scale); bands are 95% intervals over seeds.  Values are placed at equal steps (the tick labels give them).
Paper     appendix, companion of the arrival-spread figure (two columns).
"""
from . import style as st, load as ld
from .fig_s_02_race_spread import factor_rows, EXP

NAME = "fig_s_03_race_factors"
FACTORS = [("jitter", "arrival spread (think times)"), ("think", "think time (s)"), ("fan", "fan-out per fact"),
           ("conc", "target concurrency"), ("tau", r"duplicate threshold $\tau$ (cosine)"), ("k", "retrieval depth $k$"),
           ("hot", "hot anchors (topic locality)"), ("base", "initial memory size")]


OWN_SCALE = "fan"           # the factor whose panel gets its own y range


def tick(v):
    try:
        x = float(v)
    except ValueError:
        return {"mix": "mixed"}.get(v, v)
    if x >= 1000:
        return "%gk" % (x / 1000)
    return "%g" % x


def main(results="results", suffix="", out="figures"):
    R = [r for r in ld.rows(results, EXP, suffix=suffix) if r.get("dup_per_100_events") is not None]
    present = [(f, lab) for f, lab in FACTORS if factor_rows(R, f)]
    if not present:
        return ld.skip(NAME, "no cases")
    n = len(present)
    ncol = min(4, n); nrow = (n + ncol - 1) // ncol
    fig, axes = st.subplots(nrow, ncol, cols=2 if ncol > 2 else 1, aspect=(0.22 if ncol > 2 else 0.5) * nrow + 0.02, squeeze=False)
    flat = [a for row in axes for a in row]
    top = {}
    for i, (ax, (f, lab)) in enumerate(zip(flat, present)):
        F = factor_rows(R, f)
        vals = ld.natsorted({r["_value"] for r in F})
        A = ld.agg(F, "_value", "dup_per_100_events")
        S = [A[v] for v in vals]
        top[f] = max(max(x.hi for x in S), max(x.mean for x in S))
        xs = list(range(len(vals)))
        st.line_ci(ax, xs, S, color=st.INK, marker="o", ms=3.0, lw=1.1)
        ax.set_xticks(xs); ax.set_xticklabels([tick(v) for v in vals])
        ax.set_xlim(-0.4, max(len(vals) - 0.6, 0.6))
        ax.set_xlabel(lab)
        st.grid(ax, "y")
        st.panel_label(ax, "abcdefgh"[i])
        if f == OWN_SCALE:
            ax.text(0.12, 1.03, "own y scale", transform=ax.transAxes, ha="left", va="bottom", fontsize=st.SMALL, color=st.MUTED)
        if i % ncol == 0:
            ax.set_ylabel("duplicates per 100 facts")
    for ax in flat[n:]:
        ax.set_visible(False)
    # one y range for every panel but fan-out, whose largest value (several hundred) would flatten the others
    shared = max([v for f, v in top.items() if f != OWN_SCALE] or [1.0])
    for ax, (f, lab) in zip(flat, present):
        hi = top[f] if f == OWN_SCALE else shared
        ax.set_ylim(0, hi * 1.06 if hi > 0 else 1.0)
    fig.supxlabel("the values on each x axis are equally spaced steps (ordinal), not to scale",
                  fontsize=st.SMALL, color=st.MUTED)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
