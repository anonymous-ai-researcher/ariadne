"""Figure S2: duplicates against the arrival spread of the copies of a fact.

Reads     A2_race_sensitivity/summary.jsonl, cases jitter<x> (dup_per_100_events, dup_per_100_events_by_fan).
Shows     duplicate rows per 100 facts under no isolation against the arrival spread of the copies (in units of the
          mean think time, log axis): all facts (band: 95% interval over seeds) and split by fan-out.  The race
          disappears only when the copies arrive much further apart than one think time.
Paper     main text, figure "duplicates vs arrival spread" (one column).
"""
import re

from . import style as st, load as ld

EXP = "A2_race_sensitivity"
NAME = "fig_s_02_race_spread"


def factor_rows(R, factor):
    out = []
    for r in R:
        m = re.match(r"^(jitter|think|fan|tau|k|hot|base|conc)(.+)$", str(r.get("case")))
        if m and m.group(1) == factor:
            out.append(dict(r, _value=m.group(2)))
    return out


def main(results="results", suffix="", out="figures"):
    R = factor_rows(ld.rows(results, EXP, suffix=suffix), "jitter")
    R = [r for r in R if r.get("dup_per_100_events") is not None]
    if not R:
        return ld.skip(NAME, "no jitter cases")
    fig, ax = st.subplots(cols=1, aspect=0.58)
    fans = sorted({f for r in R for f in (r.get("dup_per_100_events_by_fan") or {})}, key=float)
    fans = [f for f in fans if float(f) > 1]
    shades = {f: st.darken(st.GREY, 0.10 + 0.35 * i / max(len(fans) - 1, 1)) for i, f in enumerate(fans)}
    dashes = [(0, (1, 1.2)), (0, (3.5, 1.5)), (0, (5, 1, 1, 1)), (0, (2, 1))]
    handles = []
    for i, f in list(enumerate(fans))[::-1]:
        xs, S = ld.series(R, "_value", lambda r, f=f: (r.get("dup_per_100_events_by_fan") or {}).get(f), xnum=float)
        if xs:
            handles.append(st.line_ci(ax, xs, S, band=False, color=shades[f], lw=0.9, ls=dashes[i % 4], marker="os^D"[i % 4], ms=2.6,
                                      mfc="white", mew=0.6, label="facts reported by %s agents" % f))
    xs, S = ld.series(R, "_value", "dup_per_100_events", xnum=float)
    handles.insert(0, st.line_ci(ax, xs, S, color=st.INK, lw=1.5, marker="o", ms=3.4, mfc=st.INK, mec=st.INK, label="all facts"))
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(["%g" % x for x in xs]); ax.minorticks_off()
    st.floor0(ax)
    if min(xs) <= 1 <= max(xs):
        ax.axvline(1.0, color=st.MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=0.5)
        ax.annotate("one think time", (1.0, 0.0), xycoords=("data", "axes fraction"), xytext=(2.5, 2.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=st.TINY, color=st.MUTED)
    ax.set_xlabel("arrival spread of the copies of a fact (think times)")
    ax.set_ylabel("duplicate rows per 100 facts")
    st.grid(ax, "y")
    ax.legend(handles=handles, loc="upper right", handlelength=2.6)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
