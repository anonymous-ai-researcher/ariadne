"""Figure S7: what SERIALIZABLE costs on a similarity read-think-write transaction.

Reads     A5_ssi_cost/summary.jsonl (aborts_per_task, llm_calls_per_task per protocol; the fan-out cases fan-c<n> are the
          contention levels) and A8_keyed_control/keyed.jsonl (the same check-then-insert pattern on a scalar key with a
          B-tree, READ COMMITTED and SERIALIZABLE, two table sizes).
Shows     horizontal bars per protocol: (a) aborts per task, (b) model calls per task (every retry of a transaction that
          thinks inside SERIALIZABLE is one more call); one bar per concurrency level, darker = more concurrent agents;
          caps are 95% intervals over seeds.  The keyed B-tree rows are the control: page-level predicate locks give a
          fraction of an abort per task.  The keyed test makes no model calls, so it has no bar in (b).
          The real-stream cases of A5 (A16, L4) are in the SERIALIZABLE table, not in this figure.
Paper     main text or appendix, with the table "SERIALIZABLE on similarity reads"; panel (c) of figure S8.
"""
import re

from . import style as st, load as ld

EXP, KEYED = "A5_ssi_cost", "A8_keyed_control"
NAME = "fig_s_07_ssi_cost"
ORDER = ["rc", "ssi", "ssi-backoff", "ssi-short", "lock-long", "lock-short", "av-exact", "av-cadm1.1"]
TABLE_CONC = 16          # the concurrency of the table "SERIALIZABLE on similarity reads" (tab:ssi); the numbers are at it
# row names of that table (av-exact is not in the table)
TABLE_NAMES = {"rc": "read committed", "ssi": "serializable, retry", "ssi-backoff": "serializable, backoff",
               "ssi-short": "serializable, short txn.", "lock-long": "global lock, whole txn.", "lock-short": "re-read under lock",
               "av-exact": "exact validation", "av-cadm1.1": "Ariadne, $c$=1.1"}
CONTROL_HATCH = "hollow"   # the B-tree control: grey, the larger table hollow (not hatched: hatch marks 32 agents)


def _cases(R):
    fan = [c for c in ld.values(R, "case") if re.match(r"^fan", str(c))]
    return fan or ld.values(R, "case")


def draw(ax_a, ax_c, results, suffix, label_case=None, names=None, control_hatch=None):
    """label_case: the case whose value is written next to each row (default: the largest over the cases);
    names: {proto: row name} (default: style.pretty_math); control_hatch: hatch of the larger table of the B-tree
    control, whose two bars are then grey, the larger table hollow, each with a grey mark at its end (default: light to dark
    grey, like the concurrency levels)."""
    R = ld.rows(results, EXP, suffix=suffix)
    K = ld.rows(results, KEYED, "keyed.jsonl", suffix, quiet=True)
    if not R and not K:
        return False
    cases = _cases(R)
    protos = [p for p in ORDER if ld.where(R, proto=p)] + [p for p in st.policy_order(ld.values(R, "proto")) if p not in ORDER]
    A = {m: ld.agg(ld.where(R, case=cases), ["proto", "case"], m) for m in ("aborts_per_task", "llm_calls_per_task")}
    n = max(len(cases), 1)
    rows, y = [], 0.0
    h = 0.78 / n
    for ax, metric in ((ax_a, "aborts_per_task"), (ax_c, "llm_calls_per_task")):
        y = 0.0; rows = []
        for p in protos:
            kw0 = st.policy_bar(p)
            top = None
            for j, c in enumerate(cases):
                s = A[metric].get((p, c))
                pos = y - 0.39 + (j + 0.5) * h
                if s is None or s.n == 0:
                    continue
                shade = 0.55 * (1 - (j + 1) / n)
                kw = dict(kw0); base = kw["color"] if kw["color"] != "white" else kw["edgecolor"]
                kw.update(color=st.lighten(base, shade), hatch=kw0.get("hatch") if j == n - 1 else "", edgecolor="white")
                err = [[s.mean - s.lo], [s.hi - s.mean]] if s.n > 1 and s.hi > s.lo else None
                ax.barh(pos, s.mean, height=h * 0.92, xerr=err, error_kw=dict(ecolor="#333333", elinewidth=0.6, capsize=1.2), **kw)
                top = (max(s.mean, top[0] if top else 0.0), max(s.hi if err else s.mean, top[1] if top else 0.0), y)
            if top is not None and label_case is not None:
                s = A[metric].get((p, label_case))
                top = (s.mean, top[1], y) if s is not None and s.n > 0 else None
            if top is not None:
                txt = "0" if top[0] == 0 else ("%.1f" % top[0] if top[0] < 10 else "%.0f" % top[0]) if metric.startswith("aborts") else "%.2f" % top[0]
                ax.annotate(txt, (top[1], y), xytext=(2, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=st.TINY, color=st.MUTED)
            rows.append((y, (names or {}).get(p) or st.pretty_math(p)))
            y += 1.0
        if K:
            y += 0.25
            for iso in ld.values(K, "iso"):
                KI = ld.where(K, iso=iso)
                sizes = sorted({r.get("table_rows", 0) // 1000 * 1000 or r.get("table_rows", 0) for r in KI})
                tabs = sorted(ld.values(KI, "table"), key=lambda t: min([ld.num(r.get("table_rows"), 0) for r in ld.where(KI, table=t)] or [0]))
                if metric == "aborts_per_task":
                    hk = 0.78 / max(len(tabs), 1)
                    notes, xmax, sizes_k = [], 0.0, []
                    for j, t in enumerate(tabs):
                        s = ld.stats([ld.num(r.get("aborts_per_task")) for r in ld.where(KI, table=t)])
                        pos = y - 0.39 + (j + 0.5) * hk
                        if control_hatch is None:
                            ckw = dict(color=st.lighten("#444444", 0.55 * (1 - (j + 1) / len(tabs))), edgecolor="white")
                        else:
                            # the two tables differ by fill, not by hatch (hatch marks the most concurrent protocol bar):
                            # small table solid grey, large table hollow grey; a grey mark at the end of each bar keeps
                            # bars of a fraction of an abort visible on the axis of the protocols
                            large = j == len(tabs) - 1 and j > 0
                            ckw = dict(color="white" if large else st.GREY, edgecolor=st.GREY)
                        ax.barh(pos, s.mean, height=hk * 0.92, linewidth=0.5, **ckw,
                                xerr=[[s.mean - s.lo], [s.hi - s.mean]] if s.n > 1 and s.hi > s.lo else None,
                                error_kw=dict(ecolor="#333333" if control_hatch is None else st.GREY, elinewidth=0.6, capsize=1.2))
                        if control_hatch is not None and s.mean > 0:
                            ax.plot([s.mean], [pos], ls="none", marker="s", ms=3.2, mfc="white" if large else st.GREY, mec=st.GREY,
                                    mew=0.8, zorder=4, clip_on=False)
                        rows_k = [ld.num(r.get("table_rows")) for r in ld.where(KI, table=t)]
                        size = "%gk rows" % round(sum(rows_k) / len(rows_k) / 1000) if rows_k else str(t)
                        xmax = max(xmax, s.hi if s.n > 1 else s.mean)
                        sizes_k.append(size.replace(" rows", ""))
                        if s.mean > 0:
                            notes.append("%.2g" % s.mean)
                    ax.annotate(", ".join(notes) if notes else "0", (xmax, y), xytext=(2 if control_hatch is None or not notes else 3.5, 0), textcoords="offset points",
                                ha="left", va="center", fontsize=st.TINY, color=st.MUTED)
                rows_all = sorted({round(ld.num(r.get("table_rows"), 0) / 1000) for r in KI})
                iso_name = ("SERIALIZABLE" if "SERIAL" in iso else "READ COMMITTED")
                if names is not None:
                    iso_name = iso_name.lower()
                rows.append((y, "B-tree key, " + iso_name +
                             ("\n(%s rows)" % ", ".join("%gk" % v for v in rows_all) if "SERIAL" in iso and rows_all else "")))
                y += 1.0
        ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[1] for r in rows])
        ax.set_ylim(y - 0.45, -0.6)
        ax.tick_params(axis="y", length=0)
        st.grid(ax, "x")
        ax.margins(x=0.12)
    if K and protos:
        for ax in (ax_a, ax_c):
            ax.axhline(len(protos) - 0.375, color=st.LIGHT, lw=0.5)
        ax_c.text(0.16, len(protos) + 0.25 + (len(ld.values(K, "iso")) - 1) / 2, "no model calls\nin the control", fontsize=st.TINY, color=st.MUTED,
                  ha="left", va="center", transform=ax_c.get_yaxis_transform())
    ax_c.axvline(1.0, color=st.MUTED, lw=0.5, ls=(0, (1, 1.5)), zorder=0.5)
    ax_a.set_xlabel("aborts per task"); ax_c.set_xlabel("model calls per task")
    ax_a.set_xlim(0, None); ax_c.set_xlim(0, None)
    return cases


def conc_note(cases):
    nums = [("%g" % ld.number_in(c)) for c in cases]
    if len(cases) <= 1:
        return "%s concurrent agents" % nums[0] if cases and nums[0] != "nan" else ""
    return "bars, light to dark: %s concurrent agents" % ", ".join(nums)


def main(results="results", suffix="", out="figures"):
    fig, (ax_a, ax_c) = st.subplots(1, 2, cols=1, aspect=0.74, sharey=True)
    R = ld.rows(results, EXP, suffix=suffix, quiet=True)
    ref = [c for c in _cases(R) if ld.number_in(c) == TABLE_CONC]
    ref = ref[0] if ref else None
    cases = draw(ax_a, ax_c, results, suffix, label_case=ref, names=TABLE_NAMES, control_hatch=CONTROL_HATCH)
    if cases is False:
        st.plt.close(fig)
        return ld.skip(NAME, "no A5 or A8 results")
    st.panel_label(ax_a, "a"); st.panel_label(ax_c, "b")
    note = conc_note(cases)
    if note and len(cases) > 1:
        note = "protocol bars, light to dark: %s concurrent agents" % ", ".join("%g" % ld.number_in(c) for c in cases)
        if ref is not None:
            note += "\nnumbers: the value at %g agents (as in the table)" % TABLE_CONC
        K = ld.rows(results, KEYED, "keyed.jsonl", suffix, quiet=True)
        kc = sorted({ld.num(ld.get(r, "params.conc")) for r in K} - {float("nan")})
        if K and len({r.get("table") for r in K}) > 1:
            note += "\nB-tree control%s: gray; small table solid, large table hollow" % (
                " (%g agents)" % kc[0] if len(kc) == 1 and kc[0] == kc[0] else "")
    if note:
        fig.supxlabel(note, fontsize=st.TINY, color=st.MUTED, x=0.99, ha="right")
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
