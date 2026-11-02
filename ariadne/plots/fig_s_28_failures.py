"""Figure S28: clients that fail inside the critical section.

Reads     F6_failure_injection/summary.jsonl joined with audit_run.jsonl  (grid: store, faults.crash_in_cs = probability
          that a client fails after its validation read and before its write, policy; tps, lat_p99_s, crashes;
          dup_written_per_100 and safety_theorem.violations from the audit).
Shows     against the crash probability: (a) throughput, (b) 99th percentile of the task latency, (c) duplicate writes
          per 100 tasks (expected to stay at 0: a failed client rolls back and starts over).  Colour gives the policy;
          filled marks and solid lines are the first store, hollow marks and dashed lines the second.  The stores are
          dodged slightly along x and the first store is drawn on top, so that coinciding values stay visible.
Paper     appendix table or figure, one sentence in the prototype section (failure handling).
"""
from . import style as st, load as ld, panels as pn

EXP = "F6_failure_injection"
NAME = "fig_s_28_failures"
KEY, X = "protocol.policy", "faults.crash_in_cs"


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    stores = ld.values(R, "store")
    pols = st.policy_order(ld.values(R, KEY))
    probs = sorted(ld.values(R, X))
    fig, axes = st.subplots(1, 3, cols=1, aspect=0.52)
    for ax, (key, lab) in zip(axes, (("tps", "tasks per second"), ("lat_p99_s", "p99 latency (s)"), ("dup_written_per_100", "duplicate writes\nper 100 tasks"))):
        pos = pn.ordinal(ax, probs, lambda v: "%g" % (100 * float(v)))
        n = len(stores)
        for i in reversed(range(n)):                       # the first store (filled) last, i.e. on top
            s = stores[i]
            dx = (0.5 * (n - 1) - i) * -0.09 if n > 1 else 0.0
            kw_of = lambda p, i=i: st.policy_line(p, zorder=3.2 if i == 0 else 2.4, **({} if i == 0 else dict(mfc="white", ls=(0, (4, 1.5)))))
            pn.lines(ax, ld.where(R, store=s), X, key, by=KEY, xpos={k: v + dx for k, v in pos.items()}, kw_of=kw_of)
        ax.set_ylabel(lab); st.floor0(ax); st.grid(ax, "y")
    if max(ld.num(r.get("dup_written_per_100"), 0) for r in R) == 0:
        axes[2].set_ylim(-0.05, 1.0)
    pn.letters(axes)
    fig.supxlabel("clients failing inside the critical section (%)", fontsize=st.FONT)
    from matplotlib.lines import Line2D
    H = st.legend_handles(pols, "line")
    if len(stores) > 1:
        H += [Line2D([], [], color=st.MUTED, marker="o", mfc=st.MUTED if i == 0 else "white", mec=st.MUTED, ls="-" if i == 0 else (0, (4, 1.5)),
                     ms=st.MS, label=st.store_label(s)) for i, s in enumerate(stores)]
    else:
        H += [Line2D([], [], ls="none", label="store: " + st.store_label(stores[0]))]
    fig.legend(handles=H, loc="outside upper center", ncol=4, handlelength=2.2)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
