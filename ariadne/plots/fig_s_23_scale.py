"""Figure S23: scale (size of the memory, length of the run, number of agents).

Reads     C9_scalability/summary.jsonl  (cases <data>-<initial items> for the size sweep, plus <data>-long (long stream)
          and <data>-<agents> (many agents); extra_calls_per_100, cs_vread_mean_ms, tps per policy and seed).
Shows     (a) extra model calls per 100 tasks, (b) mean time of the validation read (validating protocols only: no
          isolation has no validation read), (c) throughput (log axis: the 512-agent case runs far faster), per case;
          the cases of one data set's size sweep are joined by a line, the other cases are single marks.  The size
          sweep keeps 64 agents and 6,000 tasks; the 512-agent and long-stream cases start from the initial memory
          stored in their run configuration (<cell>.config.json: workload.base, workload.agents, workload.stream),
          which their tick labels state.  One line per policy; bands are intervals over seeds.
Paper     appendix figure "scale" (two columns), one sentence in the main text.
"""
import json
import os
import re

from . import style as st, load as ld, panels as pn

EXP = "C9_scalability"
NAME = "fig_s_23_scale"
KEY = "protocol.policy"
SHORT = {"agnews": "AG", "msmarco": "MSM", "locomo": "LoCoMo", "qqp": "QQP", "longmemeval": "LME", "synthetic": ""}


def workload_cfg(results, suffix, R, case):
    """workload section of the stored configuration of one run of a case ({} if unavailable)."""
    for r in R:
        if r.get("case") == case:
            p = os.path.join(ld.exp_dir(results, EXP, suffix), "%s.config.json" % r.get("cell"))
            try:
                with open(p) as f:
                    return json.load(f).get("workload", {}) or {}
            except (OSError, ValueError):
                return {}
    return {}


def describe(case, wl=None):
    """(group, sort value, tick label) of a case name; wl: its workload configuration (initial items, agents, tasks)."""
    ds = SHORT[st.dataset_key(case)]
    wl = wl or {}
    base = pn.fmt_num(wl["base"]) if wl.get("base") else None
    m = re.match(r"^[a-zA-Z]+-(\d+)([kKmM]?)$", str(case))
    if m:
        n = ld.number_in(case)
        if n >= 1000:
            return (ds + " size", n, "%s\n%s" % (ds, pn.fmt_num(n)))
        return (ds + " agents", n, "%s\n%s\n%g ag" % (ds, base, n) if base else "%s\n%g ag" % (ds, n))
    tail = str(case).split("-", 1)[-1]
    return (ds + " " + tail, 0, "%s\n%s\n%s" % (ds, base, tail) if base else "%s\n%s" % (ds, tail))


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    wls = {c: workload_cfg(results, suffix, R, c) for c in ld.values(R, "case")}
    info = {c: describe(c, wls[c]) for c in wls}
    order = sorted(info, key=lambda c: (not info[c][0].endswith("size"), st.DATASETS[st.dataset_key(c)]["order"], info[c][0], info[c][1]))
    pos = {c: i for i, c in enumerate(order)}
    pols = st.policy_order(ld.values(R, KEY))
    wide = len(order) > 4
    fig, axes = st.subplots(1, 3, cols=2 if wide else 1, aspect=0.30 if wide else 0.52)
    for ax, (key, lab) in zip(axes, (("extra_calls_per_100", "extra calls per 100 tasks"), ("cs_vread_mean_ms", "validation read (ms)"),
                                     ("tps", "tasks per second"))):
        only = [p for p in pols if st.family(p) != "none"] if key == "cs_vread_mean_ms" else None
        for g in dict.fromkeys(info[c][0] for c in order):
            cs = [c for c in order if info[c][0] == g]
            pn.lines(ax, ld.where(R, case=cs), "case", key, by=KEY, xpos=pos, only=only)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([info[c][2] for c in order], fontsize=st.TINY if wide else st.SMALL)
        ax.set_xlim(-0.4, len(order) - 0.6)
        ax.set_ylabel(lab); st.grid(ax, "y")
        if key == "tps":
            ax.set_yscale("log"); ax.autoscale(enable=True, axis="y")
            lo, hi = ax.get_ylim(); ax.set_ylim(lo / 1.15, hi * 1.15)
            st.log_ticks(ax, "y", steps=(1, 2, 3, 5))
        else:
            st.floor0(ax)
        if only is not None and len(only) < len(pols):
            ax.text(0.99, 0.03, "no isolation: no\nvalidation read", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=st.TINY, color=st.MUTED)
    pn.letters(axes)
    size = [wls[c] for c in order if info[c][0].endswith("size") and wls[c]]
    longs = [wls[c] for c in order if str(c).endswith("-long") and wls[c].get("stream")]
    note = ["ag: agents"]
    if size and size[0].get("agents") and size[0].get("stream"):
        note.insert(0, "size sweep: %s agents, %s tasks" % (pn.fmt_num(size[0]["agents"]), pn.fmt_num(size[0]["stream"])))
    note.append("long: %s-task stream" % pn.fmt_num(longs[0]["stream"]) if longs else "long: long stream")
    fig.supxlabel("data set and number of initial items (%s)" % "; ".join(note), fontsize=st.FONT)
    pn.policy_legend(fig, pols)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
