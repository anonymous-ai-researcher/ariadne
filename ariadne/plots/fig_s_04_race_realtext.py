"""Figure S4: the race on real text.

Reads     A3_race_realtext/audit_serial.jsonl (excess_stored_per_100_events of the duplicate-cluster cases, e.g. Quora Question Pairs)
          and A3_race_realtext/audit_sep.jsonl (redundant_per_100_tasks of the replayed streams: items stored within
          tau of an earlier stored item, which needs no ground-truth labels).
Shows     (a) duplicate rows per 100 facts on real duplicate clusters, per duplicate threshold;  (b) redundant writes
          per 100 tasks on real streams replayed in natural order.  One bar per protocol (no isolation, re-read,
          re-read under a lock); caps are 95% intervals over seeds.
Paper     appendix (the main text carries these numbers as one row group of the race table).
"""
import json
import os
import re

from . import style as st, load as ld

EXP = "A3_race_realtext"
NAME = "fig_s_04_race_realtext"


def config_tau(results, suffix, R, case):
    """Duplicate threshold of a case whose name does not carry it (the default), from the config of one of its cells."""
    for r in ld.where(R, case=case):
        p = os.path.join(ld.exp_dir(results, EXP, suffix), "%s.config.json" % r.get("cell"))
        try:
            with open(p) as f:
                t = json.load(f).get("tau_cos")
            if t is not None:
                return float(t)
        except (OSError, ValueError, TypeError):
            continue
    return None


def cluster_label(case, default_tau=None):
    m = re.search(r"tau(\d+(?:\.\d+)?)", case)
    base = st.dataset_style(case)["label"]
    if m:
        return base + "\n" + r"$\tau$=" + "%g" % float(m.group(1))
    if default_tau is not None:
        return base + "\n" + r"$\tau$=" + "%g" % default_tau + "\n(default)"
    return base


def main(results="results", suffix="", out="figures"):
    R = ld.rows(results, EXP, suffix=suffix)
    S = ld.rows(results, EXP, "audit_sep.jsonl", suffix, quiet=True)
    X = ld.rows(results, EXP, "audit_serial.jsonl", suffix, quiet=True)
    if not R:
        return ld.skip(NAME, "no rows")
    clusters = [c for c in ld.values(R, "case") if any(r.get("dup_per_100_events") is not None for r in ld.where(R, case=c))]
    streams = [c for c in st.case_order(ld.values(S, "case")) if c not in clusters]
    protos = st.policy_order(ld.values(R, "proto"))
    panels = [p for p in (("clusters", clusters), ("streams", streams)) if p[1]]
    if not panels:
        return ld.skip(NAME, "no cluster or stream cases")
    fig, axes = st.subplots(1, len(panels), cols=1, aspect=0.60, squeeze=False,
                            gridspec_kw=dict(width_ratios=[max(len(p[1]), 1.6) for p in panels]))
    for i, (ax, (kind, cases)) in enumerate(zip(axes[0], panels)):
        if kind == "clusters":
            # excess over the serial execution of the same tasks: real paraphrases can be farther apart than tau
            A = ld.agg(X, ["case", "proto"], "excess_stored_per_100_events")
            ax.set_ylabel("rows beyond the serial\nexecution per 100 facts")
            labels = [cluster_label(c, config_tau(results, suffix, R, c)) for c in cases]
        else:
            A = ld.agg(S, ["case", "proto"], "redundant_per_100_tasks")
            ax.set_ylabel("redundant writes per 100 tasks")
            labels = [cluster_label(c) if "tau" in c else st.dataset_label(c).replace(", ", "\n") for c in cases]
        st.grouped_bars(ax, cases, protos, lambda g, s: A.get((g, s)), st.policy_bar)
        # a bar whose seeds agree exactly has an interval of width 0 and hence no caps: say so, so that it does not
        # read as a missing interval
        bw = 0.82 / max(len(protos), 1)
        for gi, g in enumerate(cases):
            for j, p in enumerate(protos):
                s = A.get((g, p))
                if s is not None and s.n > 1 and s.mean == s.mean and s.hi == s.lo:
                    ax.annotate(u"\u00b10", (gi - 0.41 + (j + 0.5) * bw, s.mean), xytext=(0, 1.0), textcoords="offset points",
                                ha="center", va="bottom", fontsize=st.TINY, color=st.MUTED)
        ax.set_xticks(range(len(cases))); ax.set_xticklabels(labels)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(cases) - 0.4); st.floor0(ax)
        st.grid(ax, "y")
        if len(panels) > 1:
            st.panel_label(ax, "ab"[i], "duplicate clusters" if kind == "clusters" else "replayed streams")
    fig.legend(handles=st.legend_handles(protos, "bar"), loc="outside upper center", ncol=len(protos), handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
