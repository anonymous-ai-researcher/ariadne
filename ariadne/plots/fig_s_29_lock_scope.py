"""Figure S29: the scope of the commit lock.

Reads     E2_lock_scope/summary.jsonl joined with audit_run.jsonl  (grid: protocol.lock = global | group | none, policy;
          one case per workload and number of agents; lock_wait_mean_ms, tps; dup_written_per_100 from the audit).
Shows     per workload, one bar per lock scope: (a) mean wait for the commit lock, (b) throughput, (c) duplicate writes
          per 100 tasks.  Only the global lock is sound in general: one lock per team on a shared store and no lock at
          all are the narrower scopes whose cost in duplicates panel (c) shows.  One row per policy (titled above the
          row); colour gives the policy as everywhere else, fill and hatch the lock scope (the legend is drawn in grey
          because it encodes only the scope).  Case labels give the total number of agents and of teams (LoCoMo has
          one team per conversation; the agents per team are read from the stored run configuration).  Caps are
          intervals over seeds.
Paper     ablation (appendix figure or a row group of the ablation table).
"""
import json
import os

from . import style as st, load as ld, panels as pn

EXP = "E2_lock_scope"
NAME = "fig_s_29_lock_scope"
KEY, LOCK = "protocol.policy", "protocol.lock"
SCOPES = ["global", "group", "none"]
LABEL = {"global": "global lock (sound)", "group": "one lock per team", "none": "no lock"}


def row_title(policy):
    fam = st.family(policy)
    if fam == "exact":
        return "exact (re-execution)"
    c = st.spec_c(policy)
    if fam == "cadm" and c is not None:
        return "$c$-adm., $c$ = %g" % c
    return st.pretty_math(policy)


def case_label(results, suffix, R, case):
    """Data set, total number of agents and number of teams of a case ("LoCoMo 20, 10 teams")."""
    lab = st.dataset_label(case, short=True)
    if "teams" in lab:
        return lab
    for r in ld.where(R, case=case):
        try:
            with open(os.path.join(ld.exp_dir(results, EXP, suffix), "%s.config.json" % r.get("cell"))) as f:
                per_team = int(json.load(f)["workload"]["agents"])
            clients = int(r["clients"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if per_team > 0 and clients % per_team == 0:
            return "%s %d, %d teams" % (st.DATASETS[st.dataset_key(case)]["label"], clients, clients // per_team)
    return lab


def scope_bar(policy=None):
    """Bar style of a lock scope; with policy=None, in neutral greys (legend: it encodes only the scope)."""
    base = st.policy_style(policy)["color"] if policy is not None else st.GREY
    def kw(scope):
        if scope == "global":
            return dict(color=base, edgecolor="white", linewidth=0.5)
        if scope == "group":
            return dict(color=st.lighten(base, 0.55), edgecolor="white", linewidth=0.5, hatch="///")
        return dict(color="white", edgecolor=base, linewidth=0.6)
    return kw


def main(results="results", suffix="", out="figures"):
    R = pn.merged(results, EXP, suffix)
    if not R:
        return ld.skip(NAME, "no rows")
    cases = st.case_order(ld.values(R, "case"))
    pols = st.policy_order(ld.values(R, KEY))
    scopes = [s for s in SCOPES if ld.where(R, **{LOCK: s})]
    metrics = (("lock_wait_mean_ms", "lock wait (ms)"), ("tps", "tasks per second"), ("dup_written_per_100", "duplicate writes\nper 100 tasks"))
    labels = {c: case_label(results, suffix, R, c) for c in cases}
    fig, axes = st.subplots(len(pols), 3, cols=2 if len(cases) > 3 else 1, aspect=(0.26 if len(cases) > 3 else 0.50) * len(pols) + 0.10, sharex=True, squeeze=False)
    for j, p in enumerate(pols):
        Rp = ld.where(R, **{KEY: p})
        for i, (key, lab) in enumerate(metrics):
            ax = axes[j][i]
            pn.bars(ax, Rp, "case", LOCK, key, groups=cases, order=lambda v: [s for s in scopes if s in v], kw_of=scope_bar(p),
                    group_label=lambda c: labels[c], zero_label=(key == "dup_written_per_100"), rotate=30)
            ax.set_ylabel(lab)
            st.grid(ax, "y")
            ax.tick_params(axis="x", labelsize=st.TINY)
    pn.zero_floor([a for row in axes for a in row])
    pn.letters(axes[0])
    if len(pols) > 1:
        for j, p in enumerate(pols):
            axes[j][1].set_title("policy: " + row_title(p), fontsize=st.FONT, pad=4)
    from matplotlib.patches import Patch
    H = [Patch(label=LABEL.get(s, s), **{("facecolor" if k == "color" else k): v for k, v in scope_bar()(s).items()}) for s in scopes]
    fig.legend(handles=H, loc="outside upper center", ncol=3, handlelength=1.4)
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
