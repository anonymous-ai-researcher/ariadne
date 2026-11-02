"""Figure S8: the SERIALIZABLE story in one two-column figure.

Combines  (a) the space-time diagram of the unsoundness window (fig_s_05_ssi_window: a SCHEMATIC, no data),
          (b) the dose response of the anomaly (fig_s_06_ssi_dose: A6_ssi_soundness/hammer_grid.jsonl),
          (c, d) aborts and model calls per task per protocol with the keyed B-tree control
          (fig_s_07_ssi_cost: A5_ssi_cost/summary.jsonl, A8_keyed_control/keyed.jsonl).
Panels whose data is missing are left out; with no data at all the figure is skipped (the schematic alone is S5).
Paper     main text as a figure* if the three parts are to stay together; otherwise use S5, S6 and S7 separately.
"""
from . import style as st, load as ld
from . import fig_s_05_ssi_window as F5, fig_s_06_ssi_dose as F6, fig_s_07_ssi_cost as F7

NAME = "fig_s_08_ssi_story"


def main(results="results", suffix="", out="figures"):
    st.use()
    has_dose = bool(F6.points(results, suffix)[0])
    has_cost = bool(ld.rows(results, F7.EXP, suffix=suffix, quiet=True) or ld.rows(results, F7.KEYED, "keyed.jsonl", suffix, quiet=True))
    if not has_dose and not has_cost:
        return ld.skip(NAME, "no A5 / A6 / A8 results (the schematic alone is fig_s_05_ssi_window)")
    h = 2.15 + (1.75 if has_cost else 0.0)
    fig = st.plt.figure(figsize=(st.COL2, h))
    subs = fig.subfigures(2, 1, height_ratios=[2.15, 1.75]) if has_cost else [fig]
    top = subs[0]
    if has_dose:
        ax_d, ax_b = top.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    else:
        ax_d = top.subplots(1, 1)
    F5.draw(ax_d)
    st.panel_label(ax_d, "a", "the window (schematic)", dy=1.0)
    if has_dose:
        F6.draw(ax_b, results, suffix, compact=True)
        st.panel_label(ax_b, "b", "anomaly rate against insertion time")
    if has_cost:
        ax_a, ax_c = subs[1].subplots(1, 2, sharey=True)
        cases = F7.draw(ax_a, ax_c, results, suffix)
        st.panel_label(ax_a, "c", "aborts"); st.panel_label(ax_c, "d", "model calls")
        note = F7.conc_note(cases)
        if note:
            subs[1].supxlabel(note, fontsize=st.TINY, color=st.MUTED, x=0.995, ha="right")
    return st.save(fig, NAME, out)


if __name__ == "__main__":
    ld.cli(main, __doc__)
