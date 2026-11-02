# Theory figures

Modules `fig_t_<nn>_<slug>.py` of `ariadne/plots/`: the figures of the theory side (characterization theorem,
conflict law, lag kernel, transfers, controls).  Every module has a docstring (inputs, what is shown, place in
the paper), a `main(results, suffix, out)` function, and runs alone:

    python -m ariadne.plots.fig_t_03_law_vs_observed --results results --suffix _smoke --out figures

    for m in ariadne/plots/fig_t_*.py; do python -m ariadne.plots.$(basename $m .py) --results results --out figures; done

Results of experiment X are read from `<results>/X<suffix>/`.  Missing inputs give one printed line
("skipped ...") and no error.  Output: `<out>/<module name>.pdf` and `.png`.

Shared code is in `_theory.py` (not a figure): the encoding below, file access, aggregation over seeds, the
characterization theorem (`conflict_intervals`, `conflict_radius`), the tolerance axis, and `quick_fit`.

## Encoding (the same in every theory figure)

| what | mark |
|---|---|
| observed | black markers, 95% interval as caps or band |
| exchangeable only (background law, no parameter) | grey, dashed |
| closed form | orange, solid |
| semi-parametric form, 1 sibling scale / 2 scales | blue solid / green dash-dotted |
| sibling insertions / background insertions (densities) | purple fill / grey fill |
| conflict / admissible (regions, outcomes) | vermillion / blue |
| a data set (LoCoMo, AG News, ...) | colour and marker of `style.dataset_style`, as in the system-side figures |
| stream lag | ordered purple ramp, dark = lag 1; other team = grey |
| transfer kind | marker (`_theory.MODE_MARK`); static-stream estimates are hollow diamonds |

Tolerance axes are logarithmic in c - 1 and labelled with c.  Intervals: Student t over seeds with three or
more seeds; with two seeds the within-run intervals are combined; with one seed the interval of the result
file (bootstrap of `ariadne.law fit`, Wilson for proportions).

## Figures and their input files

| module | reads | shows |
|---|---|---|
| `fig_t_01_teaser` | `D3a_law_real/<cell>.pkl` + `.law.pkl`, the workload of that run (texts when present) | two real reads with one concurrent in-ball insertion each (a sibling at rank 1 that breaks c = 1.1; a background item at a high rank that does not): 2-D scene with exact radii, the conflict set of the theorem for that neighbour list, and the per-rank admissibility test |
| `fig_t_11_characterization` | one read of `D3a_law_real/<cell>.law.pkl`; prints the agree / tested counts of `B7_characterization/audit_law.jsonl` and `D3a_law_real/audit_law.jsonl` | the theorem rank by rank on a real neighbour list, and the conflict region of that list in the (t, c) plane |
| `fig_t_02_conflict_landscape` | `D3a_law_real/*.law.pkl` (ro, ev, nc, N) | P(conflict given u = t / r_k and c) from the theorem, averaged over all reads; on top where real insertions land (first-principles background plus sibling excess); at the right the conflict probability of the observed insertions against the background law |
| `fig_t_03_law_vs_observed` | `D3b_law_analysis/fits.json`; dense curves from `D3a_law_real/*.law.pkl` when present | conflict probability of single-insertion reads against c: observed, exchangeable only, closed form, semi-parametric with 1 and 2 scales; max absolute error per panel |
| `fig_t_04_law_heatmap` | `D1_closed_form/grid.json`, `D3b_law_analysis/fits.json` | closed form over (c, D_s) for two sibling shares, with the fitted D_s of every real configuration |
| `fig_t_08_controls` | `D4a_exchangeable_controls/audit_law.jsonl`, optionally `D4b_controls_analysis/fits.json` | natural / block-shuffled / shuffled / exchangeable order: in-ball insertions per read against the first-principles background, conflict probability at c = 1.1, stale reads |
| `fig_t_05_lag_kernel` | `D3a_law_real/*.law.pkl`, `D3b_law_analysis/lag.json`, `static_*.json`, `transfer_*.json` | ridgelines of y = ln(r_k / t) per lag bucket with the fitted exponential and the background shape; in-ball probability h per lag for every concurrency level with the static-stream estimate; predicted against observed in-ball insertions per read |
| `fig_t_06_transfers` | `D3b_law_analysis/transfer_*.json`, `static_*.json`, `fits.json` | max error of every transfer between the target's own fit and the exchangeable-only prediction; worst transfer of each kind as a curve |
| `fig_t_07_planted_recovery` | `D2b_planted_analysis/planted.json`, optionally `D4b_controls_analysis/controls.json` | fitted against planted D_s (and their ratio); observed background rate against n_c k / (N + 1) |
| `fig_t_09_rank_distribution` | `D3b_law_analysis/fits.json` ("rank") | old rank of the in-ball insertions: observed, predicted (never fitted), uniform |
| `fig_t_10_y_density` | `D3b_law_analysis/fits.json` (y_hist, y_model1, y_model2, y_bg) | density of y on a log scale with the fitted mixtures and the background part |
| `fig_t_12_tail_exponents` | `D3b_law_analysis/fits.json` ("tail") | tail exponent of the conflict curve in three ranges of c against the fitted D_s |
| `fig_t_13_lid_sensitivity` | `D3b_law_analysis/fits.json` ("lid_sensitivity") | three estimates of D and the closed-form error with each; semi-parametric error for reference |
| `fig_t_14_monte_carlo` | `D1_closed_form/mc*.json` | closed form against Monte Carlo: scatter, z-scores against the closed-form value, z histogram against N(0, 1) |
| `fig_t_15_heterogeneous_dimension` | `D5_heterogeneous_dimension/fits.json` (or the law files, fitted in process), optionally `strata_lid.json` there and in `D3b_law_analysis/` | conflict curves on stores that mix manifold dimensions; errors inside strata of the local dimension |
| `fig_t_16_k_and_embedders` | `D6_embedders_and_k/fits.json` (or the law files, fitted in process), also in `<results>_<EMB>/` | in-ball insertions, sibling share, D_s and conflict probability against k, per data set and embedding model |

## Notes

* `fig_t_03`, `fig_t_09`, `fig_t_10` take `--exp` (and `fig_t_03` `--law-exp`, `--name`), so the same figures can be
  drawn for the planted runs or the controls, for example
  `python -m ariadne.plots.fig_t_03_law_vs_observed --exp D4b_controls_analysis --law-exp D4a_exchangeable_controls --name fig_t_03b_law_controls`.
* `D5_heterogeneous_dimension` and `D6_embedders_and_k` have their analysis steps in `D5b_heterogeneous_analysis` and
  `D6b_embedders_analysis`, which write the `fits.json` (and `strata_lid.json`, `lag.json`) that the figures read.  Without
  those files the figures fit the law files in process (`_theory.quick_fit`: the library calls of `ariadne.law fit`, no bootstrap, an
  approximate interval for D_s).  For bootstrap intervals write a `fits.json` into the experiment directory:
  `python -m ariadne.law fit results/D5_heterogeneous_dimension/*.law.pkl --out results/D5_heterogeneous_dimension/fits.json`.
  Panel (b) of `fig_t_15` needs `python -m ariadne.law strata <law files> --by lid --out <experiment directory>/strata_lid.json`.
* The teaser and the schematic choose their reads automatically (the choice is printed); `--cell`, `--row-sib`,
  `--row-bg`, `--row` fix them for the final version.
* A consequence of the characterization theorem that several figures use: for one neighbour list the conflict set
  is a disc, t < R_c with R_c = max{ r_i / c : r_i > c r_{i-1} } (r_0 = 0).  `_theory.conflict_intervals` evaluates
  the theorem rank by rank and does not rely on it.
