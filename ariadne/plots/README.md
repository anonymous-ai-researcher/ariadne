# Figures and tables

Everything in this directory turns result files into the figures (`.pdf` and 300 dpi `.png`) and LaTeX tables of
the paper.  Nothing here runs an experiment or changes a result file.

    python -m ariadne.plots.all    --results results --suffix _smoke --out figures     # every figure and table
    python -m ariadne.plots.all    --results results --out figures --only fig_s_10 fig_s_14 --no-tables
    python -m ariadne.plots.fig_s_10_pareto --results results --out figures            # one figure
    python -m ariadne.plots.tables --results results --out figures                     # tables only -> figures/tables/*.tex

Results of experiment `X` are read from `<results>/X<suffix>/` (`--suffix _smoke` for the smoke runs, nothing for the
full runs).  A figure or table whose data is missing prints one line and is skipped; `all.py` ends with the list of
what was produced, what was skipped and what failed.

System-side figures are the modules `fig_s_<nn>_<slug>.py` (this file); theory-side figures are `fig_t_*.py`
(see `README_theory.md`).  Every module has a docstring that names its inputs, what it shows and where it goes, and a
`main(results, suffix, out)` function.

## Shared modules

| module | what it holds |
|---|---|
| `style.py` | sizes (single column 3.33 in, double column 7.0 in, 7.5 pt text, TrueType fonts in PDF), the Okabe-Ito palette, the ONE mapping protocol family to colour, marker, line style and hatch (`policy_style`, `policy_line`, `policy_bar`), data set to colour and marker (`dataset_style`, `case_lines`), labels (`pretty`, `dataset_label`, `store_label`), the tolerance axis (`c_axis`: log scale in c - 1, labelled with c, with break marks for c = 1 and reference columns), label placement without overlaps (`label_points`, `end_labels`), `figsize`, `save` |
| `load.py` | `rows` (JSONL to list of dicts), `where`, `values`, `agg` / `series` / `Stats` (mean and interval over seeds), `runs`, `npz`, `cell_arrays`, `pooled` (per-task arrays), `wilson`, `ccdf`, `join` (summary + audit rows), `cli` |
| `panels.py` | small-multiples helpers: one line per protocol with bands, grouped bars, shared legends, categorical axes |
| `tables.py` | the LaTeX tables |
| `all.py` | runs every `fig_*` module and the tables |

Conventions used by every system-side figure:

* **Protocol families**: grey = no isolation, black = exact (re-execution) validation, blue ramp = c-admissible (ours;
  light = strict, dark = loose), bluish green = certified variant, orange = identity tests (overlap, top-j), reddish
  purple = single-rank distance tests, yellow star = oracles, vermillion = SERIALIZABLE, burnt orange = ball predicate
  locks, dark red = application-level remedies, dark brown = lock held across the think step.  Each family also has
  its own marker, line style and hatch.  Protocols are always listed in this order.
* **Data sets**: LoCoMo blue circle, AG News vermillion square, LongMemEval green triangle, QQP purple diamond,
  MS MARCO orange triangle, synthetic grey cross; several workloads of one data set are shades of its colour.
* **Intervals**: the band or cap is the 95% Student t interval of the mean over seeds for n >= 3 seeds; for n = 2 it
  is the range of the two runs (a t interval on one degree of freedom is 12.7 standard errors wide); a single seed
  gives a bare line or mark.  Proportions from counted trials (the SERIALIZABLE stress test) use Wilson intervals.
* No titles (captions are in LaTeX), panel labels (a), (b), ..., a light grid on the value axis only.
* A figure keeps its exact width, so that text has its true size in the paper.  If something does not fit, `save`
  prints a note and stores the figure with a tight box instead of clipping it: treat that note as a defect to fix.

## Figures and the experiments they read

| figure | reads | shows |
|---|---|---|
| `fig_s_01_race_stores` | A1 | duplicates per 100 facts per store and protocol, one panel per concurrency |
| `fig_s_02_race_spread` | A2 (`jitter*`) | duplicates against the arrival spread of the copies, band over seeds, split by fan-out |
| `fig_s_03_race_factors` | A2 (all factors) | the race against each workload factor, small multiples |
| `fig_s_04_race_realtext` | A3 (`audit_serial`, `audit_sep`) | the race on real duplicate clusters (rows beyond the serial execution of the same tasks) and replayed streams |
| `fig_s_05_ssi_window` | none (schematic) | space-time diagram of the SERIALIZABLE unsoundness window on HNSW, with the place of the patched check |
| `fig_s_06_ssi_dose` | A6 (`hammer_grid.jsonl`, controls) | anomaly rate with Wilson intervals against the mean insertion time |
| `fig_s_07_ssi_cost` | A5, A8 | aborts and model calls per task per protocol and concurrency, keyed B-tree control |
| `fig_s_08_ssi_story` | A5, A6, A8 | S5, S6 and S7 as one two-column figure |
| `fig_s_09_calls_vs_c` | C1 | extra model calls against c, one line per workload; exact and overlap as reference columns |
| `fig_s_10_pareto` (+ `_all`) | C2 (`first_dense.jsonl` or `audit_first.jsonl`) | calls saved against decisions that differ, one panel per decision proxy; frontier and index floor |
| `fig_s_11_scaling`, `fig_s_11_scaling_all` | C5 | throughput and p99 latency against the number of agents, log-log; the first at thinking time 0.2 s, the second with one row per thinking time |
| `fig_s_12_open_loop` | C8b | latency and completed rate against the arrival rate (Poisson, bursty) |
| `fig_s_13_think_time` | C8 | extra calls against the mean think time for five think-time distributions |
| `fig_s_14_rho_ccdf` | C1 (`*.audit.npz`) | CCDF of the approximation ratio of committed answers per policy, lines at each c, index floor |
| `fig_s_15_contract_vs_budget` | C6 | answers above c against the search budget, per index and validation variant |
| `fig_s_16_critical_section` | G1 | stacked critical-section time (validation read, predicate, write) and lock wait |
| `fig_s_17_validation_budget` | E4 | validation read time, answers above c, throughput against the validation budget |
| `fig_s_18_opt_vs_pess` | C10 | extra calls against lock wait for validation and ball locks over c, deadlock counts |
| `fig_s_19_rounds_ccdf` | C11 (`*.audit.npz`) | CCDF of rounds per task under worst-case workloads |
| `fig_s_20_multi_read` | B3 | extra calls and critical-section time against the number of extra reads |
| `fig_s_21_k_tau` | C12 | extra calls against k and against tau |
| `fig_s_22_read_mix` | C13 | extra calls, throughput, p99 ratio against the share of read-only tasks |
| `fig_s_23_scale` | C9 | extra calls, validation read time, throughput against memory size and run variants |
| `fig_s_24_choosing_c` | C7 | extra calls and per-proxy divergence against c on a fine grid, index rho quantiles |
| `fig_s_25_separation` | B2 (`audit_sep`) | smallest stored distance / tau against c with the bound tau / c |
| `fig_s_26_long_run` | F5 (`*.pkl`) | windows of tasks over a long run: extra calls, failed first validations, validation read, p99 |
| `fig_s_27_stores` | F1 | extra calls, duplicate writes, critical section per store; safety violations |
| `fig_s_28_failures` | F6 | throughput, p99, duplicate writes against the crash probability |
| `fig_s_29_lock_scope` | E2 | lock wait, throughput, duplicate writes per lock scope |
| `fig_s_30_retry` | E3 | extra calls, capped or given up, p99 against the round limit per strategy |
| `fig_s_31_isolation_phases` | E5 | duplicate writes and extra calls per isolation level of the two phases |
| `fig_s_32_per_decision` | E6 | extra calls, link divergence, differing actions per tolerance assignment |
| `fig_s_33_predicates` | E1 | four metrics per validation predicate and workload |
| `fig_s_34_anomalies` | A4 (`audit_serial`, `audit_run`) | annotated heat table, anomalies per 100 tasks by protocol |
| `fig_s_35_broken_variants` | B5a | duplicate writes and stale validation reads of deliberately broken protocols |

`fig_s_05_ssi_window` is a schematic drawn from the mechanism (reproduced by `ariadne/pgtests/hole.py`); it contains
no measured number.  Every other mark in every figure comes from a result file.

## Tables (`<out>/tables/*.tex`, booktabs `tabular` only)

| file | reads |
|---|---|
| `tab_race.tex`, `tab_race_by_concurrency.tex` | A1, A3 |
| `tab_serializable.tex` | A5, A8 |
| `tab_ssi_soundness.tex` | A6 (locks, hole, generic, hammer) |
| `tab_cost.tex`, `tab_cost_all.tex` | C1 |
| `tab_llm.tex` | C3 (`audit_replay`), C4 (`qa.jsonl`) |
| `tab_ablation.tex` | E1 to E6 |
| `tab_safety.tex`, `tab_safety_all.tex` | B1, B5b; every experiment with a safety check |
| `tab_portability.tex` | F1 |
| `tab_opt_vs_pess.tex` | C10 |

Cells read "mean +- h" when a cell has several seeds (h as under "Intervals" above) and a bare mean otherwise.

## Options and inputs worth knowing

* `fig_s_10_pareto` reads `C2_counterfactual_pareto/first_dense.jsonl` when it exists (a dense grid of c costs
  nothing: it is a replay) and takes `--dense FILE` for a file elsewhere:

      python -m ariadne.audit first results/C2_counterfactual_pareto/*_s[0-9].pkl \
          --cs 1.005,1.01,1.02,1.03,1.05,1.07,1.1,1.15,1.2,1.3,1.5,2,3 \
          --out results/C2_counterfactual_pareto/first_dense.jsonl

  (`*_s[0-9].pkl`, not `*.pkl`: the directory also holds `<cell>.law.pkl` files, which are not run files and make the
  audit command stop.)
* `fig_s_06_ssi_dose` needs `A6_ssi_soundness/hammer_grid.jsonl` (the full run of A6); with the smoke results it shows
  the single point of `hammer.jsonl`.
* `fig_s_09_calls_vs_c` uses C1 only, the rows behind the cost table and the numbers in the text (C7's finer grid of c is in `fig_s_24`).
* `fig_s_21_k_tau` and `fig_s_26_long_run` open run files (`<cell>.pkl`); do not run the experiments with `--no-logs`
  if these figures are wanted.  `fig_s_14_rho_ccdf` and `fig_s_19_rounds_ccdf` need the `<cell>.audit.npz` files that
  the run audit writes.
