# paper/resmap: where every number of the paper comes from

One module per experiment group. Each module has one function, `resolve(R)`, that defines keys with
`R.put(key, value, src, n)` or with the shorthand `R.stat(key, experiment, field, where)`. `R` is the helper
class `Results` of `paper/make_results.py`; its docstrings are the reference.

Rules that every module follows:

1. A number in the text and the same number in a generated table or figure come from the same rows and the
   same field (the modules use the field names of `ariadne/plots/tables.py`).
2. Rows are selected by grid parameters with `R.sel` / `R.stat`, never by position, so that the same code works
   on smoke results and on the full grids. A selector that can never match the full grid is reported.
3. A value is the mean over the seeds of a cell unless the key says otherwise (`.min`, `.max`, counts, totals).
4. No data means `R.put(key, None, reason)`: the placeholder stays in the PDF. A module never invents a value
   and never falls back to another cell.
5. `src` says which file, which selector, which field, and which aggregation; it goes to `results_report.md`.

`python paper/make_results.py --selftest` checks all of this without the full results.
