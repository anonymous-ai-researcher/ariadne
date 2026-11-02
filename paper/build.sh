#!/usr/bin/env bash
# Builds the paper (main text and full version) from a results directory.
#
#   RESULTS=results bash paper/build.sh            draft: a missing number prints its shape in orange, a missing figure a frame
#   FINAL=1 RESULTS=results bash paper/build.sh    final: a missing number, figure or table is a compile error
#
# Steps: figures and generated tables from the results (ariadne.plots), every number of the text from the results
# (paper/make_results.py -> paper/results.tex), typesetting of the generated tables (paper/tidy_tables.py), LaTeX.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
cd "$ROOT"
RESULTS=${RESULTS:?a results directory, e.g. RESULTS=results}
rm -rf "$HERE/figures" "$HERE/tables" && mkdir -p "$HERE/figures"
python3 -m ariadne.plots.all --results "$RESULTS" --out "$HERE/figures" > "$HERE/figures.log" 2>&1 || { tail -20 "$HERE/figures.log"; exit 1; }
python3 -m ariadne.plots.fig_t_04_law_heatmap --results "$RESULTS" --out "$HERE/figures" --single >> "$HERE/figures.log" 2>&1
python3 -m ariadne.plots.fig_t_03_law_vs_observed --results "$RESULTS" --out "$HERE/figures" \
       --cases L4,A64 --cols 2 --width 3.33 --name fig_t_03_law_main >> "$HERE/figures.log" 2>&1
rm -f "$HERE"/figures/*.png
mv "$HERE/figures/tables" "$HERE/tables"
python3 "$HERE/make_results.py" --results "$RESULTS" || echo "make_results reports problems: see paper/results_report.md"
python3 "$HERE/tidy_tables.py"
cd "$HERE"
for f in pvldb20-ariadne pvldb20-ariadne-full; do       # the full version reads the main text's .aux (xr), so the main text goes first
  if [ "${FINAL:-0}" = 1 ]; then src="\\AtBeginDocument{\\resultsfinaltrue}\\input{$f}"; else src="$f"; fi
  pdflatex -interaction=nonstopmode -jobname=$f "$src" > /dev/null || true
  bibtex $f > /dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -jobname=$f "$src" > /dev/null || true
  pdflatex -interaction=nonstopmode -jobname=$f "$src" > /dev/null || true
  if grep -q "^! " $f.log; then grep -A3 "^! " $f.log | head -20; echo "$f: LaTeX errors (see paper/$f.log)"; exit 1; fi
  echo "$f.pdf: $(pdfinfo $f.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages"
done
