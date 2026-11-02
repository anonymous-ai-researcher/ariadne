# Paper sources

LaTeX sources of the paper (`pvldb20-ariadne.tex`, 12 pages plus references) and of its full version with
appendices (`pvldb20-ariadne-full.tex`), built with the PVLDB template (`acmart.cls`, `pvldb.sty`).

No measured number is typed into the text. Each one is a macro defined in `ariadne-preamble.tex`:

| Macro | Use |
|---|---|
| `\res[shape]{key}` | one value; the shape (`00.0`, `0.00`, `N`, ...) fixes the formatting |
| `\ra`, `\rb`, `\ri`, `\rarng` | variants for ranges and intervals |

`make_results.py` resolves every key from the result files through the map in `resmap/` (one module per
experiment group, documented in `resmap/README.md`) and writes `results.tex`; `env.yaml` holds the few values that
are not measured (software versions, machines, model names, counts). Figures and the generated tables come from
`ariadne.plots`; `tidy_tables.py` typesets the tables.

```bash
python paper/assemble_results.py --vm <base results> --perf results_vldb_perf \
       --gpt results_vldb_gpt4omini --qwen results_vldb_qwen7b --out results_vldb_merged
FINAL=1 RESULTS=results_vldb_merged bash paper/build.sh

python paper/make_results.py --list                        # every key with its shape
python paper/make_results.py --context C1.A64              # where keys with this prefix occur in the text
```

The full version references the main text through `xr`, so the main text is
compiled first. Building needs a TeX Live installation with `pdflatex` and `bibtex`.
