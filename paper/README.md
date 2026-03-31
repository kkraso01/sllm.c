# ALC publication package

## Reproduce results

```bash
./scripts/run_alc_publication_suite.sh
```

Outputs:
- `paper/results/metrics.csv`
- `paper/results/summary.json`
- `paper/tables/main_results.md`
- `paper/tables/ablation_results.csv`
- `paper/figures/FIGURE_GENERATION_WARNING.txt` (if matplotlib unavailable)

## Build LaTeX draft

From `paper/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
