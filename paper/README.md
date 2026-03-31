# ALC publication package

## Reproduce results

```bash
./scripts/run_alc_publication_suite.sh
```

Outputs:
- `paper/results/metrics.csv`
- `paper/results/summary.json`
- `paper/results/multiseed_raw.csv`
- `paper/results/multiseed_summary.csv`
- `paper/results/multiseed_summary.json`
- `paper/tables/main_results.md`
- `paper/tables/main_results.tex`
- `paper/tables/ablation_results.csv`
- `paper/tables/ablation_top10.tex`
- `paper/figures/FIGURE_GENERATION_WARNING.txt` (if matplotlib unavailable)

## Build LaTeX draft

From `paper/` directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
