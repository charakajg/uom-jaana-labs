# ASD Risk Prediction with ML-Enhanced PRS

Sumstats-only pipeline that benchmarks pruning-and-thresholding (P+T) and
PRS-CS against ML-reweighted and gene-network-aware alternatives for autism
spectrum disorder (ASD) genetic-risk prediction. The project report is
built under [`report/`](report/); the latest compiled PDF is mirrored to
[`report.pdf`](report.pdf) at the repository root.

The original research proposal is preserved as
[`proposal.pdf`](proposal.pdf) for historical context.

## Authors

258222U L. T. D. B. Fernando · 239318J M. D. C. J. Gunatillake · 258251G K. P. D. P. Lakshan

## Data

Two GWAS summary-statistics files in [`Datasets/`](Datasets/) (see
[`Datasets/README.md`](Datasets/README.md) for schemas and provenance):

- `Datasets/iPSYCH-PGC_ASD/iPSYCH-PGC_ASD_Nov2017.gz` - discovery sumstats
  (Grove et al., 2019).
- `Datasets/ASD_SPARK_iPSYCH_PGC/ASD_SPARK_iPSYCH_PGC.tsv` - Stein-Lab
  meta-analysis (Matoba et al., 2020) that *includes* the iPSYCH-PGC samples.

**Note:** These two files are **not** committed in this repository (they are
large, on the order of hundreds of MB). You need to **download them yourself**
and place them at the paths above before running the pipeline. See
[`Datasets/README.md`](Datasets/README.md) for download links and provenance
(including an md5 for the Stein-Lab file).

We derive an independent SPARK-only Z-score per SNP by inverting the
sample-size-weighted-Z meta-analysis formula; that derived signal is the
held-out target for every model.

## Repository layout

```text
Datasets/                  # raw inputs (unchanged)
code/                      # pipeline scripts 01..10 + Makefile
  env/environment.yml      # conda environment
reference/                 # downloaded reference panels (gitignored)
results/
  qc/                      # harmonised parquet
  weights/                 # per-method PRS weights
  tables/, figures/        # report-ready outputs
report/                    # LaTeX project report + Makefile
```

## Quick start

```bash
conda env create -f code/env/environment.yml
conda activate jaana-asd-prs

# (Once) download reference panels and annotation tracks
make -C code reference

# Full pipeline 01 -> 10
make -C code all

# Build the project report PDF
make -C report
```

Each stage is idempotent and writes to `results/`; rerunning a stage replaces
its outputs only.

## Project report: edit and compile

The LaTeX sources live under [`report/`](report/):

| What | Path |
|------|------|
| Main driver | [`report/main.tex`](report/main.tex) |
| Sections (body text) | [`report/sections/`](report/sections/) — `01_intro.tex` … `06_conclusions.tex` |
| Bibliography | [`report/references.bib`](report/references.bib) |
| Figures embedded in the PDF | [`report/figures/`](report/figures/) (copies of `results/figures/*.pdf`) |

**Edit:** Change the `.tex` files in `report/` (usually `sections/*.tex` or `main.tex`). Add or swap figures by placing PDFs in `report/figures/` and updating `\includegraphics{...}` in the relevant section.

**Compile:** From the repo root, run:

```bash
make -C report
```

This runs [Tectonic](https://tectonic-typesetting.github.io/) on `report/main.tex`, resolves citations from `references.bib`, and writes `report/main.pdf`. The Makefile then copies that file to [`report.pdf`](report.pdf) in the repository root so the latest build is always next to `README.md`.

**Without Make:** `cd report && tectonic main.tex` — then copy `main.pdf` to the repo root as `report.pdf` if you want the same layout as above.

**Requirements:** [Tectonic](https://tectonic-typesetting.github.io/) (`brew install tectonic` on macOS). The first run may download TeX packages; ensure you have network access.

## Pipeline stages

| # | Script                                                          | Output                              |
|---|-----------------------------------------------------------------|-------------------------------------|
| 01 | [`code/01_qc.py`](code/01_qc.py)                               | `results/qc/harmonised.parquet`     |
| 02 | [`code/02_derive_spark_only.py`](code/02_derive_spark_only.py) | `results/qc/spark_only_z.parquet`   |
| 03 | [`code/03_baseline_pt.py`](code/03_baseline_pt.py)             | `results/weights/pt_*.tsv`          |
| 04 | [`code/04_baseline_prscs.py`](code/04_baseline_prscs.py)       | `results/weights/prscs.tsv`         |
| 05 | [`code/05_features.py`](code/05_features.py)                   | `results/qc/features.parquet`       |
| 06 | [`code/06_ml_reweight.py`](code/06_ml_reweight.py)             | `results/weights/ml_*.tsv`          |
| 07 | [`code/07_gnn_gene_network.py`](code/07_gnn_gene_network.py)   | `results/weights/gnn.tsv`           |
| 08 | [`code/08_evaluate.py`](code/08_evaluate.py)                   | `results/tables/eval.tsv`           |
| 09 | [`code/09_enrichment.py`](code/09_enrichment.py)               | `results/tables/enrichment.tsv`     |
| 10 | [`code/10_figures.py`](code/10_figures.py)                     | `results/figures/*.pdf`             |

## Reproducibility notes

- All chromosomes are processed; chr 21-22 are reserved as the held-out test
  fold and chr 20 as validation. Train: chr 1-19.
- The MHC region (chr6:25-34Mb) is excluded from training and reported only
  in a sensitivity analysis.
- Reference panels are *not* committed; `make reference` fetches them.
