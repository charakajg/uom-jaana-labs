# Running the Pipeline

ML-enhanced Polygenic Risk Score pipeline for Autism Spectrum Disorder. All steps run from the repo root unless noted.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- ~20 GB free disk space for reference panels and results

---

## 1. Create the Conda environment

```bash
conda env create -f code/env/environment.yml
conda activate jaana-asd-prs
```

Key packages installed: Python 3.11, Polars, NumPy, scikit-learn, XGBoost, PyTorch, PyTorch Geometric, NetworkX.

---

## 2. Download reference data

```bash
make -C code reference
```

This prints the manual download steps. Complete them in order:

| # | What | Destination |
|---|------|-------------|
| 1 | PRS-CS source code | `reference/PRScs/` |
| 2 | PRS-CS LD reference (`ldblk_1kg_eur`) | `reference/ldblk_1kg_eur/` |
| 3 | 1000G EUR PLINK files (`.bed/.bim/.fam`) | `reference/1000G_EUR/all_phase3_EUR.{bed,bim,fam}` |
| 4 | HapMap3 SNP list | `reference/hapmap3.snplist` |
| 5 | STRING v12 protein interaction links | `reference/string_v12.tsv` |
| 6 | SFARI Gene CSV (registration required) | `reference/sfari_gene.csv` |
| 7 | Gencode v44 gene BED | `reference/gencode_v44_genes.bed` |
| 8 | GTEx v8 brain eQTL significant pairs | `reference/gtex_v8_brain/` |
| 9 | ENCODE cCRE BED | `reference/encode/` |

---

## 3. Place the GWAS summary statistics

Download both files and place them at the exact paths below:

| File | Path |
|------|------|
| iPSYCH-PGC ASD Nov2017 | `Datasets/iPSYCH-PGC_ASD/iPSYCH-PGC_ASD_Nov2017.gz` |
| SPARK + iPSYCH + PGC meta-analysis | `Datasets/ASD_SPARK_iPSYCH_PGC/ASD_SPARK_iPSYCH_PGC.tsv` |

See [Datasets/README.md](../Datasets/README.md) for download links and column schemas.

---

## 4. Run the full pipeline

```bash
make -C code all
```

The pipeline runs 10 stages in dependency order:

| Stage | Make target | Script | Output |
|-------|-------------|--------|--------|
| 01 | `qc` | `01_qc.py` | `results/qc/harmonised.parquet` |
| 02 | `spark` | `02_derive_spark_only.py` | `results/qc/spark_only_z.parquet` |
| 03 | `baselines` | `03_baseline_pt.py` | `results/weights/pt_p1e-05.tsv` |
| 04 | `baselines` | `04_baseline_prscs.py` | `results/weights/prscs.tsv` |
| 05 | `features` | `05_features.py` | `results/qc/features.parquet` |
| 06 | `ml` | `06_ml_reweight.py` | `results/weights/ml_xgb.tsv`, `ml_mlp.tsv` |
| 07 | `gnn` | `07_gnn_gene_network.py` | `results/weights/gnn.tsv` |
| 08 | `evaluate` | `08_evaluate.py` | `results/tables/eval.tsv` |
| 09 | `enrichment` | `09_enrichment.py` | `results/tables/enrichment.tsv` |
| 10 | `figures` | `10_figures.py` | `results/figures/*.pdf` |

Individual stages can be run independently:

```bash
make -C code qc          # stage 01 only
make -C code baselines   # stages 03 + 04
make -C code ml          # stage 06 only
make -C code evaluate    # stage 08 only
```

Run `make -C code help` to list all available targets.

---

## Clean up

Remove all generated outputs:

```bash
make -C code clean
```

This deletes the entire `results/` directory. Reference data and datasets are not affected.

---

## Output summary

| Path | Contents |
|------|----------|
| `results/weights/` | PRS weight TSVs for each method (P+T, PRS-CS, XGBoost, GNN) |
| `results/tables/eval.tsv` | Per-method R², correlation, explained variance on held-out chr 21-22 |
| `results/tables/enrichment.tsv` | SFARI Gene, Gene Ontology, KEGG pathway enrichment |
| `results/figures/` | Manhattan plots, QQ plots, method comparison figures (PDF + PNG) |
