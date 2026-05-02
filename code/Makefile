PYTHON ?= python3
RESULTS := ../results
QC := $(RESULTS)/qc
WEIGHTS := $(RESULTS)/weights
TABLES := $(RESULTS)/tables
FIGS := $(RESULTS)/figures
REF := ../reference

.PHONY: all qc spark baselines features ml gnn evaluate enrichment figures clean reference help

help:
	@echo "Targets:"
	@echo "  make qc         - 01: harmonise both sumstats"
	@echo "  make spark      - 02: derive SPARK-only Z"
	@echo "  make baselines  - 03+04: P+T and PRS-CS baseline weights"
	@echo "  make features   - 05: SNP feature matrix"
	@echo "  make ml         - 06: ML reweighter (XGBoost + MLP)"
	@echo "  make gnn        - 07: gene-network GNN reweighter"
	@echo "  make evaluate   - 08: per-method evaluation tables"
	@echo "  make enrichment - 09: SFARI / GO / KEGG enrichment"
	@echo "  make figures    - 10: report figures"
	@echo "  make all        - run the whole pipeline 01..10"
	@echo "  make reference  - download reference panels (multi-GB; requires net)"
	@echo "  make clean      - remove generated outputs"

all: figures enrichment

qc: $(QC)/harmonised.parquet
$(QC)/harmonised.parquet:
	$(PYTHON) 01_qc.py

spark: $(QC)/spark_only_z.parquet
$(QC)/spark_only_z.parquet: $(QC)/harmonised.parquet
	$(PYTHON) 02_derive_spark_only.py

baselines: $(WEIGHTS)/pt_p1e-05.tsv $(WEIGHTS)/prscs.tsv
$(WEIGHTS)/pt_p1e-05.tsv: $(QC)/harmonised.parquet
	$(PYTHON) 03_baseline_pt.py
$(WEIGHTS)/prscs.tsv: $(QC)/harmonised.parquet
	$(PYTHON) 04_baseline_prscs.py

features: $(QC)/features.parquet
$(QC)/features.parquet: $(QC)/harmonised.parquet
	$(PYTHON) 05_features.py

ml: $(WEIGHTS)/ml_xgb.tsv
$(WEIGHTS)/ml_xgb.tsv: $(QC)/features.parquet $(QC)/spark_only_z.parquet
	$(PYTHON) 06_ml_reweight.py

gnn: $(WEIGHTS)/gnn.tsv
$(WEIGHTS)/gnn.tsv: $(QC)/features.parquet $(QC)/spark_only_z.parquet
	$(PYTHON) 07_gnn_gene_network.py

evaluate: $(TABLES)/eval.tsv
$(TABLES)/eval.tsv: $(WEIGHTS)/pt_p1e-05.tsv $(WEIGHTS)/prscs.tsv $(WEIGHTS)/ml_xgb.tsv $(WEIGHTS)/gnn.tsv $(QC)/spark_only_z.parquet
	$(PYTHON) 08_evaluate.py

enrichment: $(TABLES)/enrichment.tsv
$(TABLES)/enrichment.tsv: $(TABLES)/eval.tsv
	$(PYTHON) 09_enrichment.py

figures: $(FIGS)/fig_method_bars.pdf
$(FIGS)/fig_method_bars.pdf: $(TABLES)/eval.tsv
	$(PYTHON) 10_figures.py

reference:
	@echo "Reference downloads must be run interactively because of size and DUAs."
	@echo "Recommended manual steps (see report/Method):"
	@echo "  1) PRS-CS:        git clone https://github.com/getian107/PRScs $(REF)/PRScs"
	@echo "  2) PRS-CS LD ref: download ldblk_1kg_eur from PRS-CS into $(REF)/ldblk_1kg_eur"
	@echo "  3) 1000G EUR PLINK files: see https://www.cog-genomics.org/plink/1.9/resources"
	@echo "     Save the .bed/.bim/.fam triple under $(REF)/1000G_EUR/all_phase3_EUR.{bed,bim,fam}"
	@echo "  4) HapMap3 SNP list: https://github.com/getian107/PRScs#getting-started -> $(REF)/hapmap3.snplist"
	@echo "  5) STRING v12 (links_full): https://string-db.org/cgi/download -> $(REF)/string_v12.tsv"
	@echo "  6) SFARI Gene CSV (registration required) -> $(REF)/sfari_gene.csv"
	@echo "  7) Gencode v44 gene BED -> $(REF)/gencode_v44_genes.bed"
	@echo "  8) GTEx v8 brain eQTLs (sig pairs) -> $(REF)/gtex_v8_brain/"
	@echo "  9) ENCODE cCRE bed -> $(REF)/encode/"

clean:
	rm -rf $(RESULTS)
	@echo "removed results/"
