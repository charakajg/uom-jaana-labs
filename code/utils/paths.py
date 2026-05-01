"""Shared filesystem locations for the pipeline.

All scripts import from here so that paths are defined exactly once. Paths are
expressed as ``pathlib.Path`` objects rooted at the repository root, which is
located at the parent of ``code/``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = REPO_ROOT / "Datasets"
IPSYCH_GZ = DATASETS / "iPSYCH-PGC_ASD" / "iPSYCH-PGC_ASD_Nov2017.gz"
META_TSV = DATASETS / "ASD_SPARK_iPSYCH_PGC" / "ASD_SPARK_iPSYCH_PGC.tsv"

REFERENCE = REPO_ROOT / "reference"
HAPMAP3_SNPLIST = REFERENCE / "hapmap3.snplist"
ONEKG_EUR = REFERENCE / "1000G_EUR"
STRING_TSV = REFERENCE / "string_v12.tsv"
SFARI_CSV = REFERENCE / "sfari_gene.csv"
GENE_BED = REFERENCE / "gencode_v44_genes.bed"
GTEX_BRAIN = REFERENCE / "gtex_v8_brain"
ENCODE_DIR = REFERENCE / "encode"
PHYLOP_BED = REFERENCE / "phyloP100way.bed"
GO_GMT = REFERENCE / "go_neurodev.gmt"
KEGG_GMT = REFERENCE / "kegg_synaptic.gmt"

RESULTS = REPO_ROOT / "results"
QC_DIR = RESULTS / "qc"
WEIGHTS_DIR = RESULTS / "weights"
TABLES_DIR = RESULTS / "tables"
FIGURES_DIR = RESULTS / "figures"

HARMONISED = QC_DIR / "harmonised.parquet"
SPARK_ONLY_Z = QC_DIR / "spark_only_z.parquet"
FEATURES = QC_DIR / "features.parquet"
PER_GENE_FEATURES = QC_DIR / "gene_features.parquet"

# Effective sample sizes used by 02_derive_spark_only.py.
N_IPSYCH = 46_350
N_META = 58_794
N_SPARK = N_META - N_IPSYCH  # 12_444


def ensure_dirs() -> None:
    """Create output directories so scripts can write to them safely."""
    for d in (QC_DIR, WEIGHTS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
