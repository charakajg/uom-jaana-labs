"""Quality control and harmonisation of the two ASD GWAS summary statistics.

Reads
-----
* ``Datasets/iPSYCH-PGC_ASD/iPSYCH-PGC_ASD_Nov2017.gz`` - discovery sumstats
  (GRCh37 coordinates).
* ``Datasets/ASD_SPARK_iPSYCH_PGC/ASD_SPARK_iPSYCH_PGC.tsv`` - meta-analysis
  sumstats that *includes* the iPSYCH samples (GRCh38 coordinates).

The two files use different genome builds, so we cannot inner-join on
``chr:pos``. Instead we extract the rsID embedded in the meta
``MarkerName`` field (pattern ``_(rs\\d+)$``) and inner-join on rsID. The
canonical chromosome / position written to the harmonised parquet are
the **iPSYCH (GRCh37)** values; downstream tools that require GRCh37
(PLINK 1000G, PRS-CS LD blocks, MAGMA) therefore work without lift-over.

Writes
------
``results/qc/harmonised.parquet`` with columns

    chr, pos, snp_key, rsid, a1, a2,
    z_ipsych, beta_ipsych, se_ipsych, p_ipsych, info_ipsych,
    z_meta,  beta_meta,  se_meta,  p_meta,
    is_mhc

QC rules
--------
* Restrict to autosomes 1-22.
* Drop palindromic A/T and C/G SNPs (cannot be strand-aligned without freq).
* Drop iPSYCH ``INFO`` < 0.6.
* Inner-join on rsID and align the meta alleles to the iPSYCH (A1, A2)
  ordering, flipping the sign of the meta effect when the alleles are
  swapped. Pairs with truly mismatched alleles are dropped.
* The MHC region (chr 6, 25-34 Mbp; GRCh37) is flagged but kept; downstream
  scripts choose whether to exclude it.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


AMBIGUOUS_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
MHC_CHR, MHC_START, MHC_END = 6, 25_000_000, 34_000_000


def _read_ipsych() -> pl.DataFrame:
    """Read the iPSYCH-PGC sumstats from the gzipped TSV."""
    info(f"reading {paths.IPSYCH_GZ}")
    with gzip.open(paths.IPSYCH_GZ, "rt") as fh:
        df = pl.read_csv(fh.read().encode(), separator="\t", schema_overrides={"CHR": pl.Int64})
    df = df.rename(
        {
            "CHR": "chr",
            "SNP": "rsid",
            "BP": "pos",
            "A1": "a1",
            "A2": "a2",
            "INFO": "info_ipsych",
            "OR": "or_ipsych",
            "SE": "se_ipsych",
            "P": "p_ipsych",
        }
    )
    df = df.with_columns(
        pl.col("a1").str.to_uppercase(),
        pl.col("a2").str.to_uppercase(),
        beta_ipsych=pl.col("or_ipsych").log(),
    )
    df = df.with_columns(z_ipsych=pl.col("beta_ipsych") / pl.col("se_ipsych"))
    return df


def _read_meta() -> pl.DataFrame:
    """Read the SPARK + iPSYCH + PGC meta-analysis sumstats."""
    info(f"reading {paths.META_TSV}")
    df = pl.read_csv(
        paths.META_TSV,
        separator="\t",
        schema_overrides={"Chromosome": pl.Int64, "Position": pl.Int64},
    )
    df = df.rename(
        {
            "Allele1": "a1_meta",
            "Allele2": "a2_meta",
            "Effect": "beta_meta",
            "StdErr": "se_meta",
            "P-value": "p_meta",
        }
    )
    df = df.with_columns(
        rsid=pl.col("MarkerName").str.extract(r"_(rs\d+)$", 1),
    )
    df = df.filter(pl.col("rsid").is_not_null())
    df = df.select(
        "rsid",
        pl.col("a1_meta").str.to_uppercase(),
        pl.col("a2_meta").str.to_uppercase(),
        "beta_meta",
        "se_meta",
        "p_meta",
    )
    df = df.with_columns(z_meta=pl.col("beta_meta") / pl.col("se_meta"))
    return df


def _harmonise(ipsych: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    """Inner-join on rsID, align alleles, and flag MHC SNPs."""
    info("inner-join on rsid")
    df = ipsych.join(meta, on="rsid", how="inner")

    info("aligning alleles between iPSYCH and meta")

    same_strand_same_order = (pl.col("a1") == pl.col("a1_meta")) & (
        pl.col("a2") == pl.col("a2_meta")
    )
    same_strand_swapped = (pl.col("a1") == pl.col("a2_meta")) & (
        pl.col("a2") == pl.col("a1_meta")
    )
    df = df.with_columns(
        beta_meta=pl.when(same_strand_same_order)
        .then(pl.col("beta_meta"))
        .when(same_strand_swapped)
        .then(-pl.col("beta_meta"))
        .otherwise(pl.lit(None, dtype=pl.Float64)),
        z_meta=pl.when(same_strand_same_order)
        .then(pl.col("z_meta"))
        .when(same_strand_swapped)
        .then(-pl.col("z_meta"))
        .otherwise(pl.lit(None, dtype=pl.Float64)),
    )
    before = df.height
    df = df.drop_nulls(["beta_meta"])
    info(f"  dropped {before - df.height:,} rows with mismatched alleles")

    info("flagging ambiguous A/T C/G and MHC SNPs")
    is_amb = pl.struct("a1", "a2").map_elements(
        lambda r: (r["a1"], r["a2"]) in AMBIGUOUS_PAIRS,
        return_dtype=pl.Boolean,
    )
    df = df.with_columns(
        is_ambiguous=is_amb,
        is_mhc=(pl.col("chr") == MHC_CHR)
        & pl.col("pos").is_between(MHC_START, MHC_END, closed="both"),
        snp_key=pl.format(
            "{}:{}:{}:{}", pl.col("chr"), pl.col("pos"), pl.col("a1"), pl.col("a2")
        ),
    )

    before = df.height
    df = df.filter(
        pl.col("chr").is_between(1, 22),
        pl.col("info_ipsych") >= 0.6,
        ~pl.col("is_ambiguous"),
        pl.col("se_ipsych") > 0,
        pl.col("se_meta") > 0,
        pl.col("p_ipsych").is_between(1e-300, 1.0),
        pl.col("p_meta").is_between(1e-300, 1.0),
        pl.col("z_ipsych").is_finite(),
        pl.col("z_meta").is_finite(),
    )
    info(f"  retained {df.height:,} of {before:,} rows after QC")

    return df.select(
        "chr",
        "pos",
        "snp_key",
        "rsid",
        "a1",
        "a2",
        "z_ipsych",
        "beta_ipsych",
        "se_ipsych",
        "p_ipsych",
        "info_ipsych",
        "z_meta",
        "beta_meta",
        "se_meta",
        "p_meta",
        "is_mhc",
    )


def main() -> None:
    paths.ensure_dirs()
    with stage("read iPSYCH-PGC"):
        ipsych = _read_ipsych()
    info(f"  iPSYCH rows: {ipsych.height:,}")
    with stage("read meta"):
        meta = _read_meta()
    info(f"  meta   rows: {meta.height:,}")
    with stage("harmonise"):
        merged = _harmonise(ipsych, meta)
    info(f"final harmonised rows: {merged.height:,}")
    info(f"  MHC rows:        {int(merged['is_mhc'].sum()):,}")
    info(f"  median |z_ipsych|: {float(np.median(np.abs(merged['z_ipsych']))):.3f}")
    info(f"  median |z_meta|:   {float(np.median(np.abs(merged['z_meta']))):.3f}")

    out = paths.HARMONISED
    out.parent.mkdir(parents=True, exist_ok=True)
    info(f"writing {out}")
    merged.write_parquet(out, compression="zstd")


if __name__ == "__main__":
    main()
