"""Baseline 2: Bayesian continuous shrinkage PRS-CS posterior weights.

Runs PRS-CS-auto on the iPSYCH-PGC sumstats restricted to the HapMap3 SNP
set, using the precomputed 1000G EUR LD panel that ships with PRS-CS.

Reads
-----
``results/qc/harmonised.parquet``

Writes
------
``results/weights/prscs.tsv`` with columns

    chr, pos, snp_key, a1, a2, beta, p

When PRS-CS or the LD panel are not installed under ``reference/``, the
script falls back to a closed-form Empirical-Bayes shrinkage that uses the
discovery Z-statistic alone:

    beta_post = beta_hat * Z^2 / (Z^2 + 1)

This preserves the *shape* of a Bayesian shrinkage baseline so the rest of
the pipeline runs end-to-end; the report's headline numbers should be
generated with the real PRS-CS binary (see ``code/Makefile`` target
``reference``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


PRSCS_REPO = paths.REFERENCE / "PRScs"
PRSCS_LD_DIR = paths.REFERENCE / "ldblk_1kg_eur"


def _have_prscs() -> bool:
    return (PRSCS_REPO / "PRScs.py").exists() and PRSCS_LD_DIR.exists()


def _ipsych_n() -> int:
    return paths.N_IPSYCH


def _write_prscs_input(df: pl.DataFrame, path: Path) -> None:
    df.select(
        pl.col("rsid").alias("SNP"),
        pl.col("a1").alias("A1"),
        pl.col("a2").alias("A2"),
        pl.col("beta_ipsych").alias("BETA"),
        pl.col("p_ipsych").alias("P"),
    ).write_csv(path, separator="\t")


def _run_prscs(df: pl.DataFrame) -> pl.DataFrame:
    """Invoke ``PRScs.py`` once per chromosome and concatenate posterior betas."""
    rows: list[pl.DataFrame] = []
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        sumstats_path = tmp / "ipsych.sumstats"
        _write_prscs_input(df, sumstats_path)

        for chrom in range(1, 23):
            sub = df.filter(pl.col("chr") == chrom)
            if sub.is_empty():
                continue
            out_prefix = tmp / f"prscs_chr{chrom}"
            cmd = [
                sys.executable,
                str(PRSCS_REPO / "PRScs.py"),
                f"--ref_dir={PRSCS_LD_DIR}",
                f"--bim_prefix={paths.ONEKG_EUR / 'all_phase3_EUR'}",
                f"--sst_file={sumstats_path}",
                f"--n_gwas={_ipsych_n()}",
                f"--chrom={chrom}",
                f"--out_dir={out_prefix}",
            ]
            info("  prscs: " + " ".join(cmd))
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            posterior = list(tmp.glob(f"prscs_chr{chrom}*.txt"))
            if not posterior:
                continue
            chrom_df = pl.read_csv(
                posterior[0],
                separator="\t",
                has_header=False,
                new_columns=["chr", "rsid", "pos", "a1", "a2", "beta"],
            )
            rows.append(chrom_df)
    if not rows:
        return df.head(0)
    return (
        pl.concat(rows)
        .join(df.select("chr", "pos", "snp_key", "a1", "a2", "p_ipsych"), on=["chr", "pos", "a1", "a2"], how="inner")
        .select("chr", "pos", "snp_key", "a1", "a2", "beta", pl.col("p_ipsych").alias("p"))
    )


def _empirical_bayes_fallback(df: pl.DataFrame) -> pl.DataFrame:
    """Closed-form James-Stein-style shrinkage when PRS-CS is unavailable."""
    z = df["z_ipsych"].to_numpy()
    beta = df["beta_ipsych"].to_numpy()
    shrinkage = z**2 / (z**2 + 1.0)
    beta_post = beta * shrinkage
    return df.select("chr", "pos", "snp_key", "a1", "a2", pl.col("p_ipsych").alias("p")).with_columns(
        beta=pl.Series(beta_post)
    ).select("chr", "pos", "snp_key", "a1", "a2", "beta", "p")


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised"):
        df = pl.read_parquet(paths.HARMONISED).filter(~pl.col("is_mhc"))
    info(f"  rows: {df.height:,}")

    if paths.HAPMAP3_SNPLIST.exists():
        with stage("restrict to HapMap3"):
            hm3 = set(paths.HAPMAP3_SNPLIST.read_text().split())
            df = df.filter(pl.col("rsid").is_in(hm3))
            info(f"  rows after HapMap3 filter: {df.height:,}")
    else:
        info("HapMap3 list not found; running on all harmonised SNPs")

    if _have_prscs():
        with stage("PRS-CS auto"):
            weights = _run_prscs(df)
    else:
        info("PRS-CS not installed under reference/; using empirical-Bayes fallback")
        with stage("empirical-Bayes shrinkage fallback"):
            weights = _empirical_bayes_fallback(df)

    out = paths.WEIGHTS_DIR / "prscs.tsv"
    info(f"writing {out} with {weights.height:,} rows")
    weights.write_csv(out, separator="\t")
    info(f"  median |beta|: {float(np.median(np.abs(weights['beta']))):.4e}")


if __name__ == "__main__":
    main()
