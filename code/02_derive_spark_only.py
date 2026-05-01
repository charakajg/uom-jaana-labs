"""Derive an independent SPARK-only Z-score per SNP from the two sumstats.

The Stein-Lab meta-analysis Z combines iPSYCH+PGC and SPARK by

.. math::

   Z_{\\text{meta}} = \\frac{\\sqrt{N_{\\text{iPSYCH}}}\\, Z_{\\text{iPSYCH}}
       + \\sqrt{N_{\\text{SPARK}}}\\, Z_{\\text{SPARK}}}{\\sqrt{N_{\\text{iPSYCH}} + N_{\\text{SPARK}}}}.

Solving for :math:`Z_{\\text{SPARK}}` gives

.. math::

   Z_{\\text{SPARK}} = \\frac{Z_{\\text{meta}}\\sqrt{N_{\\text{total}}}
       - Z_{\\text{iPSYCH}}\\sqrt{N_{\\text{iPSYCH}}}}{\\sqrt{N_{\\text{SPARK}}}}.

The script also round-trip-validates by recomputing the meta Z and reporting
the maximum absolute discrepancy.

Reads
-----
``results/qc/harmonised.parquet``

Writes
------
``results/qc/spark_only_z.parquet`` with columns

    chr, pos, snp_key, rsid, a1, a2, z_spark, p_spark, is_mhc
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


def derive_spark_z(
    z_ipsych: np.ndarray, z_meta: np.ndarray, n_ipsych: int, n_spark: int
) -> np.ndarray:
    n_total = n_ipsych + n_spark
    return (z_meta * np.sqrt(n_total) - z_ipsych * np.sqrt(n_ipsych)) / np.sqrt(n_spark)


def round_trip_z_meta(
    z_ipsych: np.ndarray, z_spark: np.ndarray, n_ipsych: int, n_spark: int
) -> np.ndarray:
    return (np.sqrt(n_ipsych) * z_ipsych + np.sqrt(n_spark) * z_spark) / np.sqrt(
        n_ipsych + n_spark
    )


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised"):
        df = pl.read_parquet(paths.HARMONISED)
    info(f"  rows: {df.height:,}")

    z_ipsych = df["z_ipsych"].to_numpy()
    z_meta = df["z_meta"].to_numpy()

    with stage("derive SPARK-only Z"):
        z_spark = derive_spark_z(z_ipsych, z_meta, paths.N_IPSYCH, paths.N_SPARK)

    with stage("round-trip validate"):
        z_meta_check = round_trip_z_meta(
            z_ipsych, z_spark, paths.N_IPSYCH, paths.N_SPARK
        )
        max_err = float(np.max(np.abs(z_meta_check - z_meta)))
        info(f"  max |z_meta - round_trip| = {max_err:.3e}")
        assert max_err < 1e-8, "Round-trip identity broken; check sample sizes."

    p_spark = 2.0 * norm.sf(np.abs(z_spark))

    info("summary statistics")
    info(f"  N_iPSYCH = {paths.N_IPSYCH:,}")
    info(f"  N_SPARK  = {paths.N_SPARK:,}")
    info(f"  N_total  = {paths.N_META:,}")
    info(f"  median |z_spark|: {np.median(np.abs(z_spark)):.3f}")
    info(f"  fraction p<5e-8: {float(np.mean(p_spark < 5e-8)):.4%}")
    info(
        f"  cor(z_ipsych, z_spark) = "
        f"{float(np.corrcoef(z_ipsych, z_spark)[0, 1]):.3f}"
    )

    out_df = df.select("chr", "pos", "snp_key", "rsid", "a1", "a2", "is_mhc").with_columns(
        z_spark=pl.Series(z_spark),
        p_spark=pl.Series(p_spark),
    )

    out = paths.SPARK_ONLY_Z
    info(f"writing {out}")
    out_df.write_parquet(out, compression="zstd")


if __name__ == "__main__":
    main()
