"""Baseline 1: pruning-and-thresholding (P+T) PRS weights.

Computes one set of per-SNP weights for each p-value threshold in
``{5e-8, 1e-5, 1e-3, 0.05, 1.0}``. LD clumping uses PLINK 1.9 against the
1000 Genomes EUR phase-3 reference panel (downloaded into ``reference/1000G_EUR/``
by ``make reference``). When PLINK is not available the script falls back to
a simple within-window greedy distance-based pruning so that the rest of the
pipeline still runs end-to-end.

Reads
-----
``results/qc/harmonised.parquet``

Writes
------
``results/weights/pt_p<thresh>.tsv`` - one TSV per threshold with columns

    chr, pos, snp_key, a1, a2, beta, p
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


P_THRESHOLDS = [5e-8, 1e-5, 1e-3, 0.05, 1.0]
CLUMP_R2 = 0.1
CLUMP_KB = 250


def _have_plink() -> bool:
    return shutil.which("plink") is not None or shutil.which("plink1.9") is not None


def _plink_binary() -> str:
    return "plink" if shutil.which("plink") else "plink1.9"


def _bfile_root() -> Path | None:
    """Return the prefix of a 1000G EUR PLINK ``.bed/.bim/.fam`` triple."""
    for cand in (
        paths.ONEKG_EUR / "all_phase3_EUR",
        paths.ONEKG_EUR / "1000G_EUR",
        paths.ONEKG_EUR / "EUR",
    ):
        if cand.with_suffix(".bed").exists():
            return cand
    return None


def _run_plink_clump(df: pl.DataFrame, p_threshold: float) -> pl.DataFrame:
    """Use PLINK ``--clump`` to produce an LD-pruned set at ``p_threshold``."""
    bfile = _bfile_root()
    assert bfile is not None
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        sumstats_path = tmp / "sumstats.assoc"
        out_prefix = tmp / "clumped"
        df.select(
            pl.col("rsid").alias("SNP"),
            pl.col("p_ipsych").alias("P"),
        ).write_csv(sumstats_path, separator=" ")

        cmd = [
            _plink_binary(),
            "--bfile",
            str(bfile),
            "--clump",
            str(sumstats_path),
            "--clump-p1",
            f"{p_threshold:.3e}",
            "--clump-p2",
            f"{p_threshold:.3e}",
            "--clump-r2",
            str(CLUMP_R2),
            "--clump-kb",
            str(CLUMP_KB),
            "--clump-snp-field",
            "SNP",
            "--clump-field",
            "P",
            "--out",
            str(out_prefix),
        ]
        info("  plink: " + " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        clumped_file = out_prefix.with_suffix(".clumped")
        if not clumped_file.exists():
            return df.head(0)
        clumped = pl.read_csv(clumped_file, separator=" ", has_header=True, ignore_errors=True)
        keep = set(clumped["SNP"].to_list())
        return df.filter(pl.col("rsid").is_in(keep))


def _greedy_window_prune(df: pl.DataFrame, window_kb: int = CLUMP_KB) -> pl.DataFrame:
    """Distance-based greedy fallback used when PLINK / 1000G are absent."""
    out: list[pl.DataFrame] = []
    for chrom in sorted(df["chr"].unique().to_list()):
        sub = df.filter(pl.col("chr") == chrom).sort("p_ipsych")
        kept: list[int] = []
        positions = sub["pos"].to_numpy()
        order = np.argsort(sub["p_ipsych"].to_numpy())
        taken_pos: list[int] = []
        for idx in order:
            p = int(positions[idx])
            if all(abs(p - q) > window_kb * 1000 for q in taken_pos):
                kept.append(int(idx))
                taken_pos.append(p)
        if kept:
            out.append(sub[kept])
    if not out:
        return df.head(0)
    return pl.concat(out)


def _weights_for_threshold(df: pl.DataFrame, p_threshold: float) -> pl.DataFrame:
    sig = df.filter(pl.col("p_ipsych") <= p_threshold)
    info(f"  threshold {p_threshold:.0e}: {sig.height:,} SNPs pre-clump")
    if sig.height == 0:
        return sig
    if _have_plink() and _bfile_root() is not None:
        clumped = _run_plink_clump(sig, p_threshold)
    else:
        info("  plink/1000G unavailable; using greedy window prune fallback")
        clumped = _greedy_window_prune(sig)
    info(f"  threshold {p_threshold:.0e}: {clumped.height:,} SNPs post-clump")
    return clumped.select(
        "chr",
        "pos",
        "snp_key",
        "a1",
        "a2",
        pl.col("beta_ipsych").alias("beta"),
        pl.col("p_ipsych").alias("p"),
    )


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised"):
        df = pl.read_parquet(paths.HARMONISED).filter(~pl.col("is_mhc"))
    info(f"  rows: {df.height:,} (excl. MHC)")

    for thr in P_THRESHOLDS:
        with stage(f"P+T at p<={thr:.0e}"):
            weights = _weights_for_threshold(df, thr)
        out = paths.WEIGHTS_DIR / f"pt_p{thr:.0e}.tsv"
        info(f"  writing {out} ({weights.height:,} weights)")
        weights.write_csv(out, separator="\t")


if __name__ == "__main__":
    main()
