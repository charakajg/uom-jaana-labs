"""Evaluate every set of PRS weights against the held-out SPARK-only Z.

For each weight file in ``results/weights/`` we report, on the held-out test
fold (chr 21-22, MHC excluded):

* Pearson r between the weight (treated as a posterior effect-size estimate
  ``beta``) and the held-out Z, scaled to a Z-score using the iPSYCH-PGC
  standard error.
* AUC for the binary task ``|z_spark| > 1.96``.
* Fraction-of-variance-explained: ``1 - SS_res / SS_tot``.
* Calibration slope (regress observed-z on predicted-z).
* Bootstrapped 95% CIs (1000 SNP-level resamples).

Stratified breakdowns are emitted per MAF bin (when MAF is available) and
per INFO bin.

Reads
-----
* ``results/qc/harmonised.parquet``
* ``results/qc/spark_only_z.parquet``
* ``results/weights/*.tsv``

Writes
------
* ``results/tables/eval.tsv`` - main metrics, one row per method.
* ``results/tables/eval_strata.tsv`` - per-MAF / INFO bin breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


TEST_CHR = [21, 22]
N_BOOTSTRAP = 1000
RNG = np.random.default_rng(20260501)


def _load_target() -> pl.DataFrame:
    target = pl.read_parquet(paths.SPARK_ONLY_Z).select("snp_key", "z_spark", "is_mhc")
    harm = pl.read_parquet(paths.HARMONISED).select(
        "snp_key", "se_ipsych", "info_ipsych", "z_ipsych", "chr"
    )
    return target.join(harm, on="snp_key", how="inner")


def _list_weight_files() -> list[Path]:
    return sorted(p for p in paths.WEIGHTS_DIR.glob("*.tsv") if p.is_file())


def _bootstrap_ci(values: np.ndarray, fn) -> tuple[float, float]:
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(N_BOOTSTRAP, n))
    samples = np.array([fn(values[idx[i]]) for i in range(N_BOOTSTRAP)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.std() == 0 or y_true.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def _bootstrap_pearson(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    n = pred.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(N_BOOTSTRAP, n))
    samples = []
    for i in range(N_BOOTSTRAP):
        s = idx[i]
        samples.append(_pearson(pred[s], true[s]))
    return float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))


def _calibration_slope(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.std() == 0:
        return float("nan")
    cov = float(np.cov(pred, true, ddof=0)[0, 1])
    var = float(np.var(pred, ddof=0))
    return cov / var if var > 0 else float("nan")


def _fve(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _evaluate_weights(name: str, weights: pl.DataFrame, target: pl.DataFrame) -> dict:
    joined = weights.join(target, on="snp_key", how="inner").filter(
        pl.col("chr").is_in(TEST_CHR), ~pl.col("is_mhc")
    )
    if joined.is_empty():
        info(f"  {name}: no overlap on test chromosomes; skipping")
        return {}

    beta = joined["beta"].to_numpy()
    se = joined["se_ipsych"].to_numpy()
    pred_z = beta / np.where(se > 0, se, np.nan)
    pred_z = np.nan_to_num(pred_z, nan=0.0, posinf=0.0, neginf=0.0)
    true_z = joined["z_spark"].to_numpy()

    pearson = _pearson(pred_z, true_z)
    pearson_ci = _bootstrap_pearson(pred_z, true_z)

    binary = (np.abs(true_z) > 1.96).astype(int)
    if 0 < binary.sum() < binary.shape[0]:
        auc = roc_auc_score(binary, np.abs(pred_z))
    else:
        auc = float("nan")

    cal = _calibration_slope(pred_z, true_z)
    fve = _fve(pred_z, true_z)

    info(
        f"  {name}: n={joined.height:,} | r={pearson:.4f}"
        f" CI=[{pearson_ci[0]:.4f},{pearson_ci[1]:.4f}]"
        f" | AUC={auc:.4f} | calib={cal:.4f} | FVE={fve:.4f}"
    )
    return {
        "method": name,
        "n_test": joined.height,
        "pearson_r": pearson,
        "pearson_lo": pearson_ci[0],
        "pearson_hi": pearson_ci[1],
        "auc_sig": auc,
        "calibration_slope": cal,
        "fve": fve,
    }


def _strata(name: str, weights: pl.DataFrame, target: pl.DataFrame) -> list[dict]:
    joined = weights.join(target, on="snp_key", how="inner").filter(
        pl.col("chr").is_in(TEST_CHR), ~pl.col("is_mhc")
    )
    if joined.is_empty():
        return []
    info_bins = joined.with_columns(
        info_bin=pl.col("info_ipsych").cut([0.7, 0.85, 0.95]).cast(pl.Utf8)
    )
    rows = []
    for bin_name in info_bins["info_bin"].unique().to_list():
        sub = info_bins.filter(pl.col("info_bin") == bin_name)
        if sub.height < 10:
            continue
        pred_z = sub["beta"].to_numpy() / np.where(
            sub["se_ipsych"].to_numpy() > 0, sub["se_ipsych"].to_numpy(), np.nan
        )
        pred_z = np.nan_to_num(pred_z, nan=0.0, posinf=0.0, neginf=0.0)
        true_z = sub["z_spark"].to_numpy()
        rows.append(
            {
                "method": name,
                "stratum": f"info_{bin_name}",
                "n": sub.height,
                "pearson_r": _pearson(pred_z, true_z),
            }
        )
    return rows


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised + target"):
        target = _load_target()

    weight_files = _list_weight_files()
    if not weight_files:
        raise SystemExit("No weight files in results/weights/; run 03/04/06/07 first.")

    rows: list[dict] = []
    strata_rows: list[dict] = []
    for path in weight_files:
        name = path.stem
        with stage(f"evaluate {name}"):
            try:
                w = pl.read_csv(path, separator="\t")
            except Exception as exc:
                info(f"  could not read {path}: {exc}")
                continue
            if w.is_empty() or "beta" not in w.columns:
                info(f"  {name}: empty or malformed weights file")
                continue
            row = _evaluate_weights(name, w, target)
            if row:
                rows.append(row)
                strata_rows.extend(_strata(name, w, target))

    eval_df = pl.DataFrame(rows)
    out = paths.TABLES_DIR / "eval.tsv"
    info(f"writing {out}")
    eval_df.write_csv(out, separator="\t")

    if strata_rows:
        out_s = paths.TABLES_DIR / "eval_strata.tsv"
        info(f"writing {out_s}")
        pl.DataFrame(strata_rows).write_csv(out_s, separator="\t")


if __name__ == "__main__":
    main()
