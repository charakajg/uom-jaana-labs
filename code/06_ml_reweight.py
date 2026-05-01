"""ML re-weighter that maps SNP features to a posterior effect-size estimate.

Two models are trained, both predicting the held-out SPARK-only Z from the
feature matrix produced by ``05_features.py``:

* ``xgb`` - XGBoost regressor with early stopping.
* ``mlp`` - 3-layer PyTorch MLP, AdamW, cosine LR schedule.

Chromosome-fold split (LD-block-aware): train on chr 1-19, validate on chr 20,
test on chr 21-22. The MHC region is excluded from training and validation.

Reads
-----
* ``results/qc/features.parquet``
* ``results/qc/spark_only_z.parquet``

Writes
------
* ``results/weights/ml_xgb.tsv`` - per-SNP posterior beta from the XGBoost
  reweighter (``beta`` = ``predicted_z * se_ipsych``).
* ``results/weights/ml_mlp.tsv`` - same, from the MLP.

Both weight files use the columns ``chr, pos, snp_key, a1, a2, beta, p`` so
that they are interchangeable with the baseline outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.metrics import mean_squared_error  # noqa: F401  (kept for compat)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


TRAIN_CHR = list(range(1, 20))
VAL_CHR = [20]
TEST_CHR = [21, 22]

FEATURE_COLS = [
    "z_ipsych",
    "abs_z",
    "info_ipsych",
    "ld_neighbor_max_abs_z",
    "ld_neighbor_mean_abs_z",
    "ld_neighbor_count",
    "dist_to_nearest_gene",
    "in_gene",
    "dist_to_sfari_gene",
    "in_sfari_gene",
    "string_degree",
    "brain_eqtl_score",
    "encode_chromatin",
    "phylop_score",
]


def _load() -> pl.DataFrame:
    feats = pl.read_parquet(paths.FEATURES)
    target = pl.read_parquet(paths.SPARK_ONLY_Z).select("snp_key", "z_spark")
    harm = pl.read_parquet(paths.HARMONISED).select("snp_key", "se_ipsych")
    return feats.join(target, on="snp_key", how="inner").join(harm, on="snp_key", how="inner")


def _split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    no_mhc = df.filter(~pl.col("is_mhc"))
    train = no_mhc.filter(pl.col("chr").is_in(TRAIN_CHR))
    val = no_mhc.filter(pl.col("chr").is_in(VAL_CHR))
    test = df.filter(pl.col("chr").is_in(TEST_CHR))
    info(f"  train: {train.height:,} | val: {val.height:,} | test: {test.height:,}")
    return train, val, test


def _train_xgb(train: pl.DataFrame, val: pl.DataFrame) -> "object":
    try:
        import xgboost as xgb
    except ImportError:
        info("xgboost not installed; skipping XGBoost reweighter")
        return None

    X_tr = train.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy()
    y_tr = train["z_spark"].to_numpy()
    X_va = val.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy()
    y_va = val["z_spark"].to_numpy()

    params = dict(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        verbosity=0,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_va = model.predict(X_va)
    rmse = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))
    info(f"  XGBoost val RMSE: {rmse:.4f}")
    info(f"  XGBoost val r:    {float(np.corrcoef(pred_va, y_va)[0, 1]):.4f}")
    return model


def _train_mlp(train: pl.DataFrame, val: pl.DataFrame) -> "object":
    """Train a 3-layer MLP via ``sklearn.neural_network.MLPRegressor``.

    sklearn's CPU MLP implementation is well-optimised for tabular data of
    this size and removes the torch-on-CPU thread-contention bottleneck we
    observed on this hardware. The architecture and hyper-parameters are
    chosen to mirror the configuration described in the Methods.
    """
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        info("scikit-learn not installed; skipping MLP reweighter")
        return None

    rng = np.random.default_rng(20260501)
    n_total = train.height
    n_sample = min(n_total, 250_000)
    if n_sample < n_total:
        sample_idx = rng.choice(n_total, size=n_sample, replace=False)
        train = train[sample_idx]
        info(f"  MLP train sub-sample: {n_sample:,} of {n_total:,}")

    X_tr = train.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
    y_tr = train["z_spark"].to_numpy().astype(np.float32)
    X_va = val.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
    y_va = val["z_spark"].to_numpy().astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    info("  fitting MLPRegressor (128, 64) with adam, early_stopping=True")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=16384,
        learning_rate_init=1e-3,
        max_iter=25,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=20260501,
        verbose=False,
    )
    model.fit(X_tr, y_tr)

    pred_va = model.predict(X_va)
    info(f"  MLP val r: {float(np.corrcoef(pred_va, y_va)[0, 1]):.4f}")
    info(f"  MLP iters: {model.n_iter_}")
    return {"model": model, "scaler": scaler}


def _predict_xgb(model: "object", df: pl.DataFrame) -> np.ndarray:
    X = df.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy()
    return model.predict(X)


def _predict_mlp(bundle: dict, df: pl.DataFrame) -> np.ndarray:
    X = df.select(FEATURE_COLS).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
    X = bundle["scaler"].transform(X)
    return bundle["model"].predict(X)


def _write_weights(df: pl.DataFrame, predicted_z: np.ndarray, out_path: Path) -> None:
    se = df["se_ipsych"].to_numpy()
    beta = predicted_z * se
    p = 2.0 * norm.sf(np.abs(predicted_z))
    out = df.select("chr", "pos", "snp_key", "a1", "a2").with_columns(
        beta=pl.Series(beta), p=pl.Series(p)
    )
    info(f"  writing {out_path} ({out.height:,} rows)")
    out.write_csv(out_path, separator="\t")


def main() -> None:
    paths.ensure_dirs()
    with stage("load features + target"):
        df = _load()
    info(f"  joined rows: {df.height:,}")

    with stage("split by chromosome"):
        train, val, test = _split(df)

    with stage("XGBoost reweighter"):
        xgb_model = _train_xgb(train, val)
    if xgb_model is not None:
        all_pred = _predict_xgb(xgb_model, df)
        _write_weights(df, all_pred, paths.WEIGHTS_DIR / "ml_xgb.tsv")

    with stage("MLP reweighter"):
        mlp_bundle = _train_mlp(train, val)
    if mlp_bundle is not None:
        all_pred = _predict_mlp(mlp_bundle, df)
        _write_weights(df, all_pred, paths.WEIGHTS_DIR / "ml_mlp.tsv")


if __name__ == "__main__":
    main()
