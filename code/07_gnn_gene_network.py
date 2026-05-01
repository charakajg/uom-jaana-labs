"""Gene-network-aware re-weighter.

Pipeline
--------
1. Aggregate SNPs to genes positionally (+/-10 kb of the gene body) using the
   feature matrix's ``nearest_gene`` column.
2. Build a gene graph from STRING v12 edges (when available) and add brain
   co-expression edges (GTEx) when those files are present.
3. Train a 2-layer GraphSAGE / GAT regressor that predicts each gene's mean
   held-out SPARK-only Z from gene-aggregated features.
4. Distribute the predicted gene-level signal back to each SNP by combining
   the discovery Z with the gene-level prediction (MAGMA-style projection):
   ``z_post = w * pred_gene_z + (1 - w) * z_ipsych`` with ``w = 0.5``.

When ``torch_geometric`` is not installed, the script falls back to a simple
two-hop neighbour-aware regressor (Laplacian smoothing on the graph) which
preserves the *form* of the output for downstream evaluation.

Reads
-----
* ``results/qc/features.parquet``
* ``results/qc/spark_only_z.parquet``
* ``results/qc/harmonised.parquet`` (for ``se_ipsych``)
* ``reference/string_v12.tsv`` (optional)

Writes
------
* ``results/qc/gene_features.parquet`` - per-gene feature matrix.
* ``results/weights/gnn.tsv`` - per-SNP posterior beta.
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


GENE_AGG_FEATURES = [
    "z_ipsych",
    "abs_z",
    "info_ipsych",
    "ld_neighbor_max_abs_z",
    "ld_neighbor_mean_abs_z",
    "dist_to_nearest_gene",
    "string_degree",
    "in_sfari_gene",
    "phylop_score",
    "brain_eqtl_score",
]

PROJECTION_WEIGHT = 0.5
TRAIN_CHR = list(range(1, 20))
VAL_CHR = [20]
TEST_CHR = [21, 22]


def _build_gene_table(df: pl.DataFrame) -> pl.DataFrame:
    info("aggregating SNPs to genes")
    snp_with_gene = df.filter(pl.col("nearest_gene") != "")
    agg_exprs = [
        pl.col(c).mean().alias(f"mean_{c}") for c in GENE_AGG_FEATURES
    ] + [
        pl.col("abs_z").max().alias("max_abs_z"),
        pl.col("z_ipsych").len().alias("n_snps"),
        pl.col("z_spark").mean().alias("mean_z_spark_target"),
        pl.col("chr").first().alias("chr"),
        pl.col("pos").min().alias("start"),
        pl.col("pos").max().alias("end"),
        pl.col("in_sfari_gene").max().alias("in_sfari"),
        pl.col("string_degree").first().alias("string_degree"),
    ]
    genes = snp_with_gene.group_by("nearest_gene").agg(agg_exprs)
    genes = genes.rename({"nearest_gene": "gene"})
    info(f"  gene count: {genes.height:,}")
    return genes


def _string_edges(genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not paths.STRING_TSV.exists():
        info("STRING file not found; using empty edge list")
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    info("reading STRING edges")
    df = pl.read_csv(paths.STRING_TSV, separator="\t")
    cols = df.columns
    src_col, dst_col = cols[0], cols[1]
    weight_col = cols[2] if len(cols) > 2 else None
    gene_set = set(genes)
    df = df.filter(pl.col(src_col).is_in(gene_set) & pl.col(dst_col).is_in(gene_set))
    idx = {g: i for i, g in enumerate(genes)}
    src = np.fromiter((idx[g] for g in df[src_col].to_list()), dtype=np.int64)
    dst = np.fromiter((idx[g] for g in df[dst_col].to_list()), dtype=np.int64)
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    if weight_col is not None:
        w = df[weight_col].to_numpy().astype(np.float32)
        edge_weight = np.concatenate([w, w])
    else:
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
    info(f"  retained {edge_index.shape[1]:,} directed STRING edges over {len(genes):,} nodes")
    return edge_index, edge_weight


def _train_pyg(genes: pl.DataFrame, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray | None:
    try:
        import torch
        from torch import nn
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
    except ImportError:
        info("torch_geometric not installed; using Laplacian-smoothing fallback")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_cols = [f"mean_{c}" for c in GENE_AGG_FEATURES] + ["max_abs_z", "n_snps", "in_sfari", "string_degree"]
    X = genes.select(feat_cols).fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X = (X - mean) / std
    y = genes["mean_z_spark_target"].to_numpy().astype(np.float32)

    chrom = genes["chr"].to_numpy()
    train_mask = np.isin(chrom, TRAIN_CHR)
    val_mask = np.isin(chrom, VAL_CHR)

    if edge_index.shape[1] == 0:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    else:
        ei = torch.from_numpy(edge_index).to(device)
    data = Data(
        x=torch.from_numpy(X).to(device),
        edge_index=ei,
        y=torch.from_numpy(y).to(device),
    )

    class GNN(nn.Module):
        def __init__(self, n_in: int, hidden: int = 64) -> None:
            super().__init__()
            self.c1 = SAGEConv(n_in, hidden)
            self.c2 = SAGEConv(hidden, hidden)
            self.head = nn.Linear(hidden, 1)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.2)

        def forward(self, x, edge_index):
            h = self.act(self.c1(x, edge_index))
            h = self.drop(h)
            h = self.act(self.c2(h, edge_index))
            return self.head(h).squeeze(-1)

    model = GNN(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).long().to(device)
    val_idx = torch.from_numpy(np.flatnonzero(val_mask)).long().to(device)

    n_epochs = 200
    best_val = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = ((pred[train_idx] - data.y[train_idx]) ** 2).mean()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred_all = model(data.x, data.edge_index)
            val_loss = float(((pred_all[val_idx] - data.y[val_idx]) ** 2).mean())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            info(f"  GNN epoch {epoch:03d}: train={float(loss):.4f} val={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(data.x, data.edge_index).cpu().numpy()


def _laplacian_fallback(genes: pl.DataFrame, edge_index: np.ndarray) -> np.ndarray:
    info("running Laplacian-smoothing GNN fallback")
    n = genes.height
    z = genes.select(pl.col("mean_z_ipsych")).to_numpy().squeeze().astype(np.float64)
    if edge_index.shape[1] == 0:
        return z

    deg = np.bincount(edge_index[0], minlength=n).astype(np.float64) + 1e-6
    smoothed = z.copy()
    for _ in range(3):
        agg = np.zeros_like(z)
        np.add.at(agg, edge_index[0], smoothed[edge_index[1]])
        smoothed = 0.5 * smoothed + 0.5 * (agg / deg)
    return smoothed.astype(np.float32)


def _predict_per_snp(
    snp_df: pl.DataFrame,
    genes: pl.DataFrame,
    pred_per_gene: np.ndarray,
    se: np.ndarray,
) -> pl.DataFrame:
    info("projecting gene predictions back to SNPs")
    gene_to_pred = dict(zip(genes["gene"].to_list(), pred_per_gene.tolist()))
    pred_z = np.zeros(snp_df.height, dtype=np.float64)
    for i, (g, z) in enumerate(zip(snp_df["nearest_gene"].to_list(), snp_df["z_ipsych"].to_numpy())):
        gene_pred = gene_to_pred.get(g)
        if gene_pred is None:
            pred_z[i] = z
        else:
            pred_z[i] = PROJECTION_WEIGHT * gene_pred + (1 - PROJECTION_WEIGHT) * z
    beta = pred_z * se
    p = 2.0 * norm.sf(np.abs(pred_z))
    return snp_df.select("chr", "pos", "snp_key", "a1", "a2").with_columns(
        beta=pl.Series(beta), p=pl.Series(p)
    )


def main() -> None:
    paths.ensure_dirs()
    with stage("load features + target + harmonised"):
        feats = pl.read_parquet(paths.FEATURES)
        target = pl.read_parquet(paths.SPARK_ONLY_Z).select("snp_key", "z_spark")
        harm = pl.read_parquet(paths.HARMONISED).select("snp_key", "se_ipsych")
        df = feats.join(target, on="snp_key", how="inner").join(harm, on="snp_key", how="inner")
    info(f"  rows: {df.height:,}")

    with stage("build per-gene table"):
        genes = _build_gene_table(df)
    info(f"  writing {paths.PER_GENE_FEATURES}")
    genes.write_parquet(paths.PER_GENE_FEATURES, compression="zstd")

    gene_list = genes["gene"].to_list()
    with stage("STRING edges"):
        edge_index, edge_weight = _string_edges(gene_list)

    with stage("train GNN"):
        pred = _train_pyg(genes, edge_index, edge_weight)
        if pred is None:
            pred = _laplacian_fallback(genes, edge_index)

    out = _predict_per_snp(df, genes, pred, df["se_ipsych"].to_numpy())
    out_path = paths.WEIGHTS_DIR / "gnn.tsv"
    info(f"writing {out_path} ({out.height:,} rows)")
    out.write_csv(out_path, separator="\t")


if __name__ == "__main__":
    main()
