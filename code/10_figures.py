"""Generate paper-ready figures.

Outputs (one PDF + one PNG per figure)
--------------------------------------
* ``fig_manhattan.pdf`` - Manhattan plot of iPSYCH-PGC discovery p-values.
* ``fig_qq.pdf``        - QQ plot of iPSYCH-PGC -log10(p).
* ``fig_method_scatter.pdf`` - per-method predicted-vs-true Z scatter on the
  held-out test fold.
* ``fig_method_bars.pdf`` - Pearson r and AUC across methods.
* ``fig_sfari_overlap.pdf`` - top-decile gene overlap with SFARI Gene tiers
  rendered as an UpSet plot when ``upsetplot`` is installed; otherwise a
  bar chart of pairwise Jaccard.
* ``fig_network_topgenes.pdf`` - network sub-graph induced by the top-50
  gene-scoring genes from the GNN method (when STRING edges are present).

Reads
-----
* ``results/qc/harmonised.parquet``
* ``results/qc/spark_only_z.parquet``
* ``results/weights/*.tsv``
* ``results/tables/eval.tsv``
* ``results/tables/gene_scores_*.tsv``
* optional ``reference/string_v12.tsv``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


TEST_CHR = [21, 22]
GW_SIG = 5e-8


def _save(fig: plt.Figure, name: str) -> None:
    pdf_path = paths.FIGURES_DIR / f"{name}.pdf"
    png_path = paths.FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    info(f"  wrote {pdf_path.name}")
    plt.close(fig)


def _manhattan(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    cum = 0
    ticks = []
    labels = []
    for chrom in range(1, 23):
        sub = df.filter(pl.col("chr") == chrom).sort("pos")
        if sub.is_empty():
            continue
        x = sub["pos"].to_numpy() + cum
        y = -np.log10(np.maximum(sub["p_ipsych"].to_numpy(), 1e-300))
        ax.scatter(
            x, y, s=2, alpha=0.6,
            color="#1f77b4" if chrom % 2 else "#888888",
            rasterized=True,
        )
        ticks.append(cum + sub["pos"].max() / 2)
        labels.append(str(chrom))
        cum += int(sub["pos"].max())
    ax.axhline(-np.log10(GW_SIG), color="red", linestyle=":", linewidth=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(p_{\mathrm{iPSYCH}})$")
    ax.set_title("iPSYCH-PGC ASD discovery Manhattan plot")
    fig.savefig(paths.FIGURES_DIR / "fig_manhattan.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(paths.FIGURES_DIR / "fig_manhattan.png", bbox_inches="tight", dpi=200)
    info("  wrote fig_manhattan.pdf")
    plt.close(fig)


def _qq(df: pl.DataFrame) -> None:
    p = df["p_ipsych"].to_numpy()
    p = np.sort(np.maximum(p, 1e-300))
    n = p.shape[0]
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(expected, observed, ".", markersize=2, color="#1f77b4", alpha=0.6, rasterized=True)
    ax.plot([0, expected.max()], [0, expected.max()], color="red", linewidth=0.8)
    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")
    ax.set_title("iPSYCH-PGC QQ plot")
    fig.savefig(paths.FIGURES_DIR / "fig_qq.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(paths.FIGURES_DIR / "fig_qq.png", bbox_inches="tight", dpi=200)
    info("  wrote fig_qq.pdf")
    plt.close(fig)


def _scatter_methods(target: pl.DataFrame) -> None:
    weight_files = sorted(paths.WEIGHTS_DIR.glob("*.tsv"))
    n = len(weight_files)
    if n == 0:
        return
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax, path in zip(axes.flat, weight_files):
        try:
            w = pl.read_csv(path, separator="\t")
        except Exception:
            continue
        if "beta" not in w.columns:
            continue
        joined = w.join(target, on="snp_key", how="inner").filter(
            pl.col("chr").is_in(TEST_CHR), ~pl.col("is_mhc")
        )
        if joined.is_empty():
            continue
        pred = joined["beta"].to_numpy() / np.where(
            joined["se_ipsych"].to_numpy() > 0, joined["se_ipsych"].to_numpy(), np.nan
        )
        pred = np.nan_to_num(pred, nan=0.0)
        true = joined["z_spark"].to_numpy()
        ax.hexbin(pred, true, gridsize=60, cmap="Blues", mincnt=1)
        if pred.std() > 0 and true.std() > 0:
            r = float(np.corrcoef(pred, true)[0, 1])
            ax.text(0.05, 0.92, f"r = {r:.3f}\nn = {joined.height:,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey"))
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("predicted Z")
        ax.set_ylabel(r"SPARK-only $Z$")
        ax.set_title(path.stem, fontsize=10)
    for ax in axes.flat[len(weight_files):]:
        ax.axis("off")
    fig.suptitle("Held-out test (chr 21-22) - predicted vs SPARK-only Z")
    fig.tight_layout()
    _save(fig, "fig_method_scatter")


def _method_bars() -> None:
    out = paths.TABLES_DIR / "eval.tsv"
    if not out.exists():
        return
    df = pl.read_csv(out, separator="\t").sort("pearson_r", descending=True)
    if df.is_empty():
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(df["method"], df["pearson_r"])
    axes[0].errorbar(
        df["pearson_r"], df["method"],
        xerr=[
            (df["pearson_r"] - df["pearson_lo"]).to_numpy(),
            (df["pearson_hi"] - df["pearson_r"]).to_numpy(),
        ],
        fmt="none", ecolor="black", capsize=2,
    )
    axes[0].set_xlabel("Pearson r vs SPARK-only Z (95% CI)")
    axes[0].set_title("Held-out predictive accuracy")
    axes[1].barh(df["method"], df["auc_sig"])
    axes[1].set_xlabel(r"AUC for $|z_{\mathrm{SPARK}}| > 1.96$")
    axes[1].set_title("Significance discrimination")
    fig.tight_layout()
    _save(fig, "fig_method_bars")


def _sfari_overlap() -> None:
    files = sorted(paths.TABLES_DIR.glob("gene_scores_*.tsv"))
    if not files:
        return
    sets: dict[str, set[str]] = {}
    for f in files:
        df = pl.read_csv(f, separator="\t").sort("gene_score", descending=True)
        top_n = max(int(df.height * 0.1), 10)
        method = f.stem.replace("gene_scores_", "")
        sets[method] = set(df.head(top_n)["gene"].to_list())
    if not sets:
        return
    try:
        from upsetplot import from_contents, UpSet  # type: ignore

        fig = plt.figure(figsize=(8, 4))
        UpSet(from_contents(sets), subset_size="count", show_counts=True).plot(fig=fig)
        _save(fig, "fig_sfari_overlap")
    except Exception:
        names = list(sets.keys())
        n = len(names)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a, b = sets[names[i]], sets[names[j]]
                mat[i, j] = len(a & b) / max(len(a | b), 1)
        fig, ax = plt.subplots(figsize=(0.6 * n + 2, 0.6 * n + 2))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Jaccard")
        ax.set_title("Top-decile gene overlap (Jaccard)")
        _save(fig, "fig_sfari_overlap")


def _network_topgenes() -> None:
    gnn_path = paths.TABLES_DIR / "gene_scores_gnn.tsv"
    if not gnn_path.exists() or not paths.STRING_TSV.exists():
        return
    try:
        import networkx as nx
    except ImportError:
        return
    scores = pl.read_csv(gnn_path, separator="\t").sort("gene_score", descending=True)
    top = scores.head(50)["gene"].to_list()
    edges = pl.read_csv(paths.STRING_TSV, separator="\t")
    src, dst = edges.columns[0], edges.columns[1]
    sub = edges.filter(pl.col(src).is_in(top) & pl.col(dst).is_in(top))
    G = nx.Graph()
    G.add_nodes_from(top)
    for s, d in zip(sub[src].to_list(), sub[dst].to_list()):
        G.add_edge(s, d)
    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=120, ax=ax, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title("Top-50 GNN gene scores - STRING sub-graph")
    ax.axis("off")
    _save(fig, "fig_network_topgenes")


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised + target"):
        harm = pl.read_parquet(paths.HARMONISED)
        spark = pl.read_parquet(paths.SPARK_ONLY_Z).select("snp_key", "z_spark", "is_mhc")
        target = spark.join(
            harm.select("snp_key", "se_ipsych", "chr"), on="snp_key", how="inner"
        )

    with stage("Manhattan"):
        _manhattan(harm)
    with stage("QQ"):
        _qq(harm)
    with stage("method scatter"):
        _scatter_methods(target)
    with stage("method bars"):
        _method_bars()
    with stage("SFARI / top-gene overlap"):
        _sfari_overlap()
    with stage("network sub-graph"):
        _network_topgenes()


if __name__ == "__main__":
    main()
