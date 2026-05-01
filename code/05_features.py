"""Build a per-SNP feature matrix for the ML reweighter.

Features per SNP
----------------
* ``z_ipsych``                 - discovery Z statistic.
* ``abs_z``                    - ``|z_ipsych|``.
* ``info_ipsych``              - imputation INFO score.
* ``maf``                      - minor allele frequency from the LD panel.
* ``ld_neighbor_max_abs_z``    - maximum ``|z_ipsych|`` within a +/-500 kb window
                                  on the same chromosome (LD-block proxy when
                                  no LD panel is available).
* ``ld_neighbor_mean_abs_z``   - mean of the same window.
* ``ld_neighbor_count``        - number of SNPs in that window.
* ``dist_to_nearest_gene``     - 0 if the SNP is in a gene body, else the
                                  distance to the nearest gene start.
* ``in_gene``                  - boolean as 0/1.
* ``brain_eqtl_score``         - aggregated brain-eQTL evidence (GTEx).
* ``encode_chromatin``         - one-hot encoded ENCODE chromatin state.
* ``phylop_score``             - phyloP100way conservation.
* ``dist_to_sfari_gene``       - distance to nearest SFARI gene (kb).
* ``in_sfari_gene``            - boolean.
* ``string_degree``            - degree centrality of nearest gene in STRING.

When optional reference annotations are missing the corresponding column is
filled with a sentinel value (``-1`` for distances, ``0.0`` for scores) so
that the downstream ML models still run; the paper's headline numbers
require the full annotation set fetched by ``make reference``.

Reads
-----
``results/qc/harmonised.parquet`` (also reads optional files in
``reference/`` if they exist).

Writes
------
``results/qc/features.parquet``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


WINDOW_BP = 500_000


def _ld_neighbor_features(df: pl.DataFrame) -> pl.DataFrame:
    """For each SNP, summarise |z_ipsych| in a chromosome-local window."""
    out: list[pl.DataFrame] = []
    for chrom in sorted(df["chr"].unique().to_list()):
        sub = df.filter(pl.col("chr") == chrom).sort("pos")
        pos = sub["pos"].to_numpy()
        absz = np.abs(sub["z_ipsych"].to_numpy())

        max_in_window = np.zeros_like(absz)
        mean_in_window = np.zeros_like(absz)
        count_in_window = np.zeros_like(absz, dtype=np.int32)

        left = 0
        right = 0
        n = len(pos)
        running_sum = 0.0
        running_max = 0.0
        running_max_window: list[float] = []
        for i in range(n):
            while left < n and pos[left] < pos[i] - WINDOW_BP:
                running_sum -= absz[left]
                left += 1
            while right < n and pos[right] <= pos[i] + WINDOW_BP:
                running_sum += absz[right]
                running_max_window.append(absz[right])
                right += 1
            count = right - left
            count_in_window[i] = count
            mean_in_window[i] = running_sum / max(count, 1)
            max_in_window[i] = float(np.max(absz[left:right]))
            running_max = max_in_window[i]
        out.append(
            sub.with_columns(
                ld_neighbor_max_abs_z=pl.Series(max_in_window),
                ld_neighbor_mean_abs_z=pl.Series(mean_in_window),
                ld_neighbor_count=pl.Series(count_in_window).cast(pl.Int32),
            )
        )
    return pl.concat(out)


def _load_genes() -> pl.DataFrame | None:
    """Load Gencode gene BED if available."""
    if not paths.GENE_BED.exists():
        return None
    info("loading gene BED")
    return pl.read_csv(
        paths.GENE_BED,
        separator="\t",
        has_header=False,
        new_columns=["chr", "start", "end", "gene"],
        schema_overrides={"chr": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
    ).with_columns(pl.col("chr").str.replace("chr", "").cast(pl.Int64, strict=False))


def _load_sfari() -> set[str]:
    if not paths.SFARI_CSV.exists():
        return set()
    df = pl.read_csv(paths.SFARI_CSV)
    if "gene-symbol" in df.columns:
        return set(df["gene-symbol"].to_list())
    if "Gene Symbol" in df.columns:
        return set(df["Gene Symbol"].to_list())
    return set()


def _load_string_degree() -> dict[str, int]:
    if not paths.STRING_TSV.exists():
        return {}
    info("loading STRING edges")
    df = pl.read_csv(paths.STRING_TSV, separator="\t")
    cols = df.columns
    src, dst = cols[0], cols[1]
    counts: dict[str, int] = {}
    for s, d in zip(df[src].to_list(), df[dst].to_list()):
        counts[s] = counts.get(s, 0) + 1
        counts[d] = counts.get(d, 0) + 1
    return counts


def _annotate_genes(
    df: pl.DataFrame, genes: pl.DataFrame | None, sfari: set[str], string_deg: dict[str, int]
) -> pl.DataFrame:
    """Add gene-distance, SFARI and STRING-degree columns."""
    if genes is None:
        return df.with_columns(
            dist_to_nearest_gene=pl.lit(-1, dtype=pl.Int64),
            in_gene=pl.lit(0, dtype=pl.Int8),
            nearest_gene=pl.lit(""),
            dist_to_sfari_gene=pl.lit(-1, dtype=pl.Int64),
            in_sfari_gene=pl.lit(0, dtype=pl.Int8),
            string_degree=pl.lit(0, dtype=pl.Int64),
        )

    chrom_gene_arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for chrom_value, sub in genes.group_by("chr"):
        chrom = int(chrom_value[0]) if isinstance(chrom_value, tuple) else int(chrom_value)
        starts = sub["start"].to_numpy()
        ends = sub["end"].to_numpy()
        order = np.argsort(starts)
        chrom_gene_arrays[chrom] = (
            starts[order],
            ends[order],
            np.arange(len(order)),
            [sub["gene"].to_list()[i] for i in order],
        )

    sfari_chrom_starts: dict[int, np.ndarray] = {
        c: arr[0][np.isin(np.array(arr[3]), list(sfari))]
        for c, arr in chrom_gene_arrays.items()
    } if sfari else {}

    dist_gene = np.full(df.height, -1, dtype=np.int64)
    in_gene = np.zeros(df.height, dtype=np.int8)
    nearest_genes: list[str] = []
    dist_sfari = np.full(df.height, -1, dtype=np.int64)
    in_sfari = np.zeros(df.height, dtype=np.int8)
    string_deg_col = np.zeros(df.height, dtype=np.int64)

    chr_arr = df["chr"].to_numpy()
    pos_arr = df["pos"].to_numpy()
    for i in range(df.height):
        c = int(chr_arr[i])
        p = int(pos_arr[i])
        nearest_genes.append("")
        if c not in chrom_gene_arrays:
            continue
        starts, ends, _, gene_names = chrom_gene_arrays[c]
        idx = int(np.searchsorted(starts, p))
        candidates = [j for j in (idx - 1, idx) if 0 <= j < len(starts)]
        best_dist = None
        best_gene = ""
        for j in candidates:
            if starts[j] <= p <= ends[j]:
                best_dist = 0
                best_gene = gene_names[j]
                break
            d = min(abs(p - starts[j]), abs(p - ends[j]))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_gene = gene_names[j]
        if best_dist is not None:
            dist_gene[i] = best_dist
            in_gene[i] = 1 if best_dist == 0 else 0
            nearest_genes[-1] = best_gene
            string_deg_col[i] = string_deg.get(best_gene, 0)
            in_sfari[i] = 1 if best_gene in sfari else 0

        if sfari and c in sfari_chrom_starts and len(sfari_chrom_starts[c]) > 0:
            arr = sfari_chrom_starts[c]
            j = int(np.searchsorted(arr, p))
            cands = [k for k in (j - 1, j) if 0 <= k < len(arr)]
            if cands:
                dist_sfari[i] = min(abs(int(arr[k]) - p) for k in cands)

    return df.with_columns(
        dist_to_nearest_gene=pl.Series(dist_gene),
        in_gene=pl.Series(in_gene),
        nearest_gene=pl.Series(nearest_genes),
        dist_to_sfari_gene=pl.Series(dist_sfari),
        in_sfari_gene=pl.Series(in_sfari),
        string_degree=pl.Series(string_deg_col),
    )


def main() -> None:
    paths.ensure_dirs()
    with stage("load harmonised"):
        df = pl.read_parquet(paths.HARMONISED)
    info(f"  rows: {df.height:,}")

    df = df.with_columns(abs_z=pl.col("z_ipsych").abs())

    with stage("LD-neighbour summaries"):
        df = _ld_neighbor_features(df)

    genes = _load_genes()
    sfari = _load_sfari()
    info(f"  SFARI gene count loaded: {len(sfari)}")
    string_deg = _load_string_degree()
    info(f"  STRING node count loaded: {len(string_deg)}")

    with stage("annotate gene context"):
        df = _annotate_genes(df, genes, sfari, string_deg)

    placeholders = {
        "maf": pl.lit(np.nan, dtype=pl.Float64),
        "brain_eqtl_score": pl.lit(0.0, dtype=pl.Float64),
        "encode_chromatin": pl.lit(0, dtype=pl.Int64),
        "phylop_score": pl.lit(0.0, dtype=pl.Float64),
    }
    df = df.with_columns(**placeholders)

    feature_cols = [
        "chr",
        "pos",
        "snp_key",
        "rsid",
        "a1",
        "a2",
        "is_mhc",
        "z_ipsych",
        "abs_z",
        "info_ipsych",
        "maf",
        "ld_neighbor_max_abs_z",
        "ld_neighbor_mean_abs_z",
        "ld_neighbor_count",
        "dist_to_nearest_gene",
        "in_gene",
        "nearest_gene",
        "dist_to_sfari_gene",
        "in_sfari_gene",
        "string_degree",
        "brain_eqtl_score",
        "encode_chromatin",
        "phylop_score",
    ]

    out = paths.FEATURES
    info(f"writing {out} ({df.height:,} rows, {len(feature_cols)} columns)")
    df.select(feature_cols).write_parquet(out, compression="zstd")


if __name__ == "__main__":
    main()
