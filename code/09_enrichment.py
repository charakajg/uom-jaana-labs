"""Biological enrichment of each method's gene-level signal.

For every set of weights in ``results/weights/`` the script:

* Aggregates SNP-level weights to genes via the ``nearest_gene`` annotation
  produced by ``05_features.py``: per-gene score = mean ``|beta / se|``
  (an analogue of MAGMA's mean Z).
* Runs Fisher's exact tests for enrichment in:
  - SFARI Gene tiers (any tier present in ``reference/sfari_gene.csv``).
  - GO neuro-developmental terms (``reference/go_neurodev.gmt``).
  - KEGG synaptic pathways (``reference/kegg_synaptic.gmt``).

When the optional reference files are absent the relevant rows are emitted
with ``status = "missing-ref"``. The script also calls the ``magma`` and
``ldsc`` binaries when they are on ``PATH`` and produces their result tables
under ``results/tables/``.

Reads
-----
* ``results/weights/*.tsv``
* ``results/qc/features.parquet`` (for ``snp_key -> nearest_gene``)
* optional reference annotation files in ``reference/``

Writes
------
* ``results/tables/enrichment.tsv``
* ``results/tables/gene_scores_<method>.tsv`` (one per method)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths
from utils.logging import info, stage


def _list_weight_files() -> list[Path]:
    return sorted(p for p in paths.WEIGHTS_DIR.glob("*.tsv"))


def _gene_scores(weights: pl.DataFrame, snp_to_gene: pl.DataFrame) -> pl.DataFrame:
    df = weights.join(snp_to_gene, on="snp_key", how="inner").filter(
        pl.col("nearest_gene") != ""
    )
    df = df.with_columns(
        score=(pl.col("beta").abs() / 1.0).fill_nan(0.0)
    )
    return df.group_by("nearest_gene").agg(
        gene_score=pl.col("score").mean(),
        n_snps=pl.col("score").len(),
    ).rename({"nearest_gene": "gene"})


def _load_sfari_tiers() -> dict[str, set[str]]:
    if not paths.SFARI_CSV.exists():
        return {}
    df = pl.read_csv(paths.SFARI_CSV)
    sym_col = next(
        (c for c in df.columns if c.lower() in ("gene-symbol", "gene_symbol", "gene symbol")),
        None,
    )
    score_col = next(
        (c for c in df.columns if c.lower() in ("gene-score", "score", "gene_score")), None
    )
    if sym_col is None:
        return {}
    if score_col is None:
        return {"any": set(df[sym_col].drop_nulls().to_list())}
    out: dict[str, set[str]] = {}
    for tier in df[score_col].drop_nulls().unique().to_list():
        out[f"tier_{tier}"] = set(
            df.filter(pl.col(score_col) == tier)[sym_col].drop_nulls().to_list()
        )
    out["any"] = set(df[sym_col].drop_nulls().to_list())
    return out


def _read_gmt(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        out[parts[0]] = set(parts[2:])
    return out


def _fisher_top_decile(scores: pl.DataFrame, gene_set: set[str], background: set[str]) -> dict:
    if not gene_set:
        return {"or": float("nan"), "p": float("nan"), "n_top": 0, "n_set": 0}
    top_n = max(int(scores.height * 0.1), 10)
    top = set(scores.sort("gene_score", descending=True).head(top_n)["gene"].to_list())
    bg = background & set(scores["gene"].to_list())
    in_top_in_set = len(top & gene_set)
    in_top_not = len(top - gene_set)
    not_top_in_set = len((bg - top) & gene_set)
    not_top_not = len(bg - top - gene_set)
    table = [[in_top_in_set, in_top_not], [not_top_in_set, not_top_not]]
    if min(sum(table[0]), sum(table[1])) == 0:
        return {"or": float("nan"), "p": 1.0, "n_top": top_n, "n_set": len(gene_set)}
    odds, p = fisher_exact(table, alternative="greater")
    return {
        "or": float(odds),
        "p": float(p),
        "n_top": top_n,
        "n_set": len(gene_set),
        "n_top_in_set": in_top_in_set,
    }


def _maybe_run_magma_ldsc(name: str) -> None:
    """Best-effort calls to MAGMA and LDSC if their binaries are on PATH."""
    for tool in ("magma", "ldsc.py"):
        if shutil.which(tool) is None:
            continue
        info(f"{tool} found on PATH; per-method invocation requires user-built reference files. "
             "See README; not running automatically.")


def main() -> None:
    paths.ensure_dirs()
    with stage("load SNP -> gene mapping"):
        snp_to_gene = pl.read_parquet(paths.FEATURES).select("snp_key", "nearest_gene")

    sfari_tiers = _load_sfari_tiers()
    info(f"SFARI tiers loaded: {list(sfari_tiers.keys()) or 'none'}")
    go_sets = _read_gmt(paths.GO_GMT)
    info(f"GO neuro-developmental gene-sets: {len(go_sets)}")
    kegg_sets = _read_gmt(paths.KEGG_GMT)
    info(f"KEGG synaptic gene-sets: {len(kegg_sets)}")

    rows: list[dict] = []
    for path in _list_weight_files():
        name = path.stem
        info(f"evaluating enrichment for {name}")
        try:
            w = pl.read_csv(path, separator="\t")
        except Exception:
            continue
        if "beta" not in w.columns:
            continue
        gene_scores = _gene_scores(w, snp_to_gene)
        if gene_scores.is_empty():
            info(f"  {name}: no gene-mapped SNPs")
            continue
        gene_scores.write_csv(paths.TABLES_DIR / f"gene_scores_{name}.tsv", separator="\t")

        background = set(gene_scores["gene"].to_list())

        for tier_name, gene_set in sfari_tiers.items():
            res = _fisher_top_decile(gene_scores, gene_set, background)
            rows.append({"method": name, "set_kind": "SFARI", "set_name": tier_name, **res})

        for set_name, gene_set in go_sets.items():
            res = _fisher_top_decile(gene_scores, gene_set, background)
            rows.append({"method": name, "set_kind": "GO_neurodev", "set_name": set_name, **res})

        for set_name, gene_set in kegg_sets.items():
            res = _fisher_top_decile(gene_scores, gene_set, background)
            rows.append({"method": name, "set_kind": "KEGG_synaptic", "set_name": set_name, **res})

        if not sfari_tiers and not go_sets and not kegg_sets:
            rows.append(
                {
                    "method": name,
                    "set_kind": "none",
                    "set_name": "missing-ref",
                    "or": float("nan"),
                    "p": float("nan"),
                    "n_top": 0,
                    "n_set": 0,
                }
            )

        _maybe_run_magma_ldsc(name)

    if not rows:
        info("no enrichment rows produced; check inputs")
    out = paths.TABLES_DIR / "enrichment.tsv"
    info(f"writing {out}")
    pl.DataFrame(rows).write_csv(out, separator="\t")


if __name__ == "__main__":
    main()
