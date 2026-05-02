# Datasets

Two GWAS summary-statistics files used by the ML-enhanced PRS pipeline. Both are **summary statistics**, not individual-level genotypes.

## `iPSYCH-PGC_ASD/iPSYCH-PGC_ASD_Nov2017.gz`

iPSYCH + Psychiatric Genomics Consortium (PGC) ASD GWAS meta-analysis released November 2017 (Grove et al., *Nat. Genet.* 2019).

| Column | Meaning |
| ------ | ------- |
| `CHR`  | Chromosome (1-22) |
| `SNP`  | rsID |
| `BP`   | GRCh37 base-pair position |
| `A1`   | Effect allele |
| `A2`   | Reference allele |
| `INFO` | Imputation INFO score |
| `OR`   | Odds ratio for `A1` |
| `SE`   | Standard error of log(OR) |
| `P`    | Two-sided p-value |

Approximate sample size: 18,381 cases + 27,969 controls (≈ N = 46,350).

Provenance: <https://www.med.unc.edu/pgc/download-results/asd/> (file `iPSYCH-PGC_ASD_Nov2017`).

## `ASD_SPARK_iPSYCH_PGC/ASD_SPARK_iPSYCH_PGC.tsv`

Stein-Lab meta-analysis combining SPARK + iPSYCH + PGC samples (Matoba et al. 2020). Plain TSV, ~8.99 M rows, ~684 MB.

| Column | Meaning |
| ------ | ------- |
| `Chromosome` | Chromosome (1-22) |
| `Position`   | GRCh37 base-pair position |
| `MarkerName` | `chrN:pos:A1:A2_rsID` |
| `Allele1`    | Effect allele (lowercase) |
| `Allele2`    | Other allele (lowercase) |
| `Effect`     | Sample-size-weighted Z meta-analysis effect (log-OR-scaled) |
| `StdErr`     | Standard error |
| `P-value`    | Two-sided p-value |
| `Direction`  | Per-cohort direction string |
| `TotalSampleSize` | 58,794 |

Provenance: <https://bitbucket.org/steinlabunc/spark_asd_sumstats>.

## How they relate

The Stein-Lab meta-analysis **includes** the iPSYCH-PGC samples by construction. To obtain an independent SPARK-only signal we invert the sample-size-weighted-Z meta formula in [`code/02_derive_spark_only.py`](../code/02_derive_spark_only.py); see that script and the report Method section for the derivation.

## md5

See `ASD_SPARK_iPSYCH_PGC/md5sum.txt` for the upstream checksum of the Stein-Lab TSV.
