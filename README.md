# cancerdata

[![Tests](https://github.com/pirl-unc/cancerdata/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/cancerdata/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cancerdata.svg)](https://pypi.org/project/cancerdata/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, anti-PD-1 response, per-cohort RNA-seq expression,
and cancer-testis antigens — behind one small Python API, a data fetch/cache
CLI, and a set of reference plots.

`cancerdata` sits at the **bottom of the openvax/PIRL stack**: it depends only on
pandas/numpy/pyarrow and is consumed by `pirlygenes` (gene-set curation and
analysis) and `tsarina` (personalized target selection). The small curated
tables ship in the wheel; the heavy per-cohort expression bundle downloads on
first use from the matching GitHub Release.

Everything keys on the cancer-type registry: per-type facts (TMB, incidence,
mortality, anti-PD-1 ORR), per-cohort expression (summary stats + percentiles +
medoid samples), and the cancer-testis-antigen definition (HPA tissue-restriction
over the candidate list) all live here, so the analyses/plots that combine them
don't need a target-selection library.

## Install

```bash
pip install cancerdata
```

## Python API

```python
import cancerdata as cd

cd.resolve_cancer_type("prostate")        # -> "PRAD"
cd.cancer_type_info("SARC_RMS_ARMS")      # full registry record + burden + tmb
cd.cancer_tmb("LUAD_EGFR")                # 6.9  (inherited from LUAD)
cd.cancer_burden("pancreas", metric="us_mortality_pct")
cd.burden_category("SARC_OS")             # -> "bone_and_joint"
cd.cancer_apd1_response("SKCM")           # 42  (anti-PD-1 monotherapy ORR %)

# Cancer-testis antigens (HPA-derived tissue-restriction):
cd.CTA_gene_names()                       # expressed CTA symbols (MAGEA4, CT83, …)
cd.CTA_evidence()                         # full HPA restriction table

# Per-cohort expression percentiles (downloads the data bundle on first use):
cd.cohort_gene_percentiles("PRAD")        # per-gene p0…p100 vector (within-cohort)
cd.within_sample_top_fraction("PRAD")     # per-gene frac of samples top-5% (within-sample)
```

### Domains

- **Ontology** — `cancer_type_registry`, `resolve_cancer_type`,
  `cancer_type_info`, `cancer_types_in_family`, `viral_status`, `fusion_status`,
  the cohort vocabulary (`cohort_registry`, `cohort_aggregates`).
- **TMB** — `cancer_tmb`, `cancer_tmb_df` (parent-chain inheritance).
- **Incidence / mortality** — `cancer_burden`, `burden_category` (ACS / GLOBOCAN).
- **Anti-PD-1 response** — `cancer_apd1_response` (monotherapy ORR per type).
- **Expression** — `cohort_gene_percentiles`, `within_sample_top_fraction`,
  `representative_cohort_samples` over the lazy-downloaded per-cohort bundle.
- **Cancer-testis antigens** — `CTA_gene_names`/`CTA_gene_ids`, `CTA_evidence`,
  `synthesize_restriction` (HPA-only tissue-restriction; MS evidence stays in the
  target-selection layer).
- **Plots** (`pip install cancerdata[plots]`) — `cancerdata.plots.apd1_vs_tmb`,
  `apd1_orr_bars`, `incidence_vs_mortality`.

## CLI

```bash
cancerdata cancer-type prostate     # registry info as JSON
cancerdata tmb LUAD_EGFR            # 6.9
cancerdata apd1 SKCM               # 42
cancerdata burden pancreas --metric us_mortality_pct
cancerdata cta --count             # number of expressed CTAs
cancerdata plot apd1-vs-tmb --out apd1_vs_tmb.png

# data bundle (per-cohort expression):
cancerdata fetch                   # download the ~340 MB bundle
cancerdata status                  # which bundle paths are cached (no download)
cancerdata cache-dir               # where the data bundle is cached
cancerdata prune --yes             # delete stale version caches
cancerdata version
```

## Development

```bash
./develop.sh   # editable install with dev extras
./format.sh    # ruff format
./lint.sh      # ruff check + format --check
./test.sh      # lint + pytest with coverage
```

## License

Apache 2.0.
