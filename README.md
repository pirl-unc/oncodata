# cancerdata

[![Tests](https://github.com/pirl-unc/cancerdata/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/cancerdata/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/cancerdata.svg)](https://pypi.org/project/cancerdata/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, anti-PD-1 response, per-cohort RNA-seq expression,
and cancer-testis antigens — behind one small Python API, a data fetch/cache
CLI, and a set of reference plots.

## cancerdata is the base layer

`cancerdata` is the **foundation of the openvax/PIRL dependency pyramid** — the
single upstream **source of truth** for cancer reference data. It depends only on
pandas / numpy / pyarrow / PyYAML, and it **never imports its consumers**: data
and logic flow only downward. It does not mirror these definitions from anywhere;
it owns them.

Anything that needs to know about

- **gene expression of cancer samples** — per-cohort RNA-seq in a normalized,
  comparable space: summary stats, tail-weighted percentiles, and medoid/exemplar
  samples per cancer type/subtype;
- **HPA protein / RNA** normal-tissue expression;
- the **definition of cancer-testis antigens** — the HPA tissue-restriction call
  over the candidate list (HPA-only; no MS/peptide layer);
- the **ontology of cancer types** — codes, the parent/child hierarchy, subtypes,
  families, characteristic driver fusions, and the cross-cutting MSI/POLE/HPV
  groupings; and
- **anti-PD-1 response rates** and **TMB** per cancer type

depends on `cancerdata` — including `pirlygenes` (gene-set curation/analysis),
`tsarina` (personalized target selection), `hitlist` (panel selection),
`trufflepig` (sample classification), and anything else downstream.

Everything keys on the cancer-type registry. The small curated tables ship in the
wheel; the heavy per-cohort expression bundle downloads on first use from
cancerdata's own GitHub Release.

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
- **HPA normal tissue** — `hpa_rna_consensus`, `hpa_normal_tissue` (IHC),
  `hpa_single_cell`, and per-gene lookups (`gene_tissue_ntpm`,
  `gene_protein_tissues`, `gene_cell_type_ntpm`) over HPA v23, fetched on demand
  (`cancerdata sources fetch`).
- **Genome reference** — `canonical_gene_id_and_name`, `find_gene_id_by_name`,
  `find_gene_name_from_ensembl_{gene,transcript}_id`, `aggregate_gene_expression`
  (pyensembl-backed symbol ↔ Ensembl-ID resolution). pyensembl ships with the
  package, but resolution needs a downloaded human release once:
  `pyensembl install --release 111 --species homo_sapiens` (the accessors return
  `None` until then).
- **Plots** (`pip install cancerdata[plots]`) — `cancerdata.plots.apd1_vs_tmb`,
  `apd1_orr_bars`, `incidence_vs_mortality`, and the CTA/coverage figures.

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
