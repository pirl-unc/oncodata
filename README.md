# oncodata

[![Tests](https://github.com/pirl-unc/oncodata/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/oncodata/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/oncodata.svg)](https://pypi.org/project/oncodata/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, anti-PD-1 response, per-cohort RNA-seq expression,
and cancer-testis antigens — behind one small Python API, a data fetch/cache
CLI, and a set of reference plots.

## oncodata is the base layer

`oncodata` is **designed as the base layer** of the openvax/PIRL stack — the
intended single upstream **source of truth** for cancer reference data, meant to
become a shared dependency of
[pirlygenes](https://github.com/pirl-unc/pirlygenes),
[trufflepig](https://github.com/pirl-unc/trufflepig), and
[tsarina](https://github.com/pirl-unc/tsarina). Adoption is still in progress —
most of these don't depend on it yet. Architecturally it stays at the bottom: it
depends only on pandas / numpy / pyarrow / PyYAML, it **never imports its
consumers** (data and logic flow only downward), and it **owns** these
definitions rather than mirroring them from elsewhere.

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

depends on `oncodata` — including `pirlygenes` (gene-set curation/analysis),
`tsarina` (personalized target selection), `hitlist` (panel selection),
`trufflepig` (sample classification), and anything else downstream.

Everything keys on the cancer-type registry. The small curated tables ship in the
wheel; the heavy per-cohort expression bundle downloads on first use from
oncodata's own GitHub Release.

## Install

```bash
pip install oncodata
```

## Python API

```python
import oncodata as cd

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
  (`oncodata sources fetch`).
- **Genome reference** — `canonical_gene_id_and_name`, `find_gene_id_by_name`,
  `find_gene_name_from_ensembl_{gene,transcript}_id`, `aggregate_gene_expression`
  (pyensembl-backed symbol ↔ Ensembl-ID resolution). pyensembl ships with the
  package, but resolution needs a downloaded human release once:
  `pyensembl install --release 111 --species homo_sapiens` (the accessors return
  `None` until then).
- **Peptides** — `cta_specific_9mer_counts`, `cta_specific_9mer_load` (per-cohort
  mean per-patient CTA-specific 9-mer load): 9-mers found in a CTA protein but in no
  non-CTA protein, enumerated from the reference proteome and cached per release.
- **Plots** (`pip install oncodata[plots]`) — `oncodata.plots.apd1_vs_tmb`,
  `apd1_orr_bars`, `incidence_vs_mortality`, and the CTA/coverage figures.

## CLI

```bash
oncodata cancer-type prostate     # registry info as JSON
oncodata tmb LUAD_EGFR            # 6.9
oncodata apd1 SKCM               # 42
oncodata burden pancreas --metric us_mortality_pct
oncodata cta --count             # number of expressed CTAs
oncodata plot apd1-vs-tmb --out apd1_vs_tmb.png

# data bundle (per-cohort expression):
oncodata fetch                   # download the ~340 MB bundle
oncodata status                  # which bundle paths are cached (no download)
oncodata cache-dir               # where the data bundle is cached
oncodata prune --yes             # delete stale version caches
oncodata version
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
