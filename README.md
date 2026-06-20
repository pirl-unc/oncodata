# oncoref

[![Tests](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/oncoref.svg)](https://pypi.org/project/oncoref/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, checkpoint-inhibitor (ICI) response, per-cohort RNA-seq expression,
and cancer-testis antigens — behind one small Python API, a data fetch/cache
CLI, and a set of reference plots.

## oncoref is the base layer

`oncoref` is **designed as the base layer** of the openvax/PIRL stack — the
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
- **checkpoint-inhibitor response rates** and **TMB** per cancer type

depends on `oncoref` — including `pirlygenes` (gene-set curation/analysis),
`tsarina` (personalized target selection), `hitlist` (panel selection),
`trufflepig` (sample classification), and anything else downstream.

Everything keys on the cancer-type registry. The small curated tables ship in the
wheel; the heavy per-cohort expression bundle downloads on first use from
oncoref's own GitHub Release.

## Install

```bash
pip install oncoref
```

## Python API

```python
import oncoref as od

od.resolve_cancer_type("prostate")        # -> "PRAD"
od.cancer_type_info("SARC_RMS_ARMS")      # full registry record + burden + tmb
od.cancer_tmb("LUAD_EGFR")                # 6.9  (inherited from LUAD)
od.cancer_burden("pancreas", metric="us_mortality_pct")
od.burden_category("SARC_OS")             # -> "bone_and_joint" (incidence/mortality bucket)
od.cancer_ici_response("SKCM")            # 42  (anti-PD-1 ORR %; fallback aPD-1 → aPD-L1 → combo)
od.cancer_ici_response("SKCM", regimen="PD-1+CTLA-4")   # 57.6  (pin a regimen)

# Cancer-testis antigens (HPA-derived tissue-restriction):
od.cta_gene_names()                       # expressed CTA symbols (MAGEA4, CT83, …)
od.cta_evidence()                         # full HPA restriction table

# Per-cohort expression percentiles (downloads the data bundle on first use):
od.cohort_gene_percentiles("PRAD")        # per-gene p0…p100 vector (within-cohort)
od.within_sample_top_fraction("PRAD")     # per-gene frac of samples top-5% (within-sample)
```

### Domains

- **Ontology** — `cancer_type_registry`, `resolve_cancer_type`,
  `cancer_type_info`, `cancer_types_in_family`, `viral_status`, `fusion_status`,
  the cohort vocabulary (`cohort_registry`, `cohort_aggregates`).
- **TMB** — `cancer_tmb`, `cancer_tmb_df` (parent-chain inheritance).
- **Incidence / mortality** — `cancer_burden`, `burden_category` (ACS / GLOBOCAN).
- **Checkpoint response** — `cancer_ici_response` (ORR per type/regimen: anti-PD-1,
  anti-PD-L1, anti-PD-1+anti-CTLA-4), with `cancer_apd1_response` the PD-1 shortcut.
- **Expression** — `cohort_gene_percentiles`, `within_sample_top_fraction`,
  `representative_cohort_samples` over the lazy-downloaded per-cohort bundle.
- **Cancer-testis antigens** — `cta_gene_names`/`cta_gene_ids`, `cta_evidence`,
  `synthesize_restriction` (HPA-only tissue-restriction; MS evidence stays in the
  target-selection layer).
- **HPA normal tissue** — `hpa_rna_consensus`, `hpa_normal_tissue` (IHC),
  `hpa_single_cell`, and per-gene lookups (`gene_tissue_ntpm`,
  `gene_protein_tissues`, `gene_cell_type_ntpm`) over HPA v23, fetched on demand
  (`oncoref hpa fetch`).
- **Genome reference** — `canonical_gene_id_and_name`, `find_gene_id_by_name`,
  `find_gene_name_from_ensembl_{gene,transcript}_id`, `aggregate_gene_expression`
  (pyensembl-backed symbol ↔ Ensembl-ID resolution). pyensembl ships with the
  package, but resolution needs a downloaded human release once:
  `pyensembl install --release 111 --species homo_sapiens` (the accessors return
  `None` until then).
- **Peptides** — `cta_specific_9mer_counts`, `cta_specific_9mer_load` (per-cohort
  mean per-patient CTA-specific 9-mer load): 9-mers found in a CTA protein but in no
  non-CTA protein, enumerated from the reference proteome and cached per release.
- **Plots** (`pip install oncoref[plots]`) — `oncoref.plots.apd1_vs_tmb`,
  `apd1_orr_bars`, `incidence_vs_mortality`, the CTA/coverage figures, and
  `oncoref.cta_curation_plots.render`.

## CLI

```bash
oncoref cancer-type prostate     # registry info as JSON
oncoref tmb LUAD_EGFR            # 6.9
oncoref ici SKCM                # 42  (--regimen to pin, --all-regimens to compare)
oncoref burden pancreas --metric us_mortality_pct
oncoref cta --count             # number of expressed CTAs
oncoref plot apd1-vs-tmb --out apd1_vs_tmb.png
oncoref plot patient-coverage --gene-set cta --out coverage_out
oncoref plot cta-curation --out cta_curation_out

# expression-bundle cache (per-cohort expression):
oncoref cache fetch             # download the ~340 MB bundle
oncoref cache status            # which bundle paths are cached (no download)
oncoref cache dir               # where the data bundle is cached
oncoref cache prune --yes       # delete stale version caches
oncoref hpa fetch               # download HPA reference data (RNA / IHC / single-cell)
oncoref version
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
