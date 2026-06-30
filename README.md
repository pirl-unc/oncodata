# oncoref

[![Tests](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/oncoref.svg)](https://pypi.org/project/oncoref/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, checkpoint-inhibitor (ICI) response, per-cohort
RNA-seq expression, HPA normal-tissue expression, and HPA-derived
cancer-testis antigen references — behind one small Python API, a data
fetch/cache CLI, and a set of reference plots.

## oncoref is the base layer

`oncoref` is **designed as the base layer** of the openvax/PIRL stack — the
intended upstream home for shared cancer reference mechanics and data, meant to
become a common dependency of
[pirlygenes](https://github.com/pirl-unc/pirlygenes),
[trufflepig](https://github.com/pirl-unc/trufflepig), and
[tsarina](https://github.com/pirl-unc/tsarina). Adoption is staged: downstream
packages can delegate parity-clean primitives while keeping their own curated
tables, packaged artifacts, and compatibility APIs until those surfaces are
ready to move. Architecturally oncoref stays at the bottom: it depends only on
pandas / numpy / pyarrow / PyYAML, it **never imports its consumers** (data and
logic flow only downward), and shared definitions should be fixed or exposed
here rather than reimplemented separately downstream.

Use oncoref for shared questions about

- **gene expression of cancer samples** — per-cohort RNA-seq in a normalized,
  comparable space: summary stats, tail-weighted percentiles, and medoid/exemplar
  samples per cancer type/subtype. Downstream packages may still keep packaged
  expression artifacts and compatibility wrappers while parity checks converge;
- **HPA protein / RNA** normal-tissue expression;
- the **HPA-derived cancer-testis antigen call** — the HPA tissue-restriction
  call over the candidate list (HPA-only; no pirlygenes therapy/MS curation
  layer);
- the **ontology of cancer types** — codes, the parent/child hierarchy, subtypes,
  families, characteristic driver fusions, and the cross-cutting MSI/POLE/HPV
  groupings; and
- **checkpoint-inhibitor response rates** and **TMB** per cancer type.

Everything keys on the cancer-type registry. The small curated tables ship in the
wheel; the heavy per-cohort expression bundle downloads on first use from
oncoref's own GitHub Release.

## Install

```bash
pip install oncoref
```

## Python API

The flat `oncoref` namespace remains available for compatibility and quick
interactive use. For new code, prefer the semantic submodules in
[docs/api.md](docs/api.md); they make it clearer whether you are working with
the cancer ontology, cohorts, ICI response, CTA coverage, generic antigen-panel
coverage, or CTA-specific peptides.

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

- **Cancer ontology** — `oncoref.cancer_ontology`: `cancer_type_registry`,
  `resolve_cancer_type`, `cancer_type_records`, `cancer_type_codes`,
  `cancer_type_path`, `cancer_type_reference_data`, tree/family/lineage helpers,
  molecular subtype groups, source-scoped evidence resolution, matched normal
  tissue helpers, `viral_status`, `fusion_status`.
- **Cohorts** — `oncoref.cohorts`: `cohort_registry`, `cohort_aggregates`,
  `cohort_source_version`, and mixture-cohort helpers.
- **TMB** — `cancer_tmb`, `cancer_tmb_df` (parent-chain inheritance).
- **Incidence / mortality** — `cancer_burden`, `burden_category` (ACS / GLOBOCAN).
- **Checkpoint response** — `oncoref.ici_response`: regimen-aware ORR anchors,
  anti-PD-1 shortcuts, endpoint estimates, and pooled response summaries.
- **Expression** — `cohort_gene_percentiles`, `within_sample_top_fraction`,
  `representative_cohort_samples` over the lazy-downloaded per-cohort bundle.
- **Clean TPM / normalization** — `oncoref.normalization` for the 16/9/75
  compartment transform and `oncoref.gene_families` for clean-TPM censored
  compartment IDs plus the biological housekeeping denominator panel.
- **Cancer-testis antigens** — `oncoref.cta`: `cta_gene_names`/`cta_gene_ids`,
  `cta_evidence`, `synthesize_restriction` (HPA-only tissue-restriction; MS
  evidence stays in the target-selection layer).
- **CTA coverage / peptides** — `oncoref.cta_coverage` for patient coverage and
  `oncoref.cta_peptides` for CTA-specific 9-mer count maps and load.
- **Generic antigen-panel coverage** — `oncoref.antigen_coverage` for explicit
  non-CTA panels.
- **HPA normal tissue** — `hpa_rna_consensus`, `hpa_normal_tissue` (IHC),
  `hpa_single_cell`, and per-gene lookups (`gene_tissue_ntpm`,
  `gene_protein_tissues`, `gene_cell_type_ntpm`) over HPA v23, fetched on demand
  (`oncoref data fetch hpa`).
- **Genome reference** — `canonical_gene_id_and_name`, `find_gene_id_by_name`,
  `find_gene_name_from_ensembl_{gene,transcript}_id`, `aggregate_gene_expression`
  (pyensembl-backed symbol ↔ Ensembl-ID resolution). pyensembl ships with the
  package, but resolution needs a downloaded human release once:
  `pyensembl install --release 111 --species homo_sapiens` (the accessors return
  `None` until then).
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

# managed data downloads/cache:
oncoref data list               # every wheel/bundle/HPA/source dataset
oncoref data status bundle      # expression-bundle cache state (no download)
oncoref data fetch bundle       # download the large expression bundle
oncoref data fetch hpa          # download HPA reference data (RNA / IHC / single-cell)
oncoref data dir bundle         # where the data bundle is cached
oncoref data prune --yes        # delete stale bundle version caches
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
