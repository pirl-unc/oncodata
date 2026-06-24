# oncoref API Guide

`oncoref` keeps its historical flat top-level imports for compatibility, but new
code should prefer the semantic submodules below. They make the domain boundary
clear and avoid guessing whether a broad name such as `coverage` or `peptides` is
general or CTA-specific.

## Cancer Vocabulary

- `oncoref.cancer_ontology` — cancer-type registry, aliases, parent/child tree,
  lineage/family groupings, molecular subtype axes, matched normal tissues,
  source-scoped evidence resolution, and display helpers.
- `oncoref.cohorts` — expression/source cohort IDs, computed aggregate cohorts,
  source versions, and mixture-cohort flags.

Use these when asking "what cancer type or cohort does this code mean?"
Prefer the DataFrame-returning query helpers when code will be passed into
other oncoref domains; they keep the result type and columns stable.

```python
from oncoref import cancer_ontology, cohorts, expression

cancer_ontology.resolve_cancer_type("prostate")
cancer_ontology.cancer_type_tree("CRC")
cancer_ontology.cancer_type_path("COAD_MSI")

# CRC plus anatomical children and molecular leaves.
crc = cancer_ontology.cancer_type_records(under="CRC")
crc["code"].tolist()

# Cross-cutting molecular axes can be intersected with hierarchy or lineage.
msi_crc = cancer_ontology.cancer_type_records(subtype_group="MSI", under="CRC")
epithelial_msi = cancer_ontology.cancer_type_records(
    subtype_group="MSI", lineage_group="Epithelial"
)

# COAD_MSI / READ_MSI keep anatomical expression context but resolve evidence
# rows through CRC_MSI when published sources are colorectal-level.
msi_crc[["code", "evidence_source_code", "normal_tissue_code", "hpa_tissues"]]

# Join scalar references for the returned codes.
cancer_ontology.cancer_type_reference_data(msi_crc)

# Use codes directly with expression accessors.
codes = cancer_ontology.cancer_type_codes(subtype_group="MSI", under="CRC")
expression.cancer_reference_expression(codes)

# Matched normal RNA expression is an explicit HPA read.
cancer_ontology.matched_normal_tissue_expression("COAD", genes=["ENSG00000141510"])

cohorts.cohort_registry_df()
```

## ICI Response

- `oncoref.ici_response` — checkpoint-inhibitor response anchors, anti-PD-1
  shortcuts, regimen-aware lookups, extracted endpoint estimates, and pooled
  response summaries.

`DEFAULT_ICI_REGIMEN_PRIORITY` is the unpinned regimen priority
(`PD-1`, then `PD-L1`, then `PD-1+CTLA-4`). The older
`REGIMEN_FALLBACK` name remains available in `oncoref.ici` for compatibility.

```python
from oncoref import ici_response

ici_response.apd1_response("SKCM")
ici_response.best_available_ici_response("SARC_ASPS")
ici_response.ici_response_by_regimen("SKCM")
ici_response.ici_response_estimates_df()
```

## CTA Antigens

- `oncoref.cta` — CTA definition, HPA restriction tiers, axes, aliases, and gene
  ID/name sets.
- `oncoref.cta_coverage` — CTA patient coverage over per-sample expression
  matrices.
- `oncoref.cta_peptides` — CTA-specific 9-mer counts and load.

`cta_specific_9mer_count_map()` returns a map from a join key to
`n_specific_9mers`; those counts are used as weights when computing
`cta_specific_9mer_load()`.

```python
from oncoref import cta, cta_coverage, cta_peptides

cta.cta_gene_names()
cta_coverage.cta_addressable_fraction("LUAD")
cta_peptides.cta_specific_9mer_count_map(by="proteoform_key")
```

## Generic Antigen Panels

- `oncoref.antigen_coverage` — explicit-gene-panel coverage helpers.

Use this when the panel is not necessarily CTA. The function names require
`gene_ids=` so a caller cannot accidentally rely on the CTA default.

```python
from oncoref import antigen_coverage

antigen_coverage.addressable_antigen_fraction("LUAD", gene_ids={"ENSG00000141510"})
antigen_coverage.greedy_antigen_coverage("LUAD", gene_ids={"ENSG00000141510"})
```

## Expression And Normalization

- `oncoref.expression` — read-time accessors for per-sample expression,
  percentile vectors, representative samples, within-sample top fractions, and
  pan-cancer reference tables. `sample_expression_qc` reports per-sample
  detected-gene counts, top-gene/top-10 concentration, biological-housekeeping
  detection, source-scale class, and source-type caveats so sparse source-matrix
  artifacts can be audited before using absolute TPM floors or housekeeping
  normalization. `per_sample_expression(..., sample_qc="pass" | "pass_or_warn"
  | "all")` filters sample columns at read time; the raw per-sample accessor
  defaults to `"all"` for forensic access, while live summaries such as
  `cohort_stats` and `pooled_cohort_stats` default to QC-passing samples.
- `oncoref.expression_builders` — pure build-time cores used by data-bundle
  generation scripts.
- `oncoref.normalization` — TPM conversion, clean TPM, technical-RNA filtering,
  log transforms, percentile ranks, and housekeeping normalization.

Clean TPM has one public compartment contract:

- `clean-tpm-censored-genes.csv:category == "ribosomal_protein"` — 16%
  ribosomal compartment.
- `clean-tpm-censored-genes.csv:category == "technical"` — 9% other-technical
  compartment.
- genes absent from the censored table — 75% biological compartment.

The category-specific helper sets are available from `oncoref.gene_families`:

```python
from oncoref import gene_families

gene_families.clean_tpm_ribosomal_gene_ids()
gene_families.clean_tpm_other_technical_gene_ids()
gene_families.clean_tpm_censored_gene_ids()
```

For clean-TPM housekeeping denominators, use the biological HPA-stable panel:

```python
gene_families.clean_tpm_biological_housekeeping_gene_ids()
gene_families.clean_tpm_biological_housekeeping_genes()
gene_families.clean_tpm_biological_housekeeping_genes(primary_only=False)
```

The legacy qPCR/reference-gene panel remains available as
`legacy_qpcr_housekeeping_*` and through the historical `housekeeping_*` helpers,
but it is not the clean-TPM biological denominator.

## Genes and Proteoforms

- `oncoref.gene_ids` — bundled canonical Ensembl gene space, cross-release alias
  resolution, symbol synonyms, and biotype checks.
- `oncoref.genome` — pyensembl-backed gene/transcript lookup against installed
  Ensembl releases.
- `oncoref.proteoforms` — identical-protein paralog grouping and expression
  collapse helpers.
- `oncoref.gene_qc` / `oncoref.gene_families` — technical-RNA and gene-family
  classification used by normalization.

## Burden, TMB, Fusions, and Signatures

- `oncoref.tmb` — tumor mutational burden reference values.
- `oncoref.incidence` — incidence/mortality burden and burden categories.
- `oncoref.fusions` — defining fusions and partner-family lookups.
- `oncoref.response_signatures` — therapy-response gene signatures and scoring.

## Data Management

- `oncoref.catalog` — unified dataset inventory and fetch/status/path operations.
- `oncoref.data_bundle` — heavy expression bundle cache.
- `oncoref.reference_data` / `oncoref.hpa` — HPA reference-data cache and HPA
  tissue/cell-type accessors.

## Compatibility Modules

These modules remain importable but are less discoverable than the organized
facades above:

- `oncoref.apd1` — legacy anti-PD-1 response slice; prefer
  `oncoref.ici_response`.
- `oncoref.ici` — core ICI implementation; prefer `oncoref.ici_response` for the
  organized public surface.
- `oncoref.coverage` — original mixed CTA/generic antigen-panel coverage module;
  prefer `oncoref.cta_coverage` or `oncoref.antigen_coverage`.
- `oncoref.peptides` — original CTA-specific 9-mer module; prefer
  `oncoref.cta_peptides`.
