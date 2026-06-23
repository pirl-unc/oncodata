# oncoref API Guide

`oncoref` keeps its historical flat top-level imports for compatibility, but new
code should prefer the semantic submodules below. They make the domain boundary
clear and avoid guessing whether a broad name such as `coverage` or `peptides` is
general or CTA-specific.

## Cancer Vocabulary

- `oncoref.cancer_ontology` — cancer-type registry, aliases, parent/child tree,
  lineage/family groupings, and display helpers.
- `oncoref.cohorts` — expression/source cohort IDs, computed aggregate cohorts,
  source versions, and mixture-cohort flags.

Use these when asking "what cancer type or cohort does this code mean?"

```python
from oncoref import cancer_ontology, cohorts

cancer_ontology.resolve_cancer_type("prostate")
cancer_ontology.cancer_type_tree("CRC")
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

## Expression

- `oncoref.expression` — read-time accessors for per-sample expression,
  percentile vectors, representative samples, within-sample top fractions, and
  pan-cancer reference tables.
- `oncoref.expression_builders` — pure build-time cores used by data-bundle
  generation scripts.
- `oncoref.normalization` — TPM conversion, clean TPM, technical-RNA filtering,
  log transforms, percentile ranks, and housekeeping normalization.

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
