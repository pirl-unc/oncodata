# oncoref

`oncoref` is the base-layer package for curated cancer reference data: cancer
ontology, cohorts, expression, clean-TPM normalization, TMB, incidence/mortality,
ICI response, and cancer-testis antigen references.

Start with the [API guide](api.md) for the organized public modules and where to
look for each data domain.

## Key Entry Points

- `oncoref.cancer_ontology` — cancer-type registry, aliases, hierarchy, and
  display helpers.
- `oncoref.cohorts` — expression/source cohort IDs and aggregate cohorts.
- `oncoref.ici_response` — anti-PD-1 and broader ICI response references.
- `oncoref.expression` — per-sample, summary, representative, and pan-cancer
  expression accessors.
- `oncoref.normalization` — clean TPM, housekeeping normalization, log transforms,
  and percentile ranks.
- `oncoref.gene_families` — clean-TPM censored compartments, biological
  housekeeping denominators, and gene-family reference sets.
- `oncoref.cta`, `oncoref.cta_coverage`, and `oncoref.cta_peptides` — CTA
  definition, patient coverage, and CTA-specific 9-mer counts/load.
