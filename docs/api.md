# oncoref API Guide

`oncoref` keeps its historical flat top-level imports for compatibility, but new
code should prefer the semantic submodules below. They make the domain boundary
clear and avoid guessing whether a broad name such as `coverage` or `peptides` is
general or CTA-specific.

Package boundary: oncoref is the upstream home for shared reference mechanics
and data that are ready to be reused across the PIRL stack. Downstream packages
can keep package-specific curation, generated artifacts, and compatibility APIs;
when a missing data field, gene universe, bundle-integrity rule, or source-QC
decision affects generated artifacts, the durable fix should live or be exposed
here rather than only in a downstream compatibility layer.

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

The CTA definition here is the HPA-derived tissue-restriction call. Broader
therapy-target curation, MS evidence, and downstream prioritization rules can
live in consumer packages while they remain package-specific.

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
  `source_matrix_sample_qc_manifest`, `expression_artifact_build_metadata`, and
  `expression_artifact_build_summary` read the optional QC/build metadata emitted
  by regenerated expression bundles. Until a regenerated heavy bundle ships those
  files, they return schema-stable empty metadata by default; use
  `on_missing="raise"` when a downstream migration requires the manifests.
- `oncoref.expression_builders` — pure build-time cores used by data-bundle
  generation scripts. `scripts/rebuild_expression_artifacts.py` applies the same
  sample-QC policy to derived shards by default (`--sample-qc pass`) and emits
  `source-matrix-sample-qc.csv` plus `expression-artifact-build-metadata.*` in
  the staging directory so bundle releases record which source samples fed
  percentiles, representatives, proteoform summaries, and within-sample
  summaries. Representative
  sample selection uses `representative_sample_columns` / `cohort_medoids` on the
  biological clean-TPM view, then stores the selected samples' full
  clean_tpm_16_9_75 vectors. Release builds retain curated cohorts that have no
  strict QC-pass samples only through explicit source-aware fallbacks recorded in
  the build metadata, and clip invalid negative source expression values to zero
  with per-cohort counts.
- `oncoref.expression_engine` — reusable low-level builder primitives for
  expression tables: identity/value column detection, transcript-to-gene
  aggregation, source row ID-type detection, source gene-row mapping audits,
  missing-vs-non-parsing numeric diagnostics, and canonical ENSG aggregation in
  linear expression space. Use these in builders before committing a source
  matrix so unresolved high-expression rows and duplicate canonical IDs are
  explicit artifacts rather than hidden cleanup. Source audit CSV contracts are
  versioned by `SOURCE_GENE_MAPPING_AUDIT_SCHEMA_VERSION` and
  `SOURCE_VALUE_PARSE_DIAGNOSTIC_SCHEMA_VERSION`, and the emitted audit frames
  include those versions as persisted columns.
- `oncoref.source_matrices` — raw per-cohort source-matrix cache/fetch helpers.
  Use `source_matrices.sample_qc(code)` for the live source-matrix QC audit and
  `source_matrices.sample_qc_manifest(...)` for the optional generated-bundle QC
  manifest that records which samples fed derived artifacts.
- `oncoref.normalization` — TPM conversion, clean TPM, technical-RNA filtering,
  log transforms, percentile ranks, and housekeeping normalization.

The normalization helpers are intended to be reusable directly. Expression
accessors and bundles are also reusable, but downstream packages may keep their
own packaged expression artifacts until row-set, value, provenance, and QC
contracts are parity-clean for the specific accessor they want to replace.

`expression.pan_cancer_expression()` defaults to oncoref's entity-first schema:
HPA normal tissue columns are `<tissue>_nTPM_raw`, TCGA source/provenance
columns are `<CODE>_FPKM_raw`, deterministic TCGA TPM companions are
`<CODE>_TPM_raw`, and analysis columns append `_clean`, `_hk`, `_percentile`, or
`_log1p`. For migration code that needs pirlygenes' unsuffixed column names, use
`column_style="pirlygenes"`; the legacy `to_tpm=True` keyword is accepted as a
compatibility alias for that view and maps the default call to `normalize="tpm"`.

`expression.cancer_reference_expression()` returns cohort-level tumor reference
expression with stable long or wide output. It accepts canonical cancer codes,
aliases, and aggregate cohorts, resolves gene filters by ENSG or symbol, and can
return one or more normalization modes in one call:

- `normalize="tpm_clean"` / `"clean_tpm"` — shipped biological clean-TPM
  percentiles.
- `normalize="tpm_clean_biological"` — explicit name for that biological-only
  reference artifact.
- `normalize="tpm_clean_log1p"` — stored log1p biological clean-TPM percentiles.
- `normalize="tpm_raw"` / `"tpm"` — source-matrix raw TPM summaries recomputed
  through `cohort_stats`.

Long output includes source/provenance columns by default, including source
cohort, source type/unit, source scale class, reference method, `DATA_VERSION`,
and `SOURCE_MATRIX_VERSION`. This accessor is the compatibility surface for
reference-expression reads; expression artifact row-set/value parity is still
tracked separately in the upstream parity issues.
Raw source-matrix summaries default to `sample_qc="pass"` so sparse or
source-QC-failed samples do not shape newly computed reference rows. Use
`sample_qc="pass_or_warn"` or `"all"` for forensic audits or exact parity with
the current unfiltered source matrices. Long-form provenance and availability
rows report the live QC mode for raw source-matrix summaries and `artifact` for
existing shard-backed clean-TPM rows.

Use `expression.cancer_reference_expression_availability()` before delegating a
downstream reference-expression accessor that must distinguish unavailable
oncoref artifacts from empty gene filters. It returns one row per requested
code/mode with `requested_code`, expanded `cancer_code`, `request_kind`,
`available`, `missing_reason`, provenance fields, and the reference-expression
schema/data versions. `expression.cancer_reference_expression(...,
on_missing="empty")` returns a schema-stable empty frame and stores the same
missing rows in `df.attrs["missing_requests"]`; `on_missing="raise"` fails fast
for required cohorts. `include_request_metadata=True` adds request/availability
columns to long expression output, which is useful when a requested aggregate
expands to child expression cohorts.

Representative and percentile artifact readers have explicit downstream-facing
contracts:

- `expression.representative_cohort_samples(..., format="long",
  include_provenance=True)` includes the representative id, source cohort/project,
  source sample id, cohort sample count, deterministic selection rank/method/basis,
  artifact schema version, `DATA_VERSION`, and `SOURCE_MATRIX_VERSION`.
  Public representative ids default to pirlygenes-compatible `CODE_rep01`
  columns/values. Pass `representative_id_style="internal"` to expose the
  underlying bundle/provenance ids (`CODE__rep1`).
  Representatives are selected by central-medoid plus farthest-first traversal in
  log1p biological clean-TPM space, with stable sample-id tie-breaking; the
  persisted vectors remain full clean_tpm_16_9_75.
- `expression.cohort_gene_percentiles(..., include_provenance=True)` appends the
  cohort code, normalization, expression unit, percentile basis, artifact schema
  version, `DATA_VERSION`, and `SOURCE_MATRIX_VERSION`.
- Missing percentile shards still raise by default. Use
  `on_missing="empty"` to return an empty but schema-stable frame with
  `df.attrs["missing_reason"]`, which is useful for compatibility adapters that
  need to distinguish unavailable upstream data from private downstream fallback
  data.

The QC-policy expression bundle ships representative, percentile,
within-sample, CTA-scope proteoform percentile, CTA-scope proteoform
within-sample, sample-QC, and build-metadata artifacts. Non-shipped proteoform
scopes can still recompute from cached source matrices. Row-set and value parity
with pirlygenes is still governed by the gene-universe and expression-artifact
parity issues.

`expression.expression_artifact_gene_universe_deltas()` exposes the known
pirlygenes/oncoref row-universe deltas from the current parity audit: canonical
remaps such as legacy `PAXX` to its oncoref ENSG, CLL percentile replacement rows
that are known but absent from the current output, representative-sample rows
missing from oncoref, and examples of oncoref-only technical/noncoding rows.
Use `expression.expression_artifact_gene_universe_delta_summary()` for counts by
product/cohort/status. This table is intentionally provenance: it makes
differences explicit for migration code, but does not synthesize missing
expression rows or alter artifact values.

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
  `tmb.cancer_tmb_df()` includes evidence-schema columns (`estimate_type`,
  `source_scope`, `missing_reason`), and `tmb.cancer_tmb_record()` /
  `tmb.resolve_tmb_source()` preserve requested-code metadata for source-scoped
  lookups such as `COAD_MSI` or `READ_MSI` resolving through `CRC_MSI`.
- `oncoref.incidence` — incidence/mortality burden and burden categories.
- `oncoref.fusions` — defining fusions and partner-family lookups.
- `oncoref.response_signatures` — therapy-response gene signatures and scoring.

## Data Management

- `oncoref.catalog` — unified dataset inventory and fetch/status/path operations.
- `oncoref.data_bundle` — heavy expression bundle cache. Use
  `data_bundle.bundle_contract()` to inspect the downstream-stable package/data
  version linkage, release asset URLs, cache environment variables, completion
  marker policy, and expected artifact inventory for the active bundle. The
  inventory includes the generated sample-QC manifest, per-cohort build metadata,
  within-sample prevalence shards, and CTA-scope proteoform percentile/prevalence
  shards, not just the legacy pirlygenes expression tables. Use
  `data_bundle.bundle_release_manifest()` to fetch and validate only the small
  release manifest/checksum for the active `DATA_VERSION`, including tarball
  sha256 plus any artifact inventory, builder commit, source-matrix version, and
  sample-QC policy metadata published with the release.
  CLI equivalents are available for CI/notebooks: `oncoref data contract` prints
  the active bundle contract as JSON, and `oncoref data release-manifest
  [oncoref|pirlygenes]` prints the validated release manifest/checksum metadata
  as JSON.
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
