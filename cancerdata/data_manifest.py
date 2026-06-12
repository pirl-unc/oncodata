# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The complete, classified inventory of cancerdata-domain data.

This is the single declarative source of truth for *what data cancerdata owns*
and how each dataset is held. It keeps the base layer honest: every dataset is
classified into exactly one bucket, and a guard test (test_data_manifest.py)
asserts the buckets partition the frozen pirlygenes inventory exhaustively and
disjointly — so as cancerdata absorbs pirlygenes, nothing is silently dropped or
double-owned.

Held buckets (cancerdata domain):
  WHEEL     — small curated tables shipped in the wheel (no fetch).
  BUNDLE    — heavy expression artifacts in the version-pinned release tarball.
  HPA       — Human Protein Atlas reference tables fetched per-source on demand.
  SOURCE    — the raw per-sample TPM matrices (a separate large optional bundle).
  PLANNED   — cancerdata-domain tables still to port (they feed the normalization
              / CTA-regeneration / ontology phases).
  SUPERSEDED— a pirlygenes table cancerdata replaced with its own regenerated one.

OUT_OF_SCOPE — pirlygenes data that is NOT cancerdata's domain (target selection,
  therapy/modality, surfaceome, analysis gene sets) and stays in tsarina/hitlist.

The catalog (:mod:`cancerdata.catalog`) manages the fetchable subset; this module
is the inventory + classification.
"""

from __future__ import annotations

#: {name: (category, description)} — small curated tables shipped in the wheel.
WHEEL: dict[str, tuple[str, str]] = {
    "cancer-type-registry": ("ontology", "cancer-type codes, hierarchy, family/tissue"),
    "cancer-subtype-groupings": ("ontology", "cross-cutting MSI/POLE/HPV/MYCN axes"),
    "cancer-cohort-aggregates": ("ontology", "computed histology rollups (SARC_RMS, …)"),
    "cohort-registry": ("ontology", "first-class cohort vocabulary"),
    "cancer-fusions": ("ontology", "characteristic driver fusions per cancer type"),
    "cancer-code-burden-map": ("ontology", "anatomic burden-category overrides"),
    "cancer-incidence-mortality": ("epidemiology", "incidence/mortality by burden category"),
    "cancer-tmb": ("genomics", "median tumor mutational burden per type"),
    "cancer-apd1-response": ("response", "anti-PD-1 monotherapy ORR per type"),
    "cancer-reference-expression-samples": ("expression", "per-sample curation manifest"),
    "expression_sources": ("expression", "cohort expression-source registry"),
    # normalization references (R-norm; consumed by the clean_tpm_v4 engine)
    "housekeeping-genes": ("normalization", "housekeeping panel for normalization"),
    "censored-gene-reference-tpm": ("normalization", "fixed surrogate TPM for censored genes"),
    "clean-tpm-censored-genes": (
        "normalization",
        "technical+ribosomal genes censored by clean_tpm_v4",
    ),
    "histone-genes": ("normalization", "histone-cluster genes"),
    "ribosomal-protein-genes": ("normalization", "ribosomal-protein genes"),
    "ribosomal-protein-pseudogenes": ("normalization", "ribosomal-protein pseudogenes"),
    "mitochondrial-genes": ("normalization", "mtDNA-encoded genes"),
    "rrna-and-pseudogenes": ("normalization", "rRNA + rRNA pseudogenes"),
    "numt-pseudogenes": ("normalization", "NUMT-like nuclear-mito pseudogenes"),
    "small-noncoding-rnas": ("normalization", "small non-coding RNA loci"),
    "nuclear-retained-lncrnas": ("normalization", "polyA-bias nuclear-retained lncRNAs"),
    "hemoglobin-genes": ("normalization", "hemoglobin genes"),
}

#: {name: (category, description)} — heavy artifacts in the release tarball.
#: Names match ``data_bundle.DOWNLOADABLE_PATHS`` (sans extension); kept in sync by a test.
BUNDLE: dict[str, tuple[str, str]] = {
    "cancer-reference-expression": ("expression", "per-cohort RNA-seq summary shards"),
    "cancer-reference-expression-percentiles": ("expression", "per-gene percentile vectors"),
    "cancer-reference-expression-representatives": ("expression", "per-cohort medoid samples"),
    "pan-cancer-expression": ("expression", "pan-cancer HPA-tissue + TCGA matrix"),
    "hpa-cell-type-expression": ("hpa", "HPA cell-type nTPM matrix"),
}

#: {name: (category, description)} — HPA tables fetched per-source (not pirlygenes files).
HPA: dict[str, tuple[str, str]] = {
    "hpa_rna_consensus": ("hpa", "HPA RNA consensus per-tissue nTPM"),
    "hpa_normal_tissue": ("hpa", "HPA IHC protein detection per tissue"),
    "hpa_single_cell": ("hpa", "HPA single-cell-type RNA nTPM"),
}

#: The raw per-sample TPM matrices — every derived artifact is built from these.
SOURCE: dict[str, tuple[str, str]] = {
    "per-sample-tpm-matrices": ("expression", "raw per-sample cohort TPM (build inputs)"),
}

#: {name: (category, description)} — cancerdata-domain tables still to port.
PLANNED: dict[str, tuple[str, str]] = {
    # gene-id / symbol resolution
    "ensembl-id-aliases": ("gene-id", "retired→current Ensembl gene-id aliases"),
    "ncbi-symbol-synonyms": ("gene-id", "NCBI gene symbol synonyms"),
    "extra-tx-mappings": ("gene-id", "supplemental transcript→gene mappings"),
    "cdna-identical-gene-groups": ("gene-id", "cDNA-identical gene groups"),
    "proteoform-collapse-overrides": ("gene-id", "manual proteoform-collapse overrides"),
    # ontology metadata (O5)
    "cancer-key-genes": ("ontology", "per-type biomarkers + therapy targets"),
    "cancer-driver-genes": ("ontology", "per-type driver genes"),
    "cancer-driver-variants": ("ontology", "per-type driver variants"),
    "cancer-type-genes": ("ontology", "role-stratified per-type genes"),
    "cancer-viral-antigens": ("ontology", "per-oncovirus targetable antigens"),
    "disease-state-rules": ("ontology", "narrative disease-state rules"),
    "narrative-gene-sets": ("ontology", "named narrative gene sets"),
    "degenerate-subtype-pairs": ("ontology", "expression-degenerate subtype pairs"),
    "rare-cancer-fusion-rules": ("ontology", "direct fusion rules for rare cancers"),
    "fusion-surrogate-expression": ("ontology", "expression surrogates for fusions"),
    "fusion-expression-effects": ("ontology", "downstream-expression rules per fusion"),
    # expression-source metadata
    "cancer-expression-source-candidates": ("expression", "candidate expression sources per type"),
    "cancer-frameshift-burden": ("genomics", "per-type frameshift-indel burden"),
}

#: pirlygenes tables cancerdata replaced with its own regenerated equivalent.
SUPERSEDED: dict[str, str] = {
    "protein-identical-gene-groups": "proteoform-groups-genome (byte-identical, regenerated)",
    "cta-protein-groups": "proteoform-groups (byte-identical; pirlygenes' is ≥90% identity)",
}

#: {name: (category, description)} — cancerdata-ORIGINATED wheel tables: derived or
#: regenerated here rather than copied from pirlygenes, so they aren't in the
#: pirlygenes snapshot but DO ship in the wheel and belong in the inventory.
CANCERDATA_ORIGINATED: dict[str, tuple[str, str]] = {
    "cancer-testis-antigens": ("cta", "CTA definition (HPA tissue-restriction over candidates)"),
    "proteoform-groups": ("gene-id", "byte-identical CTA proteoform groups"),
    "proteoform-groups-genome": ("gene-id", "byte-identical proteoform groups (genome-wide)"),
    "source-matrices": ("expression", "per-cohort raw-matrix registry (code/source/n_samples)"),
}

#: pirlygenes data that is NOT cancerdata's domain — target selection / therapy /
#: surfaceome / analysis gene sets. These stay in tsarina / hitlist.
OUT_OF_SCOPE: frozenset[str] = frozenset(
    {
        # therapeutic modality / drug data
        "ADC-approved",
        "ADC-trials",
        "ADC-withdrawn",
        "bispecific-antibodies-approved",
        "CAR-T-approved",
        "TCR-T-approved",
        "TCR-T-trials",
        "multispecific-tcell-engager-trials",
        "radioligand-targets",
        # surfaceome / targetability
        "surface-proteins",
        "cancer-surfaceome",
        # therapy signatures / evidence
        "therapy-benefit-toxicity-evidence",
        "therapy-response-signatures",
        "estimate-signatures",
        "immune-receptor-segments",
        # analysis / QC gene sets
        "tme-markers",
        "stem-cell-marker-panels",
        "culture-stress-genes",
        "ffpe-sensitive-markers",
        "degradation-gene-pairs",
        "mutation-expression-effects",
        "rare-cancer-rna-surrogates",
        # differential-expression signatures
        "tumor-up-vs-matched-normal",
        "heme-tumor-up-vs-matched-normal",
        # gene-set panels (analysis / target selection)
        "cancer-lineage-panels",
        "cancer-family-panels",
        "lineage-genes",
        "gene-sets",
        # build QC
        "artifact-expectations",
    }
)

#: Frozen snapshot of pirlygenes' shipped ``data/`` inventory (file stems + artifact
#: dirs), 2026-06. The guard test partitions this against the buckets above; a new
#: pirlygenes dataset must be consciously classified rather than silently missed.
PIRLYGENES_DATA: frozenset[str] = frozenset(
    {
        "ADC-approved",
        "ADC-trials",
        "ADC-withdrawn",
        "artifact-expectations",
        "bispecific-antibodies-approved",
        "cancer-apd1-response",
        "cancer-code-burden-map",
        "cancer-cohort-aggregates",
        "cancer-driver-genes",
        "cancer-driver-variants",
        "cancer-expression-source-candidates",
        "cancer-family-panels",
        "cancer-frameshift-burden",
        "cancer-fusions",
        "cancer-incidence-mortality",
        "cancer-key-genes",
        "cancer-lineage-panels",
        "cancer-reference-expression",
        "cancer-reference-expression-percentiles",
        "cancer-reference-expression-representatives",
        "cancer-reference-expression-samples",
        "cancer-subtype-groupings",
        "cancer-surfaceome",
        "cancer-tmb",
        "cancer-type-genes",
        "cancer-type-registry",
        "cancer-viral-antigens",
        "CAR-T-approved",
        "cdna-identical-gene-groups",
        "censored-gene-reference-tpm",
        "clean-tpm-censored-genes",
        "cohort-registry",
        "cta-protein-groups",
        "culture-stress-genes",
        "degenerate-subtype-pairs",
        "degradation-gene-pairs",
        "disease-state-rules",
        "ensembl-id-aliases",
        "estimate-signatures",
        "expression_sources",
        "extra-tx-mappings",
        "ffpe-sensitive-markers",
        "fusion-expression-effects",
        "fusion-surrogate-expression",
        "gene-sets",
        "heme-tumor-up-vs-matched-normal",
        "hemoglobin-genes",
        "histone-genes",
        "housekeeping-genes",
        "hpa-cell-type-expression",
        "immune-receptor-segments",
        "lineage-genes",
        "mitochondrial-genes",
        "multispecific-tcell-engager-trials",
        "mutation-expression-effects",
        "narrative-gene-sets",
        "ncbi-symbol-synonyms",
        "nuclear-retained-lncrnas",
        "numt-pseudogenes",
        "pan-cancer-expression",
        "protein-identical-gene-groups",
        "proteoform-collapse-overrides",
        "radioligand-targets",
        "rare-cancer-fusion-rules",
        "rare-cancer-rna-surrogates",
        "ribosomal-protein-genes",
        "ribosomal-protein-pseudogenes",
        "rrna-and-pseudogenes",
        "small-noncoding-rnas",
        "stem-cell-marker-panels",
        "surface-proteins",
        "TCR-T-approved",
        "TCR-T-trials",
        "therapy-benefit-toxicity-evidence",
        "therapy-response-signatures",
        "tme-markers",
        "tumor-up-vs-matched-normal",
    }
)


def captured() -> set[str]:
    """cancerdata-domain datasets already held (wheel + bundle + superseded)."""
    return set(WHEEL) | set(BUNDLE) | set(SUPERSEDED)


def in_scope() -> set[str]:
    """Every cancerdata-domain dataset — captured plus still-planned."""
    return captured() | set(PLANNED)
