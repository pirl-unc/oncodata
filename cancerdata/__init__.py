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

"""cancerdata — curated cancer reference data (ontology, TMB, incidence/
mortality, and expression) with a single fetch/cache surface.

Bottom-of-stack: depends only on pandas/numpy/pyarrow, never on the analysis
or target-selection libraries that consume it.
"""

from .apd1 import cancer_apd1_response, cancer_apd1_response_df
from .cancer_types import (
    CANCER_TYPE_ALIASES,
    CANCER_TYPE_NAMES,
    cancer_lineage_group,
    cancer_lineage_group_overrides,
    cancer_lineage_groups,
    cancer_subtype_group,
    cancer_subtype_groupings,
    cancer_type_ancestors,
    cancer_type_descendants,
    cancer_type_families,
    cancer_type_info,
    cancer_type_lineage,
    cancer_type_registry,
    cancer_type_subtypes_of,
    cancer_type_synonyms,
    cancer_type_tree,
    cancer_types_by_tissue,
    cancer_types_in_family,
    canonical_cancer_code,
    cohort_aggregate_members,
    cohort_aggregates,
    cohort_aggregates_df,
    cohort_kind,
    cohort_registry,
    cohort_registry_df,
    family_display_name,
    format_cancer_code_label,
    fusion_status,
    is_mixture_cohort,
    known_cohort_ids,
    mixture_cohort_codes,
    resolve_cancer_type,
    sarcoma_lineage_codes,
    tissue_of_origin,
    viral_status,
)
from .coverage import (
    addressable_fraction,
    addressable_fraction_by_cohort,
    cta_patient_fractions,
    greedy_coverage,
)
from .cta import (
    CTA_evidence,
    CTA_excluded_gene_names,
    CTA_filtered_gene_ids,
    CTA_filtered_gene_names,
    CTA_gene_id_to_name,
    CTA_gene_ids,
    CTA_gene_names,
    CTA_never_expressed_gene_names,
    CTA_unfiltered_gene_ids,
    CTA_unfiltered_gene_names,
    cta_dataframe,
)
from .expression import (
    available_percentile_cohorts,
    available_representative_cohorts,
    available_within_sample_cohorts,
    cohort_gene_percentiles,
    cohort_mean_expression,
    per_sample_expression,
    proteoform_representative_samples,
    representative_cohort_samples,
    within_sample_top_fraction,
)
from .expression_engine import aggregate_transcripts_to_genes
from .expression_registry import (
    ExpressionSource,
    expression_source,
    expression_sources,
    registry_dataframe,
    sources_for_cancer_code,
)
from .fusions import (
    cancer_fusions,
    cancer_fusions_df,
    cancer_types_with_fusion,
    fusion_partners,
    protein_family,
)
from .gene_qc import GeneQcClass, classify_gene_qc, is_rescue_feature
from .genome import (
    aggregate_gene_expression,
    canonical_gene_id_and_name,
    canonical_gene_ids_and_names,
    find_gene_id_by_name,
    find_gene_name_from_ensembl_gene_id,
    find_gene_name_from_ensembl_transcript_id,
    genomes,
)
from .hpa import (
    gene_cell_type_ntpm,
    gene_protein_tissues,
    gene_tissue_ntpm,
    hpa_normal_tissue,
    hpa_rna_consensus,
    hpa_single_cell,
)
from .incidence import (
    burden_category,
    cancer_burden,
    cancer_burden_df,
    cancer_code_burden_map,
)
from .normalization import (
    clean_tpm,
    fpkm_to_tpm,
    normalize_expression,
    normalize_technical_rna_columns,
    normalize_technical_rna_long_table,
    renormalize_to_million,
    tpm_to_housekeeping_normalized,
)
from .proteoforms import (
    collapse_to_proteoforms,
    gene_to_proteoform,
    gene_to_proteoform_id,
    proteoform_for_gene,
    proteoform_group_map,
    proteoform_groups,
    proteoform_symbol_map,
)
from .response_signatures import (
    response_signature_direction,
    response_signature_genes,
    response_signature_names,
    response_signatures_df,
    signature_score,
)
from .samples import (
    sample_counts_by_cancer_code,
    sample_manifest,
    samples_for_cancer_code,
    samples_for_cohort,
)
from .tmb import cancer_tmb, cancer_tmb_df
from .version import __version__

__all__ = [
    # ontology / registry
    "CANCER_TYPE_ALIASES",
    "CANCER_TYPE_NAMES",
    # cancer-testis antigens
    "CTA_evidence",
    "CTA_excluded_gene_names",
    "CTA_filtered_gene_ids",
    "CTA_filtered_gene_names",
    "CTA_gene_id_to_name",
    "CTA_gene_ids",
    "CTA_gene_names",
    "CTA_never_expressed_gene_names",
    "CTA_unfiltered_gene_ids",
    "CTA_unfiltered_gene_names",
    # expression sources + per-sample curation
    "ExpressionSource",
    "GeneQcClass",
    "__version__",
    # expression (read accessors over the downloadable bundle)
    "addressable_fraction",
    "addressable_fraction_by_cohort",
    "aggregate_gene_expression",
    "aggregate_transcripts_to_genes",
    "available_percentile_cohorts",
    "available_representative_cohorts",
    "available_within_sample_cohorts",
    "burden_category",
    # anti-PD-1 response
    "cancer_apd1_response",
    "cancer_apd1_response_df",
    # incidence / mortality
    "cancer_burden",
    "cancer_burden_df",
    "cancer_code_burden_map",
    "cancer_fusions",
    "cancer_fusions_df",
    "cancer_lineage_group",
    "cancer_lineage_group_overrides",
    "cancer_lineage_groups",
    "cancer_subtype_group",
    "cancer_subtype_groupings",
    # TMB
    "cancer_tmb",
    "cancer_tmb_df",
    "cancer_type_ancestors",
    "cancer_type_descendants",
    "cancer_type_families",
    "cancer_type_info",
    "cancer_type_lineage",
    "cancer_type_registry",
    "cancer_type_subtypes_of",
    "cancer_type_synonyms",
    "cancer_type_tree",
    "cancer_types_by_tissue",
    "cancer_types_in_family",
    "cancer_types_with_fusion",
    "canonical_cancer_code",
    "canonical_gene_id_and_name",
    "canonical_gene_ids_and_names",
    "classify_gene_qc",
    "clean_tpm",
    "cohort_aggregate_members",
    # cohort vocabulary
    "cohort_aggregates",
    "cohort_aggregates_df",
    "cohort_gene_percentiles",
    "cohort_kind",
    "cohort_mean_expression",
    "cohort_registry",
    "cohort_registry_df",
    "collapse_to_proteoforms",
    "cta_dataframe",
    "cta_patient_fractions",
    "expression_source",
    "expression_sources",
    "family_display_name",
    "find_gene_id_by_name",
    "find_gene_name_from_ensembl_gene_id",
    "find_gene_name_from_ensembl_transcript_id",
    "format_cancer_code_label",
    "fpkm_to_tpm",
    "fusion_partners",
    "fusion_status",
    "gene_cell_type_ntpm",
    "gene_protein_tissues",
    "gene_tissue_ntpm",
    "gene_to_proteoform",
    "gene_to_proteoform_id",
    "genomes",
    "greedy_coverage",
    "hpa_normal_tissue",
    # HPA normal-tissue reference data
    "hpa_rna_consensus",
    "hpa_single_cell",
    "is_mixture_cohort",
    "is_rescue_feature",
    "known_cohort_ids",
    "mixture_cohort_codes",
    "normalize_expression",
    "normalize_technical_rna_columns",
    "normalize_technical_rna_long_table",
    "per_sample_expression",
    "protein_family",
    "proteoform_for_gene",
    "proteoform_group_map",
    "proteoform_groups",
    "proteoform_representative_samples",
    "proteoform_symbol_map",
    "registry_dataframe",
    "renormalize_to_million",
    "representative_cohort_samples",
    "resolve_cancer_type",
    "response_signature_direction",
    "response_signature_genes",
    "response_signature_names",
    "response_signatures_df",
    "sample_counts_by_cancer_code",
    "sample_manifest",
    "samples_for_cancer_code",
    "samples_for_cohort",
    "sarcoma_lineage_codes",
    "signature_score",
    "sources_for_cancer_code",
    "tissue_of_origin",
    "tpm_to_housekeeping_normalized",
    "viral_status",
    "within_sample_top_fraction",
]
