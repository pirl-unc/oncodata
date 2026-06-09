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
    cancer_type_families,
    cancer_type_info,
    cancer_type_registry,
    cancer_type_subtypes_of,
    cancer_type_synonyms,
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
from .incidence import (
    burden_category,
    cancer_burden,
    cancer_burden_df,
    cancer_code_burden_map,
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
    "__version__",
    "burden_category",
    # anti-PD-1 response
    "cancer_apd1_response",
    "cancer_apd1_response_df",
    # incidence / mortality
    "cancer_burden",
    "cancer_burden_df",
    "cancer_code_burden_map",
    # TMB
    "cancer_tmb",
    "cancer_tmb_df",
    "cancer_type_families",
    "cancer_type_info",
    "cancer_type_registry",
    "cancer_type_subtypes_of",
    "cancer_type_synonyms",
    "cancer_types_by_tissue",
    "cancer_types_in_family",
    "canonical_cancer_code",
    "cohort_aggregate_members",
    # cohort vocabulary
    "cohort_aggregates",
    "cohort_aggregates_df",
    "cohort_kind",
    "cohort_registry",
    "cohort_registry_df",
    "cta_dataframe",
    "family_display_name",
    "format_cancer_code_label",
    "fusion_status",
    "is_mixture_cohort",
    "known_cohort_ids",
    "mixture_cohort_codes",
    "resolve_cancer_type",
    "sarcoma_lineage_codes",
    "tissue_of_origin",
    "viral_status",
]
