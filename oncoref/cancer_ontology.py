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

"""Core cancer-type ontology and registry access.

The source of truth is ``data/cancer-type-registry.csv``. Cohort vocabulary lives
in :mod:`oncoref.cohorts`; this module keeps ontology concepts easier to find.
"""

from __future__ import annotations

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
    family_display_name,
    format_cancer_code_label,
    fusion_status,
    resolve_cancer_type,
    sarcoma_lineage_codes,
    tissue_of_origin,
    viral_status,
)

__all__ = [
    "CANCER_TYPE_ALIASES",
    "CANCER_TYPE_NAMES",
    "cancer_lineage_group",
    "cancer_lineage_group_overrides",
    "cancer_lineage_groups",
    "cancer_subtype_group",
    "cancer_subtype_groupings",
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
    "canonical_cancer_code",
    "family_display_name",
    "format_cancer_code_label",
    "fusion_status",
    "resolve_cancer_type",
    "sarcoma_lineage_codes",
    "tissue_of_origin",
    "viral_status",
]
