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

"""Registry of every cohort-level expression data source.

The bundled ``expression_sources.yaml`` is the single source of truth for which
upstream datasets feed each cancer cohort — Treehouse, BEAT-AML, GDC/TCGA,
CLL-map, MMRF CoMMpass, TARGET, GEO series, recount3, … — with the provenance a
re-runner needs (project/accession/url, native unit, library prep, builder,
citation, expected size). It drives the legible ``oncoref expression-sources`` listing
and (incrementally) the fetch + regeneration of the per-cohort summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_REGISTRY_PATH = _DATA_DIR / "expression_sources.yaml"


@dataclass(frozen=True)
class ExpressionSource:
    """One upstream expression dataset feeding one or more cancer cohorts."""

    id: str
    category: str
    cancer_codes: tuple[str, ...]
    source_type: str
    builder: str | None = None
    builder_args: tuple[str, ...] = ()
    project_id: str | None = None
    accession: str | None = None
    url: str | None = None
    unit: str | None = None
    expected_size_gb: float | None = None
    citation: str | None = None
    special_handling: str | None = None
    recount3_srp: str | None = None
    source_cohort: str | None = None
    library_prep: str | None = None


def _coerce_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=1)
def load_registry() -> tuple[ExpressionSource, ...]:
    """Parse the bundled ``expression_sources.yaml`` into ``ExpressionSource``s."""
    import yaml

    doc = yaml.safe_load(_REGISTRY_PATH.read_text())
    out = []
    for entry in doc.get("sources", []):
        out.append(
            ExpressionSource(
                id=str(entry["id"]),
                category=str(entry.get("category", "expression")),
                cancer_codes=_coerce_tuple(entry.get("cancer_codes")),
                source_type=str(entry.get("source_type", "")),
                builder=_clean(entry.get("builder")),
                builder_args=_coerce_tuple(entry.get("builder_args")),
                project_id=_clean(entry.get("project_id")),
                accession=_clean(entry.get("accession")),
                url=_clean(entry.get("url")),
                unit=_clean(entry.get("unit")),
                expected_size_gb=_coerce_float(entry.get("expected_size_gb")),
                citation=_clean(entry.get("citation")),
                special_handling=_clean(entry.get("special_handling")),
                recount3_srp=_clean(entry.get("recount3_srp")),
                source_cohort=_clean(entry.get("source_cohort")),
                library_prep=_clean(entry.get("library_prep")),
            )
        )
    return tuple(out)


def expression_sources() -> tuple[ExpressionSource, ...]:
    """Every registered expression source."""
    return load_registry()


def expression_source(source_id: str) -> ExpressionSource | None:
    """Look up one source by id."""
    for s in load_registry():
        if s.id == source_id:
            return s
    return None


def sources_for_cancer_code(code: str) -> list[ExpressionSource]:
    """Every source that feeds a given cancer code."""
    return [s for s in load_registry() if code in s.cancer_codes]


def expression_sources_df() -> pd.DataFrame:
    """The registry as a tabular view (one row per source) for inspection."""
    return pd.DataFrame(
        {
            "id": s.id,
            "source_type": s.source_type,
            "cancer_codes": ";".join(s.cancer_codes),
            "n_codes": len(s.cancer_codes),
            "unit": s.unit,
            "library_prep": s.library_prep,
            "project_id": s.project_id or s.accession or s.recount3_srp,
            "expected_size_gb": s.expected_size_gb,
            "citation": s.citation,
        }
        for s in load_registry()
    )


def expression_source_candidates(cancer_code: str | None = None) -> pd.DataFrame:
    """Candidate expression sources per cancer type (``cancer_code``,
    ``source_status``, ``reference_code``, ``source_project``, ``accession``,
    ``assay``, ``processing_plan``, …) — the build-planning view of which cohort
    could back a type's reference. Filtered to one code when given. Defensive copy."""
    from .load_dataset import get_data

    df = get_data("cancer-expression-source-candidates").copy()
    if cancer_code is not None:
        from .cancer_types import resolve_cancer_type

        df = df[df["cancer_code"].astype(str) == resolve_cancer_type(cancer_code)].reset_index(
            drop=True
        )
    return df
