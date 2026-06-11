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

"""Re-sync the cancer-type registry from pirlygenes' authoritative ontology.

The cancer-type taxonomy (codes, hierarchy via ``parent_code``, family/tissue,
viral + fusion annotations) is curated in pirlygenes today; cancerdata is taking
it over (issue #27). Until ownership flips, this re-syncs cancerdata's copy from
pirlygenes' so the two don't drift — schema is identical, so it's a full-table
adoption (no projection), validated against the expected column contract.

    python scripts/sync_cancer_type_registry.py --source ../pirlygenes/pirlygenes/data/cancer-type-registry.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEST = _REPO_ROOT / "cancerdata" / "data" / "cancer-type-registry.csv"
_DEFAULT_SOURCE = (
    _REPO_ROOT.parent / "pirlygenes" / "pirlygenes" / "data" / "cancer-type-registry.csv"
)

# The registry schema contract (shipped column order). cancerdata adopts pirlygenes'
# table verbatim, so the columns must match exactly — a schema change upstream is a
# signal to reconcile the ontology accessors before syncing.
REGISTRY_COLUMNS = [
    "code",
    "name",
    "family",
    "primary_tissue",
    "primary_template",
    "parent_code",
    "subtype_key",
    "expression_source",
    "source_cohort",
    "source_pmid",
    "notes",
    "mixture_cohort",
    "pediatric",
    "differentiation",
    "viral_etiology",
    "viral_agent",
    "fusion_driven",
    "fusion_driver",
]


def sync(source: Path, dest: Path = _DEST) -> pd.DataFrame:
    """Adopt pirlygenes' cancer-type registry into cancerdata, validating schema."""
    if not source.exists():
        raise SystemExit(
            f"pirlygenes registry not found at {source} — pass --source <path to a "
            f"pirlygenes checkout's pirlygenes/data/cancer-type-registry.csv>."
        )
    src = pd.read_csv(source, dtype=str, keep_default_na=False)
    if list(src.columns) != REGISTRY_COLUMNS:
        raise SystemExit(
            f"upstream schema changed: expected {REGISTRY_COLUMNS}, got {list(src.columns)}. "
            f"Reconcile REGISTRY_COLUMNS + the cancer_types accessors before syncing."
        )
    # Referential integrity: every parent_code must name an existing code.
    codes = set(src["code"])
    orphans = sorted({p for p in src["parent_code"] if p and p not in codes})
    if orphans:
        raise SystemExit(f"parent_code(s) with no matching code (orphan ontology): {orphans}")
    src.to_csv(dest, index=False)
    print(f"wrote {len(src)} cancer-type rows × {len(REGISTRY_COLUMNS)} cols -> {dest}", flush=True)
    return src


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    args = p.parse_args(argv)
    sync(args.source)


if __name__ == "__main__":
    sys.exit(main())
