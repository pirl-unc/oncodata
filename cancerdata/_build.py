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

"""Pure build-time transforms for precomputed expression artifacts.

Kept dependency-light (pandas/numpy only) and separate from the read accessors so
the heavy artifact builders in ``scripts/`` can be unit-tested without any data
bundle. The maintainer runs the ``scripts/generate_*`` wrappers; this module is
the testable core.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_ID_COLS = ("Ensembl_Gene_ID", "Symbol")

#: threshold (within-sample percentile rank) -> output column name.
#: 0.95 = "in the top 5% of expressed genes in that sample".
WITHIN_SAMPLE_THRESHOLDS = {
    0.99: "frac_samples_top1pct",
    0.95: "frac_samples_top5pct",
    0.90: "frac_samples_top10pct",
}


def sample_columns(df: pd.DataFrame) -> list[str]:
    """Per-sample value columns (everything that isn't a gene-id column)."""
    return [c for c in df.columns if c not in _ID_COLS]


def within_sample_top_fractions(
    df: pd.DataFrame,
    sample_cols: Iterable[str] | None = None,
    *,
    thresholds: Iterable[float] = tuple(WITHIN_SAMPLE_THRESHOLDS),
) -> pd.DataFrame:
    """Per-gene fraction of a cohort's samples in which the gene is highly
    expressed *within that sample* (signal a).

    For each sample (column), genes are ranked across the whole gene axis into a
    within-sample percentile (``rank(pct=True)``). A gene clears threshold ``t``
    in a sample when its within-sample percentile ≥ ``t``. The per-gene output is
    the fraction of the cohort's samples where it clears ``t`` — i.e. "in what
    fraction of these tumors is this gene among the top ``(1-t)`` of expressed
    genes."

    Returns one row per gene with ``Ensembl_Gene_ID``, ``Symbol``,
    ``frac_samples_top{1,5,10}pct`` (per threshold), and ``n_samples``.
    """
    cols = list(sample_cols) if sample_cols is not None else sample_columns(df)
    if not cols:
        raise ValueError("no per-sample columns to rank")

    # Rank genes WITHIN each sample (axis=0 = down the gene rows, per column).
    ranks = df[cols].rank(axis=0, pct=True)

    out = pd.DataFrame(
        {
            "Ensembl_Gene_ID": df["Ensembl_Gene_ID"].astype(str).to_numpy(),
            "Symbol": df["Symbol"].astype(str).to_numpy(),
        }
    )
    for t in thresholds:
        pct = round((1.0 - t) * 100)
        out[f"frac_samples_top{pct}pct"] = (ranks >= t).mean(axis=1).to_numpy()
    out["n_samples"] = len(cols)
    return out
