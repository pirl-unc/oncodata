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

"""A small curated set of anti-PD-1 response/resistance expression signatures.

Cancer-type-level reference describing the *biology* of checkpoint response — the
companion to the aPD1 ORR table cancerdata already owns. Deliberately a **few**
foundational, well-cited signatures (T-cell-inflamed / IFN-γ, cytotoxic effector,
MHC-I antigen presentation, TGF-β immune exclusion), not pirlygenes' full
therapy-signature catalog: cancerdata stays the base layer, not the analysis layer.

``cancer-response-signatures.csv``: ``signature, gene_symbol, direction``
(``positive`` = response-associated, ``negative`` = resistance-associated),
``description``, ``source``. :func:`signature_score` scores a cohort by averaging
its member genes' (log) clean-TPM expression over the cohort's patients.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .load_dataset import get_data


@lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    return get_data("cancer-response-signatures", copy=False)


def response_signatures_df() -> pd.DataFrame:
    """The full curated signature table (one row per signature gene). Copy."""
    return _frame().copy()


def response_signature_names() -> list[str]:
    """The available signature names (sorted)."""
    return sorted(_frame()["signature"].astype(str).unique())


def response_signature_genes(name: str) -> set[str]:
    """Member gene symbols of a signature. Raises if ``name`` is unknown."""
    df = _frame()
    sub = df[df["signature"].astype(str) == name]
    if sub.empty:
        raise ValueError(f"unknown signature {name!r}; one of {response_signature_names()}")
    return set(sub["gene_symbol"].astype(str))


def response_signature_direction(name: str) -> str:
    """``"positive"`` (response-associated) or ``"negative"`` (resistance) for a
    signature."""
    df = _frame()
    sub = df[df["signature"].astype(str) == name]
    if sub.empty:
        raise ValueError(f"unknown signature {name!r}; one of {response_signature_names()}")
    return str(sub["direction"].iloc[0])


def signature_score(cancer_type, name: str, *, statistic: str = "mean") -> float:
    """Score a cohort for a signature: the average (``statistic``) of its member
    genes' **log clean-TPM** cohort expression across the cohort's patients.

    Built on :func:`cancerdata.expression.cohort_mean_expression` (so it needs the
    cohort's per-sample matrix). Genes absent from the cohort are skipped; ``nan`` if
    none of the signature's genes are present. Higher = more of that program; pair
    with :func:`response_signature_direction` for the response interpretation."""
    from .expression import cohort_mean_expression

    genes = response_signature_genes(name)
    mean = cohort_mean_expression(cancer_type, normalize="tpm_clean_log1p", statistic=statistic)
    vals = mean.loc[mean["Symbol"].astype(str).isin(genes), "expression"]
    return float(vals.mean()) if len(vals) else float("nan")


__all__ = [
    "response_signature_direction",
    "response_signature_genes",
    "response_signature_names",
    "response_signatures_df",
    "signature_score",
]
