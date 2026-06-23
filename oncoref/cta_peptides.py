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

"""CTA-specific 9-mer peptide specificity helpers.

The older :mod:`oncoref.peptides` module is intentionally still importable, but the
functions exported here make the CTA-specific scope explicit.
"""

from __future__ import annotations

from .peptides import (
    DEFAULT_K,
    cta_specific_9mer_counts,
    cta_specific_9mer_load,
    cta_specific_9mer_weights,
)


def cta_specific_9mer_count_map(*, k: int = DEFAULT_K, by: str = "proteoform_key"):
    """Map each CTA key to its ``n_specific_9mers`` count.

    ``by`` accepts ``"proteoform_key"`` (default), ``"ensembl_gene_id"``, or
    ``"symbol"``. This is the clearer name for the historical
    ``cta_specific_9mer_weights`` helper; the values are counts used as weights in
    ``cta_specific_9mer_load``.
    """
    return cta_specific_9mer_weights(k=k, by=by)


__all__ = [
    "DEFAULT_K",
    "cta_specific_9mer_count_map",
    "cta_specific_9mer_counts",
    "cta_specific_9mer_load",
    "cta_specific_9mer_weights",
]
