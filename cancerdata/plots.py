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

"""Cancer-type-level reference plots built from cancerdata's own data.

These need only cancerdata-owned data (TMB, anti-PD-1 ORR, incidence/mortality,
the cancer-type registry for lineage colors) — no CTA or peptide data — so they
have no dependency on the target-selection libraries.

matplotlib is an optional extra (``pip install cancerdata[plots]``); it is
imported lazily so the data layer stays importable without it. Every function
returns the matplotlib ``Figure`` and optionally writes a PNG when ``save`` is
given.
"""

from __future__ import annotations

from .apd1 import cancer_apd1_response
from .cancer_types import cancer_type_registry, format_cancer_code_label
from .cta import CTA_gene_id_to_name, CTA_gene_ids
from .expression import available_percentile_cohorts, cohort_gene_percentiles
from .incidence import cancer_burden
from .tmb import cancer_tmb

_PLT = None

#: stat name -> the percentile breakpoint column in the shipped percentile vector.
_STAT_PERCENTILE_COL = {"q1": "p25", "median": "p50", "q3": "p75"}


def _plt():
    """Lazy, one-time matplotlib import (headless-safe Agg default; honors a
    backend the caller already set)."""
    global _PLT
    if _PLT is not None:
        return _PLT
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:  # pragma: no cover - exercised via extras
        raise ModuleNotFoundError(
            "cancerdata plotting requires matplotlib — install with `pip install cancerdata[plots]`"
        ) from e
    _PLT = plt
    return _PLT


def _family_by_code() -> dict[str, str]:
    reg = cancer_type_registry()
    return dict(zip(reg["code"].astype(str), reg["family"].astype(str)))


def _family_colors(codes):
    """Map each code to an RGBA color by registry family (stable tab20 palette)."""
    plt = _plt()
    fam_by_code = _family_by_code()
    families = sorted({fam_by_code.get(c, "?") for c in codes})
    cmap = plt.get_cmap("tab20")
    fam_color = {f: cmap(i % 20) for i, f in enumerate(families)}
    return {c: fam_color[fam_by_code.get(c, "?")] for c in codes}, fam_color


def _save(fig, save):
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def apd1_vs_tmb(*, save=None, annotate=True):
    """Scatter of anti-PD-1 ORR (%) vs median TMB (log x), one point per cancer
    type with a curated value for both, colored by lineage family. The classic
    "more mutations -> more neoantigens -> better checkpoint response" view."""
    plt = _plt()
    tmb = cancer_tmb()
    orr = cancer_apd1_response()
    codes = sorted(set(tmb) & set(orr))
    if not codes:
        raise ValueError("no cancer types with both a TMB and an anti-PD-1 value")
    colors, fam_color = _family_colors(codes)

    fig, ax = plt.subplots(figsize=(11, 7))
    for c in codes:
        ax.scatter(
            tmb[c], orr[c], color=colors[c], s=70, edgecolor="white", linewidth=0.6, zorder=3
        )
        if annotate:
            ax.annotate(
                format_cancer_code_label(c),
                (tmb[c], orr[c]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Median tumor mutational burden (mut/Mb, log scale)")
    ax.set_ylabel("Anti-PD-1 monotherapy ORR (%)")
    ax.set_title(f"Anti-PD-1 response vs TMB ({len(codes)} cancer types)")
    ax.grid(True, which="both", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=col, label=fam)
        for fam, col in sorted(fam_color.items())
    ]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, save)


def apd1_orr_bars(*, save=None):
    """Horizontal bar chart of anti-PD-1 ORR by cancer type, sorted ascending,
    colored by lineage family."""
    plt = _plt()
    orr = cancer_apd1_response()
    codes = sorted(orr, key=lambda c: orr[c])
    colors, _fam_color = _family_colors(codes)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(codes))))
    ax.barh(
        [format_cancer_code_label(c) for c in codes],
        [orr[c] for c in codes],
        color=[colors[c] for c in codes],
    )
    ax.set_xlabel("Anti-PD-1 monotherapy ORR (%)")
    ax.set_title(f"Anti-PD-1 response by cancer type ({len(codes)} types)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return _save(fig, save)


def incidence_vs_mortality(*, region="us", save=None):
    """Scatter of mortality-share vs incidence-share (%) per burden category for
    a region (``"us"`` or ``"world"``). The diagonal separates high-lethality
    (pancreas, lung) from low-lethality (thyroid, prostate) categories."""
    plt = _plt()
    if region not in ("us", "world"):
        raise ValueError("region must be 'us' or 'world'")
    inc_metric, mort_metric = f"{region}_incidence_pct", f"{region}_mortality_pct"
    inc = cancer_burden(metric=inc_metric)
    mort = cancer_burden(metric=mort_metric)
    cats = sorted(set(inc) & set(mort))

    fig, ax = plt.subplots(figsize=(9, 8))
    lim = max(max(inc.values(), default=1), max(mort.values(), default=1)) * 1.1
    ax.plot([0, lim], [0, lim], color="0.7", linestyle="--", linewidth=1, zorder=1)
    for cat in cats:
        ax.scatter(inc[cat], mort[cat], s=60, color="tab:red", alpha=0.8, zorder=3)
        ax.annotate(
            cat, (inc[cat], mort[cat]), fontsize=6, xytext=(3, 3), textcoords="offset points"
        )
    ax.set_xlabel(f"{region.upper()} share of cancer incidence (%)")
    ax.set_ylabel(f"{region.upper()} share of cancer mortality (%)")
    ax.set_title(f"Incidence vs mortality by burden category ({region.upper()})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save)


def _cta_expression_matrix(stat, cohorts):
    """cohorts × CTA-gene matrix of the requested ``stat`` TPM (NaN where a cohort
    lacks a gene). Rows are cohort codes, columns are CTA symbols. Cohorts without
    a percentile vector (summary-only or unknown codes) are skipped with a warning
    rather than aborting the whole plot."""
    import warnings

    import pandas as pd

    col = _STAT_PERCENTILE_COL[stat]
    id_to_name = CTA_gene_id_to_name()
    cta_ids = set(CTA_gene_ids())
    rows = {}
    skipped = []
    for code in cohorts:
        try:
            df = cohort_gene_percentiles(code, as_tpm=True)
        except ValueError:
            skipped.append(str(code))
            continue
        ids = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
        mask = ids.isin(cta_ids)
        sub = df.loc[mask]
        rows[code] = pd.Series(
            sub[col].to_numpy(),
            index=ids[mask].map(id_to_name),
        )
    if skipped:
        warnings.warn(
            f"skipped {len(skipped)} cohort(s) without a percentile vector: {', '.join(skipped)}",
            stacklevel=2,
        )
    matrix = pd.DataFrame(rows).T  # cohorts (rows) × CTA symbols (cols)
    return matrix.loc[:, ~matrix.columns.duplicated()]


def cta_expression_heatmap(
    *,
    stat="median",
    cohorts=None,
    n_cohorts=30,
    n_ctas=30,
    high_tpm=30.0,
    proteoform=False,
    save=None,
):
    """Heatmap of cancer-testis-antigen expression — rows are cohorts, columns are
    CTAs, cells are the cohort's per-gene ``stat`` TPM (``"median"``/``"q1"``/``"q3"``
    = p50/p25/p75 of the shipped percentile vector).

    Rows are the ``n_cohorts`` cohorts with the highest single-CTA ``stat``; columns
    are the ``n_ctas`` CTAs with the highest peak across cohorts, ordered by how many
    cohorts express them above ``high_tpm`` (the actionable threshold) then by peak.
    Colour is a log scale (dark = silent, hot = high) anchored in the clinically
    meaningful 1–1000 TPM range.

    ``cohorts`` restricts the cohort pool (default: every cohort with a percentile
    vector). Needs the expression bundle present (percentile artifacts).

    ``proteoform=True`` is not yet available: faithful paralog summation must happen
    on per-sample matrices *before* the percentile summary (percentiles can't be
    summed), which is the proteoform-summed percentile artifact tracked in #13.
    """
    import numpy as np
    from matplotlib.colors import LogNorm

    if stat not in _STAT_PERCENTILE_COL:
        raise ValueError(f"stat must be one of {sorted(_STAT_PERCENTILE_COL)}")
    if proteoform:
        raise NotImplementedError(
            "proteoform-summed CTA expression needs the proteoform-summed percentile "
            "artifact (per-sample summation before summarizing — percentiles cannot be "
            "summed); tracked in issue #13."
        )
    plt = _plt()
    if cohorts is None:
        cohorts = available_percentile_cohorts()
    if not cohorts:
        raise ValueError("no cohorts with a percentile vector — is the expression bundle present?")

    matrix = _cta_expression_matrix(stat, cohorts)
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError(
            "no CTA expression data for the selected cohorts — none had a percentile "
            "vector, or none expressed any CTA gene."
        )
    # Columns: top CTAs by peak, ordered by breadth (#cohorts > high_tpm) then peak.
    peak = matrix.max(axis=0, skipna=True)
    breadth = (matrix > high_tpm).sum(axis=0)
    top_ctas = (
        peak.to_frame("peak")
        .assign(breadth=breadth)
        .sort_values(["breadth", "peak"], ascending=False)
        .head(n_ctas)
        .index
    )
    # Rows: cohorts with the highest single-CTA value, sorted descending.
    row_score = matrix[top_ctas].max(axis=1, skipna=True)
    top_cohorts = row_score.sort_values(ascending=False).head(n_cohorts).index
    grid = matrix.loc[top_cohorts, top_ctas]

    floor = 0.01
    data = np.clip(grid.to_numpy(dtype=float), floor, None)
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(top_ctas)), max(6, 0.34 * len(top_cohorts))))
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="magma",
        norm=LogNorm(vmin=floor, vmax=max(1000.0, float(np.nanmax(data)))),
    )
    ax.set_xticks(range(len(top_ctas)))
    ax.set_xticklabels(list(top_ctas), rotation=90, fontsize=6)
    ax.set_yticks(range(len(top_cohorts)))
    ax.set_yticklabels([format_cancer_code_label(c) for c in top_cohorts], fontsize=6)
    ax.set_title(
        f"CTA expression ({stat} TPM) — top {len(top_cohorts)} cohorts × {len(top_ctas)} CTAs"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(f"{stat} TPM (log)")
    fig.tight_layout()
    return _save(fig, save)
