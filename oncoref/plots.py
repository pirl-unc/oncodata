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

"""Cancer-type-level reference plots built from oncoref's own data.

These need only oncoref-owned data (TMB, anti-PD-1 ORR, incidence/mortality,
the cancer-type registry for lineage colors) — no CTA or peptide data — so they
have no dependency on the target-selection libraries.

matplotlib is an optional extra (``pip install oncoref[plots]``); it is
imported lazily so the data layer stays importable without it. Every function
returns the matplotlib ``Figure`` and optionally writes a PNG when ``save`` is
given.
"""

from __future__ import annotations

from functools import lru_cache

from .apd1 import cancer_apd1_response
from .cancer_types import cancer_type_registry, format_cancer_code_label
from .cta import cta_gene_id_to_name, cta_gene_ids
from .expression import (
    available_percentile_cohorts,
    available_within_sample_cohorts,
    cohort_gene_percentiles,
    within_sample_top_fraction,
)
from .incidence import burden_category, cancer_burden
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
            "oncoref plotting requires matplotlib — install with `pip install oncoref[plots]`"
        ) from e
    _PLT = plt
    return _PLT


def _family_by_code() -> dict[str, str]:
    reg = cancer_type_registry()
    return dict(zip(reg["code"].astype(str), reg["family"].astype(str)))


@lru_cache(maxsize=1)
def _stable_palette() -> tuple:
    """A long, fixed list of distinct RGBA colors (tab20 + tab20b + tab20c = 60),
    so a family/group keeps the SAME color across every plot."""
    plt = _plt()
    out: list = []
    for name in ("tab20", "tab20b", "tab20c"):
        out.extend(plt.get_cmap(name).colors)
    return tuple(out)


@lru_cache(maxsize=1)
def _family_color_map() -> dict:
    """Stable ``{registry family -> color}`` over the FULL registry, assigned in
    sorted-family order so the color is deterministic and identical in every plot
    (not dependent on which subset of cancers a given plot happens to show)."""
    palette = _stable_palette()
    families = sorted({*_family_by_code().values(), "?"})
    return {f: palette[i % len(palette)] for i, f in enumerate(families)}


def _family_colors(codes):
    """``({code -> color}, {family -> color})`` by registry family. Colors are a
    stable, deterministic per-family assignment (see :func:`_family_color_map`); the
    returned family map carries only the families present in ``codes`` (for legends)."""
    fam_by_code = _family_by_code()
    full = _family_color_map()
    present = sorted({fam_by_code.get(c, "?") for c in codes})
    fam_color = {f: full[f] for f in present}
    return {c: full[fam_by_code.get(c, "?")] for c in codes}, fam_color


def _save(fig, save):
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


@lru_cache(maxsize=1)
def _cohort_sample_counts() -> dict:
    """``{cancer_code: n_samples}`` from the per-sample matrix registry — the
    cohort size used to rank cohorts in the by-cohort figures."""
    from . import source_matrices

    reg = source_matrices.registry()
    counts: dict[str, int] = {}
    for code, n in zip(reg["cancer_code"].astype(str), reg["n_samples"]):
        counts[code] = max(counts.get(code, 0), int(n))  # largest source if several
    return counts


def _top_cohorts_by_samples(codes, top_n):
    """The ``top_n`` of ``codes`` by cohort sample count (largest first); all of them
    if ``top_n`` is ``None`` or there are no more than ``top_n``. Ties / unknown counts
    fall back to a stable code order so the selection is deterministic."""
    codes = list(codes)
    if top_n is None or len(codes) <= top_n:
        return codes
    counts = _cohort_sample_counts()
    return sorted(codes, key=lambda c: (-counts.get(c, 0), c))[:top_n]


# ---------- shared plot primitives (the centralized rendering layer) ----------
#
# Every cancer-type plot reduces to one of three shapes: a per-cancer scatter, a
# cohort×gene heatmap, or a ranked horizontal bar — all coloured by registry family
# and saved the same way. The plot functions below are thin: they assemble the data
# (points / grid / pairs) and hand it to one primitive, so adding a plot is "shape
# the data, pick a primitive", not "re-derive the matplotlib boilerplate".


def _family_legend_handles(plt, fam_color):
    return [
        plt.Line2D([], [], marker="o", linestyle="", color=col, label=fam)
        for fam, col in sorted(fam_color.items())
    ]


def _family_scatter(
    points,
    *,
    xlabel,
    ylabel,
    title,
    logx=False,
    annotate=True,
    figsize=(9, 7),
    legend_loc="best",
    save=None,
):
    """Scatter of ``(code, x, y)`` points coloured by registry family, with optional
    point annotations + a family legend. The shared per-cancer scatter scaffold."""
    plt = _plt()
    codes = [p[0] for p in points]
    colors, fam_color = _family_colors(codes)
    fig, ax = plt.subplots(figsize=figsize)
    for code, x, y in points:
        ax.scatter(x, y, color=colors[code], s=70, edgecolor="white", linewidth=0.6, zorder=3)
        if annotate:
            ax.annotate(
                format_cancer_code_label(code),
                (x, y),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both" if logx else "major", alpha=0.3)
    ax.legend(
        handles=_family_legend_handles(plt, fam_color),
        fontsize=6,
        ncol=2,
        loc=legend_loc,
        framealpha=0.9,
    )
    fig.tight_layout()
    return _save(fig, save)


def _ranked_family_barh(pairs, *, xlabel, title, legend=False, save=None):
    """Horizontal bars of ``(code, value)`` in the given top-to-bottom order,
    coloured by registry family. The shared ranked-bar scaffold."""
    import numpy as np

    plt = _plt()
    codes = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors, fam_color = _family_colors(codes)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(codes))))
    y = np.arange(len(codes))
    ax.barh(y, values, color=[colors[c] for c in codes])
    ax.set_yticks(y)
    ax.set_yticklabels([format_cancer_code_label(c) for c in codes], fontsize=7)
    ax.invert_yaxis()  # first pair at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
        ax.legend(handles, list(fam_color), fontsize=6, title="family", loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def _cohort_gene_heatmap(grid, *, title, cbar_label, cmap, lognorm=False, floor=0.01, save=None):
    """Heatmap of a cohorts (rows) × genes (cols) DataFrame — rows labelled by
    cancer code, columns by gene symbol. The shared cohort×gene heatmap scaffold."""
    import numpy as np

    plt = _plt()
    cols = list(grid.columns)
    rows = list(grid.index)
    data = grid.to_numpy(dtype=float)
    norm = None
    if lognorm:
        from matplotlib.colors import LogNorm

        data = np.clip(data, floor, None)
        norm = LogNorm(vmin=floor, vmax=max(1000.0, float(np.nanmax(data))))
    else:
        data = np.nan_to_num(data, nan=0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(cols)), max(6, 0.34 * len(rows))))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([format_cancer_code_label(c) for c in rows], fontsize=6)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return _save(fig, save)


# CTA antigen families, by symbol prefix (longest match wins). oncoref has no
# curated antigen-family table; this presentational heuristic mirrors pirlygenes and
# only drives colour grouping in the gene-level CTA plots — never analysis.
_ANTIGEN_FAMILY_PREFIXES = (
    ("MAGEA", "MAGE-A"),
    ("MAGEB", "MAGE-B"),
    ("MAGEC", "MAGE-C"),
    ("CSAG", "CSAG"),
    ("GAGE", "GAGE"),
    ("XAGE", "XAGE"),
    ("PAGE", "PAGE"),
    ("SSX", "SSX"),
    ("CT45", "CT45"),
    ("CTAG", "CTAG/NY-ESO"),
    ("LAGE", "CTAG/NY-ESO"),
    ("NY-ESO", "CTAG/NY-ESO"),  # the curated proteoform alias for CTAG1A/CTAG1B
    ("SPANX", "SPANX"),
    ("PRAME", "PRAME"),
    ("DPPA2", "DPPA"),
    ("TFDP3", "TFDP3"),
)


def _antigen_family(symbol) -> str:
    """CTA antigen family for a gene symbol via longest matching prefix; ``"other"``
    when nothing matches (e.g. CT83, CTCFL)."""
    s = str(symbol).upper()
    best = None
    for prefix, fam in _ANTIGEN_FAMILY_PREFIXES:
        if s.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, fam)
    return best[1] if best else "other"


@lru_cache(maxsize=1)
def _antigen_family_color_map() -> dict:
    """Stable ``{antigen family -> color}`` over the full antigen-family vocabulary,
    so e.g. MAGE-A is the same color in every plot."""
    palette = _stable_palette()
    families = sorted({fam for _, fam in _ANTIGEN_FAMILY_PREFIXES} | {"other"})
    return {f: palette[i % len(palette)] for i, f in enumerate(families)}


def _antigen_family_colors(symbols):
    """``({symbol -> color}, {family -> color})`` keyed by antigen family, with a
    stable per-family color assignment (see :func:`_antigen_family_color_map`). The
    returned family map carries only the families present in ``symbols``."""
    fam_by_sym = {s: _antigen_family(s) for s in symbols}
    full = _antigen_family_color_map()
    present = sorted(set(fam_by_sym.values()))
    fam_color = {f: full[f] for f in present}
    return {s: full[f] for s, f in fam_by_sym.items()}, fam_color


def _stacked_barh(rows, *, xlabel, title, legend=None, annotate=True, save=None):
    """Horizontal stacked bars. ``rows`` is ``[(row_label, [(seg_label, value,
    color), ...]), ...]`` in top-to-bottom order; each row's segments stack along x.
    ``legend`` is an optional ``{label: color}`` shown as a colour key. Segments wide
    enough are annotated with their ``seg_label``. The shared stacked-bar scaffold."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(rows))))
    total = max((sum(v for _, v, _ in segs) for _, segs in rows), default=1.0) or 1.0
    for i, (_, segs) in enumerate(rows):
        left = 0.0
        for seg_label, value, color in segs:
            ax.barh(i, value, left=left, color=color, edgecolor="white", linewidth=0.4)
            # Label every segment wide enough to fit text — denser than a fixed cut so
            # the carrying antigens are readable (the first segment is always labeled).
            if annotate and (value > 0.018 * total or left == 0.0):
                ax.text(
                    left + value / 2,
                    i,
                    seg_label,
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    rotation=90 if value < 0.04 * total else 0,
                    color="white",
                    clip_on=True,
                )
            left += value
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()  # first row at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend.values()]
        ax.legend(handles, list(legend), fontsize=6, title="antigen family", loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def _grouped_barh(categories, series, *, xlabel, title, save=None):
    """Grouped (paired) horizontal bars. ``categories`` are the row labels (top-to-
    bottom); ``series`` is ``[(name, [value per category], color), ...]`` — one bar
    per series within each category group. The shared grouped-bar scaffold."""
    import numpy as np

    plt = _plt()
    n_series = max(1, len(series))
    base = np.arange(len(categories))
    height = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(categories))))
    for k, (name, values, color) in enumerate(series):
        offset = (k - (n_series - 1) / 2) * height
        ax.barh(base + offset, values, height=height, label=name, color=color)
    ax.set_yticks(base)
    ax.set_yticklabels(categories, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def apd1_vs_tmb(*, save=None, annotate=True):
    """Scatter of anti-PD-1 ORR (%) vs median TMB (log x), one point per cancer
    type with a curated value for both, colored by lineage family. The classic
    "more mutations -> more neoantigens -> better checkpoint response" view."""
    tmb = cancer_tmb()
    orr = cancer_apd1_response()
    codes = sorted(set(tmb) & set(orr))
    if not codes:
        raise ValueError("no cancer types with both a TMB and an anti-PD-1 value")
    return _family_scatter(
        [(c, tmb[c], orr[c]) for c in codes],
        xlabel="Median tumor mutational burden (mut/Mb, log scale)",
        ylabel="Anti-PD-1 monotherapy ORR (%)",
        title=f"Anti-PD-1 response vs TMB ({len(codes)} cancer types)",
        logx=True,
        annotate=annotate,
        figsize=(11, 7),
        legend_loc="upper left",
        save=save,
    )


def apd1_orr_bars(*, save=None):
    """Horizontal bar chart of anti-PD-1 ORR by cancer type, highest at the top,
    colored by lineage family."""
    orr = cancer_apd1_response()
    codes = sorted(orr, key=lambda c: orr[c], reverse=True)  # highest at top
    return _ranked_family_barh(
        [(c, orr[c]) for c in codes],
        xlabel="Anti-PD-1 monotherapy ORR (%)",
        title=f"Anti-PD-1 response by cancer type ({len(codes)} types)",
        save=save,
    )


def ici_response_by_regimen(*, save=None, only_multi=True):
    """Grouped horizontal bars of ICI ORR by cancer type, one bar per **regimen**
    (anti-PD-1 / anti-PD-L1 / anti-PD-1+anti-CTLA-4) — so the regimens are shown as
    distinct response sources and can be compared within a cancer. With ``only_multi``
    (default) only cancers with more than one regimen are shown (where the comparison
    is meaningful); set ``False`` to include every cancer with any ICI value."""
    from .ici import REGIMEN_FALLBACK, REGIMEN_LABELS, cancer_ici_response

    by_regimen = {r: cancer_ici_response(regimen=r) for r in REGIMEN_FALLBACK}
    codes = {c for m in by_regimen.values() for c in m}
    if only_multi:
        codes = {c for c in codes if sum(c in by_regimen[r] for r in REGIMEN_FALLBACK) > 1}
    if not codes:
        raise ValueError("no cancer types to plot")
    # Order rows by best available ORR (descending), top at the top.
    best = {c: max(by_regimen[r][c] for r in REGIMEN_FALLBACK if c in by_regimen[r]) for c in codes}
    ordered = sorted(codes, key=lambda c: best[c], reverse=True)

    palette = _stable_palette()
    reg_color = {r: palette[i] for i, r in enumerate(REGIMEN_FALLBACK)}
    # Absent regimen -> NaN (no bar drawn), so a missing estimate is not confused
    # with a true 0% ORR (matplotlib draws nothing for a NaN-width bar).
    series = [
        (REGIMEN_LABELS[r], [by_regimen[r].get(c, float("nan")) for c in ordered], reg_color[r])
        for r in REGIMEN_FALLBACK
    ]
    return _grouped_barh(
        [format_cancer_code_label(c) for c in ordered],
        series,
        xlabel="Objective response rate (%)",
        title=f"ICI response by regimen ({len(ordered)} cancer types)",
        save=save,
    )


def ici_regimen_comparison(*, save=None, min_regimens=1):
    """Dumbbell/dot plot of ICI ORR — **every cancer type on its own row, one colored
    dot per available regimen** (anti-PD-1 / anti-PD-L1 / anti-PD-1+anti-CTLA-4),
    connected by a line, so all three estimates can be compared side by side and
    scanned by cancer type. Rows are sorted by best available ORR (highest at top).
    ``min_regimens`` filters to cancers with at least that many regimens (1 = show
    every cancer with any ICI value; 2 = only where regimens actually differ)."""
    from .ici import REGIMEN_FALLBACK, REGIMEN_LABELS, cancer_ici_response

    by_regimen = {r: cancer_ici_response(regimen=r) for r in REGIMEN_FALLBACK}
    codes = {c for m in by_regimen.values() for c in m}
    codes = {c for c in codes if sum(c in by_regimen[r] for r in REGIMEN_FALLBACK) >= min_regimens}
    if not codes:
        raise ValueError("no cancer types to plot")
    best = {c: max(by_regimen[r][c] for r in REGIMEN_FALLBACK if c in by_regimen[r]) for c in codes}
    ordered = sorted(codes, key=lambda c: best[c])  # ascending; highest ends up on top

    plt = _plt()
    palette = _stable_palette()
    reg_color = {r: palette[i] for i, r in enumerate(REGIMEN_FALLBACK)}
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(ordered))))
    for y, c in enumerate(ordered):
        present = [(r, by_regimen[r][c]) for r in REGIMEN_FALLBACK if c in by_regimen[r]]
        xs = [v for _, v in present]
        if len(xs) > 1:  # connect the estimates for this cancer
            ax.plot([min(xs), max(xs)], [y, y], color="#cccccc", lw=1.5, zorder=1)
        for r, v in present:
            ax.scatter(v, y, color=reg_color[r], s=48, edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([format_cancer_code_label(c) for c in ordered], fontsize=6)
    ax.set_xlabel("Objective response rate (%)")
    ax.set_title(f"ICI response by regimen and cancer type ({len(ordered)} types)")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=reg_color[r], label=REGIMEN_LABELS[r])
        for r in REGIMEN_FALLBACK
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="regimen")
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


def _cta_expression_matrix(stat, cohorts, *, proteoform=False):
    """cohorts × CTA matrix of the requested ``stat`` TPM (NaN where a cohort lacks a
    gene). Rows are cohort codes, columns are CTA symbols. With ``proteoform=True``,
    reads the proteoform-summed percentile vectors and labels columns by the proteoform
    symbol (NY-ESO-1, XAGE1A/B). Cohorts without the vector are skipped with a warning
    rather than aborting the whole plot."""
    import warnings

    import pandas as pd

    col = _STAT_PERCENTILE_COL[stat]
    id_to_name = cta_gene_id_to_name()
    cta_ids = set(cta_gene_ids())
    # In the collapsed space, a CTA group's row is keyed by its proteoform_key, and the
    # row's canonical Ensembl_Gene_ID may be an unexpressed member outside cta_gene_ids
    # — so match on the CTA *proteoform keys*, not the canonical ENSG (which would drop
    # CGB3/5/8, CT45A2/8/9, …). The keys are derived from the CTA set itself.
    cta_keys = None
    if proteoform:
        from .proteoforms import gene_to_proteoform_id

        cta_keys = set(gene_to_proteoform_id(sorted(cta_ids)).values())
    rows = {}
    skipped = []
    for code in cohorts:
        try:
            df = cohort_gene_percentiles(code, as_tpm=True, proteoform=proteoform)
        except ValueError:
            skipped.append(str(code))
            continue
        if proteoform:
            mask = df["proteoform_key"].astype(str).isin(cta_keys)
            sub = df.loc[mask]
            labels = sub["Symbol"].to_numpy()  # the proteoform symbol (NY-ESO-1, XAGE1A/B)
        else:
            ids = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
            sub = df.loc[ids.isin(cta_ids)]
            ids_kept = sub["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
            labels = ids_kept.map(id_to_name).to_numpy()
        rows[code] = pd.Series(sub[col].to_numpy(), index=labels)
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

    ``proteoform=True`` reads the **proteoform-summed** percentile vectors: identical-
    protein paralogs were summed per sample *before* the percentiles were computed, so
    a duplicated antigen appears once under its proteoform symbol (NY-ESO-1, XAGE1A/B)
    rather than as several diluted member rows. Needs the proteoform percentile shards
    (``generate_cohort_percentiles.py --proteoform``).
    """
    if stat not in _STAT_PERCENTILE_COL:
        raise ValueError(f"stat must be one of {sorted(_STAT_PERCENTILE_COL)}")
    if cohorts is None:
        cohorts = available_percentile_cohorts(proteoform=proteoform)
    if not cohorts:
        variant = "proteoform-summed " if proteoform else ""
        raise ValueError(
            f"no cohorts with a {variant}percentile vector — is the expression bundle present?"
        )

    matrix = _cta_expression_matrix(stat, cohorts, proteoform=proteoform)
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

    return _cohort_gene_heatmap(
        grid,
        title=f"CTA expression ({stat} TPM) — top {len(top_cohorts)} cohorts × {len(top_ctas)} CTAs",
        cbar_label=f"{stat} TPM (log)",
        cmap="magma",
        lognorm=True,
        save=save,
    )


def _cta_prevalence_by_cohort(threshold):
    """``{cohort code: fraction of samples where the single most-prevalent CTA is a
    top-expressed gene}`` from the within-sample artifact.

    This is the best **shipped** proxy for "fraction of patients with a targetable
    CTA": the within-sample artifact gives, per gene, the fraction of a cohort's
    samples in which it ranks in the top ``(1-threshold)`` *within that sample*; we
    take the max over the CTA set. It is a lower bound on the true "≥1 CTA expressed"
    union (different patients may express different CTAs), which needs the per-sample
    joint matrices — see :func:`oncoref.expression.proteoform_representative_samples`
    and #13. Cohorts without a within-sample shard are skipped."""
    import pandas as pd

    cta_ids = set(cta_gene_ids())
    out = {}
    for code in available_within_sample_cohorts():
        df = within_sample_top_fraction(code, threshold=threshold)
        frac_col = next((c for c in df.columns if c.startswith("frac_samples_top")), None)
        if frac_col is None:
            continue
        ids = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
        cta_frac = pd.to_numeric(df.loc[ids.isin(cta_ids), frac_col], errors="coerce")
        out[str(code)] = float(cta_frac.max()) if len(cta_frac) and cta_frac.notna().any() else 0.0
    return out


def _addressable_prevalence(source, *, threshold, threshold_tpm):
    """``({code: addressable share}, xlabel)`` for the two prevalence sources."""
    if source == "per_sample":
        # Faithful: fraction of a cohort's patients expressing >=1 CTA above
        # threshold_tpm (the union), from the per-sample matrices.
        from .coverage import addressable_fraction_by_cohort

        prev = addressable_fraction_by_cohort(threshold_tpm=threshold_tpm).to_dict()
        return prev, (
            f"Addressable burden  (incidence share × P(≥1 CTA > {threshold_tpm:g} TPM), per-patient)"
        )
    if source == "within_sample":
        prev = _cta_prevalence_by_cohort(threshold)
        return prev, f"Addressable burden  (incidence share × top-CTA prevalence, thr={threshold})"
    raise ValueError("source must be 'within_sample' or 'per_sample'")


def cta_addressable_burden(
    *,
    source="within_sample",
    threshold=0.95,
    threshold_tpm=10.0,
    metric="us_incidence_pct",
    n=25,
    save=None,
):
    """Bar chart of **CTA-addressable cancer burden** per cancer type — ``incidence
    share × CTA prevalence``, ranking cancers by how many patients a CTA-directed
    therapy could address.

    ``source`` selects the prevalence basis:
      - ``"within_sample"`` (default, portable) — the single-best-CTA within-sample
        prevalence from the shipped within-sample bundle (a *proxy*; ``threshold`` is
        the within-sample rank cut 0.99/0.95/0.90);
      - ``"per_sample"`` — the **faithful** fraction of patients expressing ≥1 CTA
        above ``threshold_tpm`` (the per-patient union), from the per-sample matrices
        (:func:`oncoref.coverage.addressable_fraction_by_cohort`); needs the
        cohorts' per-sample matrices cached.

    ``metric`` selects the incidence basis; ``n`` caps the bars; bars are coloured by
    registry family. Incidence is at burden-category granularity (several subtypes
    share a category), so read the bar as a relative prioritization."""
    prevalence, xlabel = _addressable_prevalence(
        source, threshold=threshold, threshold_tpm=threshold_tpm
    )
    if not prevalence:
        raise ValueError(
            f"no CTA prevalence available for source={source!r} — is the required data "
            "present? (within-sample bundle, or cached per-sample matrices)"
        )
    burden = cancer_burden(metric=metric)  # {burden category: share}

    rows = []
    for code, prev in prevalence.items():
        category = burden_category(code)
        inc = burden.get(category) if category else None
        if inc is None or not inc:
            continue
        rows.append((code, float(inc) * prev, prev))
    if not rows:
        raise ValueError("no cohort mapped to both a burden category and CTA prevalence")

    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:n]
    return _ranked_family_barh(
        [(r[0], r[1]) for r in rows],
        xlabel=xlabel,
        title=f"CTA-addressable cancer burden — top {len(rows)} cancers",
        legend=True,
        save=save,
    )


def _cached_per_sample_cohorts():
    """Cohort codes whose per-sample matrix is cached locally (sorted)."""
    from . import source_matrices

    return sorted(c for c in source_matrices.available_cohorts() if source_matrices.is_cached(c))


def cta_patient_count_heatmap(
    *,
    threshold_tpm=10.0,
    cohorts=None,
    n_cohorts=30,
    n_ctas=30,
    value="fraction",
    save=None,
):
    """Heatmap of **per-patient CTA prevalence** — rows are cohorts, columns are
    CTAs, each cell is the fraction (or count) of that cohort's patients expressing
    the CTA above ``threshold_tpm`` clean TPM.

    Unlike :func:`cta_expression_heatmap` (cohort summary TPM), this is computed from
    the full per-sample matrices, so it answers "in what fraction of *patients* is
    this antigen expressed". ``cohorts`` defaults to every cohort whose per-sample
    matrix is cached; ``value`` is ``"fraction"`` (0–1) or ``"count"`` (patients).
    Rows/cols are the top ``n_cohorts``/``n_ctas`` by prevalence."""
    import pandas as pd

    from .coverage import cta_patient_fractions

    if value not in ("fraction", "count"):
        raise ValueError("value must be 'fraction' or 'count'")
    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    if not cohorts:
        raise ValueError(
            "no cohorts with a cached per-sample matrix — fetch them via "
            "source_matrices.fetch(code) (or stage them) first."
        )
    col = "fraction_expressing" if value == "fraction" else "n_patients_expressing"
    rows = {}
    key_to_symbol = {}
    for code in cohorts:
        pf = cta_patient_fractions(code, threshold_tpm=threshold_tpm)
        # Index by the proteoform_key (unique per antigen after the collapse) so the
        # per-cohort Series align cleanly; the column display label comes from Symbol.
        rows[code] = pf.groupby("proteoform_key")[col].max()
        key_to_symbol.update(zip(pf["proteoform_key"].astype(str), pf["Symbol"].astype(str)))
    matrix = pd.DataFrame(rows).T  # cohorts × CTA proteoform keys
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError("no CTA expressed above the threshold in the selected cohorts")

    top_ctas = matrix.max(axis=0).sort_values(ascending=False).head(n_ctas).index
    row_score = matrix[top_ctas].max(axis=1)
    top_cohorts = row_score.sort_values(ascending=False).head(n_cohorts).index
    grid = matrix.loc[top_cohorts, top_ctas].fillna(0.0)
    # Label columns by the proteoform symbol (NY-ESO-1, CT83), not the ENSG-or-symbol key.
    grid = grid.rename(columns=lambda key: key_to_symbol.get(str(key), str(key)))

    unit = "fraction of patients" if value == "fraction" else "patients"
    return _cohort_gene_heatmap(
        grid,
        title=(
            f"CTA per-patient prevalence (> {threshold_tpm:g} TPM) — "
            f"{len(top_cohorts)} cohorts × {len(top_ctas)} CTAs"
        ),
        cbar_label=unit,
        cmap="viridis",
        save=save,
    )


def cta_coverage_curves(
    cancer_types,
    *,
    threshold_tpm=10.0,
    max_genes=20,
    n_label=5,
    top_n=40,
    save=None,
):
    """Greedy **antigen-coverage curves** — for each cohort, the cumulative fraction
    of patients covered as CTAs are added to the panel in greedy set-cover order
    (:func:`oncoref.coverage.greedy_coverage`). Answers "how many antigens does a
    panel need to reach most patients of this cancer".

    ``cancer_types`` is a code or iterable of codes (each needs a cached per-sample
    matrix). Rendered as **small multiples** — one panel per cohort, sorted by final
    coverage (broadest first) — with the leading antigens labeled on each curve, so
    you can read *which* CTAs carry the coverage and how fast it plateaus. ``n_label``
    caps how many antigen names are annotated per panel. ``top_n`` (default 40) keeps
    only the largest cohorts by sample count so the grid stays legible; pass
    ``top_n=None`` for every cohort."""
    import numpy as np

    from .coverage import greedy_coverage

    codes = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    requested = len(codes)
    codes = _top_cohorts_by_samples(codes, top_n)
    capped = len(codes) < requested
    plt = _plt()

    curves = []  # (code, x, y, symbols)
    for code in codes:
        gc = greedy_coverage(code, threshold_tpm=threshold_tpm, max_genes=max_genes)
        if gc.empty:
            continue
        x = np.concatenate([[0], gc["rank"].to_numpy()])
        y = np.concatenate([[0.0], gc["cumulative_fraction"].to_numpy()])
        curves.append((code, x, y, [str(s) for s in gc["Symbol"]]))
    if not curves:
        raise ValueError(
            "no coverage curve could be drawn — are the cohorts' per-sample matrices "
            "cached, and does any CTA clear the threshold?"
        )
    curves.sort(key=lambda t: t[2][-1], reverse=True)  # broadest coverage first

    ncol = min(6, len(curves))
    nrow = (len(curves) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * 3.0, nrow * 2.4), sharex=True, sharey=True, squeeze=False
    )
    axes = axes.ravel()
    fam_color, _ = _antigen_family_colors({s for _, _, _, syms in curves for s in syms})
    for ax, (code, x, y, syms) in zip(axes, curves):
        ax.step(x, y, where="post", color="#3b4cc0", lw=1.3)
        ax.fill_between(x, y, step="post", alpha=0.12, color="#3b4cc0")
        for rank, sym in enumerate(syms[:n_label], start=1):
            ax.annotate(
                sym,
                (x[rank], y[rank]),
                fontsize=5,
                rotation=40,
                textcoords="offset points",
                xytext=(2, 2),
                color=fam_color.get(sym, "#333333"),
            )
            ax.plot(x[rank], y[rank], "o", ms=3, color=fam_color.get(sym, "#333333"))
        ax.set_title(f"{format_cancer_code_label(code)}  ({y[-1]:.0%})", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(max_genes, x[-1]))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)
    for ax in axes[len(curves) :]:
        ax.set_visible(False)
    fig.supxlabel("number of CTAs in panel (greedy set cover)", fontsize=9)
    fig.supylabel(f"fraction of patients covered (≥1 CTA > {threshold_tpm:g} TPM)", fontsize=9)
    suffix = (
        f"top {len(curves)} of {requested} by sample count" if capped else f"{len(curves)} cohorts"
    )
    fig.suptitle(f"CTA antigen-coverage curves ({suffix})", fontsize=11)
    fig.tight_layout()
    return _save(fig, save)


def cta_coverage_stacked_bars(
    cancer_types,
    *,
    threshold_tpm=10.0,
    max_genes=12,
    top_n=40,
    save=None,
):
    """Greedy **coverage plateau** as a stacked bar per cohort — one horizontal bar
    per cancer type, split into the marginal new-patient fraction each CTA adds in
    greedy set-cover order (:func:`oncoref.coverage.greedy_coverage`). The bar's
    total length is the cohort's addressable share (≥1 CTA panel); each segment shows
    how much a single antigen contributes, coloured by antigen family.

    Complements :func:`cta_coverage_curves` (cumulative step curve) by showing *which*
    antigens carry the coverage. ``cancer_types`` is a code or iterable; each needs a
    cached per-sample matrix. ``max_genes`` caps the segments per cohort; ``top_n``
    (default 40) keeps only the largest cohorts by sample count (pass ``None`` for all)."""
    from .coverage import greedy_coverage

    codes = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    requested = len(codes)
    codes = _top_cohorts_by_samples(codes, top_n)
    capped = len(codes) < requested
    coverage_by_code = {}
    for code in codes:
        gc = greedy_coverage(code, threshold_tpm=threshold_tpm, max_genes=max_genes)
        if not gc.empty:
            coverage_by_code[code] = gc
    if not coverage_by_code:
        raise ValueError(
            "no coverage to plot — are the cohorts' per-sample matrices cached, and "
            "does any CTA clear the threshold?"
        )
    all_syms = {str(s) for gc in coverage_by_code.values() for s in gc["Symbol"]}
    sym_color, fam_color = _antigen_family_colors(all_syms)

    rows = []
    for code, gc in coverage_by_code.items():
        covered = float(gc["cumulative_fraction"].iloc[-1])
        segs = [
            (str(r.Symbol), float(r.marginal_fraction), sym_color[str(r.Symbol)])
            for r in gc.itertuples()
        ]
        rows.append((f"{format_cancer_code_label(code)}  ({covered:.0%})", segs))
    suffix = f"top {len(rows)} of {requested} by sample count" if capped else f"{len(rows)} cohorts"
    return _stacked_barh(
        rows,
        xlabel=f"fraction of patients covered (≥1 CTA > {threshold_tpm:g} TPM)",
        title=f"CTA greedy coverage by antigen — {suffix}",
        legend=fam_color,
        save=save,
    )


def burden_category_bars(*, region="us", n=None, save=None):
    """Grouped horizontal bars of cancer **incidence vs mortality** share per burden
    category for a region (``"us"``/``"world"``), categories ordered by incidence. The
    reference view behind :func:`incidence_vs_mortality` — read the gap between a
    category's two bars as its lethality."""
    if region not in ("us", "world"):
        raise ValueError("region must be 'us' or 'world'")
    plt = _plt()
    inc = cancer_burden(metric=f"{region}_incidence_pct")
    mort = cancer_burden(metric=f"{region}_mortality_pct")
    cats = sorted(set(inc) & set(mort), key=lambda c: inc.get(c, 0.0), reverse=True)
    if not cats:
        raise ValueError("no burden category with both an incidence and a mortality share")
    if n is not None:
        cats = cats[:n]
    cmap = plt.get_cmap("tab10")
    return _grouped_barh(
        cats,
        [
            ("incidence", [inc[c] for c in cats], cmap(0)),
            ("mortality", [mort[c] for c in cats], cmap(3)),
        ],
        xlabel=f"{region.upper()} share of cancer cases / deaths (%)",
        title=f"Cancer burden by category ({region.upper()}) — incidence vs mortality",
        save=save,
    )


def cta_burden_vs_response(*, against="apd1", threshold_tpm=10.0, cohorts=None, save=None):
    """Scatter of a cohort's **mean CTA antigen load** (mean number of CTAs a patient
    expresses above ``threshold_tpm``, from
    :func:`oncoref.coverage.mean_antigens_per_patient`) vs its anti-PD-1 ORR
    (``against="apd1"``) or median TMB (``against="tmb"``), one point per cancer type,
    coloured by lineage family.

    The per-patient counterpart to :func:`apd1_response_signature_scatter`: does a
    cohort's CTA antigen load track checkpoint response / mutational burden? Points are
    cohorts with BOTH a cached per-sample matrix and the chosen response metric. Needs
    the per-sample matrices cached."""
    from .coverage import mean_antigens_per_patient

    if against == "apd1":
        ymap, ylabel = cancer_apd1_response(), "Anti-PD-1 monotherapy ORR (%)"
    elif against == "tmb":
        ymap, ylabel = cancer_tmb(), "Median tumor mutational burden (mut/Mb)"
    else:
        raise ValueError("against must be 'apd1' or 'tmb'")

    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = []
    for code in cohorts:
        if code not in ymap:
            continue
        load = mean_antigens_per_patient(code, threshold_tpm=threshold_tpm)
        points.append((code, load, float(ymap[code])))
    if not points:
        raise ValueError(
            f"no cohort with both a cached per-sample matrix and a {against} value — "
            "fetch the matrices (source_matrices.fetch) for those cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"mean CTAs expressed per patient (> {threshold_tpm:g} TPM)",
        ylabel=ylabel,
        title=f"CTA antigen load vs {against} — {len(points)} cancers",
        save=save,
    )


def apd1_response_signature_scatter(signature="t_cell_inflamed", *, cohorts=None, save=None):
    """Scatter of a cohort's **aPD1 response-signature score** vs its anti-PD-1 ORR,
    one point per cancer type — the mechanism view of *why* response varies.

    The signature score is the cohort-mean log clean-TPM of the signature's genes
    (:func:`oncoref.response_signatures.signature_score`); ``signature`` is one of
    :func:`oncoref.response_signatures.response_signature_names` (e.g.
    ``"t_cell_inflamed"`` / ``"cytotoxic"`` / ``"antigen_presentation"`` →
    response-associated, ``"tgfb_exclusion"`` → resistance-associated). Points are
    cohorts that have BOTH a cached per-sample matrix and a curated aPD1 ORR,
    coloured by lineage family. Needs the per-sample matrices cached."""
    from .response_signatures import response_signature_direction, signature_score

    direction = response_signature_direction(signature)  # validates the name
    orr = cancer_apd1_response()
    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = []
    for code in cohorts:
        if code not in orr:
            continue
        score = signature_score(code, signature)
        if score == score:  # not NaN
            points.append((code, score, float(orr[code])))
    if not points:
        raise ValueError(
            "no cohort with both a cached per-sample matrix and an aPD1 ORR — fetch "
            "the matrices (source_matrices.fetch) for aPD1 cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"{signature} signature score (cohort-mean log clean TPM)",
        ylabel="Anti-PD-1 monotherapy ORR (%)",
        title=f"aPD1 response vs {signature} ({direction}-associated) — {len(points)} cancers",
        save=save,
    )


def cta_specific_9mer_load(*, against="tmb", threshold_tpm=10.0, cohorts=None, save=None):
    """Scatter of a cohort's **mean per-patient CTA-specific 9-mer load**
    (:func:`oncoref.peptides.cta_specific_9mer_load`) vs its median TMB
    (``against="tmb"``) or anti-PD-1 ORR (``against="apd1"``), one point per cancer
    type, coloured by lineage family.

    The 9-mer load is, for the average patient, the total CTA-specific 9-mers across
    the CTAs they express above ``threshold_tpm`` — a per-patient measure of
    tumor-restricted neoepitope source breadth. CTA-specific 9-mers come from the
    reference proteome (the longest protein per gene, background-subtracted against
    all non-CTA proteins); see :mod:`oncoref.peptides`.

    Points are cohorts with BOTH a cached per-sample matrix and the chosen metric.
    Needs the per-sample matrices cached and a downloaded Ensembl release with protein
    sequences (the first call builds + caches the per-CTA 9-mer table)."""
    from .peptides import cta_specific_9mer_load as _load

    if against == "tmb":
        xmap, ylabel2 = cancer_tmb(), "Median tumor mutational burden (mut/Mb)"
    elif against == "apd1":
        xmap, ylabel2 = cancer_apd1_response(), "Anti-PD-1 monotherapy ORR (%)"
    else:
        raise ValueError("against must be 'tmb' or 'apd1'")

    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = []
    for code in cohorts:
        if code not in xmap:
            continue
        load = _load(code, threshold_tpm=threshold_tpm)
        points.append((code, load, float(xmap[code])))
    if not points:
        raise ValueError(
            f"no cohort with both a cached per-sample matrix and a {against} value — "
            "fetch the matrices (source_matrices.fetch) for those cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"mean CTA-specific 9-mers per patient (> {threshold_tpm:g} TPM)",
        ylabel=ylabel2,
        title=f"CTA-specific 9-mer load vs {against} — {len(points)} cancers",
        save=save,
    )
