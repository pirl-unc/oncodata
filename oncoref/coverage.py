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

"""Per-patient antigen coverage over the full per-sample matrices.

These answer questions a per-cohort summary cannot, because they need the *joint*
per-sample matrix — which patient expresses which antigen, not just per-gene
prevalence:

  - :func:`cta_patient_fractions` — per gene, the fraction of a cohort's patients
    expressing it above a TPM threshold;
  - :func:`addressable_fraction` — the fraction of patients expressing **≥1** of a
    gene panel (the union — "what share of this cohort a CTA-directed therapy could
    address"), which is NOT the per-gene fractions summed;
  - :func:`greedy_coverage` — a minimal antigen panel by greedy set cover: at each
    step add the gene covering the most still-uncovered patients.

All operate on :func:`oncoref.expression.per_sample_expression` (clean TPM), so
they need the cohort's per-sample matrix fetched (see :mod:`oncoref.source_matrices`).
The default gene panel is the expressed CTA set (:func:`oncoref.cta.cta_gene_ids`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .cancer_genes import cancer_key_genes_df, cancer_type_genes_df
from .cancer_types import format_cancer_code_label, resolve_cancer_type
from .cta import cta_gene_id_to_name, cta_gene_ids
from .expression import per_sample_expression
from .expression_engine import ID_COLUMNS as _ANTIGEN_ID_COLS
from .gene_families import gene_families, gene_family_ids, housekeeping_gene_ids
from .gene_ids import canonical_gene_space, resolve_symbol, symbol_synonyms

#: Default clean-TPM threshold for "expressed in a patient". 10 TPM is a common
#: cut for a confidently-expressed transcript.
DEFAULT_EXPRESSED_TPM: float = 10.0

# Mirrors pirlygenes' patient-coverage tabulation cutoffs so the generated CSVs
# are comparable across packages.
DEFAULT_THRESHOLDS: tuple[float, ...] = (25, 50, 100, 200)

_BASE = ["Ensembl_Gene_ID", "Symbol"]


def _clean_id(value: str) -> str:
    return str(value).split(".", 1)[0].strip()


def _symbol_to_id_map() -> dict[str, str]:
    """Case-insensitive symbol/synonym -> canonical ENSG, using bundled data only."""
    df = canonical_gene_space()
    out = {
        str(sym).upper(): _clean_id(gid)
        for gid, sym in zip(df["ensembl_gene_id"], df["symbol"])
        if str(gid).strip() and str(sym).strip()
    }
    # Keep canonical symbols authoritative, then add NCBI synonym aliases when the
    # official symbol exists in the canonical space.
    for alias, official in symbol_synonyms().items():
        if official and official.upper() in out:
            out.setdefault(alias.upper(), out[official.upper()])
    return out


def _symbols_to_ids(symbols: Iterable[str]) -> set[str]:
    by_symbol = _symbol_to_id_map()
    out: set[str] = set()
    for symbol in symbols:
        s = str(symbol).strip()
        if not s or s.lower() == "nan":
            continue
        if s.upper().startswith("ENSG"):
            out.add(_clean_id(s))
            continue
        gid = by_symbol.get(resolve_symbol(s).upper()) or by_symbol.get(s.upper())
        if gid:
            out.add(gid)
    return out


def _ids_from_frame(df: pd.DataFrame, *, symbol_col: str = "Symbol") -> set[str]:
    ids: set[str] = set()
    if "Ensembl_Gene_ID" in df.columns:
        ids |= {
            _clean_id(v)
            for v in df["Ensembl_Gene_ID"].dropna().astype(str)
            if v.strip() and v.strip().lower() != "nan"
        }
    elif "ensembl_gene_id" in df.columns:
        ids |= {
            _clean_id(v)
            for v in df["ensembl_gene_id"].dropna().astype(str)
            if v.strip() and v.strip().lower() != "nan"
        }
    if symbol_col in df.columns:
        ids |= _symbols_to_ids(df[symbol_col].dropna().astype(str))
    return ids


def _gene_set_from_file(path: Path) -> tuple[str, set[str]]:
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    if suffix == ".txt":
        df = pd.read_csv(path, header=None, names=["Symbol"])
    else:
        try:
            df = pd.read_csv(path, sep=sep)
        except pd.errors.ParserError:
            df = pd.read_csv(path, header=None, names=["Symbol"])
    if len(df.columns) == 1 and df.columns[0] not in {"Symbol", "Ensembl_Gene_ID"}:
        df = df.rename(columns={df.columns[0]: "Symbol"})
    ids = _ids_from_frame(df)
    if not ids:
        raise ValueError(f"gene-set file {path} did not contain resolvable Ensembl IDs/symbols")
    return path.stem, ids


def resolve_gene_set(name: str) -> tuple[str, set[str]]:
    """Resolve a plot gene-set token to ``(label, unversioned Ensembl IDs)``.

    Supported tokens mirror pirlygenes where oncoref owns the data:
    ``cta``, ``mito``/``mitochondrial``, ``housekeeping``, ``family:<name>``,
    ``therapy:<agent_class>``, ``lineage:<cancer_code>``, or a CSV/TSV/TXT file
    with ``Ensembl_Gene_ID`` and/or ``Symbol``.
    """
    token = str(name).strip()
    low = token.lower()
    if low == "cta":
        return "CTA", {_clean_id(g) for g in cta_gene_ids()}
    if low in {"mito", "mitochondrial"}:
        return "mitochondrial", set(gene_family_ids("mitochondrial"))
    if low == "housekeeping":
        return "housekeeping", set(housekeeping_gene_ids())
    if low.startswith("family:"):
        family = token.split(":", 1)[1].strip()
        if family not in gene_families():
            raise ValueError(f"unknown gene family {family!r}; one of {sorted(gene_families())}")
        return f"family:{family}", set(gene_family_ids(family))
    if low.startswith("therapy:"):
        agent_class = token.split(":", 1)[1].strip().lower()
        df = cancer_key_genes_df()
        target = df["role"].astype(str).str.lower() == "target"
        cls = df["agent_class"].astype(str).str.lower() == agent_class
        ids = _symbols_to_ids(df.loc[target & cls, "symbol"].dropna().astype(str))
        if not ids:
            raise ValueError(f"no therapy target genes found for agent_class={agent_class!r}")
        return f"therapy:{agent_class}", ids
    if low.startswith("lineage:"):
        code = resolve_cancer_type(token.split(":", 1)[1])
        df = cancer_type_genes_df()
        ids = _ids_from_frame(df[df["Cancer_Type"].astype(str) == code])
        if not ids:
            raise ValueError(f"no lineage gene panel found for {code}")
        return f"lineage:{code}", ids
    path = Path(token).expanduser()
    if path.exists():
        return _gene_set_from_file(path)
    raise ValueError(
        f"unknown gene set {name!r}; use cta, mito, housekeeping, family:<name>, "
        "therapy:<agent_class>, lineage:<code>, or a path to a CSV/TSV/TXT gene panel"
    )


def _panel_ids(gene_ids: Iterable[str] | None) -> set[str]:
    ids = set(gene_ids) if gene_ids is not None else set(cta_gene_ids())
    return {str(g).split(".")[0] for g in ids}


def _hit_matrix(cancer_type, *, threshold_tpm: float, gene_ids, proteoform: bool = True):
    """``(panel rows, sample columns, gene×sample boolean hit matrix, id columns)``
    where a hit is clean TPM > ``threshold_tpm`` for one antigen in one patient.

    With ``proteoform=True`` (default), identical-protein paralogs in the panel
    (CTAG1A+CTAG1B = NY-ESO-1, XAGE1A+XAGE1B, …) are **summed per patient** into one
    antigen row before thresholding — the biologically correct unit for antigen
    coverage: RNA-seq reads multi-map between the loci (so per-gene TPM under-counts
    the proteoform), and a TCR/vaccine targets the shared protein once. With
    ``proteoform=False`` each gene is kept separate."""
    df = per_sample_expression(cancer_type, normalize="tpm_clean")
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    panel = _panel_ids(gene_ids)
    sub = df[unversioned.isin(panel)].reset_index(drop=True)
    if proteoform and len(sub):
        from .proteoforms import collapse_to_proteoforms

        sub = collapse_to_proteoforms(sub).reset_index(drop=True)
    id_cols = [c for c in _ANTIGEN_ID_COLS if c in sub.columns]
    samples = [c for c in sub.columns if c not in id_cols]
    hits = sub[samples].to_numpy(dtype=float) > float(threshold_tpm)
    return sub, samples, hits, id_cols


def cta_patient_fractions(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.DataFrame:
    """Per antigen, the fraction of a cohort's patients expressing it above
    ``threshold_tpm`` (clean TPM). Returns ``Ensembl_Gene_ID`` (a real ENSG — the
    canonical member for a collapsed proteoform), ``Symbol`` (the proteoform symbol —
    alias ``NY-ESO-1`` / contracted ``XAGE1A/B`` — when collapsed), ``proteoform_members``
    (the sorted member symbols, present when ``proteoform=True``), ``fraction_expressing``,
    ``n_patients_expressing``, ``n_patients`` — sorted by prevalence. The default
    gene panel is the expressed CTA set; identical-protein paralogs are summed to
    one antigen (``proteoform=True``)."""
    sub, samples, hits, id_cols = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    n = len(samples)
    out = sub[id_cols].copy()
    counts = hits.sum(axis=1)
    out["n_patients_expressing"] = counts
    out["fraction_expressing"] = counts / n if n else 0.0
    out["n_patients"] = n
    return out.sort_values("fraction_expressing", ascending=False).reset_index(drop=True)


def addressable_fraction(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> float:
    """Fraction of a cohort's patients expressing **at least one** antigen in the
    panel above ``threshold_tpm`` — the faithful "addressable" share (the union across
    patients, which the per-gene fractions can't be summed into). Identical-protein
    paralogs are summed to one antigen (``proteoform=True``). 0.0 for an empty
    cohort/panel."""
    _, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    if not samples or hits.size == 0:
        return 0.0
    return float(hits.any(axis=0).mean())


def greedy_coverage(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    max_genes: int | None = None,
    proteoform: bool = True,
) -> pd.DataFrame:
    """Greedy set-cover panel: at each step add the antigen covering the most
    patients not yet covered by the chosen panel, until every coverable patient is
    covered (or ``max_genes`` is reached). Identical-protein paralogs are summed to
    one antigen (``proteoform=True``), so e.g. CTAG1A/CTAG1B counts once.

    Returns one row per chosen antigen, in selection order: ``rank``,
    ``Ensembl_Gene_ID``, ``Symbol``, ``marginal_patients`` (newly covered),
    ``marginal_fraction``, ``cumulative_patients``, ``cumulative_fraction``. The
    cumulative fraction is the coverage curve; its last value equals
    :func:`addressable_fraction` once the panel is exhausted (unless ``max_genes``
    truncates it first). Ties are broken by total prevalence (deterministic)."""
    sub, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    n = len(samples)
    rows: list[dict] = []
    if n == 0 or hits.size == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "Ensembl_Gene_ID",
                "Symbol",
                "marginal_patients",
                "marginal_fraction",
                "cumulative_patients",
                "cumulative_fraction",
                "proteoform_key",
                "proteoform_members",
            ]
        )

    covered = np.zeros(n, dtype=bool)
    remaining = set(range(len(sub)))
    total_prev = hits.sum(axis=1)  # tie-break: prefer the more broadly expressed gene
    limit = max_genes if max_genes is not None else len(sub)

    while remaining and len(rows) < limit:
        # Pick the gene covering the most still-uncovered patients; ties by total
        # prevalence, then by smallest row index — fully deterministic (sorted scan +
        # strict tuple comparison), independent of set-iteration order.
        best_g, best_new, best_prev = None, 0, -1
        for g in sorted(remaining):
            new = int(np.count_nonzero(hits[g] & ~covered))
            if (new, total_prev[g]) > (best_new, best_prev):
                best_g, best_new, best_prev = g, new, int(total_prev[g])
        if best_g is None or best_new == 0:
            break  # no remaining gene covers a new patient
        covered |= hits[best_g]
        remaining.discard(best_g)
        cum = int(covered.sum())
        row = {
            "rank": len(rows) + 1,
            "Ensembl_Gene_ID": str(sub.at[best_g, "Ensembl_Gene_ID"]),
            "Symbol": str(sub.at[best_g, "Symbol"]),
            "marginal_patients": best_new,
            "marginal_fraction": best_new / n,
            "cumulative_patients": cum,
            "cumulative_fraction": cum / n,
        }
        for _idcol in ("proteoform_key", "proteoform_members"):
            if _idcol in sub.columns:
                row[_idcol] = str(sub.at[best_g, _idcol])
        rows.append(row)
    return pd.DataFrame(rows)


def mean_antigens_per_patient(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> float:
    """Mean number of panel antigens a patient in the cohort expresses above
    ``threshold_tpm`` — the per-patient antigen *load* (how many CTAs the average
    patient presents, not just whether ≥1). Equals the sum over antigens of their
    per-patient prevalence; identical-protein paralogs count once
    (``proteoform=True``). 0.0 for an empty cohort/panel."""
    _, samples, hits, _ = _hit_matrix(
        cancer_type, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
    )
    if not samples or hits.size == 0:
        return 0.0
    return float(hits.sum(axis=0).mean())


def patient_coverage(
    gene_set: str = "cta",
    cohorts: Iterable[str] | None = None,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    *,
    proteoform: bool = True,
) -> pd.DataFrame:
    """Per cohort × gene patient counts for a resolved gene panel.

    This is the oncoref counterpart to pirlygenes' patient-coverage CSV: for each
    cached/selected cohort and each panel gene, tabulate ``n_gtX`` and ``pct_gtX``
    for every requested TPM cutoff. ``gene_set`` accepts :func:`resolve_gene_set`
    tokens; explicit ``cohorts`` are used as-is, while ``None`` means every cached
    per-sample cohort.
    """
    from . import source_matrices

    _label, gene_ids = resolve_gene_set(gene_set)
    thresholds = tuple(float(t) for t in thresholds)
    codes = (
        list(cohorts)
        if cohorts is not None
        else [c for c in source_matrices.available_cohorts() if source_matrices.is_cached(c)]
    )
    rows: list[dict] = []
    for code in codes:
        sub, samples, _hits, id_cols = _hit_matrix(
            code, threshold_tpm=0.0, gene_ids=gene_ids, proteoform=proteoform
        )
        n = len(samples)
        if n == 0 or sub.empty:
            continue
        values = sub[samples].to_numpy(dtype=float)
        for i, r in sub.iterrows():
            row = {
                "cancer_code": str(code),
                "label": format_cancer_code_label(str(code)),
                "n_patients": n,
            }
            for col in id_cols:
                row[col] = r[col]
            any_hit = False
            for t in thresholds:
                count = int(np.count_nonzero(values[i] > t))
                any_hit = any_hit or count > 0
                suffix = f"{t:g}"
                row[f"n_gt{suffix}"] = count
                row[f"pct_gt{suffix}"] = 100.0 * count / n if n else 0.0
            if any_hit:
                rows.append(row)
    columns = (
        ["cancer_code", "label"]
        + [
            c
            for c in ("Ensembl_Gene_ID", "Symbol", "proteoform_key", "proteoform_members")
            if any(c in r for r in rows)
        ]
        + ["n_patients"]
    )
    for t in thresholds:
        suffix = f"{t:g}"
        columns.extend([f"n_gt{suffix}", f"pct_gt{suffix}"])
    return pd.DataFrame(rows, columns=columns)


def _scalar_by_cohort(
    scalar_fn, name, cohorts, *, threshold_tpm, gene_ids, proteoform
) -> pd.Series:
    """Map a per-cohort scalar coverage function over cohorts → ``Series``. When
    ``cohorts`` is ``None`` every cohort with a *cached* per-sample matrix is used
    (uncached ones are skipped, never fetched implicitly); an explicit ``cohorts``
    list is taken as-is."""
    from . import source_matrices

    codes = list(cohorts) if cohorts is not None else source_matrices.available_cohorts()
    out: dict[str, float] = {}
    for code in codes:
        if cohorts is None and not source_matrices.is_cached(code):
            continue
        out[str(code)] = scalar_fn(
            code, threshold_tpm=threshold_tpm, gene_ids=gene_ids, proteoform=proteoform
        )
    return pd.Series(out, name=name)


def mean_antigens_per_patient_by_cohort(
    cohorts: Iterable[str] | None = None,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.Series:
    """``{cohort code -> mean antigens per patient}`` over the cohorts that have a
    cached per-sample matrix (default: all cached ones); uncached cohorts are skipped
    rather than fetched."""
    return _scalar_by_cohort(
        mean_antigens_per_patient,
        "mean_antigens_per_patient",
        cohorts,
        threshold_tpm=threshold_tpm,
        gene_ids=gene_ids,
        proteoform=proteoform,
    )


def addressable_fraction_by_cohort(
    cohorts: Iterable[str] | None = None,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    gene_ids: Iterable[str] | None = None,
    proteoform: bool = True,
) -> pd.Series:
    """``{cohort code -> addressable fraction}`` over the cohorts that have a cached
    per-sample matrix (default: all cached ones); uncached cohorts are skipped rather
    than fetched."""
    return _scalar_by_cohort(
        addressable_fraction,
        "addressable_fraction",
        cohorts,
        threshold_tpm=threshold_tpm,
        gene_ids=gene_ids,
        proteoform=proteoform,
    )


def cta_id_to_name() -> dict[str, str]:
    """``{unversioned CTA gene id -> symbol}`` for labelling coverage outputs."""
    return cta_gene_id_to_name()


# --- pirlygenes-style rendering -------------------------------------------------------

_PALETTE = [
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#9A6324",
    "#800000",
    "#808000",
    "#000075",
    "#e6beff",
    "#aaffc3",
    "#ffd8b1",
    "#a9a9a9",
    "#fabed4",
]


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(label).lower()).strip("_")


def _gene_color_map(genes_ordered):
    seen, colors = [], {}
    for gene in genes_ordered:
        if gene not in colors:
            colors[gene] = _PALETTE[len(seen) % len(_PALETTE)]
            seen.append(gene)
    return colors


def _stacked_bar(per, label, threshold, path, plt):
    from collections import Counter

    totals = Counter()
    for _code, _n, cum, names in per:
        prev = 0.0
        for name, c in zip(names, cum):
            totals[name] += (c - prev) * 100
            prev = c
    color = _gene_color_map([g for g, _ in totals.most_common()])

    fig, ax = plt.subplots(figsize=(13, max(6, len(per) * 0.28)))
    labels = []
    for y, (code, n, cum, names) in enumerate(per):
        labels.append(f"{format_cancer_code_label(code)}  (n={n})")
        left, prev = 0.0, 0.0
        for j, (name, c) in enumerate(zip(names, cum)):
            marginal = (c - prev) * 100
            prev = c
            if marginal <= 0:
                continue
            ax.barh(
                y,
                marginal,
                left=left,
                color=color.get(name, "#cccccc"),
                edgecolor="white",
                linewidth=0.3,
            )
            if marginal >= 1.5 and (marginal >= 3.0 or j == 0):
                ax.text(
                    left + marginal / 2,
                    y,
                    name,
                    va="center",
                    ha="center",
                    fontsize=4.5,
                    clip_on=True,
                )
            left += marginal
    ax.set_yticks(range(len(per)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel(
        f"% of patients with >=1 {label} gene > {threshold:g} TPM "
        "(stacked by each gene's marginal new-patient share, greedy)"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"{label} coverage by cancer type, split by gene (> {threshold:g} TPM, {len(per)} cohorts)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _coverage_curves(per, label, threshold, path, plt):
    ordered = sorted(per, key=lambda t: t[2][-1], reverse=True)
    ncol = 6
    nrow = (len(ordered) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(ncol * 2.5, nrow * 1.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (code, n, cum, names) in zip(axes, ordered):
        xs = range(1, len(cum) + 1)
        ax.plot(xs, [c * 100 for c in cum], color="#b5179e", lw=1.2)
        ax.fill_between(xs, [c * 100 for c in cum], alpha=0.15, color="#b5179e")
        for x, (name, c) in enumerate(zip(names[:3], cum[:3]), start=1):
            ax.annotate(
                name,
                (x, c * 100),
                fontsize=4,
                rotation=45,
                textcoords="offset points",
                xytext=(1, 2),
            )
        ax.set_title(
            f"{format_cancer_code_label(code)} (n={n}) {cum[-1] * 100:.0f}%",
            fontsize=7,
        )
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25)
    for ax in axes[len(ordered) :]:
        ax.axis("off")
    fig.suptitle(
        f"{label} panel coverage by cancer type - distinct patients "
        f"with >=1 gene > {threshold:g} TPM (sorted by plateau)",
        fontsize=11,
    )
    fig.supxlabel("# genes added (greedy)", fontsize=8)
    fig.supylabel("% patients covered", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_patient_coverage(
    gene_set: str = "cta",
    *,
    cohorts: Iterable[str] | None = None,
    threshold: float = 25,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    out_dir="coverage_out",
    proteoform: bool = True,
) -> dict:
    """Write pirlygenes-style patient-coverage artifacts for a gene panel.

    Outputs a counts CSV plus, when at least one cohort has coverage, a greedy
    stacked bar and coverage-curve small-multiples PNG. Returns a dict with
    ``paths``, ``counts``, ``label``, and ``n_cohorts``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    label, gene_ids = resolve_gene_set(gene_set)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(label)

    counts = patient_coverage(
        gene_set,
        cohorts=cohorts,
        thresholds=thresholds,
        proteoform=proteoform,
    )
    csv_path = out / f"{slug}_patient_counts.csv"
    sort_col = f"n_gt{float(threshold):g}"
    if sort_col in counts.columns:
        counts = counts.sort_values(["cancer_code", sort_col], ascending=[True, False])
    counts.to_csv(csv_path, index=False)

    codes = sorted(set(counts["cancer_code"])) if not counts.empty else []
    per = []
    for code in codes:
        gc = greedy_coverage(
            code,
            threshold_tpm=threshold,
            gene_ids=gene_ids,
            proteoform=proteoform,
        )
        if gc.empty:
            continue
        n_patients = int(counts.loc[counts["cancer_code"] == code, "n_patients"].max())
        per.append(
            (
                code,
                n_patients,
                [float(v) for v in gc["cumulative_fraction"]],
                [str(v) for v in gc["Symbol"]],
            )
        )
    per.sort(key=lambda t: t[2][-1])

    paths = {"counts_csv": str(csv_path)}
    if per:
        paths["stacked_bar"] = str(
            _stacked_bar(
                per,
                label,
                threshold,
                out / f"{slug}_stacked_coverage_t{float(threshold):g}.png",
                plt,
            )
        )
        paths["coverage_curves"] = str(
            _coverage_curves(
                per,
                label,
                threshold,
                out / f"{slug}_coverage_curves_t{float(threshold):g}.png",
                plt,
            )
        )
    return {"paths": paths, "counts": counts, "label": label, "n_cohorts": len(per)}
