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

"""Regenerate the HPA-derived columns of ``cancer-testis-antigens.csv`` from HPA.

This ports the HPA-only producer logic from tsarina (``tsarina.tiers``,
``scripts/add_cta_gene.py``, ``scripts/regenerate_table.py``) into cancerdata so
the bundled table's ~47 RNA/protein/restriction/filter columns can be re-derived
from the Human Protein Atlas alone -- no mass-spec evidence and no pyensembl.

The gene list (Symbol, Ensembl_Gene_ID, Aliases, Full_Name, Function,
source_databases, Canonical_Transcript_ID, biotype) is the FROZEN candidate
universe + identity/annotation columns: it is preserved verbatim from the input
table and never recomputed. ``biotype`` and ``Canonical_Transcript_ID`` in
particular come from pyensembl in tsarina and are deliberately preserved here.

Everything else -- the per-tissue nTPM columns, reproductive fractions, the pct
filters, protein IHC columns, the per-modality and synthesized restriction axes,
``safety_flags``, ``passes_filters`` and ``never_expressed`` -- is recomputed
from HPA ``rna_tissue_consensus`` (RNA) and ``normal_tissue`` (IHC).

``restriction`` / ``restriction_confidence`` come from the HPA-only
:func:`cancerdata.cta.synthesize_restriction` (no MS contribution), applied
row-wise -- matching the provenance cancerdata owns.

Public entry point: :func:`regenerate_cta_columns`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import cta as _cta
from .cta_tissues import (
    ALL_REPRODUCTIVE_TISSUES,
    CORE_REPRODUCTIVE_TISSUES,
    HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS,
    HPA_EXPRESSION_FLOOR_NTPM,
    NON_SOMATIC_TISSUES,
    PROTEIN_DETECTED_LEVELS,
    SAFETY_NTPM_THRESHOLD,
    SAFETY_TISSUE_GROUPS,
)
from .load_dataset import _register_derived_cache, get_data

# ── Curated overrides ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def cross_reactive_ihc() -> frozenset[str]:
    """Unversioned Ensembl IDs whose v23 HPA IHC is treated as **unreliable**
    (forced to "no data"), so they fall back to the RNA-only restriction call.

    These are sequence-near-identical CT/paralog antigens whose shared antibody
    cross-reacts: HPA "detects" Low protein in tissues where the gene's RNA is
    ~0 nTPM (heart/glandular cells), the hallmark of cross-reactivity. The set is
    curation knowledge that needs a human, so it lives in an auditable data file
    (``cta-ihc-unreliable.csv``: gene id, symbol, reason) rather than a hardcoded
    list. (Ported from tsarina's ``scripts/regenerate_table.py``.)
    """
    df = get_data("cta-ihc-unreliable", copy=False)
    return frozenset(df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0])


_register_derived_cache(cross_reactive_ihc.cache_clear)


#: Reproductive (+thymus) tissues, lowercased, for the protein restriction call.
_PROTEIN_REPRODUCTIVE: frozenset[str] = frozenset(t.lower() for t in ALL_REPRODUCTIVE_TISSUES)

#: Lowercased non-somatic tissue set used to exclude reproductive/thymus tissues
#: from the somatic RNA detail columns.
_NON_SOMATIC_LOWER: frozenset[str] = frozenset(t.lower() for t in NON_SOMATIC_TISSUES)

#: Deflated-fraction pct filter thresholds (column suffix, threshold).
_PCT_FILTERS: list[tuple[str, float]] = [
    ("80", 0.80),
    ("90", 0.90),
    ("95", 0.95),
    ("97", 0.97),
    ("98", 0.98),
    ("99", 0.99),
]

#: Labels that mean "no HPA IHC protein data" for a gene.
_NO_PROTEIN = {"no data", "nan", ""}


# ── Reproductive fractions ─────────────────────────────────────────────────


def _fraction(ntpm: dict[str, float], allowed: frozenset[str], deflate: bool) -> float:
    """Fraction of (optionally deflated) nTPM signal coming from *allowed* tissues.

    Deflation subtracts a +1 pseudocount per tissue (floored at 0) before
    summing, damping near-noise detections. With deflation, a gene whose every
    tissue is below 1 nTPM has total 0 and returns 1.0 (fully restricted by
    default); without deflation an all-zero gene returns 0.0.
    """
    vals = {t: (max(0.0, v - 1.0) if deflate else v) for t, v in ntpm.items()}
    total = sum(vals.values())
    if total <= 0:
        return 1.0 if deflate else 0.0
    return sum(v for t, v in vals.items() if t in allowed) / total


# ── RNA per-tissue enrichment ──────────────────────────────────────────────


def _build_ntpm_by_gene(consensus: pd.DataFrame) -> dict[str, dict[str, float]]:
    """``{unversioned ENSG: {tissue(lower): nTPM}}`` from HPA rna consensus."""
    by_gene: dict[str, dict[str, float]] = {}
    for ensg, sub in consensus.groupby("Gene"):
        gid = str(ensg).split(".")[0]
        by_gene[gid] = {
            str(t).strip().lower(): float(v) for t, v in zip(sub["Tissue"], sub["nTPM"])
        }
    return by_gene


def _enrich_rna_per_tissue(seed: pd.DataFrame, ntpm_by_gene: dict[str, dict[str, float]]) -> None:
    """Add the per-tissue RNA nTPM detail + per-safety-tissue max columns in place.

    Faithful port of ``tsarina.tiers.enrich_rna_per_tissue``: per-gene testis /
    ovary / placenta nTPM, the max somatic tissue + its nTPM, the count of
    somatic tissues detected at >= 1 nTPM, and the per-safety-group max nTPM.
    """
    testis_vals: list[float] = []
    ovary_vals: list[float] = []
    placenta_vals: list[float] = []
    max_somatic_tissues: list[str] = []
    max_somatic_ntpms: list[float] = []
    somatic_detected_counts: list[int] = []
    safety_maxes: dict[str, list[float]] = {grp: [] for grp in SAFETY_TISSUE_GROUPS}

    for _, row in seed.iterrows():
        gid = str(row.get("Ensembl_Gene_ID", "")).split(".")[0]
        tissue_ntpm = ntpm_by_gene.get(gid)

        if not tissue_ntpm:
            testis_vals.append(0.0)
            ovary_vals.append(0.0)
            placenta_vals.append(0.0)
            max_somatic_tissues.append("")
            max_somatic_ntpms.append(0.0)
            somatic_detected_counts.append(0)
            for grp in SAFETY_TISSUE_GROUPS:
                safety_maxes[grp].append(0.0)
            continue

        testis_vals.append(tissue_ntpm.get("testis", 0.0))
        ovary_vals.append(tissue_ntpm.get("ovary", 0.0))
        placenta_vals.append(tissue_ntpm.get("placenta", 0.0))

        somatic = {t: v for t, v in tissue_ntpm.items() if t not in _NON_SOMATIC_LOWER}
        somatic_detected = {t: v for t, v in somatic.items() if v >= 1.0}

        if somatic:
            max_t = max(somatic, key=somatic.get)  # type: ignore[arg-type]
            max_somatic_tissues.append(max_t)
            max_somatic_ntpms.append(somatic[max_t])
        else:
            max_somatic_tissues.append("")
            max_somatic_ntpms.append(0.0)

        somatic_detected_counts.append(len(somatic_detected))

        for grp, tissues in SAFETY_TISSUE_GROUPS.items():
            grp_vals = [tissue_ntpm.get(t, 0.0) for t in tissues]
            safety_maxes[grp].append(max(grp_vals) if grp_vals else 0.0)

    seed["rna_testis_ntpm"] = testis_vals
    seed["rna_ovary_ntpm"] = ovary_vals
    seed["rna_placenta_ntpm"] = placenta_vals
    seed["rna_max_somatic_tissue"] = max_somatic_tissues
    seed["rna_max_somatic_ntpm"] = max_somatic_ntpms
    seed["rna_somatic_detected_count"] = somatic_detected_counts
    for grp in SAFETY_TISSUE_GROUPS:
        seed[f"rna_{grp}_max_ntpm"] = safety_maxes[grp]


# ── Protein (IHC) columns ──────────────────────────────────────────────────


def _recompute_protein_columns(seed: pd.DataFrame, normal_tissue: pd.DataFrame) -> None:
    """Recompute the four HPA IHC protein columns in place from normal_tissue.

    Faithful port of ``tsarina/scripts/regenerate_table._recompute_protein_columns``:

    * a gene absent from the IHC table, in :func:`cross_reactive_ihc`, or
      detecting protein in no tissue (Level not Low/Medium/High) -> all four
      columns are ``"no data"`` (the original pipeline collapsed "antibody
      present, nothing detected" into ``no data``);
    * otherwise ``protein_strict_expression`` is the sorted detected tissues
      joined by "; ", ``protein_reliability`` the gene's HPA reliability tier,
      ``protein_reproductive`` whether the detected tissues are all reproductive
      (+thymus), and ``protein_thymus`` whether thymus is detected.
    """
    nt = normal_tissue.copy()
    nt["gid"] = nt["Gene"].astype(str).str.split(".").str[0]
    nt["tl"] = nt["Tissue"].astype(str).str.strip().str.lower()
    detected = nt[nt["Level"].isin(PROTEIN_DETECTED_LEVELS)]
    det_by_gene = detected.groupby("gid")["tl"].apply(lambda s: sorted(set(s)))
    rel_by_gene = nt.groupby("gid")["Reliability"].first()

    cross_reactive = cross_reactive_ihc()
    for idx, row in seed.iterrows():
        gid = str(row["Ensembl_Gene_ID"]).split(".")[0]
        det = None if gid in cross_reactive else det_by_gene.get(gid)
        if not det:  # absent / cross-reactive IHC, or detected nowhere -> no data
            seed.at[idx, "protein_strict_expression"] = "no data"
            seed.at[idx, "protein_reliability"] = "no data"
            seed.at[idx, "protein_reproductive"] = "no data"
            seed.at[idx, "protein_thymus"] = "no data"
            continue
        seed.at[idx, "protein_strict_expression"] = "; ".join(det)
        seed.at[idx, "protein_reliability"] = rel_by_gene[gid]
        # Capitalized to match the shipped table's bool-string representation.
        seed.at[idx, "protein_reproductive"] = str(set(det) <= _PROTEIN_REPRODUCTIVE)
        seed.at[idx, "protein_thymus"] = str("thymus" in det)


def _parse_protein_tissues(protein_strict_expression: str) -> set[str]:
    val = str(protein_strict_expression).strip().lower()
    if val in ("no data", "not detected", "", "nan"):
        return set()
    return {t.strip() for t in val.split(";") if t.strip()}


def _assign_protein_restriction(row: pd.Series) -> str:
    """Assign tissue restriction from IHC protein data.

    Returns TESTIS / PLACENTAL / REPRODUCTIVE / SOMATIC / NO_DATA. Each single-
    tissue value means *only* that core tissue detected; REPRODUCTIVE means
    multiple/extended reproductive tissues; SOMATIC means a non-reproductive
    tissue is present; NO_DATA means no protein expression (after dropping
    thymus).
    """
    tissues = _parse_protein_tissues(str(row.get("protein_strict_expression", "")))
    if not tissues:
        return "NO_DATA"
    tissues = tissues - {"thymus"}
    if not tissues:
        return "NO_DATA"
    non_repro = tissues - _PROTEIN_REPRODUCTIVE
    if non_repro:
        return "SOMATIC"
    core = tissues & {"testis", "ovary", "placenta"}
    if core == {"testis"}:
        return "TESTIS"
    if core == {"placenta"}:
        return "PLACENTAL"
    if core:
        return "REPRODUCTIVE"
    # Only extended reproductive tissues (epididymis, etc.), no core
    return "REPRODUCTIVE"


def _protein_tissue_flag(protein_strict_expression: str, tissue: str) -> str:
    val = str(protein_strict_expression).strip().lower()
    if val in ("no data", "not detected", "", "nan"):
        return ""
    tissues = {t.strip() for t in val.split(";") if t.strip()}
    return str(tissue in tissues)


# ── RNA restriction axes ───────────────────────────────────────────────────


def _assign_rna_restriction(row: pd.Series) -> str:
    """Assign tissue restriction from RNA per-tissue nTPM data.

    Detection threshold is nTPM >= 1.0. Returns SOMATIC if any somatic tissue is
    detected, else TESTIS / PLACENTAL / REPRODUCTIVE / NO_DATA from which
    reproductive tissues are detected.
    """
    testis = float(row.get("rna_testis_ntpm", 0) or 0)
    ovary = float(row.get("rna_ovary_ntpm", 0) or 0)
    placenta = float(row.get("rna_placenta_ntpm", 0) or 0)
    somatic_count = int(row.get("rna_somatic_detected_count", 0) or 0)

    has_testis = testis >= 1.0
    has_ovary = ovary >= 1.0
    has_placenta = placenta >= 1.0
    has_somatic = somatic_count > 0

    if has_somatic:
        return "SOMATIC"
    if not (has_testis or has_ovary or has_placenta) and not has_somatic:
        return "NO_DATA"
    if has_testis and not has_ovary and not has_placenta:
        return "TESTIS"
    if has_placenta and not has_ovary and not has_testis:
        return "PLACENTAL"
    return "REPRODUCTIVE"


def _assign_rna_restriction_level(row: pd.Series) -> str:
    """Assign RNA restriction quality level from the deflated reproductive fraction.

    STRICT requires fully reproductive (somatic count 0) AND frac >= 0.99;
    MODERATE >= 0.95; PERMISSIVE >= 0.80; else LEAKY. frac < 0 -> NO_DATA.
    """
    try:
        frac = float(row.get("rna_deflated_reproductive_frac", -1))
    except (ValueError, TypeError):
        frac = -1.0

    if frac < 0:
        return "NO_DATA"

    somatic_count = int(row.get("rna_somatic_detected_count", 0) or 0)
    rna_is_reproductive = somatic_count == 0

    if rna_is_reproductive and frac >= 0.99:
        return "STRICT"
    if frac >= 0.95:
        return "MODERATE"
    if frac >= 0.80:
        return "PERMISSIVE"
    return "LEAKY"


# ── Safety flags ───────────────────────────────────────────────────────────


def _assign_safety_flags(df: pd.DataFrame, threshold: float = SAFETY_NTPM_THRESHOLD) -> pd.Series:
    """Semicolon-separated safety tissue groups with max nTPM >= *threshold*."""
    flags: list[str] = []
    for _, row in df.iterrows():
        flagged: list[str] = []
        for grp in SAFETY_TISSUE_GROUPS:
            col = f"rna_{grp}_max_ntpm"
            if col in df.columns:
                val = float(row.get(col, 0) or 0)
                if val >= threshold:
                    flagged.append(grp)
        flags.append(";".join(flagged))
    return pd.Series(flags, index=df.index)


# ── Filter / never-expressed rules ─────────────────────────────────────────


def _has_protein(reliability) -> bool:
    return str(reliability).strip().lower() not in _NO_PROTEIN


def _passes_filters_rule(row: pd.Series, missing_threshold: float) -> bool:
    """Reproductive-restriction filter (adaptive on protein reliability).

    Requires protein_coding biotype. If protein is detected it must be
    reproductive. The required deflated reproductive fraction is the adaptive
    threshold for the protein tier (no protein -> "Missing"), with Uncertain and
    Missing tiers pinned to *missing_threshold*.
    """
    if str(row["biotype"]) != "protein_coding":
        return False
    reliability = str(row["protein_reliability"]).strip()
    has_protein = _has_protein(reliability)
    if has_protein and str(row["protein_reproductive"]).strip().lower() != "true":
        return False
    tier = reliability if has_protein else "Missing"
    threshold = HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS.get(tier, missing_threshold)
    if tier in ("Uncertain", "Missing"):
        threshold = missing_threshold
    frac = row["rna_deflated_reproductive_frac"]
    return False if pd.isna(frac) else bool(float(frac) >= threshold)


def _never_expressed_rule(row: pd.Series, floor: float) -> bool:
    """True for a gene with no HPA IHC protein and max RNA nTPM below *floor*."""
    if _has_protein(row["protein_reliability"]):
        return False
    max_ntpm = row["rna_max_ntpm"]
    return True if pd.isna(max_ntpm) else bool(float(max_ntpm) < floor)


# ── Orchestration ──────────────────────────────────────────────────────────

#: Columns recomputed from HPA RNA (consensus) by :func:`regenerate_cta_columns`.
_RNA_COLUMNS: list[str] = [
    "rna_reproductive_frac",
    "rna_reproductive_and_thymus_frac",
    "rna_deflated_reproductive_frac",
    "rna_deflated_reproductive_and_thymus_frac",
    "rna_max_ntpm",
    "rna_thymus",
    "rna_80_pct_filter",
    "rna_90_pct_filter",
    "rna_95_pct_filter",
    "rna_97_pct_filter",
    "rna_98_pct_filter",
    "rna_99_pct_filter",
    "rna_testis_ntpm",
    "rna_ovary_ntpm",
    "rna_placenta_ntpm",
    "rna_max_somatic_tissue",
    "rna_max_somatic_ntpm",
    "rna_somatic_detected_count",
    "rna_brain_max_ntpm",
    "rna_heart_max_ntpm",
    "rna_lung_max_ntpm",
    "rna_liver_max_ntpm",
    "rna_pancreas_max_ntpm",
    "rna_reproductive",
    "rna_restriction",
    "rna_restriction_level",
]

#: Protein/IHC columns recomputed from HPA normal_tissue.
_PROTEIN_COLUMNS: list[str] = [
    "protein_reproductive",
    "protein_thymus",
    "protein_reliability",
    "protein_strict_expression",
    "protein_restriction",
    "protein_testis",
    "protein_ovary",
    "protein_placenta",
]

#: Synthesis + flag columns recomputed last.
_SYNTHESIS_COLUMNS: list[str] = [
    "restriction",
    "restriction_confidence",
    "safety_flags",
    "passes_filters",
    "never_expressed",
]

#: Identity / annotation columns preserved verbatim (never recomputed).
PRESERVED_COLUMNS: list[str] = [
    "Symbol",
    "Aliases",
    "Full_Name",
    "Function",
    "Ensembl_Gene_ID",
    "source_databases",
    "Canonical_Transcript_ID",
    "biotype",
]

#: Every column this module recomputes from HPA.
RECOMPUTED_COLUMNS: list[str] = _RNA_COLUMNS + _PROTEIN_COLUMNS + _SYNTHESIS_COLUMNS


def _recompute_rna_columns(
    seed: pd.DataFrame, ntpm_by_gene: dict[str, dict[str, float]]
) -> list[str]:
    """Recompute all RNA-derived columns in place; return symbols missing from HPA."""
    core = CORE_REPRODUCTIVE_TISSUES
    core_thymus = frozenset(core | {"thymus"})
    missing: list[str] = []

    for idx, row in seed.iterrows():
        gid = str(row["Ensembl_Gene_ID"]).split(".")[0]
        ntpm = ntpm_by_gene.get(gid)
        if not ntpm:
            missing.append(str(row.get("Symbol", gid)))
            continue
        deflated = _fraction(ntpm, core, deflate=True)
        seed.at[idx, "rna_reproductive_frac"] = round(_fraction(ntpm, core, deflate=False), 4)
        seed.at[idx, "rna_reproductive_and_thymus_frac"] = round(
            _fraction(ntpm, core_thymus, deflate=False), 4
        )
        seed.at[idx, "rna_deflated_reproductive_frac"] = round(deflated, 4)
        seed.at[idx, "rna_deflated_reproductive_and_thymus_frac"] = round(
            _fraction(ntpm, core_thymus, deflate=True), 4
        )
        for pct, thr in _PCT_FILTERS:
            seed.at[idx, f"rna_{pct}_pct_filter"] = bool(deflated >= thr)
        seed.at[idx, "rna_max_ntpm"] = round(max(ntpm.values()), 1)
        seed.at[idx, "rna_thymus"] = bool(ntpm.get("thymus", 0.0) >= 1.0)

    # Per-tissue detail + safety max columns, then the RNA axes.
    _enrich_rna_per_tissue(seed, ntpm_by_gene)
    seed["rna_reproductive"] = seed["rna_somatic_detected_count"].fillna(0).astype(int).eq(0)
    seed["rna_restriction"] = seed.apply(_assign_rna_restriction, axis=1)
    seed["rna_restriction_level"] = seed.apply(_assign_rna_restriction_level, axis=1)
    return missing


def regenerate_cta_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute the HPA-derived CTA columns from HPA, preserving the gene list.

    Takes the shipped ``cancer-testis-antigens.csv`` as a DataFrame (for the
    frozen candidate universe + the preserved identity/annotation columns) and
    returns a new DataFrame with the ~47 HPA columns recomputed from the current
    HPA release. Column order and the preserved columns are kept unchanged.

    Downloads HPA v23 (``rna_tissue_consensus`` + ``normal_tissue``) via the
    cancerdata accessors on first use.

    Parameters
    ----------
    table
        The shipped CTA table. Must carry ``Ensembl_Gene_ID`` and ``biotype``.

    Returns
    -------
    pd.DataFrame
        Same rows/columns as *table*, with the HPA columns recomputed.
    """
    from . import hpa  # lazy: triggers HPA download only when regenerating

    columns = list(table.columns)
    seed = table.copy()

    consensus = hpa.hpa_rna_consensus()
    normal_tissue = hpa.hpa_normal_tissue()
    ntpm_by_gene = _build_ntpm_by_gene(consensus)

    # Protein/IHC first -- the RNA restriction axes and the filters read it.
    _recompute_protein_columns(seed, normal_tissue)
    missing = _recompute_rna_columns(seed, ntpm_by_gene)
    if missing:
        import warnings

        warnings.warn(
            f"{len(missing)} gene(s) absent from HPA rna_consensus keep their input RNA "
            f"values (not recomputed): {', '.join(missing[:10])}"
            + (" …" if len(missing) > 10 else ""),
            stacklevel=2,
        )

    # Protein restriction axis + per-core-tissue flags.
    seed["protein_restriction"] = seed.apply(_assign_protein_restriction, axis=1)
    pse = seed.get("protein_strict_expression", pd.Series([""] * len(seed), index=seed.index))
    seed["protein_testis"] = pse.map(lambda v: _protein_tissue_flag(v, "testis"))
    seed["protein_ovary"] = pse.map(lambda v: _protein_tissue_flag(v, "ovary"))
    seed["protein_placenta"] = pse.map(lambda v: _protein_tissue_flag(v, "placenta"))

    # HPA-only synthesis (protein > RNA; no MS) from cancerdata.cta.
    synth = seed.apply(_cta.synthesize_restriction, axis=1, result_type="expand")
    seed["restriction"] = synth[0]
    seed["restriction_confidence"] = synth[1]

    # Safety flags + filter / never-expressed flags.
    seed["safety_flags"] = _assign_safety_flags(seed)
    missing_threshold = HPA_ADAPTIVE_PROTEIN_RNA_THRESHOLDS["Missing"]
    seed["passes_filters"] = seed.apply(
        lambda r: _passes_filters_rule(r, missing_threshold), axis=1
    )
    seed["never_expressed"] = seed.apply(
        lambda r: _never_expressed_rule(r, HPA_EXPRESSION_FLOOR_NTPM), axis=1
    )

    return seed[columns]
