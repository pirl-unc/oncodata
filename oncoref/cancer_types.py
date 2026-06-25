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

"""Cancer-type ontology: the canonical registry of cancer codes plus the
synonym/alias resolution, per-type metadata, and cohort vocabulary built on
top of it.

Everything is keyed by the canonical registry code and accepts any synonym
(common name, display name, or a pre-rename old code) on input. The facts live
in ``data/cancer-type-registry.csv`` (one row per type); the cohort vocabulary
in ``data/cohort-registry.csv`` and ``data/cancer-cohort-aggregates.csv``.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

import pandas as pd

from .load_dataset import get_data

#: Ensembl release a cohort's gene ids were harmonized to, parsed from registry provenance
#: (e.g. "…harmonized to Ensembl release 112…" / "…Ensembl release 112 gene lengths…").
_ENSEMBL_RELEASE_RE = re.compile(r"Ensembl\s+release\s+(\d{2,3})", re.I)

# Hand-curated common-name aliases. Keyed by lowercase / underscored
# variant; values are canonical codes from cancer-type-registry.csv.
# The registry CSV is the source-of-truth for valid codes and their
# display names — see :data:`CANCER_TYPE_NAMES` below.
CANCER_TYPE_ALIASES = {
    "prostate": "PRAD",
    "breast": "BRCA",
    "lung_adeno": "LUAD",
    "lung_squamous": "LUSC",
    "melanoma": "SKCM",
    "skin": "SKCM",
    "colon": "COAD",
    "colorectal": "COAD",
    "rectal": "READ",
    "pancreatic": "PAAD",
    "pancreas": "PAAD",
    "liver": "LIHC",
    "kidney_clear": "KIRC",
    "kidney_papillary": "KIRP",
    "kidney_chromophobe": "KICH",
    "kidney": "KIRC",
    "ovarian": "OV",
    "ovary": "OV",
    "cervical": "CESC",
    "cervix": "CESC",
    "bladder": "BLCA",
    "stomach": "STAD",
    "gastric": "STAD",
    "glioblastoma": "GBM",
    "gbm": "GBM",
    "head_neck": "HNSC",
    "hnscc": "HNSC",
    "thyroid": "THCA",
    "endometrial": "UCEC",
    "uterine": "UCEC",
    "testicular": "TGCT",
    "testis": "TGCT",
    "sarcoma": "SARC",
    "adrenocortical": "ACC",
    "adrenal": "ACC",
    "cholangiocarcinoma": "CHOL",
    "bile_duct": "CHOL",
    "dlbcl": "DLBC",
    "lymphoma": "DLBC",
    "esophageal": "ESCA",
    "esophagus": "ESCA",
    "aml": "LAML",
    "leukemia": "LAML",
    "low_grade_glioma": "LGG",
    "lgg": "LGG",
    "glioma": "LGG",
    "mesothelioma": "MESO",
    "pheochromocytoma": "PCPG",
    "paraganglioma": "PCPG",
    "thymoma": "THYM",
    "uterine_carcinosarcoma": "UCS",
    "uveal_melanoma": "UVM",
}

# Backward-compat aliases for the Phase-C code renames (old code -> new code).
# resolve_cancer_type consults these so trufflepig / external callers keep
# working without a literal migration. Keep entries here permanently for every
# rename wave (the audit doc promises old codes never hard-break).
_RENAMED_CODE_ALIASES = {
    "OS": "SARC_OS",
    "EWS": "SARC_EWS",
    "CHON": "SARC_CHON",
    "CHOR": "SARC_CHOR",
    "GCTB": "SARC_GCTB",
    "ESS_LG": "SARC_ESS_LG",
    "ESS_HG": "SARC_ESS_HG",
    "RMS_ERMS": "SARC_RMS_ERMS",
    "RMS_ARMS": "SARC_RMS_ARMS",
    "RMS_PRMS": "SARC_RMS_PRMS",
    "RMS_SSRMS": "SARC_RMS_SSRMS",
    "HNSC_HPV_pos": "HNSC_HPVpos",
    "HNSC_HPV_neg": "HNSC_HPVneg",
    # #288 neuroendocrine wave: NET_ (well-diff) / NEC_ (poorly-diff) scheme.
    "MID_NET": "NET_MIDGUT",
    "REC_NET": "NET_RECTAL",
    "LUNG_NET_LC": "NET_LUNG",
    "MEC": "NEC_MERKEL",
    # #288 follow-up: full NET_<site> consistency + spell out the ambiguous "LC"
    # (lung-carcinoid vs large-cell). Chain the pre-5.16 codes to the final name.
    "PANNET": "NET_PANCREAS",
    "LUNG_NET_LCNEC": "NEC_LUNG_LARGECELL",
    "NEC_LUNG_LC": "NEC_LUNG_LARGECELL",
    # #288 one-separator normalization (subtype token has no internal underscore)
    "NBL_MYCN_amp": "NBL_MYCNamp",
    "NBL_MYCN_nonamp": "NBL_MYCNnonamp",
    "LAML_ELN_Fav": "LAML_ELNfav",
    "LAML_ELN_Int": "LAML_ELNint",
    "LAML_ELN_Adv": "LAML_ELNadv",
}
_RENAMED_CODE_ALIASES_UPPER = {k.upper(): v for k, v in _RENAMED_CODE_ALIASES.items()}


class _CancerTypeNamesView:
    """Read-only ``{code: display_name}`` view backed by the registry CSV.

    Consumers treat ``CANCER_TYPE_NAMES`` as a dict (``.get(code)``,
    ``code in CANCER_TYPE_NAMES``, iteration). Loading from the CSV at first
    access keeps the dict in lock-step with the registry — adding a new subtype
    row to ``cancer-type-registry.csv`` automatically extends the dict without
    a code change here.

    The loaded mapping is cached on the instance after the first access. A
    second cached map (``_name_to_code``) holds the lowercased reverse lookup
    used by display-name resolution. Both are protected by a ``threading.Lock``
    so concurrent first-call paths don't both pay the build cost.

    Call ``clear_cache()`` (or the module-level :func:`_clear_caches`) to drop
    the caches — tests that monkey-patch ``get_data`` to swap fixture
    registries need this.
    """

    def __init__(self):
        self._cache: dict | None = None
        self._name_to_code_cache: dict | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._cache is None:
            with self._lock:
                if self._cache is None:
                    df = get_data("cancer-type-registry")
                    # DataFrame-level filter: drop NaN and empty/whitespace
                    # names before building the dict so missing values never
                    # reach ``str(NaN) == "nan"``.
                    df = df[df["name"].notna()]
                    df = df[df["name"].astype(str).str.strip().ne("")]
                    self._cache = dict(zip(df["code"].astype(str), df["name"].astype(str)))
        return self._cache

    def _name_to_code(self):
        """Lowercased ``display_name → code`` for case-insensitive display-name
        resolution. Cached alongside ``_cache``."""
        if self._name_to_code_cache is None:
            with self._lock:
                if self._name_to_code_cache is None:
                    self._name_to_code_cache = {
                        name.lower(): code for code, name in self._load().items()
                    }
        return self._name_to_code_cache

    def clear_cache(self):
        """Drop the cached dicts. Force a re-read on next access."""
        with self._lock:
            self._cache = None
            self._name_to_code_cache = None

    def __getitem__(self, key):
        return self._load()[key]

    def get(self, key, default=None):
        return self._load().get(key, default)

    def __contains__(self, key):
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()

    def __repr__(self):
        return f"_CancerTypeNamesView({len(self._load())} codes)"


# Registry-backed view of cancer code → display name. Reads
# ``cancer-type-registry.csv`` lazily on first access and caches the resolved
# mapping so adding a new subtype row automatically broadens
# ``resolve_cancer_type`` without repeated CSV-parse cost on the hot path.
CANCER_TYPE_NAMES = _CancerTypeNamesView()


def _clear_caches():
    """Reset every registry-backed cache in this module, plus the load_dataset
    frame cache and all derived caches (incidence/cta lookups).

    Test hook for swapping the registry CSV via a monkey-patched ``get_data``;
    not part of the public surface.
    """
    from .load_dataset import _clear_cache

    CANCER_TYPE_NAMES.clear_cache()
    _registry_frame.cache_clear()
    _clear_cache()  # frame cache + registered derived caches (burden maps, CTA, …)


def resolve_cancer_type(cancer_type, *, strict=True):
    """Resolve a cancer type name or alias to a registry code.

    Accepts:
    - canonical registry codes (``"PRAD"``, ``"SARC_DDLPS"``, ``"LAML_APL"``);
    - hand-curated common-name aliases (``"prostate"``, ``"melanoma"``);
    - the registry display name (``"Prostate Adenocarcinoma"``),
      case-insensitive.

    Returns the registry code, or ``None`` if ``cancer_type`` is ``None``.
    For an unknown input: raises ``ValueError`` when ``strict=True`` (default),
    or returns ``None`` when ``strict=False`` (a non-raising lookup for callers
    that want to branch instead of catch).
    """
    if cancer_type is None:
        return None
    raw = str(cancer_type).strip()
    if not raw:
        if strict:
            raise ValueError("Empty cancer type")
        return None

    alias_key = raw.lower().replace(" ", "_").replace("-", "_")
    if alias_key in CANCER_TYPE_ALIASES:
        return CANCER_TYPE_ALIASES[alias_key]

    # Backward-compat: old codes renamed in a Phase-C wave resolve to the new
    # code (case-insensitive), so external callers don't hard-break.
    if raw in _RENAMED_CODE_ALIASES:
        return _RENAMED_CODE_ALIASES[raw]
    if raw.upper() in _RENAMED_CODE_ALIASES_UPPER:
        return _RENAMED_CODE_ALIASES_UPPER[raw.upper()]

    registry = CANCER_TYPE_NAMES  # registry-backed view
    if raw in registry:
        return raw
    upper = raw.upper()
    if upper in registry:
        return upper

    # Display-name lookup (e.g. "Prostate Adenocarcinoma" → "PRAD").
    # The reverse map is cached on the view so this is O(1).
    name_to_code = registry._name_to_code()
    if raw.lower() in name_to_code:
        return name_to_code[raw.lower()]

    if not strict:
        return None
    raise ValueError(
        f"Unknown cancer type {cancer_type!r}. "
        f"Valid registry codes: {sorted(registry.keys())}. "
        f"Common-name aliases: {sorted(CANCER_TYPE_ALIASES.keys())}."
    )


def canonical_cancer_code(code):
    """Map a possibly-renamed cancer code to its canonical current code.

    Pure, registry-free alias lookup over :data:`_RENAMED_CODE_ALIASES`
    (case-insensitive): a pre-rename code like ``"MID_NET"`` or ``"PANNET"``
    returns its current name (``"NET_MIDGUT"`` / ``"NET_PANCREAS"``); any other
    value — including already-canonical codes and non-codes — is returned
    unchanged. Unlike :func:`resolve_cancer_type` this never validates against
    the registry or raises.
    """
    if code is None or code != code:  # None or NaN (NaN != NaN)
        return code
    raw = str(code).strip()
    if raw in _RENAMED_CODE_ALIASES:
        return _RENAMED_CODE_ALIASES[raw]
    return _RENAMED_CODE_ALIASES_UPPER.get(raw.upper(), raw)


_EVIDENCE_SOURCE_CODE = {
    # The source papers report biomarker-selected MSI-H/dMMR colorectal cohorts,
    # not separate colon- and rectum-specific TMB/ICI estimates.
    "COAD_MSI": "CRC_MSI",
    "READ_MSI": "CRC_MSI",
}


def cancer_evidence_source_code(cancer_type, *, strict=True):
    """Canonical code for source-scoped evidence rows.

    Most curated TMB / ICI rows are keyed directly by cancer type. A few molecular
    subtype rows are intentionally source-scoped instead: ``COAD_MSI`` and
    ``READ_MSI`` inherit biomarker-selected colorectal MSI-H/dMMR estimates from
    ``CRC_MSI`` because the published sources report mCRC-level cohorts.
    """
    code = resolve_cancer_type(cancer_type, strict=strict)
    if code is None:
        return None
    return _EVIDENCE_SOURCE_CODE.get(code, code)


def format_cancer_code_label(code):
    """Plot-friendly display label for a cancer-type code.

    A trailing ``pos`` / ``neg`` molecular-status suffix becomes a superscript
    ``⁺`` / ``⁻`` (``HNSC_HPVpos`` → ``HNSC_HPV⁺``); every other code is
    returned unchanged. Uses Unicode superscript glyphs so it renders in any
    matplotlib text without mathtext escaping.
    """
    s = str(code)
    if s.endswith("pos"):
        return s[:-3] + "⁺"  # superscript plus
    if s.endswith("neg"):
        return s[:-3] + "⁻"  # superscript minus
    return s


def cancer_type_info(cancer_type):
    """Resolve any synonym/alias/display-name to a cancer type and return its
    **canonical info** as a dict — the one call to go from a messy input to
    everything the registry knows about that type.

    Routes the input through :func:`resolve_cancer_type`, then assembles the
    registry row plus the derived fields that live in their own tables:
    ``burden_category`` and ``tmb`` (parent-inherited).

    Returns ``None`` if ``cancer_type`` is ``None``; raises ``ValueError`` for
    an unknown input (same contract as :func:`resolve_cancer_type`).

    Keys: ``code``, ``name``, ``family``, ``primary_tissue``,
    ``primary_template``, ``parent_code``, ``subtype_key``, ``pediatric``,
    ``differentiation``, ``expression_source``, ``source_cohort``,
    ``source_pmid``, ``notes``, ``viral_etiology``, ``viral_agent``,
    ``fusion_driven``, ``fusion_driver``, ``burden_category``, ``tmb``.
    """
    # Lazy imports avoid an import cycle: tmb/incidence depend on this module's
    # resolve_cancer_type + cancer_type_registry.
    from .incidence import burden_category
    from .tmb import cancer_tmb

    code = resolve_cancer_type(cancer_type)
    if code is None:
        return None
    reg = cancer_type_registry().set_index("code")
    row = reg.loc[code] if code in reg.index else None
    info = {"code": code, "name": CANCER_TYPE_NAMES.get(code) or code}
    for col in (
        "family",
        "primary_tissue",
        "primary_template",
        "parent_code",
        "subtype_key",
        "pediatric",
        "differentiation",
        "expression_source",
        "source_cohort",
        "source_pmid",
        "notes",
        "viral_etiology",
        "viral_agent",
        "fusion_driven",
        "fusion_driver",
    ):
        val = None if row is None else row.get(col)
        if val is not None and (isinstance(val, str) or not pd.isna(val)):
            # Coerce numpy scalars (e.g. numpy.bool_ for pediatric) to native
            # Python types so the dict is JSON-serializable.
            info[col] = val.item() if hasattr(val, "item") else val
        else:
            info[col] = None
    info["burden_category"] = burden_category(code)
    tmb = cancer_tmb(code)
    info["tmb"] = float(tmb) if tmb is not None else None
    return info


def cancer_type_synonyms(cancer_type):
    """Reverse synonym lookup: every alias that resolves TO a cancer code.

    Returns a sorted list of the common-name aliases (``CANCER_TYPE_ALIASES``),
    registry display name, and pre-rename old codes (``_RENAMED_CODE_ALIASES``)
    that all resolve to the canonical code — the inverse of
    :func:`resolve_cancer_type`. ``[]`` for an unknown input rather than raising.
    """
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return []
    if code is None:
        return []
    syns = {a for a, c in CANCER_TYPE_ALIASES.items() if c == code}
    syns |= {o for o, n in _RENAMED_CODE_ALIASES.items() if n == code}
    name = CANCER_TYPE_NAMES.get(code)
    if name:
        syns.add(name)
    syns.discard(code)
    return sorted(syns)


def viral_status(cancer_type):
    """``{'etiology': ..., 'agent': ...}`` for a cancer type.

    ``etiology`` ∈ {``'defining'``, ``'subset'``, ``'none'``} — whether a virus
    defines the entity/subtype, drives a meaningful subset, or has no
    established role. ``agent`` names the virus (or ``''``). Synonym-resolved;
    raises ``ValueError`` on unknown input.
    """
    info = cancer_type_info(cancer_type)
    if info is None:
        return None
    return {
        "etiology": info.get("viral_etiology") or "none",
        "agent": info.get("viral_agent") or "",
    }


def fusion_status(cancer_type):
    """``{'status': ..., 'driver': ...}`` for a cancer type.

    ``status`` ∈ {``'defining'``, ``'subtype'``, ``'rare'``, ``'none'``}.
    ``driver`` lists the canonical fusion(s). Synonym-resolved; raises
    ``ValueError`` on unknown input.
    """
    info = cancer_type_info(cancer_type)
    if info is None:
        return None
    return {
        "status": info.get("fusion_driven") or "none",
        "driver": info.get("fusion_driver") or "",
    }


def tissue_of_origin(cancer_type):
    """The cancer type's tissue/cell of origin (registry ``primary_tissue``).
    Synonym-resolved; ``None`` for unknown tissue, raises on unknown input."""
    info = cancer_type_info(cancer_type)
    return None if info is None else info.get("primary_tissue")


# Human-readable display names for the registry's ``family`` slugs, so
# consumers don't hardcode the labels.
_FAMILY_DISPLAY_NAMES = {
    "carcinoma-breast": "Breast carcinoma",
    "carcinoma-gi": "Gastrointestinal carcinoma",
    "carcinoma-gu": "Genitourinary carcinoma",
    "carcinoma-head-neck": "Head & neck carcinoma",
    "carcinoma-lung": "Lung carcinoma",
    "carcinoma-mesothelial": "Mesothelioma",
    "carcinoma-other": "Other carcinoma",
    "carcinoma-skin": "Non-melanoma skin carcinoma",
    "cns-glial": "Glial tumor",
    "cns-embryonal": "Embryonal CNS tumor",
    "cns-ependymal": "Ependymal tumor",
    "cns-sellar": "Sellar tumor",
    "cns-meningeal": "Meningeal tumor",
    "cns-choroid": "Choroid plexus tumor",
    "embryonal": "Embryonal tumor",
    "endocrine-epithelial": "Endocrine epithelial carcinoma",
    "endocrine-neuroendocrine": "Endocrine neuroendocrine tumor",
    "germ-cell": "Germ cell tumor",
    "heme-bcell": "B-cell neoplasm",
    "heme-myeloid": "Myeloid neoplasm",
    "heme-plasma": "Plasma cell neoplasm",
    "heme-tcell": "T-cell neoplasm",
    "melanoma": "Melanoma",
    "neuroendocrine": "Neuroendocrine neoplasm",
    "salivary": "Salivary gland carcinoma",
    "sarcoma": "Sarcoma",
    "thymic": "Thymic epithelial tumor",
}


def family_display_name(family):
    """Human-readable label for a registry ``family`` slug (e.g. ``"heme-bcell"``
    -> ``"B-cell neoplasm"``). Falls back to a title-cased de-slugged form for
    any family without a curated label."""
    if family is None:
        return None
    key = str(family).strip()
    if key in _FAMILY_DISPLAY_NAMES:
        return _FAMILY_DISPLAY_NAMES[key]
    return key.replace("-", " ").replace("_", " ").strip().capitalize()


def cancer_type_families():
    """``{family_slug: display_name}`` for every family present in the registry,
    so callers can render a family picker without hardcoding labels."""
    fams = cancer_type_registry()["family"].dropna().astype(str).unique().tolist()
    return {f: family_display_name(f) for f in sorted(fams)}


# ---------- Coarse histogenesis lineage groups ----------


def cancer_lineage_groups():
    """``{registry family -> coarse histogenesis lineage group}``.

    The registry ``family`` column is intentionally uneven in granularity (CNS
    splits into six neuro-lineages, carcinoma by organ system), so it ranges from
    a whole organ system down to single tumours. This rolls the 27 families up to
    8 cell-of-origin classes — **Epithelial, Sarcoma, Heme, CNS, Neuroendocrine,
    Melanoma, Germ cell, Embryonal** — the consistent level for broad
    cross-lineage reasoning and plot colouring (``cancer-lineage-groups.csv``)."""
    df = get_data("cancer-lineage-groups")
    return dict(
        df[["family", "lineage_group"]].drop_duplicates().itertuples(index=False, name=None)
    )


def cancer_lineage_group_overrides():
    """``{code -> lineage_group}`` for codes whose coarse group differs from their
    registry ``family`` default (``cancer-lineage-group-overrides.csv``). The one
    case is neuroblastoma -> ``Embryonal`` (a neural-crest embryonal tumour, not an
    epithelial neuroendocrine neoplasm); it inherits down the ``parent_code`` chain
    (so ``NBL_MYCNamp`` follows ``NBL``)."""
    df = get_data("cancer-lineage-group-overrides")
    return dict(df[["code", "lineage_group"]].itertuples(index=False, name=None))


def cancer_lineage_group(cancer_type):
    """Coarse histogenesis group (Epithelial / Sarcoma / Neuroendocrine / CNS /
    Melanoma / Heme / Germ cell / Embryonal) for a code, alias, or display name.

    Resolution: a per-code override (inherited up the ``parent_code`` chain) if one
    applies, else the registry ``family`` default. Returns ``None`` if the type or
    its family doesn't resolve."""
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return None
    if code is None:
        return None
    reg = _registry_frame().set_index("code")
    if code not in reg.index:
        return None
    overrides = cancer_lineage_group_overrides()
    cur, seen = code, set()
    while cur and cur not in seen:  # nearest override up the parent chain
        seen.add(cur)
        if cur in overrides:
            return overrides[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur, "parent_code"] or "").strip() or None
    family = str(reg.loc[code, "family"] or "")
    return cancer_lineage_groups().get(family)


# ---------- Cancer-type registry ----------


@lru_cache(maxsize=1)
def _registry_frame():
    """Cached registry frame (shared, read-only). Internal hot-path callers use
    this; the public :func:`cancer_type_registry` returns a defensive copy."""
    return get_data("cancer-type-registry", copy=False)


def cancer_type_registry():
    """Return the cancer-type registry: one row per code with family / tissue /
    template / parent / source.

    The registry is a richer superset of TCGA — it covers non-TCGA heme
    malignancies, pediatric cancers, the neuroendocrine axis, and rare
    entities. Each row carries ``code``, ``family``, ``primary_tissue``,
    ``primary_template``, ``parent_code``, ``expression_source``, ``notes`` and
    more. Returns a defensive copy so callers can mutate freely.
    """
    return _registry_frame().copy()


def _split_semicolon(value) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ()
    return tuple(x.strip() for x in str(value).split(";") if x.strip())


def cancer_normal_tissue_map():
    """Curated mapping from registry ``primary_tissue`` values to matched normal
    tissue references.

    Returns a defensive DataFrame with one row per registry primary-tissue token:
    ``primary_tissue``, stable ``normal_tissue_code``, display
    ``normal_tissue_name``, tuple-valued ``hpa_tissues``, and
    ``match_confidence`` / ``match_basis``. ``match_confidence`` is deliberately
    explicit because several cancer registry tissues are composite or do not have
    an exact HPA tissue (for example ``neuroendocrine`` or ``notochord``).
    """
    df = get_data("cancer-normal-tissue-map").copy()
    df["hpa_tissues"] = df["hpa_rna_tissues"].map(_split_semicolon)
    return df[
        [
            "primary_tissue",
            "normal_tissue_code",
            "normal_tissue_name",
            "hpa_tissues",
            "match_confidence",
            "match_basis",
        ]
    ]


def _source_matrix_frame():
    sm = get_data("source-matrices", copy=False)
    return sm.rename(
        columns={
            "source_cohort": "source_matrix_cohort",
            "n_samples": "source_matrix_n_samples",
        }
    )


def _subtype_group_maps():
    df = cancer_subtype_groupings()
    groups: dict[str, tuple[str, ...]] = {}
    axes: dict[str, tuple[str, ...]] = {}
    if df.empty:
        return groups, axes
    for code, grp in df.groupby("member_code", sort=False):
        groups[str(code)] = tuple(dict.fromkeys(grp["group_code"].astype(str)))
        axes[str(code)] = tuple(dict.fromkeys(grp["axis"].astype(str)))
    return groups, axes


def _row_is_missing(value) -> bool:
    return value is None or (not isinstance(value, (tuple, list, dict)) and pd.isna(value))


def _normalize_filter_values(values, *, resolver=None) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    out: set[str] = set()
    for value in raw:
        if _row_is_missing(value):
            continue
        item = str(value)
        out.add(resolver(item) if resolver is not None else item)
    return out


def _filter_string_values(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    wanted = _normalize_filter_values(values)
    if not wanted:
        return df
    lowered = {v.lower() for v in wanted}
    return df[df[column].astype(str).str.lower().isin(lowered)]


def _filter_tuple_values(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    wanted = _normalize_filter_values(values)
    if not wanted:
        return df
    wanted_lower = {v.lower() for v in wanted}
    return df[
        df[column].map(lambda items: bool({str(x).lower() for x in (items or ())} & wanted_lower))
    ]


def _codes_under(roots, *, include_self: bool) -> set[str]:
    root_codes = _normalize_filter_values(
        roots, resolver=lambda x: resolve_cancer_type(x, strict=False) or x
    )
    if not root_codes:
        return set()
    out: set[str] = set()
    for code in root_codes:
        out.update(cancer_type_descendants(code, include_self=include_self))
    return out


_CANCER_TYPE_RECORD_COLUMNS = [
    "code",
    "name",
    "lineage_group",
    "family",
    "family_name",
    "primary_tissue",
    "normal_tissue_code",
    "normal_tissue_name",
    "hpa_tissues",
    "normal_tissue_match_confidence",
    "normal_tissue_match_basis",
    "parent_code",
    "children",
    "ancestors",
    "path",
    "path_names",
    "is_leaf",
    "subtype_groups",
    "subtype_axes",
    "evidence_source_code",
    "evidence_source_kind",
    "burden_category",
    "tmb",
    "has_expression_matrix",
    "source_matrix_cohort",
    "source_matrix_n_samples",
    "expression_source",
    "source_cohort",
    "source_pmid",
    "notes",
]


def _cancer_type_record_frame() -> pd.DataFrame:
    from .incidence import burden_category
    from .tmb import cancer_tmb

    df = cancer_type_registry()
    normal = cancer_normal_tissue_map().rename(
        columns={
            "match_confidence": "normal_tissue_match_confidence",
            "match_basis": "normal_tissue_match_basis",
        }
    )
    df = df.merge(normal, how="left", on="primary_tissue", validate="many_to_one")
    df = df.merge(_source_matrix_frame(), how="left", left_on="code", right_on="cancer_code")
    if "cancer_code" in df.columns:
        df = df.drop(columns=["cancer_code"])

    children = _children_map()
    groups, axes = _subtype_group_maps()
    names = CANCER_TYPE_NAMES

    df["lineage_group"] = [cancer_lineage_group(code) for code in df["code"]]
    df["family_name"] = df["family"].map(family_display_name)
    df["children"] = [tuple(children.get(str(code), ())) for code in df["code"]]
    df["ancestors"] = [tuple(cancer_type_ancestors(code)) for code in df["code"]]
    df["path"] = [tuple(cancer_type_lineage(code)) for code in df["code"]]
    df["path_names"] = [tuple(names.get(code, code) for code in path) for path in df["path"]]
    df["is_leaf"] = df["children"].map(lambda x: len(x) == 0)
    df["subtype_groups"] = [groups.get(str(code), ()) for code in df["code"]]
    df["subtype_axes"] = [axes.get(str(code), ()) for code in df["code"]]
    df["evidence_source_code"] = [cancer_evidence_source_code(code) for code in df["code"]]
    df["evidence_source_kind"] = [
        "direct" if code == source else "source_scope"
        for code, source in zip(df["code"], df["evidence_source_code"])
    ]
    df["burden_category"] = [burden_category(code) for code in df["code"]]
    df["tmb"] = [cancer_tmb(code) for code in df["code"]]
    df["has_expression_matrix"] = df["source_matrix_cohort"].notna()
    for col in ("source_matrix_n_samples",):
        df[col] = df[col].where(df[col].notna(), None)
    for col in _CANCER_TYPE_RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_CANCER_TYPE_RECORD_COLUMNS].copy()


def cancer_type_records(
    cancer_types=None,
    *,
    under=None,
    include_self: bool = True,
    include_descendants: bool = False,
    lineage_group=None,
    family=None,
    primary_tissue=None,
    normal_tissue=None,
    subtype_group=None,
    subtype_axis=None,
    evidence_source=None,
    expression_only: bool = False,
    leaves_only: bool = False,
    query: str | None = None,
) -> pd.DataFrame:
    """Flexible cancer-type query returning one stable record DataFrame.

    This is the structured companion to the older single-purpose helpers. Every
    result has the same columns, including hierarchy fields (``path``,
    ``ancestors``, ``children``), semantic rollups (``lineage_group``,
    ``family``), cross-cutting molecular groupings (``subtype_groups`` /
    ``subtype_axes``), source-scoped evidence resolution
    (``evidence_source_code``), expression-matrix availability, and matched
    normal-tissue metadata.

    Examples:

    - ``cancer_type_records(under="CRC")`` gives CRC, COAD, READ, and their MSI/MSS
      leaves.
    - ``cancer_type_records(subtype_group="MSI", lineage_group="Epithelial")``
      gives curated epithelial MSI subtypes.
    - ``cancer_type_records(family="carcinoma-gi", leaves_only=True)`` gives GI
      carcinoma leaves.

    Return type is always a DataFrame, including for empty results.
    """
    df = _cancer_type_record_frame()

    exact_codes = _normalize_filter_values(
        cancer_types, resolver=lambda x: resolve_cancer_type(x, strict=False) or x
    )
    if exact_codes is not None:
        if include_descendants:
            allowed: set[str] = set()
            for code in exact_codes:
                allowed.update(cancer_type_descendants(code, include_self=True))
            exact_codes = allowed
        df = df[df["code"].isin(exact_codes)]

    if under is not None:
        df = df[df["code"].isin(_codes_under(under, include_self=include_self))]

    df = _filter_string_values(df, "lineage_group", lineage_group)
    df = _filter_string_values(df, "family", family)
    df = _filter_string_values(df, "primary_tissue", primary_tissue)

    if normal_tissue is not None:
        wanted = {x.lower() for x in (_normalize_filter_values(normal_tissue) or set())}
        df = df[
            df.apply(
                lambda r: (
                    str(r["normal_tissue_code"]).lower() in wanted
                    or bool({str(x).lower() for x in (r["hpa_tissues"] or ())} & wanted)
                ),
                axis=1,
            )
        ]

    df = _filter_tuple_values(df, "subtype_groups", subtype_group)
    df = _filter_tuple_values(df, "subtype_axes", subtype_axis)
    df = _filter_string_values(df, "evidence_source_code", evidence_source)

    if expression_only:
        df = df[df["has_expression_matrix"]]
    if leaves_only:
        df = df[df["is_leaf"]]
    if query:
        q = str(query).strip().lower()
        haystack_cols = [
            "code",
            "name",
            "lineage_group",
            "family",
            "family_name",
            "primary_tissue",
            "normal_tissue_code",
            "normal_tissue_name",
        ]
        mask = pd.Series(False, index=df.index)
        for col in haystack_cols:
            mask = mask | df[col].astype(str).str.lower().str.contains(q, regex=False)
        df = df[mask]
    return df[_CANCER_TYPE_RECORD_COLUMNS].reset_index(drop=True)


def cancer_type_codes(*args, **kwargs) -> list[str]:
    """Codes from :func:`cancer_type_records` in registry order."""
    return cancer_type_records(*args, **kwargs)["code"].tolist()


_CANCER_TYPE_PATH_COLUMNS = [
    "level",
    "kind",
    "code",
    "name",
    "lineage_group",
    "family",
    "family_name",
    "primary_tissue",
    "normal_tissue_code",
]


def cancer_type_path(cancer_type, *, include_semantic_groups: bool = True) -> pd.DataFrame:
    """Semantic path for one cancer type as a stable DataFrame.

    With ``include_semantic_groups=True`` (default), the path starts with the
    derived coarse lineage group and registry family before the strict
    ``parent_code`` chain. For ``COAD_MSI`` this makes the conceptual hierarchy
    explicit: Epithelial → Gastrointestinal carcinoma → CRC → COAD → COAD_MSI.
    """
    code = resolve_cancer_type(cancer_type)
    records = cancer_type_records(cancer_type_lineage(code)).set_index("code", drop=False)
    rows: list[dict] = []
    leaf = records.loc[code]
    if include_semantic_groups:
        if leaf.get("lineage_group"):
            rows.append(
                {
                    "kind": "lineage_group",
                    "code": leaf["lineage_group"],
                    "name": leaf["lineage_group"],
                    "lineage_group": leaf["lineage_group"],
                    "family": None,
                    "family_name": None,
                    "primary_tissue": None,
                    "normal_tissue_code": None,
                }
            )
        if leaf.get("family"):
            rows.append(
                {
                    "kind": "family",
                    "code": leaf["family"],
                    "name": leaf["family_name"],
                    "lineage_group": leaf["lineage_group"],
                    "family": leaf["family"],
                    "family_name": leaf["family_name"],
                    "primary_tissue": None,
                    "normal_tissue_code": None,
                }
            )
    for node in cancer_type_lineage(code):
        r = records.loc[node]
        rows.append(
            {
                "kind": "cancer_type",
                "code": r["code"],
                "name": r["name"],
                "lineage_group": r["lineage_group"],
                "family": r["family"],
                "family_name": r["family_name"],
                "primary_tissue": r["primary_tissue"],
                "normal_tissue_code": r["normal_tissue_code"],
            }
        )
    for idx, row in enumerate(rows):
        row["level"] = idx
    return pd.DataFrame(rows, columns=_CANCER_TYPE_PATH_COLUMNS)


def cancer_type_siblings(cancer_type, *, include_self: bool = False) -> pd.DataFrame:
    """Cancer types that share the same immediate ``parent_code``.

    For example ``cancer_type_siblings("COAD")`` returns ``READ`` because both
    are direct children of ``CRC``. The return type is the same record DataFrame
    as :func:`cancer_type_records`.
    """
    code = resolve_cancer_type(cancer_type)
    reg = _registry_frame().set_index("code")
    parent = None
    if code in reg.index:
        value = reg.loc[code].get("parent_code")
        parent = None if _row_is_missing(value) else str(value).strip() or None
    if parent is None:
        codes = [code] if include_self else []
    else:
        codes = cancer_type_subtypes_of(parent)
        if not include_self:
            codes = [c for c in codes if c != code]
    return cancer_type_records(codes)


def matched_normal_tissues(cancer_types=None, **query_kwargs) -> pd.DataFrame:
    """Matched normal-tissue rows for queried cancer types.

    Accepts the same filters as :func:`cancer_type_records` and returns one row
    per matched cancer code with normal-tissue metadata. It does not read HPA
    expression data; use :func:`matched_normal_tissue_expression` for that.
    """
    records = cancer_type_records(cancer_types, **query_kwargs)
    cols = [
        "code",
        "name",
        "primary_tissue",
        "normal_tissue_code",
        "normal_tissue_name",
        "hpa_tissues",
        "normal_tissue_match_confidence",
        "normal_tissue_match_basis",
    ]
    return records[cols].reset_index(drop=True)


def matched_normal_tissue(cancer_type) -> dict | None:
    """Single-code normal-tissue match as a dict, or ``None`` when unresolved."""
    if _row_is_missing(cancer_type):
        return None
    code = resolve_cancer_type(cancer_type, strict=False)
    if code is None:
        return None
    df = matched_normal_tissues([code])
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _filter_hpa_genes(df: pd.DataFrame, genes) -> pd.DataFrame:
    wanted = _normalize_filter_values(genes)
    if not wanted:
        return df
    ids = {g.split(".")[0] for g in wanted}
    symbols = {g.upper() for g in wanted}
    gene_ids = df["Gene"].astype(str).str.split(".").str[0]
    gene_names = df.get("Gene name", pd.Series("", index=df.index)).astype(str).str.upper()
    return df[gene_ids.isin(ids) | gene_names.isin(symbols)]


def matched_normal_tissue_expression(cancer_type, genes=None) -> pd.DataFrame:
    """HPA RNA consensus expression for a cancer type's matched normal tissue.

    Returns a long DataFrame with HPA ``Gene`` / ``Gene name`` / ``Tissue`` /
    ``nTPM`` plus ``cancer_code`` and normal-tissue mapping columns. This function
    may download/cache HPA data on first use; ordinary ontology queries do not.
    """
    match = matched_normal_tissue(cancer_type)
    if match is None:
        return pd.DataFrame(
            columns=[
                "cancer_code",
                "normal_tissue_code",
                "normal_tissue_name",
                "match_confidence",
                "Gene",
                "Gene name",
                "Tissue",
                "nTPM",
            ]
        )
    from .hpa import hpa_rna_consensus

    tissues = {str(t).lower() for t in (match.get("hpa_tissues") or ())}
    hpa = hpa_rna_consensus().copy()
    sub = hpa[hpa["Tissue"].astype(str).str.lower().isin(tissues)]
    sub = _filter_hpa_genes(sub, genes).copy()
    sub.insert(0, "cancer_code", match["code"])
    sub.insert(1, "normal_tissue_code", match["normal_tissue_code"])
    sub.insert(2, "normal_tissue_name", match["normal_tissue_name"])
    sub.insert(3, "match_confidence", match["normal_tissue_match_confidence"])
    return sub.reset_index(drop=True)


_CANCER_TYPE_REFERENCE_COLUMNS = [
    "code",
    "name",
    "evidence_source_code",
    "burden_category",
    "us_incidence_pct",
    "us_mortality_pct",
    "world_incidence_pct",
    "world_mortality_pct",
    "tmb",
    "apd1_orr_pct",
    "ici_orr_pct",
    "ici_regimen",
    "ici_response_source_code",
    "ici_inheritance_kind",
    "has_expression_matrix",
    "source_matrix_cohort",
    "source_matrix_n_samples",
    "normal_tissue_code",
    "hpa_tissues",
]

_REFERENCE_BURDEN_METRICS = (
    "us_incidence_pct",
    "us_mortality_pct",
    "world_incidence_pct",
    "world_mortality_pct",
)


def _codes_from_query_input(cancer_types) -> list[str]:
    if isinstance(cancer_types, pd.DataFrame):
        if "code" not in cancer_types.columns:
            raise ValueError("DataFrame cancer_types input must contain a 'code' column")
        return list(dict.fromkeys(cancer_types["code"].dropna().astype(str)))
    if cancer_types is None:
        return cancer_type_codes()
    if isinstance(cancer_types, str):
        return [resolve_cancer_type(cancer_types)]
    return [resolve_cancer_type(x) for x in cancer_types]


def cancer_type_reference_data(
    cancer_types=None,
    *,
    regimen=None,
    fallback: bool = True,
    inherit: bool = True,
) -> pd.DataFrame:
    """Join common scalar oncoref references for cancer-type codes.

    ``cancer_types`` may be ``None`` (all registry codes), one code/name, an
    iterable of codes/names, or the DataFrame returned by
    :func:`cancer_type_records`. The result is a stable DataFrame keyed by
    canonical ``code`` and includes incidence/mortality burden, TMB, anti-PD-1
    ORR, best-available ICI ORR + source metadata, expression-matrix availability,
    and matched normal-tissue metadata. Per-gene expression remains in the
    existing expression/HPA APIs; use the returned ``code`` values with
    ``cancer_reference_expression`` / ``per_sample_expression`` and
    ``matched_normal_tissue_expression``.
    """
    from .apd1 import cancer_apd1_response
    from .ici import cancer_ici_response, resolve_ici_response_source
    from .incidence import cancer_burden
    from .tmb import cancer_tmb

    codes = _codes_from_query_input(cancer_types)
    records = cancer_type_records(codes).set_index("code", drop=False)
    burden_maps = {metric: cancer_burden(metric=metric) for metric in _REFERENCE_BURDEN_METRICS}
    rows = []
    for code in codes:
        if code not in records.index:
            continue
        record = records.loc[code]
        burden = record["burden_category"]
        ici_source = resolve_ici_response_source(
            code, regimen=regimen, fallback=fallback, inherit=inherit
        )
        rows.append(
            {
                "code": code,
                "name": record["name"],
                "evidence_source_code": record["evidence_source_code"],
                "burden_category": burden,
                "us_incidence_pct": burden_maps["us_incidence_pct"].get(burden),
                "us_mortality_pct": burden_maps["us_mortality_pct"].get(burden),
                "world_incidence_pct": burden_maps["world_incidence_pct"].get(burden),
                "world_mortality_pct": burden_maps["world_mortality_pct"].get(burden),
                "tmb": cancer_tmb(code, inherit=inherit),
                "apd1_orr_pct": cancer_apd1_response(code, inherit=inherit),
                "ici_orr_pct": cancer_ici_response(
                    code, regimen=regimen, fallback=fallback, inherit=inherit
                ),
                "ici_regimen": ici_source.get("selected_regimen"),
                "ici_response_source_code": ici_source.get("resolved_cancer_code"),
                "ici_inheritance_kind": ici_source.get("inheritance_kind"),
                "has_expression_matrix": bool(record["has_expression_matrix"]),
                "source_matrix_cohort": record["source_matrix_cohort"],
                "source_matrix_n_samples": record["source_matrix_n_samples"],
                "normal_tissue_code": record["normal_tissue_code"],
                "hpa_tissues": record["hpa_tissues"],
            }
        )
    return pd.DataFrame(rows, columns=_CANCER_TYPE_REFERENCE_COLUMNS)


def cancer_types_in_family(family):
    """Return cancer-type codes belonging to a registry family.

    Families are lineage/organ-system only (age is a separate ``pediatric``
    flag): ``sarcoma`` spans soft-tissue + bone + rhabdomyosarcomas regardless
    of patient age; ``heme-myeloid`` covers LAML + MDS + MPN + CML etc.
    """
    df = cancer_type_registry()
    return df[df["family"] == family]["code"].tolist()


def cancer_types_by_tissue(primary_tissue):
    """Return cancer-type codes whose primary tissue matches.

    Useful for site-aware hypothesis generation — passing ``bone`` returns
    osteosarcoma, Ewing, chondrosarcoma, chordoma, etc.
    """
    df = cancer_type_registry()
    return df[df["primary_tissue"] == primary_tissue]["code"].tolist()


def cancer_type_subtypes_of(parent_code):
    """Return registry subtypes of a given parent cancer code.

    For example ``cancer_type_subtypes_of("LAML")`` returns the LAML subtypes.
    """
    df = cancer_type_registry()
    return df[df["parent_code"] == parent_code]["code"].tolist()


def _parent_of_map():
    """``{code: parent_code}`` from the registry (empty parents dropped)."""
    df = _registry_frame()
    out = {}
    for code, parent in zip(df["code"], df["parent_code"]):
        if isinstance(parent, str) and parent:
            out[str(code)] = parent
    return out


def _walk_ancestors(code, parent_of):
    """Codes on the ``parent_code`` chain above ``code`` (excluding itself)."""
    seen, cur = [], code
    while cur in parent_of:
        cur = parent_of[cur]
        if cur in seen:  # defensive: a cycle would otherwise loop forever
            break
        seen.append(cur)
    return seen


def cancer_subtype_groupings():
    """Orthogonal cross-cutting subtype groupings (``cancer-subtype-groupings.csv``).

    Axes such as microsatellite (MSI/MSS), hypermutation (POLE), viral_hpv, and
    copy_number_mycn cut *across* the organ ``parent_code`` tree — a leaf belongs to
    its organ parent **and** to a mechanism group. Returns the long-form
    ``group_code, axis, member_code, basis`` table (defensive copy).
    """
    return get_data("cancer-subtype-groupings").copy()


def cancer_subtype_group(group_code, *, under=None):
    """Member registry codes of a cross-cutting group (e.g. ``"MSI"``, ``"MSS"``,
    ``"POLE"``, ``"HPV_POS"``, ``"MYCN_AMP"``).

    ``cancer_subtype_group("MSI")`` returns every MSI subtype across cancers;
    ``cancer_subtype_group("POLE")`` the POLE-ultramutated subtypes. With ``under``,
    restrict to members that are descendants of that hierarchy node —
    ``cancer_subtype_group("MSI", under="CRC")`` -> the colorectal MSI cross-cut.
    """
    df = cancer_subtype_groupings()
    members = df[df["group_code"] == group_code]["member_code"].tolist()
    if under is not None:
        parent_of = _parent_of_map()
        members = [m for m in members if under in _walk_ancestors(m, parent_of)]
    return members


def _children_map():
    """``{parent_code: [child codes in registry order]}``."""
    df = _registry_frame()
    out: dict[str, list[str]] = {}
    for code, parent in zip(df["code"], df["parent_code"]):
        if isinstance(parent, str) and parent:
            out.setdefault(parent, []).append(str(code))
    return out


def cancer_type_ancestors(cancer_type):
    """Codes on the parent chain above ``cancer_type``, nearest parent first.

    ``cancer_type_ancestors("COAD_MSI")`` -> ``["COAD", "CRC"]``. Empty for a
    top-level code. Accepts any alias/display name.
    """
    code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
    return _walk_ancestors(code, _parent_of_map())


def cancer_type_descendants(cancer_type, *, include_self=False):
    """Every registry code transitively beneath ``cancer_type`` (depth-first,
    registry order).

    ``cancer_type_descendants("CRC")`` -> ``["COAD", "COAD_MSI", "COAD_MSS",
    "READ", "READ_MSI", "READ_MSS"]`` — the whole subtree, so you can roll a parent
    down to its leaves. ``include_self`` prepends the node itself.
    """
    code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
    children = _children_map()
    out: list[str] = []
    seen: set[str] = set()

    def _visit(node):
        for child in children.get(node, []):
            if child in seen:  # defensive against a malformed cycle
                continue
            seen.add(child)
            out.append(child)
            _visit(child)

    _visit(code)
    return [code, *out] if include_self else out


def cancer_type_lineage(cancer_type):
    """Path from the root down to ``cancer_type`` (root first), e.g.
    ``cancer_type_lineage("COAD_MSI")`` -> ``["CRC", "COAD", "COAD_MSI"]``."""
    code = resolve_cancer_type(cancer_type, strict=False) or cancer_type
    return [*reversed(cancer_type_ancestors(code)), code]


def cancer_type_tree(root=None):
    """The hierarchy as nested ``{code: {child: {...}}}`` dicts.

    ``root=None`` returns the whole forest (every code with no parent at the top
    level); pass a code to get just that subtree. Leaves map to ``{}``.
    """
    children = _children_map()

    def _subtree(node):
        return {child: _subtree(child) for child in children.get(node, [])}

    if root is not None:
        code = resolve_cancer_type(root, strict=False) or root
        return {code: _subtree(code)}
    df = cancer_type_registry()
    roots = [
        str(c) for c, p in zip(df["code"], df["parent_code"]) if not (isinstance(p, str) and p)
    ]
    return {r: _subtree(r) for r in roots}


def mixture_cohort_codes():
    """Return parent codes flagged as mixture cohorts in the registry.

    A mixture cohort is a parent code whose reference median is a biological
    union of lineage-distinct subtypes.
    """
    df = cancer_type_registry()
    if "mixture_cohort" not in df.columns:
        return []
    flag = df["mixture_cohort"].fillna(False).astype(bool)
    return df.loc[flag, "code"].tolist()


def is_mixture_cohort(code):
    """True when ``code`` is a mixture cohort per the registry."""
    return code in set(mixture_cohort_codes())


def sarcoma_lineage_codes(*, with_expression_only=False):
    """Return every registry code that is a sarcoma by lineage.

    Every sarcoma is in the ``SARC_`` namespace under one ``family == "sarcoma"``
    bucket: soft-tissue, bone, rhabdomyosarcomas, and endometrial stromal
    sarcomas. This is the membership the ``SARC`` grand-union aggregate pools
    over.

    ``with_expression_only`` drops codes with no per-sample expression source
    (the literature-curated entries), leaving only codes that contribute
    samples to a pooled aggregate.
    """
    df = cancer_type_registry()
    sub = df[df["family"].astype(str) == "sarcoma"]
    if with_expression_only:
        src = sub["expression_source"].astype(str).str.lower()
        sub = sub[~src.isin(["curated", "nan", ""])]
    return sub["code"].tolist()


# Computed cohort aggregates: "view" cohorts that pool the per-sample values of
# several atom cohorts by histology or source, rather than being a single frozen
# matrix. Backed by ``cancer-cohort-aggregates.csv``; the pan-sarcoma ``SARC``
# grand union is computed from the registry family (so it tracks new atoms
# automatically) rather than enumerated.
def cohort_aggregates_df():
    """Return the curated ``cancer-cohort-aggregates.csv`` long table
    (``aggregate_code, member_code, basis``) — the explicit histology rollup
    cohorts (e.g. ``SARC_RMS`` ← the four rhabdomyosarcoma subtypes)."""
    return get_data("cancer-cohort-aggregates")


def cohort_registry_df():
    """The first-class cohort vocabulary: one row per ``cohort_id`` with
    ``prefix, kind, source_project, assay, n_samples, n_codes, is_computed,
    member_cohorts, provenance``. The authority to validate any
    ``source_cohort`` against — includes the computed aggregates and
    literature-curated cohorts."""
    return get_data("cohort-registry")


def cohort_registry():
    """``{cohort_id: {column: value}}`` view of :func:`cohort_registry_df`."""
    df = cohort_registry_df()
    return {
        str(r["cohort_id"]): {k: r[k] for k in df.columns if k != "cohort_id"}
        for _, r in df.iterrows()
    }


def cohort_kind(cohort_id):
    """The ``kind`` (pipeline family) of a cohort_id (``treehouse``, ``geo``,
    ``computed``, …), or ``None`` if unknown."""
    df = cohort_registry_df()
    hit = df.loc[df["cohort_id"].astype(str) == str(cohort_id), "kind"]
    return str(hit.iloc[0]) if len(hit) else None


def cohort_source_version(cancer_type):
    """The Ensembl release a cohort's gene ids were harmonized to (e.g. ``"112"``), or
    ``None`` if its registry provenance doesn't record one.

    The per-cohort ``source_version`` for auditing the canonical gene-ID space
    (oncoref#135 item 6): accepts a cancer code (resolved to its source cohort via
    ``source-matrices.csv``) or a ``cohort_id`` directly, and parses the harmonized
    Ensembl release from the registry ``provenance`` text."""
    code = resolve_cancer_type(cancer_type, strict=False) or str(cancer_type)
    sm = get_data("source-matrices", copy=False)
    hit = sm.loc[sm["cancer_code"].astype(str) == str(code), "source_cohort"]
    cohort_id = str(hit.iloc[0]) if len(hit) else str(cancer_type)
    reg = cohort_registry_df()
    prov = reg.loc[reg["cohort_id"].astype(str) == cohort_id, "provenance"]
    m = _ENSEMBL_RELEASE_RE.search(str(prov.iloc[0]) if len(prov) else "")
    return m.group(1) if m else None


def known_cohort_ids():
    """Frozenset of every valid ``cohort_id`` (the validation authority)."""
    return frozenset(cohort_registry_df()["cohort_id"].astype(str))


def cohort_aggregates():
    """``{aggregate_code: [member_code, ...]}`` for every computed-aggregate
    cohort: the curated histology rollups (``SARC_RMS``, ``SARC_LPS``) plus the
    pan-sarcoma ``SARC`` grand union (every ``family == 'sarcoma'`` atom that is
    not itself an aggregate)."""
    df = cohort_aggregates_df()
    out = {}
    for agg, grp in df.groupby("aggregate_code"):
        out[str(agg)] = list(dict.fromkeys(grp["member_code"].astype(str)))
    # pan-sarcoma grand union under the bare SARC code, computed from family;
    # exclude the aggregates AND SARC itself (no self-membership / circularity).
    aggs = set(out) | {"SARC"}
    out["SARC"] = [c for c in sarcoma_lineage_codes() if c not in aggs]
    return out


def cohort_aggregate_members(aggregate_code):
    """Member atom codes pooled by a computed-aggregate cohort, or ``None`` if
    ``aggregate_code`` is not an aggregate."""
    return cohort_aggregates().get(str(aggregate_code))
