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

import threading
from functools import lru_cache

from .load_dataset import get_data

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
    """Reset every registry-backed cache in this module.

    Test hook for swapping the registry CSV via a monkey-patched ``get_data``;
    not part of the public surface.
    """
    CANCER_TYPE_NAMES.clear_cache()
    _registry_frame.cache_clear()


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
    import pandas as pd

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
    "cns": "CNS tumor",
    "embryonal": "Embryonal tumor",
    "endocrine": "Endocrine tumor",
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
