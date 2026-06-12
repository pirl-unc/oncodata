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

"""Cancer incidence / mortality burden reference data and the registry-driven
resolution of a cancer type to its anatomic burden category."""

from __future__ import annotations

from .cancer_types import cancer_type_registry, resolve_cancer_type
from .load_dataset import get_data

_BURDEN_METRICS = (
    "us_incidence_pct",
    "us_mortality_pct",
    "world_incidence_pct",
    "world_mortality_pct",
)


def cancer_burden_df():
    """Return the curated ``cancer-incidence-mortality.csv`` reference: each
    cancer **burden category**'s share (%) of annual cancer incidence and
    mortality, for the US and worldwide, cited per row (ACS Cancer Facts &
    Figures 2024 / GLOBOCAN 2022). Rows flagged in ``notes`` as subsets/rollups
    overlap others — don't sum them blindly."""
    return get_data("cancer-incidence-mortality")


def cancer_code_burden_map():
    """Return ``{cancer_code: burden_category}`` from
    ``cancer-code-burden-map.csv``. This is now only the small set of
    **overrides** the registry ontology can't express on its own (e.g.
    ``SARC_KS`` -> Kaposi rather than soft-tissue; ``LAML`` -> AML rather than
    other-leukemia; ``HL`` -> Hodgkin; ``CTCL`` -> non-Hodgkin). Everything else
    is resolved by :func:`burden_category` from the registry's family +
    primary_tissue."""
    df = get_data("cancer-code-burden-map")
    return dict(zip(df["cancer_code"].astype(str), df["burden_category"].astype(str)))


def cancer_burden(category=None, *, metric="us_incidence_pct"):
    """Burden share (%) for one category and metric, or the whole
    ``{category: pct}`` map. ``metric`` is one of ``us_incidence_pct``,
    ``us_mortality_pct``, ``world_incidence_pct``, ``world_mortality_pct``."""
    if metric not in _BURDEN_METRICS:
        raise ValueError(f"metric must be one of {_BURDEN_METRICS}")
    df = cancer_burden_df()
    mapping = dict(zip(df["burden_category"].astype(str), df[metric].astype(float)))
    if category is None:
        return mapping
    return mapping.get(category)


# Burden categories are anatomic-site shares (how ACS/GLOBOCAN tabulate), so a
# cohort is resolved to its category straight from the **cancer-type registry
# ontology** — one source of truth — rather than a parallel hand-map: the
# sarcoma family splits bone vs soft tissue on primary_tissue, plasma-cell and a
# handful of leukemia/lymphoma exceptions resolve by family, and primary tissue
# decides everything else. ``cancer-code-burden-map.csv`` now holds only the few
# true exceptions the ontology can't express.

# Sarcoma family -> bone_and_joint when its primary_tissue is skeletal, else
# soft_tissue_sarcoma.
_BONE_SARCOMA_TISSUES = {"bone", "cartilage", "notochord"}

# Registry primary_tissue -> burden category. Covers every non-heme tissue in
# the registry; heme tissues are routed by :data:`_HEME_TISSUE_BURDEN` below.
_PRIMARY_TISSUE_BURDEN = {
    "lung": "lung",
    "breast": "breast",
    "prostate": "prostate",
    "colon": "colorectal",
    "rectum": "colorectal",
    "colorectum": "colorectal",  # the CRC parent node (subtypes split colon/rectum)
    "pancreas": "pancreas",
    "liver": "liver",
    "bile_duct": "gallbladder_biliary",
    "stomach": "stomach",
    "esophagus": "esophagus",
    "small_intestine": "small_intestine",
    "bladder": "bladder",
    "kidney": "kidney",
    "kidney_cns_soft": "kidney",
    "ovary": "ovary",
    "endometrium": "uterus_endometrium",
    "cervix": "cervix",
    "vulva": "vulva",
    "vagina": "vagina",
    "penis": "penis",
    "urethra": "bladder",
    "anal_canal": "anus",
    "fallopian_tube": "ovary",
    "peritoneum_serous": "ovary",  # HGSC pooled with OV
    "gallbladder": "gallbladder_biliary",
    "thyroid": "thyroid",
    "thyroid_c_cell": "thyroid",
    "testis": "testicular_germ_cell",
    "pleura": "mesothelioma",
    "peritoneum": "mesothelioma",
    "oral_cavity": "head_and_neck",
    "oropharynx": "head_and_neck",
    "pharynx": "head_and_neck",
    "nasopharynx": "head_and_neck",
    "larynx": "head_and_neck",
    "salivary_gland": "head_and_neck",
    "midline_structures": "head_and_neck",
    "thymus": "head_and_neck",
    "thorax": "head_and_neck",
    "cerebrum": "brain_cns",
    "cerebellum": "brain_cns",
    "eye": "eye_ocular",
    "retina": "eye_ocular",
    "skin": "melanoma",
    "epidermis": "non_melanoma_skin",  # BCC / cSCC keratinocyte carcinomas
    "ependyma": "brain_cns",
    "meninges": "brain_cns",
    "choroid_plexus": "brain_cns",
    "sellar_suprasellar": "brain_cns",
    "pons_midline": "brain_cns",
    "pituitary": "brain_cns",
    "adrenal_cortex": "adrenal",
    "adrenal_medulla": "adrenal",
    "sympathetic_ganglia": "adrenal",
    "bone": "bone_and_joint",
    "cartilage": "bone_and_joint",
    "notochord": "bone_and_joint",
    "soft_tissue": "soft_tissue_sarcoma",
    "smooth_muscle": "soft_tissue_sarcoma",
    "skeletal_muscle": "soft_tissue_sarcoma",
    "adipose": "soft_tissue_sarcoma",
    "nerve_sheath": "soft_tissue_sarcoma",
    "vascular_endothelium": "soft_tissue_sarcoma",
    "gi_wall": "soft_tissue_sarcoma",
    # Site-agnostic NET parent node; organ-specific NET subtypes carry their own
    # primary_tissue (pancreas, lung, ...) and route there instead.
    "neuroendocrine": "other_and_unknown_primary",
}
# Heme (non-plasma): lymph node -> lymphoma; marrow/blood/spleen -> leukemia.
# AML and Hodgkin are exceptions carried in cancer-code-burden-map.csv.
_HEME_TISSUE_BURDEN = {
    "lymph_node": "non_hodgkin_lymphoma",
    "bone_marrow": "leukemia_all_other",
    "peripheral_blood": "leukemia_all_other",
    "spleen_marrow": "leukemia_all_other",
}
# Last-resort family fallback when primary_tissue is blank/unmapped.
_FAMILY_BURDEN = {
    "sarcoma": "soft_tissue_sarcoma",
    "melanoma": "melanoma",
    "cns": "brain_cns",
    "carcinoma-skin": "non_melanoma_skin",
}


def burden_category(cancer_type):
    """Robustly resolve a cancer type (code, alias, or display name) to an
    anatomic burden category, driven by the cancer-type registry ontology.
    Order: normalize via :func:`resolve_cancer_type`; the small explicit
    ``cancer-code-burden-map`` *override* (walking the ``parent_code`` chain);
    then registry-driven — sarcoma family splits bone vs soft tissue, plasma
    cell -> myeloma, other heme by tissue, then ``primary_tissue``, then
    ``family``. Returns ``None`` only when nothing matches — callers should
    **warn**, not silently skip (an unmapped cohort is a coverage gap)."""
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return None
    if code is None:
        return None
    override = cancer_code_burden_map()
    reg = cancer_type_registry().set_index("code")
    # 1. explicit override (true exceptions only), walking up the parent chain
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in override:
            return override[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    # 2. registry-driven, walking up the parent chain for blank tissue/family
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur not in reg.index:
            break
        row = reg.loc[cur]
        family = str(row.get("family", "") or "")
        tissue = str(row.get("primary_tissue", "") or "")
        if family == "sarcoma":
            return "bone_and_joint" if tissue in _BONE_SARCOMA_TISSUES else "soft_tissue_sarcoma"
        if family == "heme-plasma":
            return "multiple_myeloma"
        if family.startswith("heme") and tissue in _HEME_TISSUE_BURDEN:
            return _HEME_TISSUE_BURDEN[tissue]
        if tissue in _PRIMARY_TISSUE_BURDEN:
            return _PRIMARY_TISSUE_BURDEN[tissue]
        if family in _FAMILY_BURDEN:
            return _FAMILY_BURDEN[family]
        cur = str(row.get("parent_code", "") or "").strip() or None
    return None
