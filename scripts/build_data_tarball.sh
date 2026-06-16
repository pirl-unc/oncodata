#!/usr/bin/env bash
# Build the oncodata expression-data tarball for upload to a GitHub Release.
#
# The wheel ships only the small curated tables; the heavy per-cohort expression
# artifacts (data_bundle.DOWNLOADABLE_PATHS) are distributed as a version-pinned
# tarball attached to the pirl-unc/oncodata release. This script packages those
# paths from a source directory that already contains them — e.g. a populated
# cache dir (`oncodata fetch` then `oncodata status` for the path) or a
# pirlygenes data checkout during the migration.
#
# Usage:
#   scripts/build_data_tarball.sh <source-dir> [output-dir]
#
# Then: upload <output-dir>/oncodata-data-v<DATA_VERSION>.tar.gz to the
# `v<DATA_VERSION>` release on pirl-unc/oncodata, and only THEN bump DATA_VERSION
# (never before the upload — a 404 on the primary URL falls back to pirlygenes,
# but a version with neither published hangs the fetch).
set -euo pipefail

SRC="${1:?usage: build_data_tarball.sh <source-dir> [output-dir]}"
OUT_DIR="${2:-.}"

# Resolve user-supplied paths to absolute BEFORE cd'ing into the repo, so a
# relative source/output dir stays relative to the caller's CWD.
SRC="$(cd "$SRC" 2>/dev/null && pwd)" || {
    echo "error: source dir '$1' not found" >&2
    exit 1
}
mkdir -p "${OUT_DIR%/}"
OUT_DIR="$(cd "${OUT_DIR%/}" && pwd)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_VERSION="$(python -c 'from oncodata.version import DATA_VERSION; print(DATA_VERSION)')"
read -r -a PATHS <<<"$(python -c 'from oncodata.data_bundle import DOWNLOADABLE_PATHS; print(" ".join(DOWNLOADABLE_PATHS))')"
OUT="${OUT_DIR%/}/oncodata-data-v${DATA_VERSION}.tar.gz"

missing=()
for p in "${PATHS[@]}"; do
    [[ -e "$SRC/$p" ]] || missing+=("$p")
done
if ((${#missing[@]})); then
    echo "error: source dir '$SRC' is missing required bundle paths:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

echo "packaging ${#PATHS[@]} bundle paths from $SRC -> $OUT"
tar -czf "$OUT" -C "$SRC" "${PATHS[@]}"
SIZE="$(du -h "$OUT" | cut -f1)"
echo "wrote $OUT ($SIZE)"
echo
echo "next: gh release upload v${DATA_VERSION} '$OUT' --repo pirl-unc/oncodata"
echo "      (create the v${DATA_VERSION} release first if it doesn't exist)"
