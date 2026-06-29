#!/usr/bin/env bash
# Build the oncoref expression-data tarball for upload to a GitHub Release.
#
# The wheel ships only the small curated tables; the heavy per-cohort expression
# artifacts (data_bundle.DOWNLOADABLE_PATHS) are distributed as a version-pinned
# tarball attached to the pirl-unc/oncoref release. This script packages those
# paths from a source directory that already contains them — e.g. a populated
# cache dir (`oncoref data fetch bundle` then `oncoref data dir bundle` for the
# path) or a pirlygenes data checkout during the migration.
#
# Usage:
#   scripts/build_data_tarball.sh <source-dir> [output-dir]
#
# Then: upload <output-dir>/oncoref-data-v<DATA_VERSION>.tar.gz plus the emitted
# .sha256 and .manifest.json files to the `v<DATA_VERSION>` release on
# pirl-unc/oncoref, and only THEN bump DATA_VERSION.
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

DATA_VERSION="$(python -c 'from oncoref.version import DATA_VERSION; print(DATA_VERSION)')"
read -r -a PATHS <<<"$(python -c 'from oncoref.data_bundle import DOWNLOADABLE_PATHS; print(" ".join(DOWNLOADABLE_PATHS))')"
OUT="${OUT_DIR%/}/oncoref-data-v${DATA_VERSION}.tar.gz"
SHA_OUT="${OUT}.sha256"
MANIFEST_OUT="${OUT_DIR%/}/oncoref-data-v${DATA_VERSION}.manifest.json"

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
python - "$OUT" "$SRC" "$MANIFEST_OUT" "$DATA_VERSION" "${PATHS[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

tarball = Path(sys.argv[1])
source = Path(sys.argv[2])
manifest = Path(sys.argv[3])
data_version = sys.argv[4]
paths = sys.argv[5:]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(rel: str) -> dict:
    path = source / rel
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    return {
        "path": rel,
        "file_count": len(files),
        "size_bytes": sum(p.stat().st_size for p in files),
    }


payload = {
    "manifest_version": 1,
    "data_version": data_version,
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "tarball": {
        "filename": tarball.name,
        "bytes": tarball.stat().st_size,
        "sha256": sha256_file(tarball),
        "downloadable_paths": paths,
    },
    "inventory": {rel: inventory(rel) for rel in paths},
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
python - "$OUT" "$SHA_OUT" <<'PY'
import hashlib
import sys
from pathlib import Path

tarball = Path(sys.argv[1])
out = Path(sys.argv[2])
h = hashlib.sha256()
with tarball.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        h.update(chunk)
out.write_text(f"{h.hexdigest()}  {tarball.name}\n")
PY
echo "wrote $SHA_OUT"
echo "wrote $MANIFEST_OUT"
echo
echo "next: gh release upload v${DATA_VERSION} '$OUT' '$SHA_OUT' '$MANIFEST_OUT' --repo pirl-unc/oncoref"
echo "      (create the v${DATA_VERSION} release first if it doesn't exist)"
