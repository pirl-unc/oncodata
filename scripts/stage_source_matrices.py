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

"""Stage the per-cohort source matrices for ``oncoref.source_matrices``.

For each cancer code in ``source-matrices.csv``, find its single selected source
cohort's raw per-sample matrix in a local matrix cache and copy it to:

  - the ``source_matrices`` on-disk cache (``<CODE>.parquet``) so the accessor +
    per-patient coverage analyses work locally without a release; and
  - (with ``--release-dir``) an upload staging dir as ``<CODE>_per_sample_tpm.parquet``,
    the exact asset names ``source_matrices.release_url`` expects on the
    ``source-v<DATA_VERSION>`` GitHub release.

The matrix cache is laid out ``<cache>/<cohort>/derived/<NAME>_per_sample_tpm.parquet``.
The cohort directory is matched to the registry's ``source_cohort`` by GSE accession
or normalized name (the same single-source choice the rebuild driver makes), so a
multi-source code stages exactly the one matrix the shipped artifacts were built on.

Run:
    python scripts/stage_source_matrices.py --cache ~/.cache/pirlygenes/expression \
        [--release-dir ~/oncoref-source-upload] [--codes LUAD,SKCM] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref import source_matrices as sm


def _source_key(name: str) -> str:
    m = re.search(r"GSE\d+", name.upper())
    return m.group() if m else re.sub(r"[^A-Z0-9]", "", name.upper())


def _matrices_by_source(cache: Path) -> dict[str, list[Path]]:
    """``{source_key -> [matrix paths]}`` for every per-sample matrix in the cache
    (sorted, so candidate order is deterministic across machines)."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(cache.glob("*/derived/*_per_sample_tpm.parquet")):
        out[_source_key(p.parent.parent.name)].append(p)
    return out


def stage(cache: Path, *, release_dir: Path | None, codes: list[str] | None, limit: int | None):
    reg = sm.registry()
    by_source = _matrices_by_source(cache)
    rows = reg.to_dict("records")
    if codes:
        wanted = {c.upper() for c in codes}
        rows = [r for r in rows if str(r["cancer_code"]).upper() in wanted]
    if limit:
        rows = rows[:limit]

    cache_out = sm.cache_dir()
    if release_dir is not None:
        release_dir.mkdir(parents=True, exist_ok=True)

    staged, missing = 0, []
    for r in rows:
        code = str(r["cancer_code"])
        src_key = _source_key(str(r["source_cohort"]))
        candidates = by_source.get(src_key, [])
        if not candidates:
            missing.append((code, r["source_cohort"]))
            continue
        # If the source has multiple code-matrices, pick the one whose stem maps to
        # this code (the rebuild driver's code key); else the sole candidate.
        chosen = candidates[0] if len(candidates) == 1 else _match_code(code, candidates)
        if chosen is None:
            missing.append((code, r["source_cohort"]))
            continue
        shutil.copyfile(chosen, cache_out / f"{code}.parquet")
        if release_dir is not None:
            shutil.copyfile(chosen, release_dir / f"{code}_per_sample_tpm.parquet")
        staged += 1
        print(f"  {code}: <- {chosen.parent.parent.name}/{chosen.name}", flush=True)

    print(f"\nstaged {staged}/{len(rows)} cohorts -> {cache_out}", flush=True)
    if release_dir is not None:
        print(f"release assets -> {release_dir}", flush=True)
    if missing:
        print(f"MISSING ({len(missing)}): {missing}", flush=True)


def _match_code(code: str, candidates: list[Path]) -> Path | None:
    """The candidate matrix whose filename stem maps to ``code`` (a source dir with
    several code-matrices, e.g. treehouse). Returns ``None`` if the match isn't
    unique — the caller reports that as missing rather than staging an arbitrary
    matrix (silently shipping the wrong cohort would corrupt every analysis)."""

    def code_key(name: str) -> str:
        n = name[5:] if name.lower().startswith("tcga_") else name
        return n.replace("_", "").lower()

    want = code_key(code)
    hits = [
        p for p in candidates if code_key(p.name.replace("_per_sample_tpm.parquet", "")) == want
    ]
    return hits[0] if len(hits) == 1 else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path, help="Local per-sample matrix cache root")
    p.add_argument("--release-dir", type=Path, default=None, help="Also write upload-named assets")
    p.add_argument("--codes", type=str, default=None, help="Comma-separated codes (default all)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N registry rows")
    args = p.parse_args(argv)
    stage(
        args.cache.expanduser(),
        release_dir=args.release_dir.expanduser() if args.release_dir else None,
        codes=args.codes.split(",") if args.codes else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
