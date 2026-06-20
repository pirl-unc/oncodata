#!/usr/bin/env python
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

"""Regenerate **all** oncoref figures into one timestamped run directory.

Mirrors the pirlygenes ``analyses/regenerate_plots.py`` convention: every run
writes into a fresh ``outputs/run_<YYYYMMDD-HHMMSS>/`` snapshot (gitignored),
organised by plot family in subfolders, so a new run never overwrites an older
one. A ``latest`` symlink points at the most recent run; an ``index.md`` lists
what was produced (and what was skipped).

Each figure is independent: one that can't be drawn (missing per-sample matrix,
empty data) is reported and skipped, never aborting the batch. Expression-backed
figures need the relevant shards/matrices cached locally; the coverage/response
panels in particular need per-sample matrices (see
``plots._cached_per_sample_cohorts``), so they cover only the cached cohorts.

Usage::

    python scripts/regenerate_plots.py                 # all figures -> new run dir
    python scripts/regenerate_plots.py --out-dir DIR   # base dir override
    python scripts/regenerate_plots.py --no-timestamp  # write straight into base
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from oncoref import cta_curation_plots, plots  # noqa: E402
from oncoref.plots import _cached_per_sample_cohorts  # noqa: E402


def _jobs() -> list[tuple[str, str, str, dict]]:
    """``(family, name, fn_attr, kwargs)`` for every figure, with sensible
    defaults. Coverage/response panels are restricted to the cohorts that
    actually have a cached per-sample matrix."""
    cached = sorted(_cached_per_sample_cohorts())
    jobs: list[tuple[str, str, str, dict]] = [
        # aPD-1 response
        ("apd1", "apd1_vs_tmb", "apd1_vs_tmb", {}),
        ("apd1", "apd1_orr_bars", "apd1_orr_bars", {}),
        (
            "apd1",
            "apd1_response_signature_t_cell_inflamed",
            "apd1_response_signature_scatter",
            {"signature": "t_cell_inflamed"},
        ),
        # ICI regimens as distinct response sources
        ("ici", "ici_response_by_regimen", "ici_response_by_regimen", {}),
        ("ici", "ici_regimen_comparison", "ici_regimen_comparison", {}),
        ("ici", "ici_orr_pooled_forest", "ici_orr_pooled_forest", {}),
        # incidence / burden
        ("burden", "incidence_vs_mortality_us", "incidence_vs_mortality", {"region": "us"}),
        ("burden", "burden_category_bars_us", "burden_category_bars", {"region": "us"}),
        # CTA expression
        (
            "cta_expression",
            "cta_expression_heatmap_median",
            "cta_expression_heatmap",
            {"stat": "median"},
        ),
        (
            "cta_expression",
            "cta_expression_heatmap_q3",
            "cta_expression_heatmap",
            {"stat": "q3"},
        ),
        # CTA addressable burden + per-patient prevalence. Uses the faithful
        # per-sample source (cached cohorts); the portable within_sample source
        # needs the within-sample bundle, which isn't always cached locally.
        (
            "cta_addressable",
            "cta_addressable_burden_per_sample",
            "cta_addressable_burden",
            {"source": "per_sample"},
        ),
        ("cta_patient", "cta_patient_count_heatmap", "cta_patient_count_heatmap", {}),
        # CTA burden / neoantigen load vs response
        (
            "cta_response",
            "cta_burden_vs_apd1",
            "cta_burden_vs_response",
            {"against": "apd1"},
        ),
        ("cta_response", "cta_burden_vs_tmb", "cta_burden_vs_response", {"against": "tmb"}),
        (
            "cta_response",
            "cta_specific_9mer_load_vs_tmb",
            "cta_specific_9mer_load",
            {"against": "tmb"},
        ),
    ]
    # Coverage panels need explicit cohort codes with cached per-sample matrices.
    if cached:
        jobs += [
            (
                "cta_coverage",
                "cta_coverage_curves",
                "cta_coverage_curves",
                {"cancer_types": cached},
            ),
            (
                "cta_coverage",
                "cta_coverage_stacked_bars",
                "cta_coverage_stacked_bars",
                {"cancer_types": cached},
            ),
        ]
    return jobs


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    base = args.out_dir.resolve() if args.out_dir else _REPO_ROOT / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    if args.no_timestamp:
        return base
    run = args.run_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = base / run
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _update_latest(run_dir: Path) -> None:
    """Best-effort ``latest`` symlink next to the run dir."""
    link = run_dir.parent / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks may be unavailable (e.g. some Windows setups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="base output dir (default: ./outputs)"
    )
    ap.add_argument(
        "--run-name", default=None, help="run subfolder name (default: run_<timestamp>)"
    )
    ap.add_argument("--no-timestamp", action="store_true", help="write straight into the base dir")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args)
    jobs = _jobs()
    print(f"Regenerating {len(jobs)} figures into {run_dir}")

    done: list[str] = []
    skipped: list[tuple[str, str]] = []
    for family, name, fn_attr, kwargs in jobs:
        fam_dir = run_dir / family
        fam_dir.mkdir(parents=True, exist_ok=True)
        out = fam_dir / f"{name}.png"
        try:
            getattr(plots, fn_attr)(save=out, **kwargs)
            done.append(f"{family}/{name}.png")
            print(f"  ok    {family}/{name}.png")
        except Exception as e:
            skipped.append((f"{family}/{name}", f"{type(e).__name__}: {e}"))
            print(f"  SKIP  {family}/{name}  ({type(e).__name__}: {e})", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    curation_dir = run_dir / "cta_curation"
    try:
        result = cta_curation_plots.render(out_dir=curation_dir)
        for path in result["paths"].values():
            done.append(f"cta_curation/{path.name}")
            print(f"  ok    cta_curation/{path.name}")
    except Exception as e:
        skipped.append(("cta_curation", f"{type(e).__name__}: {e}"))
        print(f"  SKIP  cta_curation  ({type(e).__name__}: {e})", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    index = run_dir / "index.md"
    lines = [
        f"# oncoref figures — {run_dir.name}",
        "",
        f"{len(done)} generated, {len(skipped)} skipped.",
        "",
        "## Generated",
        *(f"- `{p}`" for p in done),
    ]
    if skipped:
        lines += ["", "## Skipped", *(f"- `{n}` — {why}" for n, why in skipped)]
    index.write_text("\n".join(lines) + "\n")

    _update_latest(run_dir)
    print(f"\n{len(done)} figures written, {len(skipped)} skipped. Index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
