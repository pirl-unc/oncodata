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

"""``cancerdata`` command-line interface.

Reference lookups over the bundled tables (cancer-type / TMB / burden) plus the
data-bundle fetch/cache surface (fetch / status / cache-dir / prune) for the
heavy per-cohort expression bundle.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import (
    apd1,
    cancer_types,
    cta,
    data_bundle,
    expression_registry,
    incidence,
    reference_data,
    samples,
    tmb,
)
from .version import __version__


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"cancerdata v{__version__}")
    return 0


def _cmd_cache_dir(args: argparse.Namespace) -> int:
    print(data_bundle.cache_dir())
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        if args.force or not data_bundle.is_local():
            data_bundle.fetch()
        else:
            print(f"Already present at {data_bundle.cache_dir()}")
    except Exception as e:  # network / extract failure
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    snap = data_bundle.status()
    print(f"Data version: {snap['data_version']}")
    print(f"Cache dir:    {snap['cache_dir']}")
    print(f"Release URL:  {snap['release_url']}")
    print(f"All local:    {'yes' if snap['all_local'] else 'no'}")
    print("-" * 60)
    for name, item in snap["items"].items():
        mark = "present" if item["present"] else "missing"
        print(f"  {name:<48} {mark:>8}  {_fmt_bytes(item['size_bytes']):>9}")
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    deleted = data_bundle.prune_cache(keep_current=not args.include_current, dry_run=not args.yes)
    if not deleted:
        print("Nothing to prune.")
        return 0
    verb = "Would delete" if not args.yes else "Deleted"
    for entry in deleted:
        print(f"{verb} {entry['version']}  ({_fmt_bytes(entry['size_bytes'])})  {entry['path']}")
    if not args.yes:
        print("\n(dry run — pass --yes to delete)")
    return 0


def _cmd_cancer_type(args: argparse.Namespace) -> int:
    try:
        info = cancer_types.cancer_type_info(args.query)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if info is None:
        print("Error: no cancer type given", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, default=str))
    return 0


def _cmd_tmb(args: argparse.Namespace) -> int:
    if args.code is None:
        for code, value in sorted(tmb.cancer_tmb().items()):
            print(f"{code}\t{value:g}")
        return 0
    try:
        value = tmb.cancer_tmb(args.code, inherit=not args.no_inherit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No TMB value for {args.code!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _cmd_apd1(args: argparse.Namespace) -> int:
    if args.code is None:
        for code, value in sorted(apd1.cancer_apd1_response().items()):
            print(f"{code}\t{value:g}")
        return 0
    try:
        value = apd1.cancer_apd1_response(args.code, inherit=not args.no_inherit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No anti-PD-1 ORR for {args.code!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _cmd_cta(args: argparse.Namespace) -> int:
    if args.unfiltered:
        genes = cta.CTA_unfiltered_gene_ids() if args.ids else cta.CTA_unfiltered_gene_names()
    else:
        genes = cta.CTA_gene_ids() if args.ids else cta.CTA_gene_names()
    if args.count:
        print(len(genes))
    else:
        for g in sorted(genes):
            print(g)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from . import plots

    fns = {
        "apd1-vs-tmb": plots.apd1_vs_tmb,
        "apd1-orr-bars": plots.apd1_orr_bars,
        "incidence-vs-mortality": plots.incidence_vs_mortality,
    }
    try:
        kwargs = {"region": args.region} if args.which == "incidence-vs-mortality" else {}
        fns[args.which](save=args.out, **kwargs)
    except (ValueError, ModuleNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out}")
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    if args.action == "list" or args.action == "status":
        rows = reference_data.status()
        print(f"{'Source':<20} {'Cached':<8} {'Version':<10} {'Size':>9}  Description")
        print("-" * 92)
        for r in rows:
            cached = "yes" if r["cached"] else "no"
            ver = r["cached_version"] or f"({r['default_version']})"
            print(
                f"{r['name']:<20} {cached:<8} {ver:<10} {_fmt_bytes(r['bytes']):>9}  {r['description']}"
            )
        print(f"\nCache directory: {reference_data.cache_dir()}")
        return 0
    if args.action == "fetch":
        names = [args.name] if args.name else list(reference_data.REFERENCE_SOURCES)
        failures = []
        for name in names:
            try:
                path = reference_data.download(name, force=args.force)
            except reference_data.ReferenceDataError as e:
                print(f"Error: {name}: {e}", file=sys.stderr)
                failures.append(name)
                continue
            print(f"Ready: {name} -> {path}")
        return 1 if failures else 0
    if args.action == "path":
        if not args.name:
            print("Error: 'sources path' requires a source name", file=sys.stderr)
            return 1
        try:
            print(reference_data.ensure(args.name))
        except reference_data.ReferenceDataError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    return 1


def _cmd_expression_sources(args: argparse.Namespace) -> int:
    srcs = expression_registry.expression_sources()
    if args.code:
        srcs = [s for s in srcs if args.code in s.cancer_codes]
    if args.type:
        srcs = [s for s in srcs if s.source_type == args.type]
    if not srcs:
        print("No matching expression sources.", file=sys.stderr)
        return 1
    print(f"{'Source':<26} {'Type':<20} {'Unit':<7} {'Codes':<6} Cancer codes")
    print("-" * 100)
    for s in sorted(srcs, key=lambda x: (x.source_type, x.id)):
        codes = ";".join(s.cancer_codes)
        codes = codes if len(codes) <= 40 else codes[:37] + "..."
        print(
            f"{s.id:<26} {s.source_type:<20} {(s.unit or ''):<7} {len(s.cancer_codes):<6} {codes}"
        )
    print(f"\n{len(srcs)} sources. Registry: cancerdata/data/expression_sources.yaml")
    return 0


def _cmd_samples(args: argparse.Namespace) -> int:
    if args.counts or (not args.code and not args.cohort):
        counts = samples.sample_counts_by_cancer_code(included_only=not args.all)
        for code, n in counts.items():
            print(f"{code}\t{n}")
        print(f"\n{int(counts.sum())} samples across {len(counts)} cancer codes", file=sys.stderr)
        return 0
    if args.code:
        df = samples.samples_for_cancer_code(args.code, included_only=not args.all)
    else:
        df = samples.samples_for_cohort(args.cohort, included_only=not args.all)
    cols = [c for c in ("sample_id", "source_cohort", "sample_type", "raw_unit") if c in df.columns]
    if df.empty:
        print("No matching samples.", file=sys.stderr)
        return 1
    print(df[cols].to_string(index=False))
    return 0


def _cmd_burden(args: argparse.Namespace) -> int:
    try:
        if args.category is None:
            for cat, pct in sorted(incidence.cancer_burden(metric=args.metric).items()):
                print(f"{cat}\t{pct:g}")
            return 0
        value = incidence.cancer_burden(args.category, metric=args.metric)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No burden value for category {args.category!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cancerdata",
        description="Curated cancer reference data: ontology, TMB, incidence/mortality, expression.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the installed cancerdata version").set_defaults(
        func=_cmd_version
    )

    # --- data bundle (fetch / cache) ---
    sub.add_parser(
        "cache-dir", help="Print the on-disk cache dir for the downloadable data bundle"
    ).set_defaults(func=_cmd_cache_dir)

    p_fetch = sub.add_parser("fetch", help="Download the per-cohort expression data bundle")
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if present")
    p_fetch.set_defaults(func=_cmd_fetch)

    sub.add_parser(
        "status", help="Report which bundle paths are cached locally (no download)"
    ).set_defaults(func=_cmd_status)

    p_prune = sub.add_parser("prune", help="Delete stale version-pinned cache dirs")
    p_prune.add_argument("--yes", action="store_true", help="Actually delete (default: dry run)")
    p_prune.add_argument(
        "--include-current", action="store_true", help="Also delete the current version's cache"
    )
    p_prune.set_defaults(func=_cmd_prune)

    # --- reference lookups ---
    p_ct = sub.add_parser(
        "cancer-type", help="Resolve a cancer type/alias/name and print its registry info"
    )
    p_ct.add_argument("query", help="Cancer code, alias, or display name (e.g. PRAD, prostate)")
    p_ct.set_defaults(func=_cmd_cancer_type)

    p_tmb = sub.add_parser("tmb", help="Median TMB (mut/Mb) for a code, or the full map")
    p_tmb.add_argument("code", nargs="?", default=None, help="Cancer code/alias (omit for all)")
    p_tmb.add_argument("--no-inherit", action="store_true", help="Do not inherit an ancestor's TMB")
    p_tmb.set_defaults(func=_cmd_tmb)

    p_apd1 = sub.add_parser(
        "apd1", help="Anti-PD-1 monotherapy ORR (%) for a code, or the full map"
    )
    p_apd1.add_argument("code", nargs="?", default=None, help="Cancer code/alias (omit for all)")
    p_apd1.add_argument(
        "--no-inherit", action="store_true", help="Do not inherit an ancestor's ORR"
    )
    p_apd1.set_defaults(func=_cmd_apd1)

    p_cta = sub.add_parser("cta", help="List cancer-testis antigens (expressed set by default)")
    p_cta.add_argument("--unfiltered", action="store_true", help="Full candidate universe")
    p_cta.add_argument("--ids", action="store_true", help="Ensembl gene IDs instead of symbols")
    p_cta.add_argument("--count", action="store_true", help="Print the count only")
    p_cta.set_defaults(func=_cmd_cta)

    p_plot = sub.add_parser("plot", help="Render a cancer-type reference plot to a PNG")
    p_plot.add_argument(
        "which",
        choices=["apd1-vs-tmb", "apd1-orr-bars", "incidence-vs-mortality"],
        help="Which plot to render",
    )
    p_plot.add_argument("--out", required=True, help="Output PNG path")
    p_plot.add_argument(
        "--region", default="us", choices=["us", "world"], help="Region for incidence-vs-mortality"
    )
    p_plot.set_defaults(func=_cmd_plot)

    p_burden = sub.add_parser(
        "burden", help="Incidence/mortality share for a burden category, or the full map"
    )
    p_burden.add_argument(
        "category", nargs="?", default=None, help="Burden category (omit for all)"
    )
    p_burden.add_argument(
        "--metric",
        default="us_incidence_pct",
        choices=list(incidence._BURDEN_METRICS),
        help="Which share to report (default: us_incidence_pct)",
    )
    p_burden.set_defaults(func=_cmd_burden)

    p_sources = sub.add_parser(
        "sources", help="Manage HPA normal-tissue reference sources (RNA / IHC / single-cell)"
    )
    p_sources.add_argument(
        "action",
        choices=["list", "status", "fetch", "path"],
        help="list/status (show cache state), fetch (download), path (print local path)",
    )
    p_sources.add_argument("name", nargs="?", default=None, help="Source name (omit to fetch all)")
    p_sources.add_argument("--force", action="store_true", help="Re-download even if cached")
    p_sources.set_defaults(func=_cmd_sources)

    p_exprsrc = sub.add_parser(
        "expression-sources",
        help="List the cohort expression sources (Treehouse, GDC, GEO, recount3, …)",
    )
    p_exprsrc.add_argument("--code", default=None, help="Only sources feeding this cancer code")
    p_exprsrc.add_argument("--type", default=None, help="Only this source_type (e.g. gdc, geo)")
    p_exprsrc.set_defaults(func=_cmd_expression_sources)

    p_samples = sub.add_parser(
        "samples", help="Per-sample curation manifest (counts, or rows for a code/cohort)"
    )
    p_samples.add_argument("--code", default=None, help="Samples assigned to this cancer code")
    p_samples.add_argument("--cohort", default=None, help="Samples from this source cohort")
    p_samples.add_argument("--counts", action="store_true", help="Per-cancer-code sample counts")
    p_samples.add_argument(
        "--all", action="store_true", help="Include excluded samples (default: included only)"
    )
    p_samples.set_defaults(func=_cmd_samples)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
