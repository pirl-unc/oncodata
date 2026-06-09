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
data cache surface. The heavy per-cohort expression bundle and its
``fetch``/``status``/``prune`` subcommands are added in a later milestone; the
cache-dir resolution here is already the path that bundle will populate.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import apd1, cancer_types, cta, incidence, tmb
from .cache import bundle_cache_dir
from .version import __version__


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"cancerdata v{__version__}")
    return 0


def _cmd_cache_dir(args: argparse.Namespace) -> int:
    print(bundle_cache_dir())
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
        description="Curated cancer reference data: ontology, TMB, incidence/mortality.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the installed cancerdata version").set_defaults(
        func=_cmd_version
    )

    sub.add_parser(
        "cache-dir", help="Print the on-disk cache dir for the downloadable data bundle"
    ).set_defaults(func=_cmd_cache_dir)

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

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
