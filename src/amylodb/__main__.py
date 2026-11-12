"""amylodb command-line interface.

Two ways to build:

  # from a config file (recommended)
  python -m amylodb build --config configs/default.yaml

  # explicit source paths
  python -m amylodb build --waltzdb waltzdb.csv --amyload amyload.csv --out out/

The literature-mining stages run separately on a machine with PubMed / UniProt /
model access (see amylodb.mining); their output is written to a JSON list of
extraction dicts and fed back in with ``--mined mined.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional

from .pipeline import run

# Curated source keys the pipeline understands (map to curated parser registry).
CURATED_KEYS = [
    "waltzdb", "crossbeta", "amyload", "amyloid_explorer", "amyloid_atlas",
    "amylobase", "amypro", "amylograph", "cpad_peptides", "cpad_structures",
]


def _load_config(path: str) -> Dict:
    import yaml  # lazy: only needed when --config is used
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_mined(path: Optional[str]):
    """(aggregator records, modifier dicts) from a mined.json, or (None, None)."""
    if not path:
        return None, None
    from .mining.adapter import mined_batch_to_records, split_mined
    with open(path, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list):
        raise SystemExit(f"--mined file must be a JSON list of extraction dicts: {path}")
    aggregators, modifiers = split_mined(items)
    return mined_batch_to_records(aggregators), modifiers


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="amylodb", description="Unified amyloid database builder")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build the unified dataset")
    b.add_argument("--config", help="YAML config with sources/output/options")
    for key in CURATED_KEYS:
        b.add_argument(f"--{key.replace('_', '-')}", dest=key, help=f"{key} source file")
    b.add_argument("--mined", help="JSON list of literature-mined extraction dicts")
    b.add_argument("-o", "--out", dest="output_dir", help="Output directory")
    b.add_argument("--exclude-transmissible-prion", action="store_true",
                   help="Drop transmissible mammalian prion records (keeps prion-like domains)")
    b.add_argument("--no-features", action="store_true", help="Skip the feature matrix")
    b.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    cfg: Dict = _load_config(args.config) if args.config else {}
    sources: Dict[str, str] = dict(cfg.get("sources", {}))
    for key in CURATED_KEYS:                       # CLI overrides config
        if getattr(args, key, None):
            sources[key] = getattr(args, key)

    output_dir = args.output_dir or cfg.get("output_dir")
    if not output_dir:
        parser.error("output directory required (--out or config output_dir)")
    if not sources and not (args.mined or cfg.get("mined")):
        parser.error("at least one curated source or --mined input required")

    options = cfg.get("options", {})
    exclude_prion = args.exclude_transmissible_prion or options.get("exclude_transmissible_prion", False)
    compute_features = not args.no_features and options.get("compute_features", True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    mined, modifiers = _load_mined(args.mined or cfg.get("mined"))
    result = run(sources, output_dir, mined=mined,
                 exclude_transmissible_prion=exclude_prion,
                 compute_features=compute_features, modifiers=modifiers)
    print(json.dumps(result.stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
