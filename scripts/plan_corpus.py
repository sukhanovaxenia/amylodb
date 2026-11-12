#!/usr/bin/env python3
"""Report how large an expanded mining corpus would be, fetching nothing.

Queries PubMed for hit counts per term group and publication-year bin (one
`esearch` with `retmax=0` each) and prints a table plus the totals. The sum over
bins double-counts papers that several groups match, so the figure to plan
against is the per-group total, not the grand total; `--per-bin` shows what the
fetch budget would actually retrieve.

Consumes nothing but the settings; emits a table, a JSON (`--out`) and, with
`--estimate-overlap`, the distinct-PMID count for a sample of bins so the
double-counting can be quantified before the full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")

from amylodb.mining.pubmed.searcher import PubMedSearcher
from amylodb.mining.settings import CORPUS_PLAN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--email", required=True, help="contact address required by the NCBI API")
    ap.add_argument("--pubmed-key", default=None, help="NCBI API key (raises the rate limit)")
    ap.add_argument("--groups", nargs="+", default=None, help="term groups (default: all)")
    ap.add_argument("--date-from", default=CORPUS_PLAN["date_from"])
    ap.add_argument("--date-to", default=None)
    ap.add_argument("--bin-years", type=int, default=CORPUS_PLAN["bin_years"])
    ap.add_argument("--per-bin", type=int, default=CORPUS_PLAN["per_bin"])
    ap.add_argument("--estimate-overlap", type=int, default=0,
                    help="retrieve PMIDs for the N most recent bins of each group to "
                         "measure how much the groups overlap")
    ap.add_argument("--out", default="corpus_plan.json")
    args = ap.parse_args()

    searcher = PubMedSearcher(email=args.email, api_key=args.pubmed_key)
    rows = searcher.plan_corpus(groups=args.groups, date_from=args.date_from,
                                date_to=args.date_to, bin_years=args.bin_years)

    per_group = defaultdict(int)
    budget = defaultdict(int)
    for r in rows:
        per_group[r["group"]] += r["count"]
        budget[r["group"]] += min(r["count"], args.per_bin)
        print(f"{r['group']:<24} {r['from']}-{r['to']}  hits {r['count']:>7}  "
              f"fetched {min(r['count'], args.per_bin):>5}")

    print("\nper group (hits / what --per-bin would fetch):")
    for g in per_group:
        print(f"  {g:<24} {per_group[g]:>8} / {budget[g]:>7}")
    print(f"\nsum over groups and bins: {sum(per_group.values())} hits, "
          f"{sum(budget.values())} would be fetched (papers matched by several "
          f"groups are counted once per group, so the distinct total is lower)")
    print(f"current corpus for comparison: 2906 papers, 1234 of them with a protein")

    report = {"plan": rows, "per_group_hits": dict(per_group),
              "per_group_fetched": dict(budget), "bin_years": args.bin_years,
              "per_bin": args.per_bin, "date_from": args.date_from}

    if args.estimate_overlap:
        seen, counted = set(), 0
        for group, terms in searcher.term_groups(args.groups).items():
            bins = searcher.year_bins(args.date_from, args.date_to, args.bin_years)
            for start, end in bins[-args.estimate_overlap:]:
                q = searcher.build_query(include_terms=terms, date_from=start, date_to=end)
                pmids = searcher.search(q, max_results=args.per_bin)
                counted += len(pmids)
                seen.update(pmids)
        overlap = 1 - len(seen) / counted if counted else 0.0
        print(f"\noverlap over the last {args.estimate_overlap} bins per group: "
              f"{counted} retrieved, {len(seen)} distinct ({overlap:.1%} duplicated)")
        report["overlap_sample"] = {"retrieved": counted, "distinct": len(seen),
                                    "duplicate_share": round(overlap, 4)}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nplan -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
