#!/usr/bin/env python3
"""Fit the aggregator / modifier decision to manual review marks.

Consumes a re-validated mining DB whose proteins carry ``role_scores`` (raw NLI
role scores per abstract, written by ``revalidate_mining.py --rescore-claims``)
and one or more marked review sheets (``audit_mining.py --review-sample``).
Rows marked as aggregators in a sheet give the labels: ``y`` = the protein
forms the aggregates itself, ``n`` = it does not. Sheets are joined by UniProt
accession; a later sheet overrides an earlier one.

The rule searched is
    aggregator  iff  S(aggregates) >= a  and  S(aggregates) - max_k S(k) >= d
with S = mean or max over abstracts and k over the modifier claims. The
reported estimate is leave-one-out: thresholds chosen on n-1 proteins predict
the held-out one, so it is not fitted to the rows it scores.

Emits a printed comparison with the current rule and a JSON with the chosen
thresholds (``--out``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from statistics import mean
from typing import Dict, List, Tuple

MODIFIER_KEYS = ("disaggregase", "inhibits", "regulates", "co_deposits")
YES, NO = {"y", "yes", "1", "+", "да"}, {"n", "no", "0", "-", "нет"}


def load_labels(paths: List[str]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for path in paths:
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
            if (r.get("role") or "aggregator") != "aggregator" or not r.get("uniprot_id"):
                continue
            mark = (r.get("correct") or "").strip().lower()
            # A wrong identity, an off-topic co-mention or a condensate protein is
            # not a role error: the NLI role scores describe the paper's protein,
            # whoever UniProt says it is, so such rows would teach the threshold
            # the wrong lesson. Only unmarked-type 'y' and role errors are kept.
            etype = (r.get("error_type") or "").strip().lower()
            if mark in NO and etype not in ("", "role"):
                continue
            if mark in YES or mark in NO:
                labels[r["uniprot_id"]] = int(mark in YES)
    return labels


def load_features(db: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """accession -> {'mean': {key: v}, 'max': {key: v}} from the abstracts' role scores."""
    out = {}
    con = sqlite3.connect(db)
    for acc, raw in con.execute("SELECT uniprot_id, role_scores FROM proteins "
                                "WHERE uniprot_id IS NOT NULL AND role_scores IS NOT NULL"):
        rows = json.loads(raw)
        if not rows:
            continue
        keys = rows[0].keys()
        out[acc] = {"mean": {k: mean(r.get(k, 0.0) for r in rows) for k in keys},
                    "max": {k: max(r.get(k, 0.0) for r in rows) for k in keys}}
    con.close()
    return out


def predict(f: Dict[str, Dict[str, float]], agg: str, a: float, d: float) -> int:
    s = f[agg]
    lead = s.get("aggregates", 0.0) - max(s.get(k, 0.0) for k in MODIFIER_KEYS)
    return int(s.get("aggregates", 0.0) >= a and lead >= d)


def metrics(pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    """pairs of (label, prediction); precision over predicted aggregators."""
    tp = sum(1 for y, p in pairs if y and p)
    fp = sum(1 for y, p in pairs if not y and p)
    fn = sum(1 for y, p in pairs if y and not p)
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "kept": tp + fp, "n": len(pairs)}


GRID = [(agg, a / 100, d / 100) for agg in ("mean", "max")
        for a in range(50, 100, 5) for d in range(-30, 55, 5)]


def best_params(items, min_recall: float):
    """Highest precision with recall >= min_recall (ties: higher recall)."""
    best, best_key = None, None
    for agg, a, d in GRID:
        m = metrics([(y, predict(f, agg, a, d)) for y, f in items])
        if m["recall"] < min_recall:
            continue
        key = (m["precision"], m["recall"])
        if best_key is None or key > best_key:
            best, best_key = (agg, a, d), key
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="marked review sheets, oldest first")
    ap.add_argument("--min-recall", type=float, default=0.9)
    ap.add_argument("--out", default="role_thresholds.json")
    args = ap.parse_args()

    labels, feats = load_labels(args.labels), load_features(args.db)
    items = [(labels[acc], feats[acc]) for acc in labels if acc in feats]
    print(f"labelled aggregator rows {len(labels)}, with role scores {len(items)} "
          f"(y {sum(y for y, _ in items)}, n {sum(1 - y for y, _ in items)})")
    if len(items) < 10 or len({y for y, _ in items}) < 2:
        print("too few labelled rows with scores to calibrate")
        return 1

    current = metrics([(y, predict(f, "max", 0.7, -1.0)) for y, f in items])
    print("current rule (every scored protein kept):", current)

    loo = []
    for i, (y, f) in enumerate(items):
        params = best_params(items[:i] + items[i + 1:], args.min_recall)
        loo.append((y, predict(f, *params) if params else 1))
    print("leave-one-out estimate of the fitted rule:", metrics(loo))

    params = best_params(items, args.min_recall)
    if not params:
        print(f"no threshold keeps recall >= {args.min_recall}")
        return 1
    fit = metrics([(y, predict(f, *params)) for y, f in items])
    print(f"fitted on all rows: S={params[0]}, a={params[1]}, d={params[2]} -> {fit}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"statistic": params[0], "min_aggregates": params[1], "min_lead": params[2],
                   "fit": fit, "leave_one_out": metrics(loo), "n": len(items)}, fh, indent=2)
    print(f"thresholds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
