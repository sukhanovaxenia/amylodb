#!/usr/bin/env python3
"""Compare the current UniProt resolution with the one stored in a re-validated DB.

Measures a lookup change on the whole population before a full re-validation:
each protein is resolved again (same name preparation, fallbacks and gates as
scripts/revalidate_mining.py, no NLI) and compared with its stored accession.

Consumes: a re-validated mining SQLite (status, uniprot_name columns present).
Emits:    <out>.tsv        one row per protein whose resolution changed, compared
                           with the DB (not with an earlier diff): a protein stays
                           in it until the DB itself is re-validated;
          <out>_review.tsv a seeded sample of the changes with an empty
                           `correct` column (y = the new state is right: the right
                           entry, or a right drop), without the ids of the
                           sheets given to --exclude;
          a summary on stdout: unchanged / reidentified / lost / gained, by
          match type, and lookup errors.
Invariant: the database is opened read-only; nothing is written to it.

Example:
    python scripts/compare_lookup.py --db mining_revalidated_v3.db --out lookup_diff
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import sqlite3
import sys
from collections import Counter
from typing import Dict, List


def _revalidate_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("revalidate_mining", os.path.join(here, "revalidate_mining.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(db_path: str, limit: int = 0) -> List[Dict]:
    from amylodb.mining.ner.normalizer import ProteinNormalizer, UniProtClient, UniProtUnavailable
    from amylodb.mining.resolution import gate_resolved, prepare_candidates
    from amylodb.mining.validation import find_abbreviations
    rv = _revalidate_module()

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = [dict(r) for r in cur.execute(
        "SELECT id, protein_name, status, uniprot_id, uniprot_name, match_type FROM proteins "
        "WHERE status = 'kept' OR status LIKE 'dropped:no_uniprot_match%' OR status = 'dropped:lookup_error'")]
    if limit:
        rows = rows[:limit]
    out: List[Dict] = []
    for n, p in enumerate(rows, 1):
        pid = p["id"]
        texts = [f"{r['title'] or ''}. {r['abstract'] or ''}" for r in cur.execute(
            "SELECT pa.title, pa.abstract FROM protein_papers pp JOIN papers pa ON pa.pmid = pp.pmid "
            "WHERE pp.protein_id = ?", (pid,))]
        text = " ".join(texts)
        surfaces = Counter(r[0] for r in cur.execute(
            "SELECT surface_form FROM protein_claims WHERE protein_id = ?", (pid,)) if r[0])
        surface = surfaces.most_common(1)[0][0] if surfaces else p["protein_name"]
        abbrev: Dict[str, str] = {}
        for t in texts:
            for k, v in find_abbreviations(t).items():
                abbrev.setdefault(k, v)
        pd = {"protein_name": surface, "surface_form": surface}
        _, dropped = prepare_candidates([pd], text, abbrev)
        new_status, meta = "kept", None
        if dropped:
            new_status = f"dropped:{dropped[0][1]}"
        else:
            try:
                meta, _ = rv._resolve(UniProtClient.lookup, pd, ProteinNormalizer.normalize(pd["protein_name"]),
                                      surface, text, {"expanded": []}, pid)
            except UniProtUnavailable:
                pd["lookup_error"] = True
            for key in ("uniprot_id", "gene_name", "organism", "uniprot_name", "match_type",
                        "lookup_stage", "organism_in_text"):
                pd[key] = (meta or {}).get(key)
            reason = gate_resolved(pd, text)
            if reason:
                new_status = f"dropped:{reason}"
        old_acc = p["uniprot_id"] if p["status"] == "kept" else None
        new_acc = pd.get("uniprot_id") if new_status == "kept" else None
        if old_acc == new_acc:
            change = "unchanged"
        elif old_acc and new_acc:
            change = "reidentified"
        elif old_acc:
            change = "lost"
        else:
            change = "gained"
        out.append({"id": pid, "name": surface, "change": change,
                    "old_status": p["status"], "old_uniprot_id": old_acc or "",
                    "old_uniprot_name": p["uniprot_name"] or "", "old_match_type": p["match_type"] or "",
                    "new_status": new_status, "new_uniprot_id": new_acc or "",
                    "new_uniprot_name": pd.get("uniprot_name") or "", "new_organism": pd.get("organism") or "",
                    "new_match_type": pd.get("match_type") or ""})
        if n % 100 == 0:
            print(f"  {n}/{len(rows)}", file=sys.stderr)
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="re-validated mining SQLite")
    ap.add_argument("--out", default="lookup_diff", help="output prefix")
    ap.add_argument("--sample", type=int, default=40, help="changed rows in the review sheet")
    ap.add_argument("--limit", type=int, default=0, help="first N proteins only (smoke test)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="earlier review sheets: their ids are left out of the new sample")
    args = ap.parse_args()

    rows = compare(args.db, args.limit)
    changed = [r for r in rows if r["change"] != "unchanged"]
    fields = list(rows[0]) if rows else []
    with open(f"{args.out}.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t"); w.writeheader(); w.writerows(changed)
    seen = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as fh:
            seen |= {r["id"] for r in csv.DictReader(fh, delimiter="\t") if r.get("id")}
    fresh = [r for r in changed if str(r["id"]) not in seen]
    sample = random.Random(7).sample(fresh, min(args.sample, len(fresh)))
    with open(f"{args.out}_review.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["correct", "note"], delimiter="\t"); w.writeheader()
        w.writerows({**r, "correct": "", "note": ""} for r in sample)

    print(f"proteins compared: {len(rows)}")
    print("change:", dict(Counter(r["change"] for r in rows)))
    print("lost, by old match type:", dict(Counter(r["old_match_type"] for r in rows if r["change"] == "lost")))
    print("lost, new status:", dict(Counter(r["new_status"] for r in rows if r["change"] == "lost")))
    print("gained, by new match type:", dict(Counter(r["new_match_type"] for r in rows if r["change"] == "gained")))
    print("lookup errors:", sum(r["new_status"] == "dropped:lookup_error" for r in rows))
    if seen:
        print(f"already reviewed and left out of the sample: {len(changed) - len(fresh)}")
    print(f"-> {args.out}.tsv ({len(changed)} changed), {args.out}_review.tsv ({len(sample)} to mark)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
