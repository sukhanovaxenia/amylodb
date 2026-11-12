#!/usr/bin/env python3
"""Quality audit of a mining SQLite database (any version of the schema).

Measures the failure modes found on the first mining run, so a re-run can be
compared with it on the same instrument:

  * fill rate of the categorical fields (decided values / all proteins)
  * template-echo rate (values such as "beta-sheet|alpha-helical|unknown")
  * confidence distribution (a floor at 0.45 indicated the pattern bug)
  * spurious links: for proteins named after a known canonical, the share of
    linked abstracts that never name that protein (canonical, synonym or gene
    symbol, boundary-delimited) -- the same rule as the pipeline's mention gate
  * noise names (methods, diseases, reagents, generic phrases stored as proteins)
  * triage-score and per-paper reversibility vote summaries when present

Usage:
    python scripts/audit_mining.py --db data/mining/amyloid_aggregation.db [--json audit.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter

FIELDS = ["reversibility", "aggregate_structure", "functional_class", "assembly_mechanism", "undergoes_llps"]


def _cols(cur, table):
    return {r[1] for r in cur.execute(f"PRAGMA table_info({table})")}


def _has(cur, table):
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def audit(db_path: str) -> dict:
    from amylodb.mining.ner.normalizer import ProteinNormalizer
    from amylodb.mining.validation import genuine_mention, looks_like_noise

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    pcols = _cols(cur, "proteins")
    # Every rate below is computed over kept proteins only; dropped rows are
    # counted separately under "status".
    total = cur.execute("SELECT count(*) FROM proteins").fetchone()[0]
    where = " WHERE status IS NULL OR status = 'kept'" if "status" in pcols else ""
    proteins = [dict(r) for r in cur.execute("SELECT * FROM proteins" + where)]
    n = len(proteins)
    out = {"db": db_path, "proteins": n, "proteins_total": total,
           "papers": cur.execute("SELECT count(*) FROM papers").fetchone()[0]}

    fill, echo = {}, {}
    for f in FIELDS:
        if f not in pcols:
            continue
        vals = [p.get(f) for p in proteins]
        echo[f] = sum(1 for v in vals if v and "|" in str(v))
        fill[f] = round(sum(1 for v in vals if v and v != "unknown" and "|" not in str(v)) / n, 4) if n else 0.0
    out["fill_rate"], out["template_echoes"] = fill, echo

    conf = [p["extraction_confidence"] for p in proteins if p.get("extraction_confidence") is not None]
    if conf:
        out["confidence"] = {"min": round(min(conf), 4), "median": round(statistics.median(conf), 4),
                             "max": round(max(conf), 4),
                             "share_ge_0.5": round(sum(c >= 0.5 for c in conf) / len(conf), 4)}

    noise = [p["protein_name"] for p in proteins if looks_like_noise(p["protein_name"])]
    out["noise_names"] = {"count": len(noise), "share": round(len(noise) / n, 4) if n else 0.0,
                          "examples": noise[:15]}

    linked = spurious = 0
    per_protein = {}
    for p in proteins:
        syn = ProteinNormalizer.CANONICAL_NAMES.get(p["protein_name"])
        if syn is None:
            continue
        names = [p["protein_name"], p.get("gene_name")] + list(syn)
        rows = cur.execute("SELECT coalesce(pa.title,'') || '. ' || coalesce(pa.abstract,'') "
                           "FROM protein_papers pp JOIN papers pa ON pa.pmid = pp.pmid "
                           "WHERE pp.protein_id = ?", (p["id"],)).fetchall()
        bad = sum(1 for (t,) in rows if not genuine_mention(t, [x for x in names if x]))
        linked += len(rows); spurious += bad
        if rows:
            per_protein[p["protein_name"]] = {"linked": len(rows), "spurious": bad}
    out["canonical_links"] = {"linked": linked, "spurious": spurious,
                              "spurious_share": round(spurious / linked, 4) if linked else 0.0,
                              "by_protein": per_protein}

    if "triage_score" in _cols(cur, "papers"):
        ts = [r[0] for r in cur.execute("SELECT triage_score FROM papers WHERE triage_score IS NOT NULL")]
        if ts:
            out["triage"] = {"scored": len(ts), "median": round(statistics.median(ts), 4),
                             "share_ge_0.5": round(sum(t >= 0.5 for t in ts) / len(ts), 4)}
    if _has(cur, "protein_claims"):
        out["paper_claims"] = {
            "rows": cur.execute("SELECT count(*) FROM protein_claims").fetchone()[0],
            "reversibility": dict(Counter(r[0] for r in cur.execute("SELECT reversibility FROM protein_claims"))),
            "llps_yes": cur.execute("SELECT count(*) FROM protein_claims WHERE undergoes_llps='yes'").fetchone()[0],
        }
    # Independent counters (do not depend on the name filter).
    kept_where = " WHERE status IS NULL OR status = 'kept'" if "status" in pcols else ""
    kept_ids = [r[0] for r in cur.execute("SELECT id FROM proteins" + kept_where)]
    npap = {r[0]: r[1] for r in cur.execute("SELECT protein_id, count(*) FROM protein_papers GROUP BY protein_id")}
    out["kept"] = len(kept_ids)
    out["single_paper_share"] = round(sum(1 for i in kept_ids if npap.get(i, 0) <= 1) / len(kept_ids), 4) if kept_ids else 0.0
    for col in ("status", "protein_role", "role_subtype", "match_type", "lookup_stage"):
        if col in pcols:
            out[col] = dict(Counter(r[0] or "unset" for r in cur.execute(f"SELECT {col} FROM proteins")))
    con.close()
    return out


# Sentence vocabulary that supports the role a review sheet is judging.
_SHEET_CUES = {
    "kept": re.compile(r"\b(?:amyloid\w*|fibril\w*|aggregat\w*|oligomer\w*|prion\w*|misfold\w*|"
                       r"plaques?|deposits?|inclusions?|cross-?β|cross-?beta|β-sheet|beta-sheet)\b", re.I),
    "modifiers": re.compile(r"\b(?:inhibit\w*|prevent\w*|suppress\w*|attenuat\w*|revers\w*|"
                            r"disaggregat\w*|dissolv\w*|degrad\w*|clear\w*|chaperon\w*|sequest\w*|"
                            r"regulat\w*|modulat\w*|promot\w*|bind\w*|co-?locali\w*|associat\w*)\b", re.I),
    "condensates": re.compile(r"\b(?:stress granules?|p-bod(?:y|ies)|condensat\w*|phase[- ]separat\w*|"
                              r"llps|droplets?|nucleol\w*|paraspeckles?|granules?)\b", re.I),
    "dropped": None,
}


def review_sample(db_path: str, n: int = 50, seed: int = 7, which: str = "kept"):
    """Random sample for manual marking, one row per UniProt accession.

    which='kept'    -- kept aggregators (the exported amyloid records); 'correct' =
                       the protein forms the aggregates itself, with the right identity.
    which='modifiers' -- kept modifiers and dual-role aggregators; 'correct' =
                       identity and subtype are right.
    which='condensates' -- condensate candidates.

    A role counts as right when any of the protein's papers supports it, so
    each row shows up to three evidence sentences from different papers, those
    using the sheet's vocabulary first, and how many papers have one.
    which='dropped' -- rejected proteins; 'correct' = the rejection is right, so
                       1 - score estimates how many real proteins were lost.
    """
    import random
    import re
    from amylodb.mining.validation import genuine_mention
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    pcols = _cols(cur, "proteins")
    if which == "dropped":
        if "status" not in pcols:
            return []
        where = " WHERE status LIKE 'dropped:%'"
    else:
        where = " WHERE (status IS NULL OR status = 'kept')" if "status" in pcols else " WHERE 1"
        if "protein_role" in pcols:
            # modifiers are not exported as amyloid records: they get their own sheet
            dual = (" OR (modifier_roles IS NOT NULL AND modifier_roles != '[]')"
                    if "modifier_roles" in pcols else "")
            where += {"modifiers": " AND (protein_role = 'modifier'" + dual + ")",
                      "condensates": " AND protein_role = 'condensate'"}.get(
                which, " AND (protein_role IS NULL OR protein_role = 'aggregator')")
    rows = [dict(r) for r in cur.execute("SELECT * FROM proteins" + where + " ORDER BY id")]
    random.Random(seed).shuffle(rows)
    # rows that resolve to the same accession are one protein after the build
    seen, picked = set(), []
    for p in rows:
        key = p.get("uniprot_id") or f"row{p['id']}"
        if key not in seen:
            seen.add(key)
            picked.append(p)
    out = []
    for p in picked[:n]:
        names = [p.get("resolved_name"), p["protein_name"], p.get("resolved_from"), p.get("gene_name")]
        if _has(cur, "protein_claims") and "surface_form" in _cols(cur, "protein_claims"):
            names += [r[0] for r in cur.execute("SELECT DISTINCT surface_form FROM protein_claims WHERE protein_id=?", (p["id"],))]
        names = [x for x in names if x]
        cue = _SHEET_CUES.get(which)
        best, other, supporting = [], [], 0
        for pm, text in cur.execute("SELECT pa.pmid, coalesce(pa.title,'') || '. ' || coalesce(pa.abstract,'') "
                                    "FROM protein_papers pp JOIN papers pa ON pa.pmid = pp.pmid WHERE pp.protein_id=?", (p["id"],)):
            mention = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if genuine_mention(x, names, min_len=2)]
            if not mention:
                continue
            hit = [x for x in mention if cue and cue.search(x)]
            if hit:
                supporting += 1
                best.append(f"[{pm}] {hit[0][:300]}")
            else:
                other.append(f"[{pm}] {mention[0][:300]}")
        evidence = (best + other)[:3] + ["", "", ""]
        out.append({"id": p["id"], "name": p.get("resolved_name") or p["protein_name"],
                    "uniprot_id": p.get("uniprot_id") or "", "uniprot_name": p.get("uniprot_name") or "",
                    "organism": p.get("resolved_organism") or p.get("organism") or "",
                    "role": p.get("protein_role") or "", "subtype": p.get("role_subtype") or "",
                    "role_votes": p.get("role_votes") or "", "match_type": p.get("match_type") or "",
                    "status": p.get("status") or "",
                    "papers": cur.execute("SELECT count(*) FROM protein_papers WHERE protein_id=?", (p["id"],)).fetchone()[0],
                    "modifier_roles": ";".join(json.loads(p.get("modifier_roles") or "[]")),
                    "supporting_papers": supporting,
                    "evidence_1": evidence[0], "evidence_2": evidence[1], "evidence_3": evidence[2],
                    "correct": "",
                    # for an 'n': identity | role | offtopic | condensate | subtype.
                    # calibrate_roles.py learns only from role errors; an identity
                    # error would teach the role threshold the wrong lesson
                    "error_type": ""})
    con.close()
    return out


def score_review(path: str) -> dict:
    """Precision from a marked review sheet ('correct' = y/n) with a Wilson 95 % interval."""
    import csv
    import math
    marks = [r["correct"].strip().lower() for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")]
    yes = sum(m in ("y", "yes", "1", "+", "да") for m in marks)
    no = sum(m in ("n", "no", "0", "-", "нет") for m in marks)
    n = yes + no
    if not n:
        return {"marked": 0}
    phat, z = yes / n, 1.96
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"marked": n, "correct": yes, "precision": round(phat, 4),
            "ci95": [round(centre - half, 4), round(centre + half, 4)], "unmarked": len(marks) - n}


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit a mining database")
    ap.add_argument("--db", help="mining SQLite (required except with --score-review)")
    ap.add_argument("--json", help="also write the full audit to this JSON file")
    ap.add_argument("--review-sample", type=int, default=0,
                    help="write N random kept proteins with an evidence sentence for manual marking")
    ap.add_argument("--review-out", default="review.tsv")
    ap.add_argument("--review-modifiers", type=int, default=0,
                    help="write N random modifiers (chaperones, inhibitors, regulators) for marking")
    ap.add_argument("--review-modifiers-out", default="review_modifiers.tsv")
    ap.add_argument("--review-condensates", type=int, default=0,
                    help="write N random condensate candidates (literature without aggregation)")
    ap.add_argument("--review-condensates-out", default="review_condensates.tsv")
    ap.add_argument("--review-dropped", type=int, default=0,
                    help="write N random dropped proteins for checking the rejections (recall)")
    ap.add_argument("--review-dropped-out", default="review_dropped.tsv")
    ap.add_argument("--score-review", help="precision from a marked review sheet (column 'correct': y/n)")
    args = ap.parse_args()
    if args.score_review:
        print("review:", score_review(args.score_review))
        return 0
    if not args.db:
        ap.error("--db is required unless --score-review is given")
    a = audit(args.db)
    print(f"proteins kept {a['proteins']} of {a['proteins_total']}, papers {a['papers']}")
    print("fill rate:", a["fill_rate"])
    print("template echoes:", a["template_echoes"])
    print("confidence:", a.get("confidence"))
    print("names failing the pipeline's own filter (consistency check, NOT an independent "
          "estimate -- use --review-sample):",
          {k: v for k, v in a["noise_names"].items() if k != "examples"}, "e.g.", a["noise_names"]["examples"][:8])
    for key in ("kept", "single_paper_share", "status", "protein_role", "role_subtype", "match_type",
                "lookup_stage"):
        if key in a:
            print(f"{key}:", a[key])
    cl = a["canonical_links"]
    print(f"canonical links: {cl['linked']} linked, {cl['spurious']} spurious ({cl['spurious_share']:.1%})")
    if "triage" in a:
        print("triage:", a["triage"])
    if "paper_claims" in a:
        print("per-paper claims:", a["paper_claims"])
    import csv
    for n_rows, which, path, meaning in (
            (args.review_sample, "kept", args.review_out,
             "the protein forms the aggregates itself and the identity is right"),
            (args.review_modifiers, "modifiers", args.review_modifiers_out,
             "identity and subtype are right"),
            (args.review_condensates, "condensates", args.review_condensates_out,
             "the protein is a condensate component and the identity is right"),
            (args.review_dropped, "dropped", args.review_dropped_out, "the rejection is right")):
        if not n_rows:
            continue
        rows = review_sample(args.db, n_rows, which=which)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"], delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"Review sheet: {path} ({len(rows)} rows) -- 'correct' = y if {meaning}; "
              f"then run --score-review {path}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(a, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
