#!/usr/bin/env python3
"""Export the mining SQLite database to a mined.json for `amylodb build --mined`.

The mining pipeline (``python -m amylodb.mining.pipeline``) writes its results to
a SQLite database (proteins / papers / protein_papers / cofactors /
string_interactions). This script flattens that into a JSON list of extraction
dicts in exactly the shape `amylodb.mining.adapter.mined_to_record` expects, so
the literature-mined records can be reconciled with the curated ones.

Consumes: the mining SQLite path.
Emits: a JSON list written to --out.
Invariant: one JSON object per row in the proteins table; pmids, cofactors and
STRING partners are gathered per protein.

Usage:
    python scripts/export_mined.py --db data/amyloid_aggregation.db --out mined.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from typing import Dict, List

DECIDED_REVERSIBILITY = ("reversible", "irreversible", "conditionally_reversible")


def resolve_reversibility(votes: Dict[str, int], stored: str) -> str:
    """Protein-level reversibility from per-paper decisions.

    * one class holds >= 2/3 of the decided votes -> that class;
    * reversible and irreversible both present without such a majority ->
      conditionally_reversible (different papers report different states,
      e.g. wild type vs disease mutant);
    * no per-paper decisions -> the value stored on the protein row.
    """
    decided = {k: v for k, v in votes.items() if k in DECIDED_REVERSIBILITY and v > 0}
    total = sum(decided.values())
    if total:
        top, n = max(decided.items(), key=lambda kv: kv[1])
        if n / total >= 2 / 3:
            return top
        if "reversible" in decided and "irreversible" in decided:
            return "conditionally_reversible"
        return top
    return stored or "unknown"


def _has_table(cur, name: str) -> bool:
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def unify_roles(records: List[Dict]) -> List[Dict]:
    """Give every row of one UniProt accession the same role fields, in place.

    Rows are per written name, so one protein can carry several roles: TDP-43
    was an aggregator under "TDP-43" and a modifier under "TAR DNA binding
    protein" on the second v3 sheet. The accession's role is aggregator if any
    row is, else modifier, else condensate; modifier_roles is the union over the
    rows (with each modifier row's subtype); condensate evidence is any row's.
    """
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        if r.get("uniprot_id"):
            groups.setdefault(r["uniprot_id"], []).append(r)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        roles = {r["protein_role"] for r in rows}
        primary = next(x for x in ("aggregator", "modifier", "condensate") if x in roles) \
            if roles & {"aggregator", "modifier", "condensate"} else rows[0]["protein_role"]
        mods = set()
        for r in rows:
            mods.update(r.get("modifier_roles") or [])
            if r["protein_role"] == "modifier" and r.get("role_subtype"):
                mods.add(r["role_subtype"])
        subtypes = Counter(r["role_subtype"] for r in rows if r["protein_role"] == "modifier" and r.get("role_subtype"))
        evidence = any(r.get("condensate_evidence") or r["protein_role"] == "condensate" for r in rows)
        # a condensate candidate stays a "regulator" only if no row is a component
        cond_regulator = all(r.get("role_subtype") == "regulator" for r in rows if r["protein_role"] == "condensate")
        rank = {"curated": 3, "text": 2, "model_only": 1}
        tier = max((r.get("aggregation_evidence") or "" for r in rows), key=lambda t: rank.get(t, 0))
        for r in rows:
            r["protein_role"] = primary
            r["role_subtype"] = subtypes.most_common(1)[0][0] if primary == "modifier" and subtypes else \
                ("regulator" if primary == "condensate" and cond_regulator else
                 "" if primary != "modifier" else r.get("role_subtype", ""))
            r["aggregation_evidence"] = tier if primary == "aggregator" else ""
            r["modifier_roles"] = sorted(mods) if primary != "condensate" else []
            r["condensate_evidence"] = evidence
            r["forms_aggregates"] = primary == "aggregator"
    return records


def export(db_path: str) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    pcols = {r[1] for r in cur.execute("PRAGMA table_info(proteins)")}
    # Re-validated databases mark dropped records instead of deleting them.
    where = " WHERE status IS NULL OR status = 'kept'" if "status" in pcols else ""
    proteins = [dict(r) for r in cur.execute("SELECT * FROM proteins" + where)]
    has_claims = _has_table(cur, "protein_claims")
    has_methods = _has_table(cur, "experimental_methods")
    out: List[Dict] = []
    for p in proteins:
        pid = p["id"]
        pmids = [r["pmid"] for r in cur.execute(
            "SELECT pmid FROM protein_papers WHERE protein_id = ?", (pid,)) if r["pmid"]]
        cof = {"assembly": [], "disassembly": []}
        for r in cur.execute(
                "SELECT cofactor_name, cofactor_type FROM cofactors WHERE protein_id = ?", (pid,)):
            cof.setdefault(r["cofactor_type"], []).append(r["cofactor_name"])
        partners = [r["partner_name"] for r in cur.execute(
            "SELECT partner_name FROM string_interactions WHERE protein_id = ?", (pid,))
            if r["partner_name"]]
        methods: List[str] = []
        if has_methods:
            for r in cur.execute("SELECT method_name FROM experimental_methods WHERE protein_id = ?", (pid,)):
                if r["method_name"] and r["method_name"] not in methods:
                    methods.append(r["method_name"])
        votes: Counter = Counter()
        scores: Dict[str, float] = {}
        llps_yes = (p.get("undergoes_llps") == "yes")
        if has_claims:
            for r in cur.execute("SELECT reversibility, undergoes_llps, claim_scores "
                                 "FROM protein_claims WHERE protein_id = ?", (pid,)):
                if r["reversibility"]:
                    votes[r["reversibility"]] += 1
                llps_yes = llps_yes or r["undergoes_llps"] == "yes"
                for k, v in json.loads(r["claim_scores"] or "{}").items():
                    scores[k] = max(scores.get(k, 0.0), float(v))

        out.append({
            "protein_name": p.get("resolved_name") or p.get("protein_name", ""),
            "gene_name": p.get("gene_name") or "",
            "uniprot_id": p.get("uniprot_id") or "",
            "organism": p.get("resolved_organism") or p.get("organism") or "",
            "protein_role": p.get("protein_role") or "aggregator",
            "role_subtype": p.get("role_subtype") or "",
            # modifier subtypes any paper supports, beside the primary role
            # (gelsolin: aggregator, and inhibitor of Abeta fibrillisation)
            "modifier_roles": json.loads(p.get("modifier_roles") or "[]"),
            "condensate_evidence": bool(p.get("condensate_evidence")),
            # curated | text (a sentence says the protein aggregates) | model_only
            "aggregation_evidence": p.get("aggregation_evidence") or "",
            "uniprot_name": p.get("uniprot_name") or "",
            "match_type": p.get("match_type") or "",
            "resolved_from": p.get("resolved_from") or "",
            "forms_aggregates": bool(p.get("forms_aggregates", 1)),
            "aggregate_structure": p.get("aggregate_structure") or "unknown",
            "reversibility": resolve_reversibility(dict(votes), p.get("reversibility")),
            "reversibility_votes": dict(votes),
            "assembly_mechanism": p.get("assembly_mechanism") or "unknown",
            "functional_class": p.get("functional_class") or "unknown",
            "biological_role": p.get("biological_role") or "",
            "associated_disease": p.get("associated_disease") or "",
            "evidence_strength": p.get("evidence_strength") or "unknown",
            "extraction_confidence": p.get("extraction_confidence") or 0.0,
            "assembly_cofactors": cof.get("assembly", []),
            "disassembly_cofactors": cof.get("disassembly", []),
            "string_partners": partners,
            "experimental_methods": methods,
            "reported_method_tier": p.get("reported_method_tier"),
            "undergoes_llps": "yes" if llps_yes else "unknown",
            "claim_scores": scores,
            "source_pmids": pmids,
            "pmid": pmids[0] if pmids else "",
        })
    conn.close()
    unify_roles(out)

    # For every modifier, the aggregators mentioned in the same papers: the
    # candidates for what that chaperone / disaggregase acts on.
    by_pmid: Dict[str, List[str]] = {}
    for item in out:
        if item["protein_role"] == "aggregator":
            for pm in item["source_pmids"]:
                by_pmid.setdefault(pm, []).append(item["protein_name"])
    for item in out:
        if item["protein_role"] == "modifier" or item["modifier_roles"]:
            seen: List[str] = []
            for pm in item["source_pmids"]:
                for name in by_pmid.get(pm, []):
                    if name not in seen and name != item["protein_name"]:
                        seen.append(name)
            item["co_mentioned_aggregators"] = seen
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export mining SQLite to mined.json")
    ap.add_argument("--db", required=True, help="Mining SQLite path")
    ap.add_argument("--out", default="mined.json", help="Output JSON path")
    args = ap.parse_args()

    records = export(args.db)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    role = lambda r: r.get("protein_role")
    n_agg = sum(role(r) == "aggregator" for r in records)
    n_dual = sum(role(r) == "aggregator" and bool(r.get("modifier_roles")) for r in records)
    n_mod = sum(role(r) == "modifier" for r in records)
    n_cond = sum(role(r) == "condensate" for r in records)
    n_ev = sum(bool(r.get("condensate_evidence")) and role(r) != "condensate" for r in records)
    print(f"Exported {len(records)} mined proteins -> {args.out}: {n_agg} aggregators "
          f"({n_dual} also modifiers), {n_mod} modifiers, {n_cond} condensate candidates; "
          f"{n_ev} aggregators/modifiers with condensate evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
