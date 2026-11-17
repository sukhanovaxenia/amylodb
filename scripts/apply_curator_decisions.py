#!/usr/bin/env python3
"""Apply curator decisions to a copy of a revalidated mining database.

    python scripts/apply_curator_decisions.py \
        --db data/mining_v3/amyloid_aggregation.revalidated.db \
        --out data/mining_v3/amyloid_aggregation.v3f.db \
        --decisions configs/curator_decisions.tsv \
        --snapshot configs/curator_decisions.json \
        --cache data/reviews/amylodb-005b/uniprot_cache \
        --report data/reviews/amylodb-005b/apply_report.json [--offline]

Consumes: the source SQLite (opened read-only; never migrated or written), a
decision table (one row per decision) and a snapshot manifest holding the
SHA-256 the decisions were checked against.
Emits: a new SQLite at --out, published only after every check passed, and a
JSON report.

Invariants (docs/changes_from_source.md 9.17):
  * the source hash equals the manifest's; any other snapshot -- including the
    output of an earlier run -- is refused, so decisions are never re-applied
    blindly to a re-validated database;
  * each decision addresses exactly one row: id, protein_name, organism and
    status must all match, and (protein_name, organism, status) must be unique;
  * every supporting PMID is linked to that row;
  * a restored accession is checked against UniProt (primary accession, TaxID,
    scientific name, protein name, gene name); answers are cached on disk, one
    file per accession, and --offline uses the cache only;
  * all decisions are applied in one transaction to a private temporary copy,
    checked, then published with os.link, which fails if --out exists (no
    overwrite mode). On any failure nothing appears at --out and the source is
    unchanged; only this run's own temporary file is removed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

ACTIONS = {"restore", "drop", "unresolved"}
BASES = {"restore": {"aggregation_text", "LCR_text"}, "unresolved": {"unresolved_attribution"}, "drop": {""}}
COLUMNS = ["row_id", "protein_name", "baseline_organism", "baseline_status", "action", "role",
           "aggregation_evidence", "evidence_basis", "drop_status", "uniprot_id", "expected_name",
           "expected_gene", "expected_taxid", "verified_organism", "support_pmids", "scope_note",
           "reason", "decision_source", "decided"]
NEW_COLUMNS = (("evidence_basis", "TEXT"), ("curator_decision", "TEXT"))  # as in storage.py
UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.json"


class DecisionError(Exception):
    """A decision cannot be applied; nothing is written."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_decisions(path: Path) -> List[Dict[str, str]]:
    """Rows of the decision table, with the columns and values checked."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise DecisionError(f"decision columns {reader.fieldnames} != {COLUMNS}")
        rows = list(reader)
    seen = set()
    for r in rows:
        tag = f"decision row_id={r['row_id']}"
        try:
            row_id = int(r["row_id"])
        except (ValueError, TypeError) as exc:
            raise DecisionError(f"{tag}: expected a positive integer") from exc
        if row_id <= 0:
            raise DecisionError(f"{tag}: expected a positive integer")
        if row_id in seen:
            raise DecisionError(f"{tag}: duplicate")
        seen.add(row_id)
        r["row_id"] = str(row_id)
        if r["action"] not in ACTIONS:
            raise DecisionError(f"{tag}: action {r['action']!r}")
        if r["evidence_basis"] not in BASES[r["action"]]:
            raise DecisionError(f"{tag}: evidence_basis {r['evidence_basis']!r} for {r['action']}")
        for col in ("reason", "decision_source", "decided"):
            if not r[col].strip():
                raise DecisionError(f"{tag}: empty {col}")
        if r["action"] == "restore":
            for col in ("uniprot_id", "expected_name", "expected_gene", "expected_taxid",
                        "verified_organism", "support_pmids", "scope_note"):
                if not r[col].strip():
                    raise DecisionError(f"{tag}: restore without {col}")
            if (r["role"], r["aggregation_evidence"]) != ("aggregator", "curator_review"):
                raise DecisionError(f"{tag}: restore must be aggregator/curator_review")
        if r["action"] == "drop" and not r["drop_status"].startswith("dropped:curator_"):
            raise DecisionError(f"{tag}: drop_status {r['drop_status']!r}")
        if r["action"] == "unresolved" and r["aggregation_evidence"] != "unresolved":
            raise DecisionError(f"{tag}: unresolved must set aggregation_evidence unresolved")
    return rows


def _names(entry: Dict) -> List[str]:
    pd = entry.get("proteinDescription", {}) or {}
    out = []
    for block in [pd.get("recommendedName")] + (pd.get("alternativeNames") or []) + \
            (pd.get("submissionNames") or []):
        if block:
            out.append(block.get("fullName", {}).get("value", ""))
            out += [s.get("value", "") for s in block.get("shortNames", []) or []]
    return [n for n in out if n]


def _genes(entry: Dict) -> List[str]:
    out = []
    for g in entry.get("genes", []) or []:
        if g.get("geneName"):
            out.append(g["geneName"]["value"])
        out += [s.get("value", "") for s in g.get("synonyms", []) or []]
    return [x for x in out if x]


def fetch_uniprot(acc: str) -> bytes:
    """Raw JSON of one UniProtKB entry. Generic client name, no personal data."""
    req = urllib.request.Request(UNIPROT.format(acc), headers={"User-Agent": "amylodb-curator-check"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def uniprot_entry(acc: str, cache: Path, offline: bool, fetch: Callable[[str], bytes],
                  usage: Dict[str, str]) -> Dict:
    """The cached entry for ``acc``, fetching (and caching) it unless offline."""
    path = cache / f"{acc}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        body = rec["response"].encode()
        if hashlib.sha256(body).hexdigest() != rec["sha256"]:
            raise DecisionError(f"{acc}: cache checksum mismatch")
        usage[acc] = "cache"
    else:
        if offline:
            raise DecisionError(f"{acc}: not in cache and --offline")
        body = fetch(acc)
        cache.mkdir(parents=True, exist_ok=True)
        entry = json.loads(body)
        rec = {"accession": acc, "url": UNIPROT.format(acc),
               "retrieved": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "taxid": entry.get("organism", {}).get("taxonId"),
               "sha256": hashlib.sha256(body).hexdigest(), "response": body.decode()}
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        os.replace(tmp, path)
        usage[acc] = "fetched"
    return json.loads(body)


def check_identity(d: Dict[str, str], entry: Dict) -> Dict[str, str]:
    """Verified identity fields for a restore, or DecisionError on any mismatch."""
    tag = f"decision row_id={d['row_id']} {d['uniprot_id']}"
    if entry.get("primaryAccession") != d["uniprot_id"]:
        raise DecisionError(f"{tag}: primary accession {entry.get('primaryAccession')}")
    org = entry.get("organism", {}) or {}
    if str(org.get("taxonId")) != d["expected_taxid"]:
        raise DecisionError(f"{tag}: TaxID {org.get('taxonId')} != {d['expected_taxid']}")
    if org.get("scientificName") != d["verified_organism"]:
        raise DecisionError(f"{tag}: organism {org.get('scientificName')!r}")
    names, genes = _names(entry), _genes(entry)
    if d["expected_name"].lower() not in {n.lower() for n in names}:
        raise DecisionError(f"{tag}: name {d['expected_name']!r} not in {names[:5]}")
    if d["expected_gene"].lower() not in {g.lower() for g in genes}:
        raise DecisionError(f"{tag}: gene {d['expected_gene']!r} not in {genes[:5]}")
    pd = entry.get("proteinDescription", {}) or {}
    rec = (pd.get("recommendedName") or {}).get("fullName", {}).get("value") or names[0]
    gene = entry["genes"][0]["geneName"]["value"] if entry.get("genes") and \
        entry["genes"][0].get("geneName") else d["expected_gene"]
    return {"uniprot_name": rec, "gene_name": gene, "organism": org["scientificName"],
            "taxid": str(org["taxonId"])}


def decision_record(d: Dict[str, str], ident: Optional[Dict[str, str]], source_sha: str) -> Dict:
    """The JSON stored in proteins.curator_decision (becomes evidence_items)."""
    rec = {"source_row_id": int(d["row_id"]), "decision": d["action"],
           "aggregation_evidence": d["aggregation_evidence"] or None,
           "evidence_basis": d["evidence_basis"] or None,
           "support_pmids": [p for p in d["support_pmids"].split(";") if p],
           "scope_note": d["scope_note"], "reason": d["reason"],
           "decision_source": d["decision_source"], "decided": d["decided"],
           "source_snapshot_sha256": source_sha}
    if ident:
        rec.update({"uniprot_id": d["uniprot_id"], "uniprot_name": ident["uniprot_name"],
                    "verified_organism": ident["organism"], "taxid": ident["taxid"]})
    if d["action"] == "drop":
        rec["status"] = d["drop_status"]
    return rec


def planned_updates(d: Dict[str, str], ident: Optional[Dict], source_sha: str) -> Dict[str, object]:
    """Column -> new value for one decision (the transition table of 9.17)."""
    dec = json.dumps(decision_record(d, ident, source_sha), ensure_ascii=False, sort_keys=True)
    if d["action"] == "restore":
        return {"status": "kept", "protein_role": "aggregator", "role_subtype": None,
                "modifier_roles": None, "uniprot_id": d["uniprot_id"], "gene_name": ident["gene_name"],
                "uniprot_name": ident["uniprot_name"], "resolved_name": d["protein_name"],
                "resolved_organism": ident["organism"],
                "forms_aggregates": 1 if d["evidence_basis"] == "aggregation_text" else None,
                "aggregation_evidence": "curator_review", "evidence_basis": d["evidence_basis"],
                "curator_decision": dec}
    if d["action"] == "drop":
        return {"status": d["drop_status"], "protein_role": None, "role_subtype": None,
                "modifier_roles": None, "uniprot_id": None, "gene_name": None,
                "resolved_name": None, "resolved_organism": None, "aggregation_evidence": None,
                "condensate_evidence": None, "evidence_basis": None, "curator_decision": dec}
    return {"forms_aggregates": None, "aggregation_evidence": "unresolved",
            "evidence_basis": d["evidence_basis"], "curator_decision": dec}


def validate_against_source(src: sqlite3.Connection, decisions: List[Dict[str, str]]) -> None:
    """Addresses unique and current; supporting PMIDs linked to the row."""
    for d in decisions:
        tag = f"decision row_id={d['row_id']}"
        row = src.execute("SELECT protein_name, organism, status FROM proteins WHERE id = ?",
                          (int(d["row_id"]),)).fetchone()
        if row is None:
            raise DecisionError(f"{tag}: no such row")
        # an empty organism in the table stands for NULL; UNIQUE(protein_name,
        # organism) does not hold for NULL organisms, so uniqueness is checked
        org = d["baseline_organism"] or None
        if tuple(row) != (d["protein_name"], org, d["baseline_status"]):
            raise DecisionError(f"{tag}: row is {tuple(row)}, decision expects "
                                f"{(d['protein_name'], org, d['baseline_status'])}")
        n = src.execute("SELECT COUNT(*) FROM proteins WHERE protein_name = ? AND organism IS ? AND status = ?",
                        (d["protein_name"], org, d["baseline_status"])).fetchone()[0]
        if n != 1:
            raise DecisionError(f"{tag}: address matches {n} rows")
        for pm in [p for p in d["support_pmids"].split(";") if p]:
            if not src.execute("SELECT 1 FROM protein_papers WHERE protein_id = ? AND pmid = ?",
                               (int(d["row_id"]), pm)).fetchone():
                raise DecisionError(f"{tag}: PMID {pm} not linked to the row")


def _private_file(destination: Path) -> Path:
    """Reserve an exclusive staging file beside its eventual destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    os.close(fd)
    return Path(name)


def apply(db: Path, out: Path, decisions_path: Path, snapshot_path: Path, cache: Path,
          offline: bool = False, fetch: Callable[[str], bytes] = fetch_uniprot,
          report_path: Optional[Path] = None) -> Dict:
    """Validate everything, build the output privately, publish it; return the report."""
    if out.exists():
        raise DecisionError(f"{out} exists; this script never overwrites")
    if os.path.realpath(db) == os.path.realpath(out):
        raise DecisionError("--db and --out are the same file")
    if report_path is not None:
        # Never let a report overwrite an input, a cache entry or the SQLite
        # being published. lexists also refuses dangling symlinks.
        if os.path.lexists(report_path):
            raise DecisionError(f"{report_path} exists; the report is never overwritten")
        if report_path.resolve() == out.resolve():
            raise DecisionError("--report and --out are the same path")
        if report_path.resolve().is_relative_to(cache.resolve()):
            raise DecisionError("--report must be outside the UniProt cache")
    snapshot = json.loads(snapshot_path.read_text())
    source_sha = sha256(db)
    if source_sha != snapshot["db_sha256"]:
        raise DecisionError(f"source SHA-256 {source_sha} != manifest {snapshot['db_sha256']} "
                            f"({snapshot.get('snapshot')}); rebind the decisions in a new stage")
    decisions = read_decisions(decisions_path)
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        validate_against_source(src, decisions)
        before = {int(d["row_id"]): dict(zip([c[0] for c in src.execute("SELECT * FROM proteins LIMIT 0").description],
                                              src.execute("SELECT * FROM proteins WHERE id = ?",
                                                          (int(d["row_id"]),)).fetchone()))
                  for d in decisions}
    finally:
        src.close()
    usage: Dict[str, str] = {}
    idents = {}
    for d in decisions:
        if d["action"] == "restore":
            idents[d["row_id"]] = check_identity(d, uniprot_entry(d["uniprot_id"], cache, offline, fetch, usage))
    updates = {int(d["row_id"]): planned_updates(d, idents.get(d["row_id"]), source_sha) for d in decisions}

    tmp = _private_file(out)
    report_tmp = None
    published = []
    con = None
    try:
        shutil.copyfile(db, tmp)
        if sha256(tmp) != source_sha:
            raise DecisionError("temporary copy differs from the source")
        con = sqlite3.connect(tmp, isolation_level=None)
        cols = {r[1] for r in con.execute("PRAGMA table_info(proteins)")}
        con.execute("BEGIN")
        for col, decl in NEW_COLUMNS:
            if col not in cols:
                con.execute(f"ALTER TABLE proteins ADD COLUMN {col} {decl}")
        for pid, upd in updates.items():
            sets = ", ".join(f"{c} = ?" for c in upd)
            cur = con.execute(f"UPDATE proteins SET {sets} WHERE id = ?", list(upd.values()) + [pid])
            if cur.rowcount != 1:
                raise DecisionError(f"row {pid}: {cur.rowcount} rows updated")
        # post-checks inside the transaction: every planned value landed
        names = [c[0] for c in con.execute("SELECT * FROM proteins LIMIT 0").description]
        for pid, upd in updates.items():
            row = dict(zip(names, con.execute("SELECT * FROM proteins WHERE id = ?", (pid,)).fetchone()))
            for c, v in upd.items():
                if row[c] != v:
                    raise DecisionError(f"row {pid}: {c} is {row[c]!r}, planned {v!r}")
        status = dict(con.execute("SELECT status = 'kept', COUNT(*) FROM proteins GROUP BY status = 'kept'"))
        con.execute("COMMIT")
        con.close(); con = None
        report = {"source": str(db), "source_sha256": source_sha, "source_sha256_after": sha256(db),
                  "out": str(out), "out_sha256": sha256(tmp), "snapshot": snapshot,
                  "decisions": len(decisions),
                  "by_action": {a: sum(d["action"] == a for d in decisions) for a in sorted(ACTIONS)},
                  "row_ids": sorted(updates), "kept": status.get(1, 0), "dropped": status.get(0, 0),
                  "uniprot": {"checked": len(idents), "unique_accessions": len(usage), "usage": usage},
                  "changes": {pid: {c: {"before": before[pid].get(c), "after": v} for c, v in upd.items()
                                    if before[pid].get(c) != v}
                              for pid, upd in updates.items()}}
        if report["source_sha256_after"] != source_sha:
            raise DecisionError("source changed during the run")
        if report_path is not None:
            report_tmp = _private_file(report_path)
            report_tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
        os.link(tmp, out)           # fails if out appeared meanwhile: never overwrites
        published.append((out, tmp))
        if report_path is not None:
            os.link(report_tmp, report_path)
            published.append((report_path, report_tmp))
    except BaseException:
        # A report publication failure must not leave an apparently successful
        # database. Remove only links created by this attempt, never a rival's.
        for destination, staging in reversed(published):
            if destination.exists() and os.path.samefile(destination, staging):
                destination.unlink()
        raise
    finally:
        if con is not None:
            con.close()
        for staging in (tmp, report_tmp):
            if staging is not None:
                try:
                    staging.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"warning: could not remove private file {staging}: {exc}", file=sys.stderr)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--decisions", required=True, type=Path)
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = apply(a.db, a.out, a.decisions, a.snapshot, a.cache,
                    offline=a.offline, report_path=a.report)
    except (DecisionError, OSError, sqlite3.Error) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(f"applied {rep['decisions']} decisions {rep['by_action']}: kept {rep['kept']} / "
          f"dropped {rep['dropped']}; UniProt {rep['uniprot']['usage']}; out {rep['out_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
