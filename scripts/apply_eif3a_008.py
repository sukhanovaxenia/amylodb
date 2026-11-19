"""Create a v3g snapshot from the reviewed v3f and verify the eIF3a build outputs."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

SOURCE = Path("data/mining_v3/amyloid_aggregation.v3f.db")
DIGEST = "49ee95a24ee40f9938aac98d1a6104577e5b04c5861eaff9c0362ee36244fd4b"
ACCESSION = "Q14152"
NAME = "Eukaryotic translation initiation factor 3 subunit A"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def create(output):
    decision = dict(source_row_id=1399, decision="retain_as_condensate_component",
                    component="stress granule", modifier_status="possible_unresolved",
                    amyloid_evidence="not_established", pmids=["23437357"],
                    correction_pmid="26914026", curator="Ksenia", decision_date="2026-11-17",
                    identity_source="UniProtKB Q14152 entry version 228", taxid=9606)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        copy = Path(staging) / "snapshot.db"
        shutil.copyfile(SOURCE, copy)
        require(hashlib.sha256(copy.read_bytes()).hexdigest() == DIGEST, "Unexpected v3f")
        with closing(sqlite3.connect(copy)) as db, db:
            row = db.execute("SELECT protein_name, organism, uniprot_name, status FROM proteins WHERE id=1399").fetchone()
            require(row == ("eukaryotic initiation factor 3a", "Homo sapiens", NAME,
                            "dropped:no_aggregation_mention"), "Identity/status mismatch")
            pmids = db.execute("SELECT pmid FROM protein_papers WHERE protein_id=1399").fetchall()
            require([str(p[0]) for p in pmids] == ["23437357"], "Evidence mismatch")
            db.execute("""UPDATE proteins SET status='kept', uniprot_id=?, gene_name='EIF3A',
                       resolved_name=?, resolved_organism='Homo sapiens', protein_role='condensate',
                       forms_aggregates=0, condensate_evidence=1, role_subtype=NULL,
                       modifier_roles='[]', aggregation_evidence=NULL, curator_decision=?
                       WHERE id=1399""", (ACCESSION, NAME, json.dumps(decision)))
            require(db.total_changes == 1, "Unexpected change count")
        os.link(copy, output)  # Publish atomically and refuse an existing output.


def verify(mined, build):
    rows = json.loads(mined.read_text())
    hits = [r for r in rows if r["uniprot_id"] == ACCESSION]
    require(len(rows) == 1470 and len(hits) == 1, "Unexpected export population")
    hit = hits[0]
    require(hit["protein_role"] == "condensate" and not hit["forms_aggregates"]
            and not hit["modifier_roles"], "Unexpected JSON role")
    for filename in ("condensate_candidates.tsv", "modifiers.tsv", "records.tsv"):
        with (build / filename).open() as f:
            matches = [r for r in csv.DictReader(f, delimiter="\t") if r["uniprot_id"] == ACCESSION]
        if filename == "condensate_candidates.tsv":
            require(len(matches) == 1, "Expected one condensate row")
            row = matches[0]
            items = json.loads(row["evidence_items"])
            require(row["condensate_role"] == "component" and not row["modifier_roles"]
                    and items == hit["evidence_items"] and len(items) == 1
                    and items[0]["source_row_id"] == 1399
                    and items[0]["modifier_status"] == "possible_unresolved"
                    and items[0]["correction_pmid"] == "26914026"
                    and row["source_pmids"] == "23437357", "Missing role or provenance")
        else:
            require(not matches, f"Unexpected Q14152 in {filename}")
    print("Verified 1470 exported rows and Q14152 across all three TSVs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("apply").add_argument("output", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("mined", type=Path)
    check.add_argument("build", type=Path)
    args = parser.parse_args()
    if args.command == "apply":
        create(args.output)
    else:
        verify(args.mined, args.build)
