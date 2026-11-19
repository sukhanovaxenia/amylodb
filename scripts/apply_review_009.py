"""Record eight claim-level non-restoration decisions on a copy of v3g."""
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

IDS = {77, 78, 347, 591, 677, 822, 857, 1734}
V3G_SHA = "3522cb8539a9fbae83ee85e930d96ee243e8171592aca224d2ef240f87da585c"


def require(value, message):
    if not value:
        raise ValueError(message)


def read_sheet(path, expected):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    indexed = {int(row["id"]): row for row in rows}
    require(len(rows) == len(expected) and set(indexed) == expected, f"Unexpected IDs: {path}")
    return indexed


def apply(db, master, review, out):
    full = read_sheet(master, IDS | {1399})
    selected = read_sheet(review, IDS)
    require(full[1399]["curator_decision"] == "retain_as_condensate_component", "Unexpected 1399")
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "snapshot.db"
        shutil.copyfile(db, staged)
        require(hashlib.sha256(staged.read_bytes()).hexdigest() == V3G_SHA, "Unexpected v3g")
        with closing(sqlite3.connect(staged)) as conn, conn:
            for pid, row in selected.items():
                require(row == full[pid], f"Review and master disagree for {pid}")
                expected = ("do_not_restore_mixed_entity_taxon_claim" if pid == 1734
                            else "do_not_restore_from_this_claim")
                require(row["curator_decision"] == expected and row["curator_reason"].strip()
                        and row["curator_source"].strip(), f"Unexpected decision for {pid}")
                pmids = [str(p[0]) for p in conn.execute(
                    "SELECT pmid FROM protein_papers WHERE protein_id=?", (pid,))]
                require(pmids == [row["linked_pmids"]], f"PMID mismatch for {pid}")
                item = {k: v for k, v in row.items() if k.startswith(("curator_", "context_"))
                        or k == "assembly_context"}
                item.update(source_row_id=pid, pmids=pmids, decision=expected)
                change = conn.execute("""UPDATE proteins SET curator_decision=?
                    WHERE id=? AND protein_name=? AND status=? AND status LIKE 'dropped:%'
                    AND COALESCE(curator_decision,'')=''""",
                    (json.dumps(item, ensure_ascii=False, sort_keys=True), pid, row["name"], row["status"]))
                require(change.rowcount == 1, f"Unexpected source row {pid}")
            require(conn.total_changes == 8, "Unexpected update count")
        os.link(staged, out)
    print(f"Saved {out}: 8 decisions, 0 status changes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "master", "review", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    apply(**vars(parser.parse_args()))
