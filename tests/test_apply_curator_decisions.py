"""apply_curator_decisions (stage amylodb-005b) on a fixture database.

Every refusal must leave the source byte-identical, publish nothing new and
keep an existing --out unchanged; no temporary file may remain.
"""

import csv
import hashlib
import importlib.util
import json
import os
import sqlite3
from pathlib import Path

import pytest

from amylodb.mining.database.storage import AmyloidDatabase

_spec = importlib.util.spec_from_file_location(
    "apply_curator_decisions",
    Path(__file__).resolve().parents[1] / "scripts/apply_curator_decisions.py")
acd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acd)

ENTRY = {"primaryAccession": "P04279", "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
         "proteinDescription": {"recommendedName": {"fullName": {"value": "Semenogelin-1"}}},
         "genes": [{"geneName": {"value": "SEMG1"}}]}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "src.db"
    AmyloidDatabase(str(db))
    con = sqlite3.connect(db)
    for pid, name, org, status, role, acc, tier in [
            (1, "semenogelin i", "Homo sapiens", "dropped:regulator_only", None, None, None),
            (2, "v8 protease", "Staphylococcus aureus", "kept", "aggregator", "P0C1U8", "model_only"),
            (3, "protein desmin", "Homo sapiens", "kept", "aggregator", "P17661", "model_only"),
            # NULL organisms escape UNIQUE(protein_name, organism): an ambiguous address
            (4, "csga", None, "dropped:regulator_only", None, None, None),
            (5, "csga", None, "dropped:regulator_only", None, None, None)]:
        con.execute("INSERT INTO proteins (id, protein_name, organism, status, protein_role, uniprot_id, "
                    "aggregation_evidence, forms_aggregates) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (pid, name, org, status, role, acc, tier))
        con.execute("INSERT INTO papers (pmid, title) VALUES (?, 't')", (f"10{pid}",))
        con.execute("INSERT INTO protein_papers (protein_id, pmid) VALUES (?, ?)", (pid, f"10{pid}"))
    con.commit(); con.close()
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"snapshot": "fixture", "db_sha256": _sha(db)}))
    cache = tmp_path / "cache"
    return {"db": db, "snap": snap, "cache": cache, "tmp": tmp_path, "out": tmp_path / "out" / "v3f.db"}


def _row(**kw):
    base = {c: "" for c in acd.COLUMNS}
    base.update({"decision_source": "amylodb-004", "decided": "2026-11-16", "reason": "r"})
    base.update(kw)
    return base


def _decisions(env, rows, name="dec.tsv"):
    p = env["tmp"] / name
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=acd.COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return p


RESTORE = _row(row_id="1", protein_name="semenogelin i", baseline_organism="Homo sapiens",
               baseline_status="dropped:regulator_only", action="restore", role="aggregator",
               aggregation_evidence="curator_review", evidence_basis="aggregation_text",
               uniprot_id="P04279", expected_name="Semenogelin-1", expected_gene="SEMG1",
               expected_taxid="9606", verified_organism="Homo sapiens", support_pmids="101",
               scope_note="peptide GSFSIQYTYHV")
DROP = _row(row_id="2", protein_name="v8 protease", baseline_organism="Staphylococcus aureus",
            baseline_status="kept", action="drop", drop_status="dropped:curator_offtopic", support_pmids="102")
UNRES = _row(row_id="3", protein_name="protein desmin", baseline_organism="Homo sapiens", baseline_status="kept",
             action="unresolved", aggregation_evidence="unresolved", evidence_basis="unresolved_attribution")


def _fetch(entry=ENTRY):
    calls = []
    def f(acc):
        calls.append(acc)
        return json.dumps(entry).encode()
    f.calls = calls
    return f


def _run(env, dec, out=None, offline=False, fetch=None, db=None):
    return acd.apply(db or env["db"], out or env["out"], dec, env["snap"], env["cache"],
                     offline=offline, fetch=fetch or _fetch())


def test_applies_the_transition_table(env):
    rep = _run(env, _decisions(env, [RESTORE, DROP, UNRES]))
    assert rep["by_action"] == {"drop": 1, "restore": 1, "unresolved": 1}
    con = sqlite3.connect(env["out"]); con.row_factory = sqlite3.Row
    r = {row["id"]: dict(row) for row in con.execute("SELECT * FROM proteins")}
    assert (r[1]["status"], r[1]["uniprot_id"], r[1]["gene_name"], r[1]["forms_aggregates"],
            r[1]["aggregation_evidence"], r[1]["evidence_basis"]) == \
        ("kept", "P04279", "SEMG1", 1, "curator_review", "aggregation_text")
    item = json.loads(r[1]["curator_decision"])
    assert item["scope_note"] == "peptide GSFSIQYTYHV" and item["support_pmids"] == ["101"]
    assert (r[2]["status"], r[2]["uniprot_id"], r[2]["aggregation_evidence"]) == \
        ("dropped:curator_offtopic", None, None)
    assert (r[3]["aggregation_evidence"], r[3]["forms_aggregates"], r[3]["status"]) == ("unresolved", None, "kept")
    assert r[4]["curator_decision"] is None and r[4]["evidence_basis"] is None
    assert not list(env["out"].parent.glob(".*tmp*"))
    # the cache is reused offline
    out2 = env["tmp"] / "again" / "v3f.db"
    rep2 = _run(env, _decisions(env, [RESTORE]), out=out2, offline=True, fetch=_fetch({}))
    assert rep2["uniprot"]["usage"] == {"P04279": "cache"}


def _refused(env, dec, **kw):
    src = _sha(env["db"])
    out = kw.pop("out", env["out"])
    before = out.read_bytes() if out.exists() else None
    with pytest.raises(acd.DecisionError):
        _run(env, dec, out=out, **kw)
    assert _sha(env["db"]) == src
    assert (out.read_bytes() if out.exists() else None) == before
    assert not list(out.parent.glob(f".{out.name}.tmp-*")) if out.parent.exists() else True


def test_01_foreign_snapshot_hash(env):
    env["snap"].write_text(json.dumps({"snapshot": "x", "db_sha256": "0" * 64}))
    _refused(env, _decisions(env, [RESTORE]))


def test_02_stale_baseline_status(env):
    _refused(env, _decisions(env, [dict(RESTORE, baseline_status="dropped:off_topic")]))


def test_03_unknown_protein_name(env):
    _refused(env, _decisions(env, [dict(RESTORE, protein_name="semenogelin ii")]))


def test_04_duplicate_decision(env):
    _refused(env, _decisions(env, [RESTORE, RESTORE]))


def test_05_unlinked_pmid(env):
    _refused(env, _decisions(env, [dict(RESTORE, support_pmids="101;999")]))


def test_06_uniprot_name_mismatch(env):
    _refused(env, _decisions(env, [dict(RESTORE, expected_name="Semenogelin-2")]))


def test_07_error_only_in_the_last_row(env):
    _refused(env, _decisions(env, [RESTORE, DROP, dict(UNRES, reason="")]))


def test_08_offline_without_cache(env):
    _refused(env, _decisions(env, [RESTORE]), offline=True)


def test_09_db_is_out(env):
    _refused(env, _decisions(env, [RESTORE]), out=env["db"])


def test_10_existing_out_is_kept(env):
    env["out"].parent.mkdir(parents=True)
    env["out"].write_bytes(b"earlier result")
    _refused(env, _decisions(env, [RESTORE]))
    assert env["out"].read_bytes() == b"earlier result"


def test_11_ambiguous_address(env):
    amb = _row(row_id="4", protein_name="csga", baseline_organism="", baseline_status="dropped:regulator_only",
               action="drop", drop_status="dropped:curator_offtopic", support_pmids="104")
    _refused(env, _decisions(env, [amb]))


def test_12_taxid_mismatch_with_matching_name(env):
    mouse = dict(ENTRY, organism={"scientificName": "Homo sapiens", "taxonId": 10090})
    _refused(env, _decisions(env, [RESTORE]), fetch=_fetch(mouse))


def test_13_reapplying_to_the_output(env):
    dec = _decisions(env, [RESTORE, DROP, UNRES])
    _run(env, dec)
    _refused(env, dec, out=env["tmp"] / "second.db", db=env["out"])


def test_restore_without_scope_note_is_refused(env):
    _refused(env, _decisions(env, [dict(RESTORE, scope_note="")]))


@pytest.mark.parametrize("alias", ["01", "+1", " 1 "])
def test_numeric_duplicate_addresses_are_refused(env, alias):
    _refused(env, _decisions(env, [RESTORE, dict(RESTORE, row_id=alias, evidence_basis="LCR_text")]))


@pytest.mark.parametrize("value", ["bad", "0", "-1"])
def test_row_id_must_be_a_positive_integer(env, value):
    _refused(env, _decisions(env, [dict(RESTORE, row_id=value)]))


def _cached_restore(env):
    body = json.dumps(ENTRY).encode()
    env["cache"].mkdir()
    (env["cache"] / "P04279.json").write_text(json.dumps({
        "response": body.decode(), "sha256": hashlib.sha256(body).hexdigest()}))


@pytest.mark.parametrize("target", ["source", "output", "decisions", "snapshot", "cache",
                                     "existing", "symlink", "hardlink"])
def test_cli_report_cannot_replace_inputs_or_the_database(env, target):
    dec = _decisions(env, [RESTORE])
    _cached_restore(env)
    paths = {"source": env["db"], "output": env["out"], "decisions": dec,
             "snapshot": env["snap"], "cache": env["cache"] / "P04279.json"}
    report = paths.get(target, env["tmp"] / "report.json")
    if target == "existing":
        report.write_bytes(b"earlier report")
    elif target == "symlink":
        report.symlink_to(env["db"])
    elif target == "hardlink":
        os.link(env["db"], report)
    protected = [env["db"], dec, env["snap"], env["cache"] / "P04279.json"]
    if report.exists():
        protected.append(report)
    before = {p: p.read_bytes() for p in protected}
    rc = acd.main(["--db", str(env["db"]), "--out", str(env["out"]),
                   "--decisions", str(dec), "--snapshot", str(env["snap"]),
                   "--cache", str(env["cache"]), "--offline", "--report", str(report)])
    assert rc == 1
    assert {p: p.read_bytes() for p in protected} == before
    assert not env["out"].exists()


def test_report_and_database_are_published_together(env):
    report = env["tmp"] / "reports" / "apply.json"
    rep = acd.apply(env["db"], env["out"], _decisions(env, [RESTORE]), env["snap"],
                    env["cache"], fetch=_fetch(), report_path=report)
    saved = json.loads(report.read_text())
    assert saved["out_sha256"] == rep["out_sha256"] == _sha(env["out"])
    assert saved["source_sha256"] == saved["source_sha256_after"] == _sha(env["db"])
    assert saved["decisions"] == 1
    assert not list(env["tmp"].rglob(".*.tmp-*"))


def test_report_creation_race_preserves_other_report_and_rolls_back_output(env, monkeypatch):
    report = env["tmp"] / "report.json"
    original_link = acd.os.link
    def racing_link(src, dst):
        if Path(dst) == report:
            report.write_bytes(b"another writer's report")
        return original_link(src, dst)
    monkeypatch.setattr(acd.os, "link", racing_link)
    before = _sha(env["db"])
    with pytest.raises(FileExistsError):
        acd.apply(env["db"], env["out"], _decisions(env, [RESTORE]), env["snap"],
                  env["cache"], fetch=_fetch(), report_path=report)
    assert report.read_bytes() == b"another writer's report"
    assert not env["out"].exists()
    assert _sha(env["db"]) == before
    assert not list(env["tmp"].rglob(".*.tmp-*"))


def test_existing_temporary_name_is_not_owned_by_this_run(env):
    env["out"].parent.mkdir(parents=True)
    previous = env["out"].parent / f".{env['out'].name}.tmp-{os.getpid()}"
    previous.write_bytes(b"earlier temporary data")
    _run(env, _decisions(env, [RESTORE]))
    assert previous.read_bytes() == b"earlier temporary data"


def test_source_change_is_detected_before_publishing(env, monkeypatch):
    original_copy = acd.shutil.copyfile
    def source_changed_after_copy(src, dst):
        result = original_copy(src, dst)
        # Simulate another writer, solely on this test's source fixture.
        with Path(src).open("ab") as f:
            f.write(b"external change")
        return result
    monkeypatch.setattr(acd.shutil, "copyfile", source_changed_after_copy)
    with pytest.raises(acd.DecisionError, match="source changed"):
        _run(env, _decisions(env, [RESTORE]))
    assert not env["out"].exists()
    assert not list(env["out"].parent.glob(".*tmp*"))
