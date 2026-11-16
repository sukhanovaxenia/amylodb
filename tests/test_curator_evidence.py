"""Curator-decision supports (stage amylodb-005a): labels, weights, provenance.

Decisions of 2026-11-16 (docs/changes_from_source.md 9.16):
  * curator_review / aggregation_text -> amyloid, full mined weight (0.25)
  * curator_review / LCR_text         -> candidate, is_amyloid None, half (0.125)
  * unresolved                        -> candidate, is_amyloid None, weight 0
  * curated, text, model_only, ""     -> label from forms_aggregates, as before
  * protein support: curated > curator/aggregation > text > model_only
    > curator/LCR > unresolved; the decisions of every row are kept.
"""

import csv
import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from amylodb.curated.models import AmyloidEntry
from amylodb.mining.adapter import mined_batch_to_records, mined_to_record
from amylodb.pipeline import run
from amylodb.provenance import (EvidenceTier, Provenance, RecordLevel, aggregation_rank,
                                aggregation_weight)
from amylodb.reconcile import reconcile
from amylodb.schema import AmyloidRecord

_spec = importlib.util.spec_from_file_location(
    "export_mined", Path(__file__).resolve().parents[1] / "scripts/export_mined.py")
export_mined = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_mined)


def _item(row, decision, basis, pmids, scope):
    return {"source_row_id": row, "decision": decision, "evidence_basis": basis,
            "support_pmids": pmids, "scope_note": scope, "decision_source": "amylodb-004"}


SEMG1 = _item(602, "restore", "aggregation_text", ["25408475"], "peptide GSFSIQYTYHV of semenogelin-1")
HNRNPDL = _item(1040, "restore", "LCR_text", ["30279180"], "low-complexity region; polymer data concern hnRNPA2")
DESMIN = _item(465, "unresolved", "unresolved_attribution", [], "desmin vs associated mutant proteins")


def _mined(name, uni, tier="", basis="", forms=True, items=None, pmids=("1",)):
    return {"protein_name": name, "uniprot_id": uni, "protein_role": "aggregator",
            "aggregation_evidence": tier, "evidence_basis": basis,
            "evidence_items": list(items or []), "forms_aggregates": forms,
            "source_pmids": list(pmids), "pmid": pmids[0], "modifier_roles": [],
            "role_subtype": "", "condensate_evidence": False}


# --- labels and weights ---------------------------------------------------

@pytest.mark.parametrize("tier", ["curated", "text", "model_only", ""])
@pytest.mark.parametrize("forms", [True, False])
def test_older_tiers_keep_the_forms_aggregates_label(tier, forms):
    rec = mined_to_record(_mined("x", "P1", tier, forms=forms))
    assert rec.is_amyloid is forms
    assert rec.experimental_label == ("amyloid" if forms else "non-amyloid")
    assert rec.evidence_weight == (0.125 if tier == "model_only" else 0.25)


@pytest.mark.parametrize("tier,basis,forms,label,amyloid,weight", [
    ("curator_review", "aggregation_text", False, "amyloid", True, 0.25),
    ("curator_review", "LCR_text", True, "candidate", None, 0.125),
    ("curator_review", "LCR_text", None, "candidate", None, 0.125),
    ("unresolved", "unresolved_attribution", True, "candidate", None, 0.0),
])
def test_curator_states_take_the_label_from_the_support(tier, basis, forms, label, amyloid, weight):
    rec = mined_to_record(_mined("x", "P1", tier, basis, forms=forms))
    assert rec.experimental_label == label
    assert rec.is_amyloid is amyloid
    assert rec.evidence_weight == weight


def test_unknown_curator_basis_fails_loudly():
    with pytest.raises(ValueError):
        mined_to_record(_mined("x", "P1", "curator_review", "hunch"))


def test_rank_order():
    order = [("curated", ""), ("curator_review", "aggregation_text"), ("text", ""),
             ("model_only", ""), ("curator_review", "LCR_text"), ("unresolved", "x"), ("", "")]
    ranks = [aggregation_rank(*o) for o in order]
    assert ranks == sorted(ranks, reverse=True) and len(set(ranks)) == len(ranks)
    assert aggregation_weight("unresolved") == 0.0


# --- weight sentinel ------------------------------------------------------

def test_weight_unset_takes_the_tier_weight_explicit_zero_is_kept():
    assert AmyloidRecord(evidence_tier=EvidenceTier.TEXT_MINED).evidence_weight == 0.25
    zero = AmyloidRecord(evidence_tier=EvidenceTier.TEXT_MINED, evidence_weight=0.0)
    assert zero.evidence_weight == 0.0
    assert dataclasses.replace(zero).evidence_weight == 0.0
    assert AmyloidRecord(evidence_tier=EvidenceTier.KINETIC, evidence_weight=0.7).evidence_weight == 0.7


def test_lifting_a_curated_entry_still_takes_the_tier_weight():
    entry = AmyloidEntry(source_db="Atlas", protein_name="tau", is_amyloid=False,
                         experimental_label="non-amyloid", evidence_type="structural")
    rec = AmyloidRecord.from_curated(entry)
    assert rec.evidence_weight == 3.0 and rec.is_amyloid is False


# --- alias unification (export_mined.unify_roles) -------------------------

@pytest.mark.parametrize("reverse", [False, True])
def test_hnrnpdl_pair_text_wins_lcr_decision_kept(reverse):
    rows = [_mined("hnrpdl", "O14979", "text"),
            _mined("hnrnpdl", "O14979", "curator_review", "LCR_text", items=[HNRNPDL])]
    rows = rows[::-1] if reverse else rows
    export_mined.unify_roles(rows)
    for r in rows:
        assert (r["aggregation_evidence"], r["evidence_basis"]) == ("text", "")
        assert r["evidence_items"] == [HNRNPDL]
    recs = mined_batch_to_records(rows)
    assert {(r.experimental_label, r.evidence_weight) for r in recs} == {("amyloid", 0.25)}


@pytest.mark.parametrize("reverse", [False, True])
def test_desmin_pair_text_wins_unresolved_decision_kept(reverse):
    rows = [_mined("desmin", "P17661", "text"),
            _mined("protein desmin", "P17661", "unresolved", "unresolved_attribution", items=[DESMIN])]
    rows = rows[::-1] if reverse else rows
    export_mined.unify_roles(rows)
    assert {(r["aggregation_evidence"], r["evidence_basis"]) for r in rows} == {("text", "")}
    assert all(r["evidence_items"] == [DESMIN] for r in rows)


@pytest.mark.parametrize("reverse", [False, True])
def test_aggregation_text_outranks_lcr_text_both_decisions_kept(reverse):
    a = _item(1, "restore", "aggregation_text", ["11"], "full length")
    rows = [_mined("a", "P9", "curator_review", "aggregation_text", items=[a]),
            _mined("b", "P9", "curator_review", "LCR_text", items=[HNRNPDL])]
    rows = rows[::-1] if reverse else rows
    export_mined.unify_roles(rows)
    assert {(r["aggregation_evidence"], r["evidence_basis"]) for r in rows} == \
        {("curator_review", "aggregation_text")}
    assert rows[0]["evidence_items"] == sorted([a, HNRNPDL], key=lambda i: str(i["source_row_id"]))
    # the same through the consensus, without unify_roles
    res = reconcile(mined_batch_to_records(
        [_mined("a", "P8", "curator_review", "aggregation_text", items=[a]),
         _mined("b", "P8", "curator_review", "LCR_text", items=[HNRNPDL])][::-1 if reverse else 1]))
    rec = res.consensus[0]
    assert (rec.aggregation_evidence, rec.evidence_basis, rec.experimental_label) == \
        ("curator_review", "aggregation_text", "amyloid")
    assert len(rec.evidence_items) == 2


# --- consensus: decisions survive a lost vote and an unresolved conflict ---

def _curated_protein(uni):
    return AmyloidRecord(source_db="Atlas", protein_name="desmin", uniprot_id=uni,
                         experimental_label="amyloid", is_amyloid=True,
                         record_level=RecordLevel.PROTEIN, evidence_tier=EvidenceTier.STRUCTURAL,
                         provenance=Provenance.CURATED, source_pmids=["9"])


@pytest.mark.parametrize("reverse", [False, True])
def test_curated_plus_mined_candidate_keeps_the_candidate_decision(reverse):
    cand = mined_to_record(_mined("protein desmin", "P17661", "unresolved",
                                  "unresolved_attribution", items=[DESMIN]))
    recs = [_curated_protein("P17661"), cand]
    res = reconcile(recs[::-1] if reverse else recs)
    assert res.conflicts == [] and len(res.consensus) == 1
    rec = res.consensus[0]
    assert rec.experimental_label == "amyloid" and rec.evidence_tier == EvidenceTier.STRUCTURAL
    assert rec.evidence_items == [DESMIN]


def test_unresolved_conflict_keeps_the_decisions_of_the_group():
    a = mined_to_record(_mined("x", "P5", "text", forms=True, items=[SEMG1]))
    b = mined_to_record(_mined("x2", "P5", "text", forms=False))
    res = reconcile([a, b])
    assert res.consensus == [] and len(res.conflicts) == 1
    assert res.conflicts[0]["evidence_items"] == [SEMG1]


# --- serialisation: records.tsv and graph.json ----------------------------

def test_end_to_end_outputs_keep_basis_pmids_and_scope(tmp_path):
    rows = [_mined("human semenogelin i", "Q02383", "curator_review", "aggregation_text",
                   items=[SEMG1], pmids=("25408475",)),
            _mined("srrd", "Q9UH36", "curator_review", "LCR_text",
                   items=[_item(1139, "restore", "LCR_text", ["38315730"], "N-terminal LCD")]),
            _mined("protein desmin", "P17661", "unresolved", "unresolved_attribution", items=[DESMIN]),
            _mined("old text", "P00001", "text", forms=False)]
    export_mined.unify_roles(rows)
    run({}, str(tmp_path), mined=mined_batch_to_records(rows), compute_features=False)
    recs = {r["uniprot_id"]: r for r in csv.DictReader((tmp_path / "records.tsv").open(), delimiter="\t")}
    semg = recs["Q02383"]
    assert (semg["experimental_label"], semg["is_amyloid"], semg["evidence_basis"]) == \
        ("amyloid", "True", "aggregation_text")
    assert json.loads(semg["evidence_items"]) == [SEMG1]
    srrd, des, old = recs["Q9UH36"], recs["P17661"], recs["P00001"]
    assert (srrd["experimental_label"], srrd["is_amyloid"], srrd["evidence_weight"]) == ("candidate", "", "0.125")
    assert (des["experimental_label"], des["is_amyloid"], des["evidence_weight"]) == ("candidate", "", "0.000")
    assert json.loads(des["evidence_items"]) == [DESMIN]
    assert (old["experimental_label"], old["is_amyloid"], old["evidence_items"]) == ("non-amyloid", "False", "")
    nodes = {n["uniprot_id"]: n for n in json.loads((tmp_path / "graph.json").read_text())["nodes"]
             if n.get("level") == "protein" and n.get("provenance") != "derived"}
    assert nodes["Q9UH36"]["is_amyloid"] is None and nodes["Q9UH36"]["experimental_label"] == "candidate"
    assert nodes["P00001"]["is_amyloid"] is False
    assert nodes["Q02383"]["evidence_items"] == [SEMG1]
    assert nodes["Q02383"]["evidence_items"][0]["scope_note"].startswith("peptide")


# --- migration and export on a fixture database (never on v3e) ------------

def test_migration_adds_columns_and_export_reads_the_decision(tmp_path):
    import sqlite3
    from amylodb.mining.database.storage import AmyloidDatabase
    db = tmp_path / "fixture.db"
    AmyloidDatabase(str(db))
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(proteins)")}
    assert {"evidence_basis", "curator_decision"} <= cols
    con.execute("INSERT INTO proteins (protein_name, organism, status, protein_role, uniprot_id, "
                "forms_aggregates, aggregation_evidence, evidence_basis, curator_decision) "
                "VALUES (?, ?, 'kept', 'aggregator', 'Q02383', 1, 'curator_review', "
                "'aggregation_text', ?)", ("human semenogelin i", "Homo sapiens", json.dumps(SEMG1)))
    con.execute("INSERT INTO proteins (protein_name, organism, status, protein_role, uniprot_id, "
                "forms_aggregates, aggregation_evidence) VALUES ('tau', 'Homo sapiens', 'kept', "
                "'aggregator', 'P10636', 1, 'curated')")
    con.commit(); con.close()
    out = {r["uniprot_id"]: r for r in export_mined.export(str(db))}
    assert out["Q02383"]["evidence_items"] == [SEMG1]
    assert out["Q02383"]["evidence_basis"] == "aggregation_text"
    assert out["P10636"]["evidence_items"] == [] and out["P10636"]["evidence_basis"] == ""
