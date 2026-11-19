"""Curator provenance survives side-list serialization and alias unification."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from amylodb.pipeline import _write_modifiers_tsv, run
from amylodb.mining.adapter import split_mined


def _read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _decision(row_id):
    return {"source_row_id": row_id, "decision": "retain_as_condensate_component",
            "support_pmids": ["23437357"], "scope_note": 'Клетки\tMCF7\n"стресс-гранулы"',
            "modifier_status": "possible_unresolved"}


@pytest.mark.parametrize("role,output", [("condensate", "condensate_candidates.tsv"),
                                         ("modifier", "modifiers.tsv")])
def test_side_provenance_round_trip_through_pipeline(tmp_path, role, output):
    item = {"protein_name": "fixture", "protein_role": role,
            "role_subtype": "component" if role == "condensate" else "inhibitor",
            "source_pmids": ["23437357"], "evidence_items": [_decision(1399)]}
    original = copy.deepcopy(item)
    _, side = split_mined([item])
    result = run({}, str(tmp_path), modifiers=side, compute_features=False)
    rows = _read(tmp_path / output)
    assert len(rows) == 1
    assert json.loads(rows[0]["evidence_items"]) == item["evidence_items"]
    assert rows[0]["source_pmids"] == "23437357"
    assert rows[0]["primary_role"] == role
    assert not result.consensus
    assert item == original
    other = "modifiers.tsv" if role == "condensate" else "condensate_candidates.tsv"
    assert _read(tmp_path / other) == []


@pytest.mark.parametrize("extra", [{}, {"evidence_items": None}, {"evidence_items": []}])
def test_absent_decisions_are_json_empty_array(tmp_path, extra):
    path = tmp_path / "side.tsv"
    _write_modifiers_tsv([{"protein_name": "without decision", **extra}], str(path))
    rows = _read(path)
    assert rows[0]["evidence_items"] == "[]"
    assert len(rows[0]) == 13


@pytest.mark.parametrize("reverse", [False, True])
def test_losing_alias_keeps_decision_in_both_side_lists(tmp_path, reverse):
    spec = importlib.util.spec_from_file_location("side_export", Path(__file__).parents[1] / "scripts/export_mined.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    rows = [{"protein_name": "component", "uniprot_id": "FIXTURE", "protein_role": "condensate",
             "role_subtype": "component", "evidence_items": [_decision(1399)]},
            {"protein_name": "modifier", "uniprot_id": "FIXTURE", "protein_role": "modifier",
             "role_subtype": "inhibitor", "evidence_items": [_decision(1400)]}]
    if reverse:
        rows.reverse()
    exporter.unify_roles(rows)
    _, side = split_mined(rows)
    run({}, str(tmp_path), modifiers=side, compute_features=False)
    for filename in ("modifiers.tsv", "condensate_candidates.tsv"):
        emitted = _read(tmp_path / filename)
        assert len(emitted) == 2
        for row in emitted:
            decisions = json.loads(row["evidence_items"])
            assert {x["source_row_id"] for x in decisions} == {1399, 1400}
            assert all(x["scope_note"] == _decision(1399)["scope_note"] for x in decisions)
            assert row["primary_role"] == "modifier"
