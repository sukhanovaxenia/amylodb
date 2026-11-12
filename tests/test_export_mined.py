"""The mining-SQLite -> mined.json exporter (scripts/export_mined.py)."""

import importlib.util
import os
import sqlite3

_HERE = os.path.dirname(__file__)
_SPEC = importlib.util.spec_from_file_location(
    "export_mined", os.path.join(_HERE, "..", "scripts", "export_mined.py"))
export_mined = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_mined)


def _make_mining_db(path):
    c = sqlite3.connect(path)
    c.executescript(
        """
        CREATE TABLE proteins(id INTEGER PRIMARY KEY AUTOINCREMENT, protein_name TEXT,
          gene_name TEXT, uniprot_id TEXT, organism TEXT, forms_aggregates BOOLEAN,
          aggregate_structure TEXT, reversibility TEXT, assembly_mechanism TEXT,
          functional_class TEXT, biological_role TEXT, associated_disease TEXT,
          evidence_strength TEXT, extraction_confidence REAL);
        CREATE TABLE papers(pmid TEXT PRIMARY KEY, title TEXT);
        CREATE TABLE protein_papers(protein_id INTEGER, pmid TEXT);
        CREATE TABLE cofactors(id INTEGER PRIMARY KEY AUTOINCREMENT, protein_id INTEGER,
          cofactor_name TEXT, cofactor_type TEXT);
        CREATE TABLE string_interactions(id INTEGER PRIMARY KEY AUTOINCREMENT,
          protein_id INTEGER, partner_name TEXT);
        """
    )
    c.execute("INSERT INTO proteins(protein_name,uniprot_id,forms_aggregates,"
              "aggregate_structure,functional_class,extraction_confidence) "
              "VALUES(?,?,?,?,?,?)", ("FUS", "P35637", 1, "beta-sheet", "both", 0.8))
    c.executemany("INSERT INTO protein_papers(protein_id,pmid) VALUES(?,?)",
                  [(1, "30000"), (1, "30001")])
    c.executemany("INSERT INTO cofactors(protein_id,cofactor_name,cofactor_type) VALUES(?,?,?)",
                  [(1, "RNA", "assembly"), (1, "HSP70", "disassembly")])
    c.execute("INSERT INTO string_interactions(protein_id,partner_name) VALUES(?,?)", (1, "G3BP1"))
    c.commit()
    c.close()


def test_export_gathers_pmids_cofactors_partners(tmp_path):
    db = tmp_path / "mining.db"
    _make_mining_db(str(db))
    items = export_mined.export(str(db))
    assert len(items) == 1
    fus = items[0]
    assert set(fus["source_pmids"]) == {"30000", "30001"}
    assert fus["assembly_cofactors"] == ["RNA"]
    assert fus["disassembly_cofactors"] == ["HSP70"]
    assert fus["string_partners"] == ["G3BP1"]
    assert fus["forms_aggregates"] is True


def test_export_round_trips_through_adapter(tmp_path):
    from amylodb.mining.adapter import mined_batch_to_records
    from amylodb.provenance import PrionClass, Provenance

    db = tmp_path / "mining.db"
    _make_mining_db(str(db))
    recs = mined_batch_to_records(export_mined.export(str(db)))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.provenance == Provenance.MINED
    assert rec.prion_class == PrionClass.PRION_LIKE_DOMAIN
    assert set(rec.source_pmids) == {"30000", "30001"}
    assert rec.string_partners == ["G3BP1"]


# --- model-based fields: per-paper claims, voting, methods ---------------------------

def test_resolve_reversibility_rules():
    r = export_mined.resolve_reversibility
    assert r({"irreversible": 3, "reversible": 1}, "unknown") == "irreversible"     # 3/4 >= 2/3
    assert r({"irreversible": 1, "reversible": 1}, "unknown") == "conditionally_reversible"
    assert r({}, "reversible") == "reversible"                                       # falls back
    assert r({"unknown": 4}, None) == "unknown"


def test_export_with_protein_claims(tmp_path):
    import json as _json
    db = str(tmp_path / "m.db")
    _make_mining_db(db)
    c = sqlite3.connect(db)
    c.executescript("""
        ALTER TABLE proteins ADD COLUMN undergoes_llps TEXT;
        ALTER TABLE proteins ADD COLUMN reported_method_tier INTEGER;
        CREATE TABLE experimental_methods(id INTEGER PRIMARY KEY, protein_id INTEGER, pmid TEXT,
          method_name TEXT, method_category TEXT);
        CREATE TABLE protein_claims(id INTEGER PRIMARY KEY, protein_id INTEGER, pmid TEXT,
          surface_form TEXT, extraction_method TEXT, forms_aggregates BOOLEAN, aggregate_structure TEXT,
          reversibility TEXT, functional_class TEXT, undergoes_llps TEXT, reported_method_tier INTEGER,
          extraction_confidence REAL, claim_scores TEXT);
    """)
    c.execute("UPDATE proteins SET reported_method_tier = 5 WHERE id = 1")
    c.executemany("INSERT INTO experimental_methods(protein_id,pmid,method_name) VALUES(?,?,?)",
                  [(1, "30000", "Cryo-EM"), (1, "30001", "ThT binding"), (1, "30001", "Cryo-EM")])
    c.executemany("INSERT INTO protein_claims(protein_id,pmid,reversibility,undergoes_llps,claim_scores) "
                  "VALUES(?,?,?,?,?)",
                  [(1, "30000", "reversible", "yes", _json.dumps({"llps": 0.91, "reversible": 0.8})),
                   (1, "30001", "irreversible", "unknown", _json.dumps({"llps": 0.2, "irreversible": 0.88}))])
    c.commit(); c.close()

    fus = export_mined.export(db)[0]
    assert fus["reversibility_votes"] == {"reversible": 1, "irreversible": 1}
    assert fus["reversibility"] == "conditionally_reversible"
    assert fus["undergoes_llps"] == "yes"
    assert fus["claim_scores"] == {"llps": 0.91, "reversible": 0.8, "irreversible": 0.88}
    assert fus["experimental_methods"] == ["Cryo-EM", "ThT binding"]
    assert fus["reported_method_tier"] == 5

    from amylodb.mining.adapter import mined_to_record
    rec = mined_to_record(fus)
    assert rec.undergoes_llps is True and rec.reported_method_tier == 5
    assert rec.reported_methods == ["Cryo-EM", "ThT binding"]
    assert rec.reversibility == "conditionally_reversible"
    assert int(rec.evidence_tier) == 1      # reported evidence does not raise the tier


def test_adapter_prnp_gene_is_transmissible():
    from amylodb.mining.adapter import mined_to_record
    from amylodb.provenance import PrionClass
    rec = mined_to_record({"protein_name": "prion", "gene_name": "PRNP", "uniprot_id": "P04156"})
    assert rec.prion_class == PrionClass.TRANSMISSIBLE_PRION


def test_reconcile_merges_model_fields():
    from amylodb.mining.adapter import mined_to_record
    from amylodb.reconcile import reconcile
    a = mined_to_record({"protein_name": "FUS", "uniprot_id": "P35637", "undergoes_llps": "yes",
                         "reversibility_votes": {"reversible": 2}, "claim_scores": {"llps": 0.9},
                         "experimental_methods": ["FRAP"], "reported_method_tier": 3, "pmid": "1"})
    b = mined_to_record({"protein_name": "FUS", "uniprot_id": "P35637",
                         "reversibility_votes": {"irreversible": 1}, "claim_scores": {"llps": 0.4, "beta": 0.8},
                         "experimental_methods": ["Cryo-EM"], "reported_method_tier": 5, "pmid": "2"})
    rec = reconcile([a, b]).consensus[0]
    assert rec.undergoes_llps is True
    assert rec.reversibility_votes == {"reversible": 2, "irreversible": 1}
    assert rec.claim_scores == {"llps": 0.9, "beta": 0.8}
    assert rec.reported_methods == ["FRAP", "Cryo-EM"] or set(rec.reported_methods) == {"FRAP", "Cryo-EM"}
    assert rec.reported_method_tier == 5
