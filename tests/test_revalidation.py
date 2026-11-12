"""Precision rules added for re-validation of an existing mining DB.

Covers name cleaning, the non-protein lists, abbreviation expansion, the
UniProt match rules, role assignment, the resolution gates, the
`revalidate_mining.py` script on a synthetic DB with a stub lookup, the
export status/role filter, the modifier split and modifiers.tsv, and the
manual-review scoring in `audit_mining.py`.
"""

import csv
import importlib.util
import json
import os
import sqlite3

import pytest

from amylodb.mining.ner.normalizer import UniProtClient
from amylodb.mining.resolution import assign_role, gate_resolved, prepare_candidates
from amylodb.mining.validation import (clean_name, corroborated, find_abbreviations,
                                       is_short_symbol, looks_like_noise, resolve_abbreviation)

_HERE = os.path.dirname(__file__)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- names
@pytest.mark.parametrize("raw,expected", [
    ("tau (", "tau"),
    ("FUS's", "FUS"),
    ("- alpha-synuclein –", "alpha-synuclein"),
    (") TDP-43", "TDP-43"),
])
def test_clean_name(raw, expected):
    assert clean_name(raw) == expected


@pytest.mark.parametrize("name", [
    "urea", "CTAB", "dimethyl sulfoxide", "GM1", "heparan sulfate", "green fluorescent protein",
    "Alzheimer's disease", "NAC domain", "BRICHOS domain", "amyloid-like fibrils",
    "transcription factors", "antimicrobial peptides", "class I", "protein-43", "b-a",
    "lysine residue at position 1", "PTMs",
])
def test_noise_names(name):
    assert looks_like_noise(name)


@pytest.mark.parametrize("name", [
    "DJ-1", "YB-1", "NLRP3", "HSP27", "TDP-43", "alpha-synuclein", "Bcl-2",
    "heterogeneous nuclear ribonucleoprotein A1", "Abeta42",
])
def test_real_proteins_not_noise(name):
    assert not looks_like_noise(name)


def test_short_symbol():
    assert is_short_symbol("PDI") and is_short_symbol("Sup35") is False


# ------------------------------------------------------------ abbreviations
def test_abbreviation_expansion_and_rejection():
    text = ("We used perylene diimide (PDI) as a probe. Human cystatin C (HCC) "
            "forms fibrils. The heat shock protein (HSP27 (B1)) was absent.")
    ab = find_abbreviations(text)
    assert ab.get("PDI", "").lower() == "perylene diimide"
    assert ab.get("HCC", "").lower() == "human cystatin c"
    assert resolve_abbreviation("PDI", ab).lower() == "perylene diimide"
    assert resolve_abbreviation("unknown", ab) is None


def test_corroborated_requires_content_words():
    assert corroborated("Cystatin-C", "human cystatin c (HCC) aggregates")
    assert not corroborated("Protein disulfide-isomerase", "perylene diimide (PDI) fibrils")


# ------------------------------------------------------------ UniProt rules
@pytest.mark.parametrize("query,rec,genes,expected", [
    ("SNCA", "Alpha-synuclein", ["SNCA"], "gene"),
    ("alpha-synuclein", "Alpha-synuclein", ["SNCA"], "name"),
    ("carbonic anhydrase", "Carbonic anhydrase 2", ["CA2"], "run"),
    ("equine skeletal myoglobin", "Myoglobin", ["MB"], "run"),
    ("human serum albumin", "Albumin", ["ALB"], "run"),
    ("dimethyl sulfoxide", "Dimethyl sulfoxide reductase", ["dmsA"], None),
    ("factor", "Coagulation factor VIII", ["F8"], None),
])
def test_match_type(query, rec, genes, expected):
    assert UniProtClient._match_type(query, query, rec, genes) == expected


def test_gene_like():
    assert UniProtClient._gene_like("Sup35")
    assert UniProtClient._gene_like("HSPB1")
    assert not UniProtClient._gene_like("tau")
    assert not UniProtClient._gene_like("lap")


def test_role_from():
    assert UniProtClient.role_from(["Chaperone"], "Heat shock 70 kDa protein") == "modifier"
    assert UniProtClient.role_from([], "DnaJ homolog subfamily B member 6") == "modifier"
    assert UniProtClient.role_from(["Amyloid"], "Microtubule-associated protein tau") == "aggregator"


# ---------------------------------------------------------------- resolution
def test_prepare_candidates_drops_noise_and_expanded_non_proteins():
    text = "Perylene diimide (PDI) binds tau. Dimethyl sulfoxide (DMSO2) was the solvent."
    cands = [{"protein_name": "PDI"}, {"protein_name": "urea"}, {"protein_name": "tau"},
             {"protein_name": "CTAB"}, {"protein_name": "DMSO2"}]
    kept, dropped = prepare_candidates(cands, text, {"PDI": "perylene diimide",
                                                     "DMSO2": "dimethyl sulfoxide"})
    reasons = dict(dropped)
    assert reasons == {"urea": "noise", "CTAB": "noise", "DMSO2": "noise_abbreviation"}
    # an expansion that is not on the lists goes on to the UniProt gate, carrying its short form
    pdi = next(k for k in kept if k.get("resolved_from") == "PDI")
    assert pdi["protein_name"] == "perylene diimide"
    assert [k["protein_name"] for k in kept] == ["perylene diimide", "tau"]


def test_gate_resolved():
    assert gate_resolved({"uniprot_id": None}, "") == "no_uniprot_match"
    assert gate_resolved({"uniprot_id": None}, "", require_uniprot=False) is None
    short = {"uniprot_id": "P07237", "match_type": "gene", "lookup_stage": "human",
             "surface_form": "PDI", "uniprot_name": "Protein disulfide-isomerase"}
    assert gate_resolved(dict(short), "PDI fibrils") == "short_symbol_uncorroborated"
    assert gate_resolved(dict(short), "protein disulfide-isomerase (PDI) binds") is None


def test_assign_role_margin():
    assert assign_role({"claim_scores": {"disaggregase": 0.9, "aggregates": 0.3}}) == "modifier"
    assert assign_role({"claim_scores": {"disaggregase": 0.6, "aggregates": 0.5}}) == "aggregator"
    assert assign_role({"protein_role": "modifier", "claim_scores": {}}) == "modifier"


# ---------------------------------------------------------- revalidate script
def _synthetic_mining_db(path):
    from amylodb.mining.database.storage import AmyloidDatabase
    AmyloidDatabase(path)
    c = sqlite3.connect(path)
    papers = [("1", "Tau fibrils", "Microtubule-associated protein tau (tau) forms amyloid fibrils."),
              ("2", "Probe", "Perylene diimide (PDI) binds amyloid. Urea disassembles them."),
              ("3", "Hsp70", "Heat shock 70 kDa protein (HSP70) disaggregates tau fibrils.")]
    c.executemany("INSERT INTO papers(pmid,title,abstract) VALUES(?,?,?)", papers)
    prots = [("tau", "P10636", "Homo sapiens", "1"), ("pdi", None, None, "2"),
             ("urea", None, None, "2"), ("hsp70", None, None, "3")]
    for i, (name, uni, org, pmid) in enumerate(prots, 1):
        c.execute("INSERT INTO proteins(id,protein_name,uniprot_id,organism) VALUES(?,?,?,?)", (i, name, uni, org))
        c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(?,?)", (i, pmid))
    c.execute("INSERT INTO protein_claims(protein_id,pmid,surface_form,claim_scores) VALUES(?,?,?,?)",
              (4, "3", "HSP70", json.dumps({"disaggregase": 0.95, "aggregates": 0.2})))
    c.commit()
    c.close()


def _stub_lookup(name, surface=None):
    table = {"tau": ("P10636", "MAPT", "Microtubule-associated protein tau", None),
             "heat shock 70 kda protein": ("P0DMV8", "HSPA1A", "Heat shock 70 kDa protein 1A", "modifier"),
             "hsp70": ("P0DMV8", "HSPA1A", "Heat shock 70 kDa protein 1A", "modifier")}
    hit = table.get((name or "").lower())
    if not hit:
        return None
    return {"uniprot_id": hit[0], "gene_name": hit[1], "organism": "Homo sapiens",
            "uniprot_name": hit[2], "match_type": "name", "lookup_stage": "human", "protein_role": hit[3]}


def test_revalidate_synthetic_db(tmp_path):
    reval = _load_script("revalidate_mining")
    src, out = str(tmp_path / "m.db"), str(tmp_path / "m_rv.db")
    _synthetic_mining_db(src)
    report = reval.revalidate(src, out, lookup=_stub_lookup)
    status = {r[0]: r[1] for r in sqlite3.connect(out).execute("SELECT protein_name, status FROM proteins")}
    assert status["tau"] == "kept"
    assert status["urea"] == "dropped:noise"
    assert status["pdi"].startswith("dropped:")           # expands to perylene diimide, no UniProt
    assert status["hsp70"] == "kept"
    assert [m["uniprot_id"] for m in report["modifiers"]] == ["P0DMV8"]
    # the source DB is untouched
    assert "status" not in [r[1] for r in sqlite3.connect(src).execute("PRAGMA table_info(proteins)")] \
        or sqlite3.connect(src).execute("SELECT count(*) FROM proteins WHERE status IS NOT NULL").fetchone()[0] == 0


def test_export_filters_status_and_splits_roles(tmp_path):
    reval = _load_script("revalidate_mining")
    exp = _load_script("export_mined")
    src, out = str(tmp_path / "m.db"), str(tmp_path / "m_rv.db")
    _synthetic_mining_db(src)
    reval.revalidate(src, out, lookup=_stub_lookup)
    records = exp.export(out)
    names = {r["protein_name"]: r for r in records}
    assert "urea" not in names and "pdi" not in names
    roles = {r["protein_role"] for r in records}
    assert roles == {"aggregator", "modifier"}

    from amylodb.mining.adapter import split_mined
    aggs, mods = split_mined(records)
    assert len(mods) == 1 and all(a["protein_role"] == "aggregator" for a in aggs)


def test_modifiers_tsv_written(tmp_path):
    from amylodb.pipeline import _write_modifiers_tsv
    path = str(tmp_path / "modifiers.tsv")
    _write_modifiers_tsv([{"protein_name": "hsp70", "uniprot_id": "P0DMV8",
                           "co_mentioned_aggregators": ["tau"], "source_pmids": ["3"]}], path)
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    assert rows[0]["uniprot_id"] == "P0DMV8" and "tau" in rows[0]["co_mentioned_aggregators"]


# --------------------------------------------------------------- review score
def test_score_review_wilson(tmp_path):
    audit = _load_script("audit_mining")
    path = tmp_path / "review.tsv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "name", "correct"])
        for i in range(40):
            w.writerow([i, f"p{i}", "y" if i < 36 else "n"])
        w.writerow([99, "unmarked", ""])
    s = audit.score_review(str(path))
    assert s["marked"] == 40 and s["correct"] == 36 and s["precision"] == 0.9
    lo, hi = s["ci95"]
    assert 0.76 < lo < 0.78 and 0.95 < hi < 0.97 and s["unmarked"] == 1


# ------------------------------------------------- noise found by manual review
@pytest.mark.parametrize("name", ["glu8", "Lys-1", "Tyr39", "polycomb repressive complex 1"])
def test_residues_and_complexes_are_noise(name):
    assert looks_like_noise(name)


def test_reversed_abbreviation():
    ab = find_abbreviations("the chromatin-modifying complex PRC1 (Polycomb repressive complex 1) binds")
    assert ab["PRC1"] == "Polycomb repressive complex 1"


def test_review_sample_one_row_per_accession(tmp_path):
    audit = _load_script("audit_mining")
    reval = _load_script("revalidate_mining")
    src, out = str(tmp_path / "m.db"), str(tmp_path / "m_rv.db")
    _synthetic_mining_db(src)
    c = sqlite3.connect(src)                     # a second row resolving to tau
    c.execute("INSERT INTO proteins(id,protein_name,uniprot_id,organism) VALUES(5,'tau','P10636','Mus')")
    c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(5,'1')")
    c.commit(); c.close()
    reval.revalidate(src, out, lookup=_stub_lookup)
    kept = audit.review_sample(out, 50)
    accs = [r["uniprot_id"] for r in kept]
    assert len(accs) == len(set(accs))
    dropped = audit.review_sample(out, 50, which="dropped")
    assert {r["status"] for r in dropped} <= {"dropped:noise", "dropped:noise_abbreviation",
                                               "dropped:no_uniprot_match"}
    assert all(r["uniprot_id"] == "" for r in dropped)


def test_construct_words_and_cdr():
    assert clean_name("tau variant") == "tau" and clean_name("SOD1 mutants") == "SOD1"
    assert looks_like_noise("CDR1") and looks_like_noise("cdr-h3")


def test_extend_truncated_names():
    from amylodb.mining.validation import extend_truncated
    assert extend_truncated("Prostatic Acid", "derived from Prostatic Acid Phosphatase (PAP)") == \
        "Prostatic Acid Phosphatase"
    assert extend_truncated("Tar-DNA", "The Tar-DNA Binding Protein-43 C-terminal") == \
        "Tar-DNA Binding Protein-43"
    assert extend_truncated("tau", "tau forms fibrils in the brain") is None
    assert looks_like_noise("paired helical filament 6")
