"""Fixes from the second v3 review (aggregators, condensates, modifiers, dropped).

Each case is a protein the sheets showed mis-resolved, mis-routed or lost:
family names resolved to one paralog (carbonic anhydrase, tubulin), "-like"
runs (ataxin-3 -> ATXN3L), run matches ranked above the exact name, human
primary symbols lost in model-organism papers (AKT1 in C. elegans), UniProt
failures read as "no match", spelling variants missed by the mention test
(beta(2) microglobulin), missing vocabulary (curli, polar granules), and one
accession carrying different roles under different names (TDP-43).
"""

import json
import re
import sqlite3

import pytest

from amylodb.mining.ner.normalizer import UniProtClient, UniProtUnavailable


def _entry(acc, rec, gene, organism="Homo sapiens", synonyms=()):
    return {"primaryAccession": acc, "organism": {"scientificName": organism},
            "proteinDescription": {"recommendedName": {"fullName": {"value": rec}}},
            "genes": [{"geneName": {"value": gene}, "synonyms": [{"value": s} for s in synonyms]}],
            "keywords": []}


HUMAN = {
    "carbonic anhydrase": [_entry("P00915", "Carbonic anhydrase 1", "CA1"),
                           _entry("P00918", "Carbonic anhydrase 2", "CA2"),
                           _entry("Q9Y2D0", "Carbonic anhydrase 5A, mitochondrial", "CA5A")],
    "tubulin": [_entry("Q9UJT1", "Tubulin delta chain", "TUBD1"),
                _entry("Q9UJT0", "Tubulin epsilon chain", "TUBE1")],
    "synphilin": [_entry("Q9Y4G2", "Synphilin-1", "SNCAIP")],
    "ataxin-3": [_entry("Q9H3M9", "Ataxin-3-like protein", "ATXN3L"),
                 _entry("P54252", "Ataxin-3", "ATXN3")],
    "akt1": [_entry("P31749", "RAC-alpha serine/threonine-protein kinase", "AKT1")],
    "caly": [_entry("Q9NYX4", "Neuron-specific vesicular protein calcyon", "CALY")],
    "drosha": [_entry("Q9NRR4", "Ribonuclease 3", "DROSHA")],
}
ANY = {
    "carbonic anhydrase": [_entry("P00921", "Carbonic anhydrase 2", "CA2", "Bos taurus")],
    "akt1": [_entry("P31750", "RAC-alpha serine/threonine-protein kinase", "Akt1", "Mus musculus")],
    "pdk-1": [_entry("Q9Y1J3", "3-phosphoinositide-dependent protein kinase 1", "pdk-1", "Caenorhabditis elegans")],
    "perfringolysin": [_entry("P0C2E9", "Perfringolysin O", "pfo", "Clostridium perfringens")],
    "caly": [],
}


class _Resp:
    def __init__(self, results, code=200):
        self._results, self.status_code = results, code

    def json(self):
        return {"results": self._results}


TREMBL = {  # (name, taxon) -> unreviewed entries
    ("drosha", 7227): [_entry("Q7KNF1", "Ribonuclease III", "drosha", "Drosophila melanogaster"),
                       _entry("Q960Y4", "Ribonuclease III", "drosha", "Drosophila melanogaster")],
}


def _fake_get(url, params=None, timeout=None):
    query = params["query"]
    q = re.search(r'protein_name:"(.*?)"', query).group(1).lower()
    if "reviewed:true" in query and "organism_id:9606" not in query and "organism_id:" in query:
        return _Resp([])                                     # no Swiss-Prot entry in that taxon
    if "reviewed:false" in query:
        taxon = int(re.search(r"organism_id:(\d+)", query).group(1))
        return _Resp(TREMBL.get((q, taxon), []))
    table = HUMAN if "organism_id:9606" in query else ANY
    return _Resp(table.get(q, []))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import amylodb.mining.ner.normalizer as nz
    monkeypatch.setattr(nz.requests, "get", _fake_get)
    monkeypatch.setattr(UniProtClient, "BACKOFF", 0.0)
    UniProtClient.candidates.cache_clear(); UniProtClient.in_taxon.cache_clear()
    yield
    UniProtClient.candidates.cache_clear(); UniProtClient.in_taxon.cache_clear()


# --- identity --------------------------------------------------------------

def test_family_name_with_several_paralogs_is_unresolved():
    assert UniProtClient.lookup("carbonic anhydrase", "carbonic anhydrase", context="") is None
    assert UniProtClient.lookup("tubulin", "tubulin", context="") is None


def test_unique_family_member_is_kept():
    assert UniProtClient.lookup("synphilin", "synphilin", context="")["uniprot_id"] == "Q9Y4G2"
    hit = UniProtClient.lookup("perfringolysin", "perfringolysin", context="Clostridium perfringens toxin")
    assert hit["uniprot_id"] == "P0C2E9"


def test_like_protein_never_matches_and_exact_name_wins():
    assert UniProtClient.lookup("ataxin-3", "ataxin-3", context="")["uniprot_id"] == "P54252"


def test_ranked_puts_name_before_run():
    hits = [dict(uniprot_id="A", match_type="run", gene_name="X"),
            dict(uniprot_id="B", match_type="name", gene_name="Y")]
    assert [h["uniprot_id"] for h in UniProtClient._ranked(hits, set())] == ["B", "A"]


def test_hyphenated_worm_gene_is_searched():
    assert UniProtClient._gene_alternatives("akt1") == ["akt1", "akt-1"]
    assert UniProtClient._gene_alternatives("tau") == ["tau"]


def test_model_organism_paper_takes_its_own_unreviewed_entry():
    # reviewed: Drosha in a fly paper is the fly protein, not human DROSHA
    ctx = "Drosha-dependent microRNAs modulate FUS-mediated neurodegeneration in Drosophila."
    hit = UniProtClient.lookup("drosha", "Drosha", context=ctx)
    assert hit["uniprot_id"] == "Q7KNF1" and hit["lookup_stage"] == "unreviewed_named"


def test_human_symbol_not_taken_for_a_worm_paper():
    ctx = "In C. elegans, AKT1 signalling controls polyglutamine aggregation in muscle."
    assert UniProtClient.lookup("akt1", "AKT1", context=ctx) is None


def test_named_organism_entry_still_wins():
    # the worm's own entry, written "pdk-1" in UniProt, is found and preferred
    ctx = "In C. elegans, loss of pdk1 reduces polyglutamine aggregation."
    assert UniProtClient.lookup("pdk-1", "pdk1", context=ctx)["uniprot_id"] == "Q9Y1J3"


def test_human_symbol_in_bacterial_paper_still_rejected():
    ctx = "CalY of Bacillus cereus assembles into functional amyloid fibrils."
    assert UniProtClient.lookup("caly", "CalY", context=ctx) is None


def test_unreachable_uniprot_raises_and_is_not_cached(monkeypatch):
    import amylodb.mining.ner.normalizer as nz
    calls = []

    def down(url, params=None, timeout=None):
        calls.append(1)
        return _Resp([], code=503)
    monkeypatch.setattr(nz.requests, "get", down)
    with pytest.raises(UniProtUnavailable):
        UniProtClient.lookup("synphilin", "synphilin", context="")
    monkeypatch.setattr(nz.requests, "get", _fake_get)
    assert UniProtClient.lookup("synphilin", "synphilin", context="")["uniprot_id"] == "Q9Y4G2"
    assert len(calls) == UniProtClient.RETRIES


def test_enrichment_marks_lookup_error(monkeypatch):
    import amylodb.mining.ner.normalizer as nz
    monkeypatch.setattr(nz.requests, "get", lambda *a, **k: _Resp([], code=500))
    out = UniProtClient.enrich_proteins([{"protein_name": "synphilin", "surface_form": "synphilin"}])
    from amylodb.mining.resolution import gate_resolved
    assert out[0]["lookup_error"] and gate_resolved(out[0], "") == "lookup_error"


def test_revalidate_marks_lookup_error(tmp_path):
    import importlib.util, os
    from amylodb.mining.database.storage import AmyloidDatabase
    path = str(tmp_path / "m.db")
    AmyloidDatabase(path)
    c = sqlite3.connect(path)
    c.execute("INSERT INTO proteins(id,protein_name) VALUES(1,'synphilin')")
    c.execute("INSERT INTO papers(pmid,title,abstract) VALUES('1','T','Synphilin forms aggregates.')")
    c.execute("INSERT INTO protein_papers VALUES(1,'1')")
    c.commit(); c.close()
    spec = importlib.util.spec_from_file_location(
        "rv", os.path.join(os.path.dirname(__file__), "..", "scripts", "revalidate_mining.py"))
    rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)

    def down(name, surface=None, context=None):
        raise UniProtUnavailable("503")
    rep = rv.revalidate(path, str(tmp_path / "rv.db"), lookup=down)
    assert rep["status"] == {"dropped:lookup_error": 1}


# --- mentions and vocabulary --------------------------------------------------

@pytest.mark.parametrize("sentence,name", [
    ("Beta(2)-microglobulin amyloidosis develops in dialysis.", "beta(2) microglobulin"),
    ("β2-microglobulin fibrils were seeded.", "beta2 microglobulin"),
    ("Class I hydrophobins self-assemble into amyloid-like fibrils.", "class i hydrophobin"),
    ("Oligomers of human stefin B precede fibrils.", "human stefin b.. oligomers"),
])
def test_spelling_variants_count_as_mentions(sentence, name):
    from amylodb.mining.validation import relevance
    assert relevance([sentence], [name])["agg_mention"]


def test_flex_mention_needs_a_separator_between_words():
    from amylodb.mining.validation import flex_mention
    assert not flex_mention("a classic hydrophobin", ["class i hydrophobin"])
    assert not flex_mention("amyloid proteins aggregate", ["proteins.. amyloid proteins"])


def test_curli_and_polar_granules_in_vocabulary():
    from amylodb.mining.validation import relevance
    assert relevance(["Cra represses curli biosynthesis in E. coli."], ["Cra"])["agg_mention"]
    r = relevance(["Vasa localises to polar granules in the germ plasm."], ["Vasa"])
    assert r["cond_mention"] and not r["agg_abstract"]


def test_bovine_serum_albumin_does_not_name_cattle():
    from amylodb.mining.validation import names_nonhuman_organism, organism_mentioned
    text = "Fibrils were blocked with bovine serum albumin."
    assert not names_nonhuman_organism(text) and not organism_mentioned("Bos taurus", text)
    assert organism_mentioned("Bos taurus", "bovine carbonic anhydrase forms fibrils")


def test_paralog_extension_needs_one_member():
    from amylodb.mining.validation import paralog_extension
    assert paralog_extension("syntaxin", "PS1- or syntaxin 6-positive membranes") == "syntaxin 6"
    assert paralog_extension("cathepsin", "Cathepsin S, cathepsin B and cathepsin D") is None
    assert paralog_extension("caspase", "caspase activity in HD") is None


def test_roman_numeral_variant():
    qs = [q for q, _ in UniProtClient._query_variants("semenogelin I")]
    assert "semenogelin 1" in qs


def test_family_test_groups_strains_of_one_species():
    hits = [dict(uniprot_id="A9JX06", match_type="run", added=["alpha", "2"],
                 organism="Staphylococcus aureus (strain USA300)"),
            dict(uniprot_id="P0C802", match_type="run", added=["alpha", "3"],
                 organism="Staphylococcus aureus (strain Mu3)")]
    assert UniProtClient._filter_runs(hits) == []


def test_alias_ignores_the_species_word():
    from amylodb.mining.resolution import prepare_candidates
    kept, _ = prepare_candidates([{"protein_name": "human beta-secretase", "surface_form": "human beta-secretase"}],
                                 "human beta-secretase (BACE-1)", {})
    assert kept[0]["protein_name"] == "BACE1"


def test_aliases_route_unreachable_names():
    from amylodb.mining.validation import ALIAS_TO_GENE
    assert ALIAS_TO_GENE["beta-site app-cleaving enzyme"] == "BACE1"
    assert ALIAS_TO_GENE["g3bp"] == "G3BP1" and ALIAS_TO_GENE["staufen1"] == "STAU1"
    assert ALIAS_TO_GENE["von hippel-lindau protein"] == "VHL"
    assert ALIAS_TO_GENE["immunoglobulin binding protein"] == "HSPA5"


# --- export -------------------------------------------------------------------

def test_one_accession_gets_one_role_on_export():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "ex", os.path.join(os.path.dirname(__file__), "..", "scripts", "export_mined.py"))
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    rows = [{"uniprot_id": "Q13148", "protein_role": "aggregator", "role_subtype": "", "modifier_roles": [],
             "condensate_evidence": False},
            {"uniprot_id": "Q13148", "protein_role": "modifier", "role_subtype": "associated",
             "modifier_roles": ["associated"], "condensate_evidence": False},
            {"uniprot_id": "Q13148", "protein_role": "condensate", "role_subtype": "", "modifier_roles": [],
             "condensate_evidence": True},
            {"uniprot_id": "P10636", "protein_role": "modifier", "role_subtype": "chaperone",
             "modifier_roles": ["chaperone"], "condensate_evidence": False}]
    ex.unify_roles(rows)
    tdp = rows[:3]
    assert {r["protein_role"] for r in tdp} == {"aggregator"}
    assert all(r["modifier_roles"] == ["associated"] and r["condensate_evidence"] for r in tdp)
    assert rows[3]["protein_role"] == "modifier" and rows[3]["role_subtype"] == "chaperone"
