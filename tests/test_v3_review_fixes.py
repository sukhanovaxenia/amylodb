"""Fixes driven by the review of the expanded (12873-paper) corpus.

Each case reproduces a row that the manual review of review_v3 / review_modifiers
/ review_dropped found wrong: gene-synonym collisions and family names in the
UniProt lookup, size labels, sample descriptions, the chaperone-noun override
that flipped Sup35, NLI abstentions that dropped real proteins, and
stress-granule literature recorded as amyloid modifiers.
"""

import csv
import importlib.util
import os
import re
import sqlite3

import pytest

from amylodb.mining.ner.normalizer import UniProtClient
from amylodb.mining.resolution import gate_resolved, lexical_modifier, prepare_candidates
from amylodb.mining.validation import (corroborated, is_size_label, looks_like_noise,
                                       nonhuman_only, relevance)

_HERE = os.path.dirname(__file__)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------- UniProt lookup
def _entry(acc, rec, gene, organism="Homo sapiens", synonyms=(), keywords=()):
    return {"primaryAccession": acc, "organism": {"scientificName": organism},
            "proteinDescription": {"recommendedName": {"fullName": {"value": rec}}},
            "genes": [{"geneName": {"value": gene}, "synonyms": [{"value": s} for s in synonyms]}],
            "keywords": [{"name": k} for k in keywords]}


YEAST = "Saccharomyces cerevisiae (strain ATCC 204508 / S288c)"
HUMAN = {
    "rim4": [_entry("Q9H426", "Regulating synaptic membrane exocytosis protein 4", "RIMS4", synonyms=["RIM4"])],
    "hba": [_entry("Q99250", "Sodium channel protein type 2 subunit alpha", "SCN2A", synonyms=["HBA"]),
            _entry("P69905", "Hemoglobin subunit alpha", "HBA1", synonyms=["HBA"])],
    "slug": [_entry("O43623", "Zinc finger protein SNAI2", "SNAI2", synonyms=["SLUG"])],
    "usp10": [_entry("Q8NFA0", "Ubiquitin carboxyl-terminal hydrolase 32", "USP32", synonyms=["USP10"]),
              _entry("Q14694", "Ubiquitin carboxyl-terminal hydrolase 10", "USP10")],
    "caly": [_entry("Q9NYX4", "Neuron-specific vesicular protein calcyon", "CALY")],
    "pat1": [_entry("Q7Z2H8", "Proton-coupled amino acid transporter 1", "SLC36A1", synonyms=["PAT1"])],
}
OTHER = {
    "rim4": [_entry("P38741", "Meiotic activator RIM4", "RIM4", YEAST)],
    "pat1": [_entry("P25644", "Topoisomerase II-associated protein PAT1", "PAT1", YEAST)],
}


class _Resp:
    status_code = 200

    def __init__(self, results):
        self._r = results

    def json(self):
        return {"results": self._r}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import amylodb.mining.ner.normalizer as nz

    def get(url, params=None, timeout=None):
        q = re.search(r'protein_name:"(.*?)"', params["query"]).group(1).lower()
        table = HUMAN if "organism_id:9606" in params["query"] else OTHER
        return _Resp(table.get(q, []))
    monkeypatch.setattr(nz.requests, "get", get)
    UniProtClient.candidates.cache_clear()
    yield
    UniProtClient.candidates.cache_clear()


def test_named_organism_beats_human_synonym():
    """Rim4 in a yeast paper is the yeast meiotic activator, not human RIMS4."""
    hit = UniProtClient.lookup("rim4", "Rim4", context="In yeast, the RNA-binding protein Rim4 forms amyloid.")
    assert hit["uniprot_id"] == "P38741"
    hit = UniProtClient.lookup("pat1", "PAT1", context="P-body components in yeast such as PAT1 and LSM1.")
    assert hit["uniprot_id"] == "P25644"


def test_ambiguous_synonym_needs_corroboration():
    """HBA is carried by haemoglobin alpha and by SCN2A; the abstract decides."""
    hit = UniProtClient.lookup("hba", "HBA", context="a C-terminal fragment of alpha-hemoglobin, HBA(111-142)")
    assert hit["uniprot_id"] == "P69905"
    assert UniProtClient.lookup("hba", "HBA", context="HBA fragments aggregate.") is None


def test_unambiguous_synonym_accepted_without_corroboration():
    """Authors write Slug, not 'zinc finger protein SNAI2'; one entry carries it."""
    hit = UniProtClient.lookup("slug", "Slug", context="Slug expression rises in amyloid-laden tissue.")
    assert hit["uniprot_id"] == "O43623"


def test_primary_gene_beats_synonym():
    hit = UniProtClient.lookup("usp10", "USP10", context="USP10 promotes stress granule aggregation.")
    assert hit["uniprot_id"] == "Q14694"


def test_human_gene_in_nonhuman_paper_needs_corroboration():
    """CalY of Bacillus cereus is not the human calcyon gene CALY."""
    assert UniProtClient.lookup("caly", "CalY", context="CalY is a Bacillus cereus biofilm matrix protein.") is None
    assert nonhuman_only("CalY is a Bacillus cereus biofilm matrix protein.")
    assert not nonhuman_only("human TDP-43 expressed in yeast")


# ------------------------------------------------------- names and labels
@pytest.mark.parametrize("name", ["importin", "AAA+ ATPase", "AAA+", "PI3K", "phosphatidylinositol 3-kinase",
                                  "TORC1", "mTORC1", "extracellular polysaccharide", "LF-MF-treated PG"])
def test_family_chemical_and_sample_names_are_noise(name):
    assert looks_like_noise(name)


@pytest.mark.parametrize("name", ["lipopolysaccharide binding protein", "casein kinase 2", "USP10", "Sup35"])
def test_real_proteins_survive_new_noise_rules(name):
    assert not looks_like_noise(name)


def test_size_labels_and_aliases():
    kept, dropped = prepare_candidates([{"protein_name": "p62"}, {"protein_name": "p25"},
                                        {"protein_name": "PHF6"}], "text")
    got = {k["protein_name"]: (k.get("resolved_from"), k.get("size_label")) for k in kept}
    assert got == {"SQSTM1": ("p62", None), "p25": (None, True), "MAPT": ("PHF6", None)}
    assert is_size_label("p110") and not is_size_label("p5") and not is_size_label("Sup35")
    tmed9 = {"uniprot_id": "Q9BVK6", "uniprot_name": "Transmembrane emp24 domain-containing protein 9",
             "size_label": True, "match_type": "name", "lookup_stage": "human", "surface_form": "p25"}
    assert gate_resolved(tmed9, "TPPP/p25 seeding activity in MSA") == "size_label_uncorroborated"


def test_generic_words_do_not_corroborate():
    assert not corroborated("Sodium channel protein type 2 subunit alpha", "a fragment of alpha-hemoglobin")
    assert corroborated("Hemoglobin subunit alpha", "a fragment of alpha-hemoglobin")
    assert not corroborated("Regulating synaptic membrane exocytosis protein 4", "the prospore membrane")


# ------------------------------------------------------ chaperone override
@pytest.mark.parametrize("sent,name,expected", [
    ("One metal-binding chaperone metallothionein-3 binds fibrils.", "metallothionein-3", "chaperone"),
    ("Btn2 and Hsp42 are compartment-specific sequestrases.", "Btn2", "chaperone"),
    ("bovine fibrinogen (bFg) is known to exhibit chaperone-like activities", "bovine fibrinogen", "chaperone"),
    ("Hsp104 is a chaperone that dissolves Sup35 fibrils.", "Sup35", None),
    ("Sup35 fibrils require the chaperone activity of Hsp104.", "Sup35", None),
    ("The co-chaperone Sis1 binds Rnq1.", "Rnq1", None),
])
def test_chaperone_override_needs_the_noun_applied(sent, name, expected):
    assert lexical_modifier([sent], [name]) == expected


def test_chaperone_override_never_beats_aggregator_majority():
    sent = ["The chaperone Sup35 fragment binds."]
    assert lexical_modifier(sent, ["Sup35"]) == "chaperone"
    assert lexical_modifier(sent, ["Sup35"], votes={"aggregator": 15, "regulator": 3}) is None


# ----------------------------------------------------------- relevance
def test_relevance_vocabulary():
    zap = relevance(["ZAP stress granule localization is correlated with antiviral activity."], ["ZAP"])
    assert zap == {"agg_abstract": False, "cond_abstract": True, "agg_mention": False,
                   "cond_mention": True}
    nucleolus = relevance(["NPM1 drives nucleolar phase separation."], ["NPM1"])
    assert nucleolus["cond_abstract"] and nucleolus["cond_mention"]
    cd20 = relevance(["Congo red, CD45 and CD20 were used. Amyloid deposits were typed."], ["CD20"])
    assert cd20["agg_abstract"] and not cd20["agg_mention"]


def _db(tmp_path, papers, proteins):
    from amylodb.mining.database.storage import AmyloidDatabase
    path = str(tmp_path / "m.db")
    AmyloidDatabase(path)
    c = sqlite3.connect(path)
    c.executemany("INSERT INTO papers(pmid,title,abstract) VALUES(?,?,?)", papers)
    for i, (name, pmid) in enumerate(proteins, 1):
        c.execute("INSERT INTO proteins(id,protein_name) VALUES(?,?)", (i, name))
        c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(?,?)", (i, pmid))
    c.commit(); c.close()
    return path


def _lk(table):
    def lookup(name, surface=None, context=None):
        acc = table.get((name or "").lower())
        return acc and {"uniprot_id": acc, "gene_name": name.upper(), "organism": "Homo sapiens",
                        "uniprot_name": name, "match_type": "gene", "lookup_stage": "human",
                        "protein_role": "aggregator", "organism_in_text": None}
    return lookup


class _Abstain:
    def score(self, premise, protein, keys=None):
        return {k: 0.1 for k in (keys or ())}


def test_routing_and_abstention(tmp_path):
    papers = [("1", "Vitronectin", "Human vitronectin forms spherical oligomers and typical amyloid fibrils."),
              ("2", "G3BP1", "G3BP1 localises to stress granules during viral infection."),
              ("3", "Insulin", "Insulin secretion rises after a meal."),
              ("4", "Typing", "Congo red, CD45 and CD20 were used. Amyloid deposits were typed by mass spectrometry.")]
    src = _db(tmp_path, papers, [("vitronectin", "1"), ("g3bp1", "2"), ("insulin", "3"), ("cd20", "4")])
    out = str(tmp_path / "rv.db")
    rep = _script("revalidate_mining").revalidate(
        src, out, lookup=_lk({"vitronectin": "P04004", "g3bp1": "Q13283", "insulin": "P01308", "cd20": "P11836"}),
        rescorer=_Abstain())
    rows = dict((r[0], r[1:]) for r in sqlite3.connect(out).execute(
        "SELECT protein_name, status, protein_role FROM proteins"))
    assert rows["vitronectin"] == ("kept", "aggregator")      # NLI abstained; the text says it
    assert rows["g3bp1"] == ("kept", "condensate")             # stress-granule literature only
    assert rows["insulin"][0] == "dropped:off_topic"           # no aggregation, no condensate
    assert rows["cd20"][0] == "dropped:no_aggregation_mention" # a marker in an amyloid paper
    assert "dropped:no_role_claim" not in rep["status"]
    assert rep["condensates"] == 1


def test_no_claim_role_without_rescorer(tmp_path):
    """Stored per-paper claims must not decide the role (GAPDH became a 'regulator')."""
    import json
    src = _db(tmp_path, [("1", "GAPDH", "Thermal aggregation of glyceraldehyde 3-phosphate dehydrogenase (GAPDH) was studied.")],
              [("gapdh", "1")])
    c = sqlite3.connect(src)
    c.execute("INSERT INTO protein_claims(protein_id,pmid,surface_form,claim_scores) VALUES(1,'1','GAPDH',?)",
              (json.dumps({"aggregates": 0.4, "regulates": 0.95}),))
    c.commit(); c.close()
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk({"gapdh": "P04406"}))
    row = sqlite3.connect(out).execute("SELECT status, protein_role, role_subtype FROM proteins").fetchone()
    assert row == ("kept", "aggregator", None)


# ------------------------------------------------- review and calibration
def test_calibration_ignores_non_role_errors(tmp_path):
    cal = _script("calibrate_roles")
    path = tmp_path / "r.tsv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["uniprot_id", "role", "correct", "error_type"])
        w.writerows([["P1", "aggregator", "y", ""], ["P2", "aggregator", "n", "role"],
                     ["P3", "aggregator", "n", "identity"], ["P4", "aggregator", "n", "offtopic"],
                     ["P5", "aggregator", "n", ""]])
    assert cal.load_labels([str(path)]) == {"P1": 1, "P2": 0, "P5": 0}


def test_build_writes_condensate_candidates(tmp_path):
    from amylodb.pipeline import run
    mods = [{"protein_name": "hsp70", "protein_role": "modifier", "uniprot_id": "P0DMV8"},
            {"protein_name": "g3bp1", "protein_role": "condensate", "uniprot_id": "Q13283"}]
    fixture = os.path.join(_HERE, "fixtures", "amyload_mini.csv")
    res = run({"amyload": fixture}, str(tmp_path), compute_features=False, modifiers=mods)
    read = lambda f: [r["uniprot_id"] for r in csv.DictReader(open(tmp_path / f), delimiter="\t")]
    assert read("modifiers.tsv") == ["P0DMV8"]
    assert read("condensate_candidates.tsv") == ["Q13283"]
    assert res.stats["modifiers"] == 1 and res.stats["condensate_candidates"] == 1
