"""UniProt resolution against recorded-shape responses (no network).

Each case reproduces a protein that the first re-validation run on the v2
mining database lost or mis-identified: gene synonyms (p53, Hsc70),
alternative and processed-product names (SOD1, glucagon), qualifier-stripped
queries (human serum albumin), single-word queries against numbered names
(calmodulin), and gene-only matches in other organisms that need the organism
named in the abstract (FadA, Hfq, Glu8).
"""

import re

import pytest

from amylodb.mining.ner.normalizer import UniProtClient
from amylodb.mining.resolution import gate_resolved
from amylodb.mining.validation import organism_mentioned


def _entry(acc, rec, genes, organism="Homo sapiens", alt=(), contains=(), synonyms=(), keywords=()):
    return {
        "primaryAccession": acc,
        "organism": {"scientificName": organism},
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": rec}},
            "alternativeNames": [{"fullName": {"value": a}} for a in alt],
            "contains": [{"recommendedName": {"fullName": {"value": c}}} for c in contains],
        },
        "genes": [{"geneName": {"value": genes[0]}, "synonyms": [{"value": s} for s in synonyms]}],
        "keywords": [{"name": k} for k in keywords],
    }


HUMAN = {
    "p53": [_entry("P04637", "Cellular tumor antigen p53", ["TP53"], synonyms=["P53"])],
    "fibronectin": [_entry("Q5T4X1", "Fibronectin type III domain-containing protein 3B", ["FNDC3B"]),
                    _entry("P02751", "Fibronectin", ["FN1"])],
    "albumin": [_entry("P02768", "Albumin", ["ALB"])],
    "myoglobin": [_entry("P02144", "Myoglobin", ["MB"])],
    "calmodulin": [_entry("P0DP23", "Calmodulin-1", ["CALM1"])],
    "glucagon": [_entry("P01275", "Pro-glucagon", ["GCG"], contains=["Glucagon", "Glucagon-like peptide 1"])],
    "superoxide dismutase-1": [_entry("P00441", "Superoxide dismutase [Cu-Zn]", ["SOD1"],
                                      alt=["Superoxide dismutase 1"])],
    "hsc70": [_entry("P11142", "Heat shock cognate 71 kDa protein", ["HSPA8"], synonyms=["HSC70"],
                     keywords=["Chaperone"])],
    "factor": [_entry("P00451", "Coagulation factor VIII", ["F8"])],
}
ANY = {
    "fada": [_entry("P21151", "3-ketoacyl-CoA thiolase FadA", ["fadA"], "Escherichia coli (strain K12)")],
    "hfq": [_entry("P0A6X3", "RNA-binding protein Hfq", ["hfq"], "Escherichia coli (strain K12)")],
    "myoglobin": [_entry("P02144", "Myoglobin", ["MB"]),
                  _entry("P68082", "Myoglobin", ["MB"], "Equus caballus")],
    "glu8": [_entry("Q8LQ92", "Endoglucanase 3", ["GLU8"], "Oryza sativa subsp. japonica")],
}


class _Resp:
    status_code = 200

    def __init__(self, results):
        self._results = results

    def json(self):
        return {"results": self._results}


def _fake_get(url, params=None, timeout=None):
    q = re.search(r'protein_name:"(.*?)"', params["query"]).group(1).lower()
    table = HUMAN if "organism_id:9606" in params["query"] else ANY
    return _Resp(table.get(q, []))


@pytest.fixture(autouse=True)
def offline_uniprot(monkeypatch):
    import amylodb.mining.ner.normalizer as nz
    monkeypatch.setattr(nz.requests, "get", _fake_get)
    UniProtClient.candidates.cache_clear(); UniProtClient.in_taxon.cache_clear()
    yield
    UniProtClient.candidates.cache_clear(); UniProtClient.in_taxon.cache_clear()


@pytest.mark.parametrize("name,acc,mt", [
    ("p53", "P04637", "gene"),                       # gene synonym
    ("fibronectin", "P02751", "name"),               # real entry below a crowding hit
    ("human serum albumin", "P02768", "name"),       # qualifier-stripped query
    ("calmodulin", "P0DP23", "run"),                 # single word vs "Calmodulin-1"
    ("glucagon", "P01275", "name"),                  # processed product of Pro-glucagon
    ("superoxide dismutase-1", "P00441", "name"),    # alternative name
])
def test_recovered_human_proteins(name, acc, mt):
    hit = UniProtClient.lookup(name, name, context="")
    assert hit and hit["uniprot_id"] == acc and hit["match_type"] == mt


def test_hsc70_synonym_is_modifier():
    hit = UniProtClient.lookup("hsc70", "Hsc70", context="")
    assert hit["uniprot_id"] == "P11142" and hit["protein_role"] == "modifier"


def test_generic_head_not_matched():
    assert UniProtClient.lookup("factor", "factor", context="") is None


def test_gene_only_other_organism_needs_organism_in_text():
    fuso = "Fusobacterium nucleatum secretes amyloid-like FadA to enhance pathogenicity."
    assert UniProtClient.lookup("fada", "FadA", context=fuso) is None
    ecoli = "The E. coli Hfq C-terminal region forms amyloid."
    hit = UniProtClient.lookup("hfq", "Hfq", context=ecoli)
    assert hit and hit["uniprot_id"] == "P0A6X3" and hit["organism_in_text"]
    assert UniProtClient.lookup("hfq", "Hfq", context="Hfq forms amyloid.") is None
    assert UniProtClient.lookup("glu8", "Glu8", context="protonated Glu8 in the fibril core") is None


def test_no_context_keeps_gene_like_rule():
    assert UniProtClient.lookup("fada", "FadA")["uniprot_id"] == "P21151"
    assert UniProtClient.lookup("hfq", "Hfq") is None          # 3 characters, not gene-like


@pytest.mark.parametrize("organism,text,expected", [
    ("Escherichia coli (strain K12)", "curli fibres of E. coli", True),
    ("Escherichia coli (strain K12)", "Fusobacterium nucleatum FadA", False),
    ("Saccharomyces cerevisiae (strain ATCC 204508 / S288c)", "the yeast prion [PSI+]", True),
    ("Caenorhabditis elegans", "a C. elegans model", True),
    ("Oryza sativa subsp. japonica", "beta-endorphin amyloid", False),
])
def test_organism_mentioned(organism, text, expected):
    assert organism_mentioned(organism, text) is expected


def test_short_form_fallback_needs_corroboration():
    from amylodb.mining.validation import looks_like_noise
    # the FAP long form is itself a class phrase, so it never reaches the lookup
    assert looks_like_noise("functional amyloid in pseudomonas")
    pd = {"uniprot_id": "Q9NVQ4", "uniprot_name": "Fas apoptotic inhibitory molecule 1",
          "gene_name": "FAIM", "short_form_fallback": True, "resolved_from": "FAM",
          "match_type": "gene", "lookup_stage": "human", "surface_form": "FAM"}
    assert gate_resolved(dict(pd), "fluorescent amyloid marker (FAM)") == "short_form_fallback_uncorroborated"
    assert gate_resolved(dict(pd), "Fas apoptotic inhibitory molecule (FAM)") is None
    bcl2 = {"uniprot_id": "P10415", "uniprot_name": "Apoptosis regulator Bcl-2", "gene_name": "BCL2",
            "short_form_fallback": True, "resolved_from": "Bcl-2", "match_type": "gene",
            "lookup_stage": "human", "surface_form": "Bcl-2"}
    assert gate_resolved(bcl2, "B-cell lymphoma 2 (Bcl-2) binds") is None   # symbol equals gene


def test_species_word_selects_that_organism():
    hit = UniProtClient.lookup("equine skeletal myoglobin", "equine skeletal myoglobin", context="")
    assert hit["uniprot_id"] == "P68082"
    assert UniProtClient.lookup("myoglobin", "myoglobin", context="")["uniprot_id"] == "P02144"


def test_core_variant_never_matches_a_gene(monkeypatch):
    HUMAN["gla"] = [_entry("P06280", "Alpha-galactosidase A", ["GLA"])]
    try:
        UniProtClient.candidates.cache_clear()
        assert UniProtClient.lookup("gla protein", "Gla protein", context="") is None
    finally:
        HUMAN.pop("gla")


def test_protein_letter_needs_exact_name():
    HUMAN["protein c"] = [_entry("Q9NQP4", "Prefoldin subunit 4", ["PFDN4"], alt=["Protein C-1"]),
                          _entry("P04070", "Vitamin K-dependent protein C", ["PROC"])]
    try:
        UniProtClient.candidates.cache_clear()
        assert UniProtClient.lookup("protein c", "protein C", context="") is None
    finally:
        HUMAN.pop("protein c")


def test_alternative_name_needs_exact_match():
    HUMAN["fatty acid-binding protein"] = [_entry(
        "P00505", "Aspartate aminotransferase, mitochondrial", ["GOT2"],
        alt=["Plasma membrane-associated fatty acid-binding protein"])]
    try:
        UniProtClient.candidates.cache_clear()
        assert UniProtClient.lookup("fatty acid-binding protein", "FABP", context="") is None
    finally:
        HUMAN.pop("fatty acid-binding protein")


def test_known_and_organism_named_pass_gates():
    known = {"uniprot_id": "P05067", "match_type": "known", "lookup_stage": "known",
             "short_form_fallback": True, "resolved_from": "Aβ42", "surface_form": "Aβ42"}
    assert gate_resolved(known, "amylin or Aβ42 aggregation") is None
    hfq = {"uniprot_id": "P0A6X3", "match_type": "gene", "lookup_stage": "reviewed_any",
           "surface_form": "Hfq", "uniprot_name": "RNA-binding protein Hfq", "organism_in_text": True}
    assert gate_resolved(hfq, "E. coli nucleoid remodeling by Hfq master regulator") is None


@pytest.mark.parametrize("organism,text", [
    ("Triticum aestivum", "globulin 1 and 3 in wheat"),
    ("Pisum sativum", "Pea and Soy 7S globulin amyloid fibrils"),
])
def test_crop_common_names(organism, text):
    assert organism_mentioned(organism, text)


def test_short_name_is_a_symbol():
    e = _entry("Q9H813", "Proton-activated chloride channel", ["PACC1"])
    e["proteinDescription"]["recommendedName"]["shortNames"] = [{"value": "PAC"}]
    hit = UniProtClient._parse_hit(e)
    assert "PAC" in hit["genes"] and "PAC" not in hit["names"]
    pd = {"uniprot_id": "Q9H813", "match_type": "gene", "lookup_stage": "human", "surface_form": "PAc",
          "uniprot_name": "Proton-activated chloride channel"}
    assert gate_resolved(pd, "Streptococcus mutans adhesin P1 (PAc)") == "short_symbol_uncorroborated"


def test_digit_symbol_passes_short_gate():
    pd = {"uniprot_id": "P67809", "match_type": "gene", "lookup_stage": "human", "surface_form": "YB-1",
          "uniprot_name": "Y-box-binding protein 1"}
    assert gate_resolved(pd, "YB-1 AP-CSD forms cross-beta amyloid fibrils") is None


def test_heat_shock_factor_is_regulator():
    from amylodb.mining.resolution import assign_role
    pd = {"protein_role": "modifier", "uniprot_name": "Heat shock factor protein 1", "claim_scores": {}}
    assert assign_role(pd) == "modifier" and pd["role_subtype"] == "regulator"
