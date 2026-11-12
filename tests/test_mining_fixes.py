"""Regression tests for mining defects measured on a real run (444 abstracts).

Each test pins one correction:
  * normaliser no longer maps names *inside* synonyms (53.5 % spurious links)
  * pattern confidence no longer credits absent keys (min confidence was 0.45)
  * template echoes from the LLM are rejected
  * noise names (methods, diseases, reagents, generic phrases) are rejected
  * prion policy: prion-like domains kept, PRPF8 kept, only PrP pathology dropped
  * mention gate drops links whose abstract never names the protein
"""

from types import SimpleNamespace

from amylodb.mining.hybrid.extractor import HybridExtractor, llm_fill_gaps
from amylodb.mining.llm.extractor import LLMExtractor, RuleBasedFilter
from amylodb.mining.ner.normalizer import ProteinNormalizer as N
from amylodb.mining.validation import clean_enum, genuine_mention, looks_like_noise


# --- normaliser -------------------------------------------------------------

def test_generic_or_unrelated_names_are_not_absorbed():
    assert N.normalize("amyloid") == "amyloid"                       # was -> amyloid-beta
    assert N.normalize("PRPF8") == "prpf8"                           # was -> prion
    assert N.normalize("GFP fusion protein") == "gfp fusion protein"  # was -> fus
    assert N.normalize("insulin-degrading enzyme") == "insulin-degrading enzyme"
    assert N.normalize("attractin") == "attractin"                   # was -> transthyretin


def test_genuine_synonyms_still_resolve():
    assert N.normalize("tdp43") == "tdp-43"
    assert N.normalize("TDP-43 proteinopathy") == "tdp-43"
    assert N.normalize("human alpha-synuclein fibrils") == "alpha-synuclein"
    assert N.normalize("FUS LC domain") == "fus"
    assert N.normalize("Tau fibrils") == "tau"


# --- confidence ---------------------------------------------------------------

def test_pattern_confidence_no_credit_for_absent_keys():
    ctx = {"forms_aggregates": True, "aggregation_type": ["fibril"]}
    assert HybridExtractor._calculate_pattern_confidence(HybridExtractor, ctx) == 0.3


def test_pattern_confidence_counts_real_decisions():
    ctx = {"forms_aggregates": True, "reversibility": "irreversible",
           "experimental_methods": ["ThT", "TEM"]}
    assert abs(HybridExtractor._calculate_pattern_confidence(HybridExtractor, ctx) - 0.55) < 1e-9


# --- LLM output validation ------------------------------------------------------

def test_template_echoes_become_unknown():
    assert clean_enum("aggregate_structure", "beta-sheet|alpha-helical|mixed|unknown") == "unknown"
    assert clean_enum("reversibility", "irreversible|unknown") == "unknown"
    assert clean_enum("reversibility", "Conditionally reversible") == "conditionally_reversible"
    assert clean_enum("functional_class", "Pathological") == "pathological"
    assert clean_enum("functional_class", "chaperone") == "unknown"


def test_llm_fill_gaps_fills_only_requested_undecided_fields():
    class FakeLLM:
        provider = "ollama"

        def _call_ollama(self, prompt):
            return ('{"aggregate_structure": "beta-sheet|alpha-helical|mixed|unknown",'
                    ' "reversibility": "irreversible", "functional_class": "functional",'
                    ' "confidence": 0.6}')

    data = {"protein_name": "tau", "reversibility": "unknown", "aggregate_structure": "unknown",
            "functional_class": "pathological", "extraction_confidence": 0.8}
    out = llm_fill_gaps(FakeLLM(), data, "abstract", only={"reversibility", "aggregate_structure"})
    assert out["reversibility"] == "irreversible"          # filled
    assert out["aggregate_structure"] == "unknown"         # echo rejected
    assert out["functional_class"] == "pathological"       # decided value not overwritten
    assert out["extraction_confidence"] == 0.8


# --- noise names ----------------------------------------------------------------

def test_noise_names_rejected():
    for name in ("afm", "AD", "abts", "4-methyl coumarin", "aggregating protein", "ThT", "x"):
        assert looks_like_noise(name), name


def test_real_proteins_not_rejected():
    for name in ("FUS", "alpha-synuclein", "Sup35", "CsgA", "TDP-43", "PRRC2A"):
        assert not looks_like_noise(name), name


# --- prion policy in the mining stack ---------------------------------------------

def test_prefilter_keeps_prion_like_and_drops_scrapie():
    keep, _ = RuleBasedFilter.should_process(
        "FUS phase separation", "The prion-like domain of FUS forms fibrils and aggregates.")
    drop, _ = RuleBasedFilter.should_process(
        "Scrapie", "PrPSc fibrils and aggregates propagate in scrapie-infected sheep.")
    assert keep is True and drop is False


def test_prefilter_counts_plurals():
    _, conf = RuleBasedFilter.should_process("x", "Fibrils and aggregates were seen by ThT.")
    assert conf >= 0.5


def test_parse_response_keeps_prpf8_and_drops_prp():
    llm = LLMExtractor.__new__(LLMExtractor)
    resp = ('{"proteins": [{"protein_name": "PRPF8", "aggregate_structure": "beta-sheet|unknown"},'
            ' {"protein_name": "PrPSc"}], "paper_relevance": "high"}')
    out = llm._parse_response(resp, "1")
    names = [p["protein_name"] for p in out["proteins"]]
    assert names == ["PRPF8"]
    assert out["proteins"][0]["aggregate_structure"] == "unknown"


# --- mention gate -------------------------------------------------------------------

def test_genuine_mention_boundaries():
    assert genuine_mention("Aggregation of amyloid-β was measured.", ["amyloid-β"])
    assert genuine_mention("TDP-43 inclusions were found.", ["TDP-43"])
    assert not genuine_mention("A GFP fusion protein was used.", ["FUS"])


def test_mention_gate_drops_spurious_canonical_link():
    from amylodb.mining.pipeline import AmyloidPipeline

    fake = SimpleNamespace(config=SimpleNamespace(require_uniprot=True, mention_gate=True),
                           stats={"uniprot_dropped": 0, "mention_gate_dropped": 0})
    text = "Amyloid fibrils of lysozyme were imaged by AFM."
    proteins = [
        {"protein_name": "amyloid-beta", "surface_form": "amyloid", "uniprot_id": "P05067"},
        {"protein_name": "lysozyme", "surface_form": "lysozyme", "uniprot_id": "P61626"},
        {"protein_name": "mystery", "surface_form": "mystery", "uniprot_id": ""},
    ]
    kept = AmyloidPipeline._apply_gates(fake, proteins, text)
    assert [p["protein_name"] for p in kept] == ["lysozyme"]
    assert fake.stats == {"uniprot_dropped": 1, "mention_gate_dropped": 1}


def test_mention_gate_keeps_genuine_synonym_and_unknown_identity():
    from amylodb.mining.pipeline import AmyloidPipeline

    fake = SimpleNamespace(config=SimpleNamespace(require_uniprot=False, mention_gate=True),
                           stats={"uniprot_dropped": 0, "mention_gate_dropped": 0})
    text = "α-synuclein and CsgA both form amyloid fibrils in vitro."
    proteins = [
        {"protein_name": "alpha-synuclein", "surface_form": "α-synuclein"},   # known canonical
        {"protein_name": "csga", "surface_form": "CsgA"},                      # not in synonym table
    ]
    kept = AmyloidPipeline._apply_gates(fake, proteins, text)
    assert [p["protein_name"] for p in kept] == ["alpha-synuclein", "csga"]
    assert fake.stats["mention_gate_dropped"] == 0
