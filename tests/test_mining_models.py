"""Model wrappers and the model-based extractor, with stub models (no weights)."""

import math
from types import SimpleNamespace

import pytest

from amylodb.mining.models.nli import CLAIMS, ClaimClassifier, NLIThresholds, decide, focus_premise
from amylodb.mining.models.openbioner import OpenBioNERExtractor
from amylodb.mining.models.structner import (ProteinStructureNER, Span, method_texts,
                                             methods_from_spans, protein_names_from_spans)
from amylodb.mining.models.triage import RelevanceTriage, detect_positive_index
from amylodb.mining.ner.extractor import BioEntity, NERResult
from amylodb.provenance import EvidenceTier


def _scores(**kw):
    base = {k: 0.05 for k in CLAIMS}
    base.update(kw)
    return base


# --- NLI decision rules -------------------------------------------------------------

def test_clear_irreversible_pathological_beta():
    d = decide(_scores(aggregates=0.95, irreversible=0.9, reversible=0.1, beta=0.88,
                       pathological=0.92, autocatalytic=0.9))
    assert d["forms_aggregates"] is True
    assert d["reversibility"] == "irreversible"
    assert d["aggregate_structure"] == "beta-sheet"
    assert d["functional_class"] == "pathological"
    assert d["assembly_mechanism"] == "autocatalytic"
    assert d["undecided"] == []


def test_assembly_mechanism_cofactor_and_unknown():
    assert decide(_scores(aggregates=0.9, cofactor_dependent=0.85))["assembly_mechanism"] == "cofactor_dependent"
    d = decide(_scores(aggregates=0.9))                      # no assembly claim entailed
    assert d["assembly_mechanism"] == "unknown"
    assert "assembly_mechanism" in d["undecided"]


def test_disaggregase_is_not_called_an_aggregator():
    # Hsp70-like: strong disaggregation signal, present in an aggregation abstract
    d = decide(_scores(aggregates=0.75, disaggregase=0.97))
    assert d["forms_aggregates"] is False
    assert d["is_disaggregase"] is True
    # dual-role protein (both strong) is kept as an aggregator
    d2 = decide(_scores(aggregates=0.95, disaggregase=0.9))
    assert d2["forms_aggregates"] is True


def test_token_budgeted_chunking_respects_limit_and_offsets():
    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return list(range(len(s.split())))                     # 1 token/word
        def __call__(self, s, add_special_tokens=False, return_offsets_mapping=True):
            offs, i = [], 0
            for w in s.split():
                j = s.find(w, i); offs.append((j, j + len(w))); i = j + len(w)
            return {"offset_mapping": offs}

    class _Pipe:
        tokenizer = _Tok()
        def __call__(self, chunk):
            k = chunk.find("FUS")
            return [{"entity_group": "protein", "score": 0.9, "start": k, "end": k + 3}] if k >= 0 else []

    text = "Alpha synuclein aggregates here. FUS forms droplets there. Tau binds microtubules now."
    ner = ProteinStructureNER(pipe=_Pipe(), max_tokens=4)
    chunks = ner._chunks(text)
    assert chunks and all(len(c.split()) <= 4 for _, c in chunks)   # no chunk exceeds the budget
    assert all(text[o:o + len(c)] == c for o, c in chunks)          # offsets reconstruct the original
    spans = ner.extract(text)
    assert any(s.text == "FUS" and text[s.start:s.end] == "FUS" for s in spans)


def test_both_reversibility_claims_entailed_is_conditional():
    d = decide(_scores(aggregates=0.9, reversible=0.8, irreversible=0.85))
    assert d["reversibility"] == "conditionally_reversible"


def test_close_call_is_undecided_not_guessed():
    d = decide(_scores(aggregates=0.9, reversible=0.75, irreversible=0.6))
    assert d["reversibility"] == "unknown"
    assert "reversibility" in d["undecided"]


def test_low_llps_is_unknown_never_no():
    assert decide(_scores(aggregates=0.9, llps=0.01))["undergoes_llps"] == "unknown"
    assert decide(_scores(llps=0.9))["undergoes_llps"] == "yes"


def test_both_entailed_structure_and_function():
    d = decide(_scores(aggregates=0.9, beta=0.8, alpha=0.75, functional=0.8, pathological=0.8))
    assert d["aggregate_structure"] == "mixed" and d["functional_class"] == "both"


def test_no_claims_low_confidence_and_nothing_undecided():
    d = decide(_scores(aggregates=0.3))
    assert d["forms_aggregates"] is False
    assert d["undecided"] == []
    assert d["extraction_confidence"] == pytest.approx(0.15)


def test_threshold_is_configurable():
    d = decide(_scores(aggregates=0.65), NLIThresholds(positive=0.6))
    assert d["forms_aggregates"] is True


def test_focus_premise_selects_protein_sentences():
    text = "Tau binds microtubules. FUS forms droplets. FUS droplets harden over time."
    assert focus_premise(text, ["FUS"]) == "FUS forms droplets. FUS droplets harden over time."
    assert focus_premise(text, ["SOD1"]) == text


def test_focus_premise_includes_anaphoric_next_sentence():
    # the evidence continues in the next sentence without repeating the name
    text = ("Alpha-synuclein fibrils were solved by cryo-EM. The fibrils resisted disaggregation "
            "and remained stable in 4 M guanidine. An unrelated assay followed.")
    prem = focus_premise(text, ["alpha-synuclein"])
    assert "resisted disaggregation" in prem           # anaphoric evidence captured
    assert "unrelated assay" not in prem               # window does not run on


def test_claim_classifier_calls_zero_shot_correctly():
    calls = {}

    def fake_pipe(premise, candidate_labels, hypothesis_template, multi_label, batch_size):
        calls.update(premise=premise, template=hypothesis_template, multi=multi_label)
        return {"labels": candidate_labels,
                "scores": [0.9 if "fibrils" in h else 0.1 for h in candidate_labels]}

    scores = ClaimClassifier(pipe=fake_pipe).score("FUS forms fibrils.", "FUS")
    assert calls == {"premise": "FUS forms fibrils.", "template": "{}", "multi": True}
    assert scores["aggregates"] == 0.9 and scores["llps"] == 0.1
    assert set(scores) == set(CLAIMS)


# --- structure NER ----------------------------------------------------------------------

def _fake_token_pipe(keywords):
    """Token-classification stub: finds keywords in the chunk, chunk-relative offsets."""
    def pipe(chunk):
        out = []
        for word, label in keywords.items():
            i = chunk.find(word)
            if i >= 0:
                out.append({"entity_group": label, "word": word, "score": 0.97,
                            "start": i, "end": i + len(word)})
        return out
    return pipe


def test_structner_offsets_and_labels():
    text = ("Alpha-synuclein fibrils were solved by cryo-EM. " * 30) + "ThT binding confirmed SOD1 aggregation."
    ner = ProteinStructureNER(pipe=_fake_token_pipe({"cryo-EM": "experimental_method",
                                                     "ThT": "experimental_method",
                                                     "SOD1": "protein"}))
    spans = ner.extract(text)
    assert all(text[s.start:s.end] == s.text for s in spans)      # offsets map back to the original
    assert "SOD1" in protein_names_from_spans(spans)
    assert {"cryo-em", "tht"} <= method_texts(spans)


def test_methods_mapped_to_universal_names_and_tier():
    spans = [Span("cryo-EM", "experimental_method", 0, 7), Span("ThT", "experimental_method", 8, 11),
             Span("SPR", "experimental_method", 12, 15), Span("FUS", "protein", 16, 19)]
    methods, tier = methods_from_spans(spans)
    assert "Cryo-EM" in methods and "ThT binding" in methods and "SPR" in methods
    assert tier == int(EvidenceTier.STRUCTURAL)


def test_no_recognised_method_gives_no_tier():
    assert methods_from_spans([Span("SPR", "experimental_method", 0, 3)]) == (["SPR"], -1)


# --- triage ------------------------------------------------------------------------------

@pytest.mark.parametrize("id2label,expected", [
    ({0: "LABEL_0", 1: "LABEL_1"}, 1),
    ({0: "positive", 1: "negative"}, 0),
    ({0: "not useful", 1: "useful"}, 1),
    ({"0": "Negative (not useful)", "1": "Positive (useful)"}, 1),
])
def test_positive_label_detection(id2label, expected):
    assert detect_positive_index(id2label) == expected


def test_annotate_never_drops_gate_does():
    scorer = lambda texts: [0.1 for _ in texts]
    annotate = RelevanceTriage(mode="annotate", scorer=scorer)
    gate = RelevanceTriage(mode="gate", threshold=0.5, scorer=scorer)
    s = annotate.score(["a"])[0]
    assert annotate.keep(s) is True and gate.keep(s) is False


def test_off_mode_scores_nan_and_keeps():
    t = RelevanceTriage(mode="off")
    s = t.score(["a"])[0]
    assert math.isnan(s) and t.keep(s)


def test_invalid_triage_mode_rejected():
    with pytest.raises(ValueError):
        RelevanceTriage(mode="drop-everything")


# --- OpenBioNER -------------------------------------------------------------------------

def test_openbioner_separates_types_and_filters_noise():
    ents = [SimpleNamespace(label_="protein", text="FUS", start_char=0, end_char=3),
            SimpleNamespace(label_="protein", text="aggregating protein", start_char=4, end_char=23),
            SimpleNamespace(label_="experimental_method", text="FRAP", start_char=24, end_char=28),
            SimpleNamespace(label_="disease", text="ALS", start_char=29, end_char=32),
            SimpleNamespace(label_="chemical", text="ATP", start_char=33, end_char=36),
            SimpleNamespace(label_="condensate", text="stress granule", start_char=37, end_char=51),
            SimpleNamespace(label_="peptide", text="Abeta42", start_char=52, end_char=59)]
    ner = OpenBioNERExtractor(nlp=lambda text: SimpleNamespace(ents=ents))
    res = ner.extract("irrelevant")
    assert sorted(res.protein_names()) == ["Abeta42", "FUS"]
    assert [e.text for e in res.methods] == ["FRAP"]
    assert [e.text for e in res.condensates] == ["stress granule"]
    assert [e.text for e in res.diseases] == ["ALS"] and [e.text for e in res.chemicals] == ["ATP"]


# --- model-based extractor end to end ----------------------------------------------------

class _FakeNER:
    def __init__(self, names):
        self.names = names

    def extract(self, text):
        return NERResult(proteins=[BioEntity(n, "PROTEIN", 0, 0) for n in self.names])


class _FakeNLI:
    def score(self, premise, protein):
        if protein == "alpha-synuclein":
            return _scores(aggregates=0.96, irreversible=0.91, reversible=0.05, beta=0.9,
                           pathological=0.93, autocatalytic=0.9)
        if protein == "FUS":
            return _scores(aggregates=0.8, llps=0.92, reversible=0.85, irreversible=0.82)
        if protein == "tau":
            return _scores(aggregates=0.9)          # everything else undecided -> LLM
        return _scores()                            # HSP70: no claim


class _FakeLLM:
    provider = "ollama"

    def __init__(self):
        self.prompts = 0

    def _call_ollama(self, prompt):
        self.prompts += 1
        return '{"reversibility": "irreversible", "aggregate_structure": "beta-sheet", "functional_class": "pathological"}'


def test_model_extractor_end_to_end():
    from amylodb.mining.models_extractor import ModelHybridExtractor

    text_title = "Aggregation of alpha-synuclein, FUS and tau"
    abstract = ("Alpha-synuclein fibrils were solved by cryo-EM and bind ThT. "
                "FUS droplets formed by phase separation. Tau aggregates were detected. "
                "HSP70 was used as a control. AFM imaging was performed.")
    structner = ProteinStructureNER(pipe=_fake_token_pipe({"cryo-EM": "experimental_method",
                                                           "ThT": "experimental_method",
                                                           "AFM": "experimental_method"}))
    llm = _FakeLLM()
    ex = ModelHybridExtractor(ner=_FakeNER(["alpha-synuclein", "FUS", "tau", "HSP70", "AFM",
                                            "aggregating protein", "SOD1"]),
                              nli=_FakeNLI(), structner=structner, llm=llm)
    res = ex.extract(text_title, abstract, "123")
    by = {p["protein_name"]: p for p in res.proteins}

    assert set(by) == {"alpha-synuclein", "FUS", "tau"}
    a = by["alpha-synuclein"]
    assert (a["reversibility"], a["aggregate_structure"], a["functional_class"]) == \
        ("irreversible", "beta-sheet", "pathological")
    assert a["reported_method_tier"] == int(EvidenceTier.STRUCTURAL)
    assert "Cryo-EM" in a["experimental_methods"]
    assert by["FUS"]["undergoes_llps"] == "yes"
    assert by["FUS"]["reversibility"] == "conditionally_reversible"   # NLI decision kept
    # fully decided by NLI -> no LLM call; partly undecided -> LLM for the gaps only
    assert a["extraction_method"] == "models"
    assert by["tau"]["extraction_method"] == "models+llm"
    assert by["tau"]["reversibility"] == "irreversible"
    assert by["FUS"]["extraction_method"] == "models+llm"
    assert llm.prompts == 2
    st = ex.get_stats()
    assert st["dropped_noise"] == 2          # AFM, aggregating protein
    assert st["dropped_no_mention"] == 1     # SOD1 is not in the text
    assert st["dropped_no_claim"] == 1       # HSP70
