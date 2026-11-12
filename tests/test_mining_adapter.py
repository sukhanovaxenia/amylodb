"""Mining adapter: mined extraction -> unified record (pure, no network)."""

from amylodb.mining.adapter import mined_batch_to_records, mined_to_record
from amylodb.provenance import EvidenceTier, PrionClass, Provenance, RecordLevel


def test_mined_record_is_protein_level_text_mined_no_sequence():
    rec = mined_to_record({
        "protein_name": "Alpha-synuclein", "uniprot_id": "P37840",
        "forms_aggregates": True, "aggregate_structure": "beta-sheet",
        "reversibility": "irreversible", "functional_class": "pathological",
        "associated_disease": "Parkinson", "pmid": "12345",
        "extraction_confidence": 0.8,
    })
    assert rec.record_level == RecordLevel.PROTEIN
    assert rec.provenance == Provenance.MINED
    assert rec.evidence_tier == EvidenceTier.TEXT_MINED
    assert rec.sequence == ""              # fetched downstream
    assert rec.secondary_structure_class == "cross_beta"
    assert rec.is_pathological is True and rec.is_functional is False
    assert rec.source_pmids == ["12345"]


def test_prion_like_domain_is_tagged_not_dropped():
    rec = mined_to_record({"protein_name": "FUS", "biological_role": "prion-like domain RBP",
                           "forms_aggregates": True})
    assert rec.prion_class == PrionClass.PRION_LIKE_DOMAIN


def test_accepts_dataclass_like_object():
    class P:
        protein_name = "TDP-43"
        uniprot_id = "Q13148"
        forms_aggregates = True
        functional_class = "both"
        aggregate_structure = "mixed"
        pmid = "999"
        extraction_confidence = 0.5
    rec = mined_to_record(P())
    assert rec.is_functional is True and rec.is_pathological is True
    assert rec.prion_class == PrionClass.PRION_LIKE_DOMAIN  # TDP-43 is a known PLD RBP


def test_batch():
    recs = mined_batch_to_records([{"protein_name": "A"}, {"protein_name": "B"}])
    assert len(recs) == 2 and all(r.provenance == Provenance.MINED for r in recs)
