"""Reconciliation: merge, tier-weighted conflict resolution, prion handling."""

from amylodb.provenance import EvidenceTier, PrionClass, Provenance, RecordLevel
from amylodb.reconcile import reconcile
from amylodb.schema import AmyloidRecord
from amylodb.mining.adapter import mined_to_record


def _region(seq, uni, label, tier, src, prion=PrionClass.NOT_PRION, pmids=None):
    return AmyloidRecord(
        source_db=src, sequence=seq, uniprot_id=uni, region_start=1, region_end=len(seq),
        experimental_label=label, is_amyloid=(label == "amyloid"),
        record_level=RecordLevel.REGION, evidence_tier=tier,
        provenance=Provenance.CURATED, prion_class=prion, source_pmids=pmids or [],
    )


def test_agreeing_duplicates_merge_to_max_tier_and_union_pmids():
    a = _region("STVIIE", "P1", "amyloid", EvidenceTier.STAINING_BINDING, "WALTZ", pmids=["111"])
    b = _region("STVIIE", "P1", "amyloid", EvidenceTier.STRUCTURAL, "Atlas", pmids=["222"])
    res = reconcile([a, b])
    assert res.stats["consensus"] == 1
    rec = res.consensus[0]
    assert rec.evidence_tier == EvidenceTier.STRUCTURAL
    assert set(rec.source_pmids) == {"111", "222"}


def test_conflict_resolved_by_tier_weight():
    a = _region("GNNQQNY", "P2", "amyloid", EvidenceTier.STRUCTURAL, "CrossBeta")
    b = _region("GNNQQNY", "P2", "non-amyloid", EvidenceTier.STAINING_BINDING, "AmyLoad")
    res = reconcile([a, b])
    assert res.stats["conflicts"] == 0
    assert res.consensus[0].experimental_label == "amyloid"


def test_too_close_conflict_is_reported_not_guessed():
    a = _region("AAAAAA", "P3", "amyloid", EvidenceTier.STAINING_BINDING, "X")
    b = _region("AAAAAA", "P3", "non-amyloid", EvidenceTier.STAINING_BINDING, "Y")
    res = reconcile([a, b])
    assert res.stats["conflicts"] == 1
    assert res.stats["consensus"] == 0


def test_transmissible_exclusion_keeps_prion_like():
    fus = AmyloidRecord(source_db="mined", uniprot_id="P35637", protein_name="FUS",
                        record_level=RecordLevel.PROTEIN, provenance=Provenance.MINED,
                        evidence_tier=EvidenceTier.TEXT_MINED, experimental_label="amyloid",
                        prion_class=PrionClass.PRION_LIKE_DOMAIN)
    prp = AmyloidRecord(source_db="mined", uniprot_id="P04156", protein_name="PrP",
                        record_level=RecordLevel.PROTEIN, provenance=Provenance.MINED,
                        evidence_tier=EvidenceTier.TEXT_MINED, experimental_label="amyloid",
                        prion_class=PrionClass.TRANSMISSIBLE_PRION)
    res = reconcile([fus, prp], exclude_transmissible_prion=True)
    kinds = {r.uniprot_id for r in res.consensus}
    assert "P35637" in kinds and "P04156" not in kinds


def _mined(support, forms=True, confidence=0.0):
    return mined_to_record({
        "uniprot_id": "P12345", "protein_name": "example",
        "aggregation_evidence": support, "forms_aggregates": forms,
        "extraction_confidence": confidence,
    })


def test_model_only_duplicates_keep_discounted_weight():
    result = reconcile([_mined("model_only"), _mined("model_only")])
    assert len(result.consensus) == 1
    assert result.consensus[0].aggregation_evidence == "model_only"
    assert result.consensus[0].evidence_weight == 0.125


def test_text_support_restores_full_weight_in_mixed_group():
    result = reconcile([_mined("model_only", confidence=0.9),
                        _mined("text", confidence=0.1)])
    assert len(result.consensus) == 1
    assert result.consensus[0].aggregation_evidence == "text"
    assert result.consensus[0].evidence_weight == 0.25


def test_text_label_outvotes_conflicting_model_only_label():
    result = reconcile([_mined("model_only"), _mined("text", forms=False)])
    assert result.conflicts == []
    assert len(result.consensus) == 1
    assert result.consensus[0].experimental_label == "non-amyloid"
    assert result.consensus[0].evidence_weight == 0.25
