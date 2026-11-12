"""Provenance, evidence tiers and prion classification."""

from amylodb.provenance import (
    EvidenceTier,
    PrionClass,
    classify_prion,
    evidence_tier_from_curated,
    tier_weight,
)


def test_tier_order_and_weights_monotone():
    tiers = list(EvidenceTier)
    assert tiers == sorted(tiers)
    weights = [tier_weight(t) for t in tiers]
    assert weights == sorted(weights)
    # structural strictly dominates text-mined
    assert tier_weight(EvidenceTier.STRUCTURAL) > tier_weight(EvidenceTier.TEXT_MINED)


def test_curated_mapping_defaults_to_weakest():
    assert evidence_tier_from_curated("structural") == EvidenceTier.STRUCTURAL
    assert evidence_tier_from_curated("kinetic") == EvidenceTier.KINETIC
    assert evidence_tier_from_curated("") == EvidenceTier.COMPUTATIONAL
    assert evidence_tier_from_curated("nonsense") == EvidenceTier.COMPUTATIONAL


def test_prion_substring_false_positives_are_not_flagged():
    # finding 2: naive substring matching wrongly caught these
    for name in ("PRPF8", "PRPF40A", "PRPS1", "SPRPP"):
        assert classify_prion(name) == PrionClass.NOT_PRION, name


def test_prion_like_domains_are_kept_and_tagged():
    for text in ("FUS prion-like domain", "hnRNPA1 low-complexity domain",
                 "TDP-43 PLD", "G3BP1"):
        assert classify_prion(text) == PrionClass.PRION_LIKE_DOMAIN, text


def test_transmissible_prion_detected():
    for text in ("PrPSc scrapie", "Creutzfeldt-Jakob disease", "bovine spongiform encephalopathy"):
        assert classify_prion(text) == PrionClass.TRANSMISSIBLE_PRION, text


def test_prion_like_wins_over_transmissible_mention():
    # a paper contrasting a prion-like domain with PrP is kept
    assert classify_prion("prion-like domain compared to PrPSc") == PrionClass.PRION_LIKE_DOMAIN
