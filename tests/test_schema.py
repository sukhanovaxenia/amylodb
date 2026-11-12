"""Unified record: lift from curated, evidence/weight coupling, dedup levels."""

from amylodb.curated.models import AmyloidEntry
from amylodb.provenance import EvidenceTier, Provenance, RecordLevel
from amylodb.schema import AmyloidRecord


def test_from_curated_sets_region_level_and_tier():
    e = AmyloidEntry(source_db="WALTZ-DB", sequence="STVIIE",
                     region_start=1, region_end=6, experimental_label="amyloid",
                     is_amyloid=True, evidence_type="structural")
    r = AmyloidRecord.from_curated(e)
    assert r.record_level == RecordLevel.REGION
    assert r.provenance == Provenance.CURATED
    assert r.evidence_tier == EvidenceTier.STRUCTURAL
    # tier is the single source of truth for the voting weight
    assert r.evidence_weight == 3.0


def test_protein_and_region_keys_never_collide():
    region = AmyloidRecord(source_db="x", sequence="STVIIE", uniprot_id="P1",
                           region_start=1, region_end=6, record_level=RecordLevel.REGION)
    protein = AmyloidRecord(source_db="y", uniprot_id="P1",
                            record_level=RecordLevel.PROTEIN)
    assert region.dedup_key()[0] == "region"
    assert protein.dedup_key()[0] == "protein"
    assert region.dedup_key() != protein.dedup_key()
