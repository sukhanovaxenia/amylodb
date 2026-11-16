"""Unified amyloid record: the reconciliation schema.

`AmyloidRecord` is a superset of the curated pipeline's `AmyloidEntry`. It adds
the fields a literature-mined, protein-level record needs (reversibility,
assembly mechanism, functional/pathological stance, cofactors, interaction
partners), plus the cross-origin bookkeeping the reconciliation layer depends
on: provenance, record level, prion class and a single ordered evidence tier.

Reusing `AmyloidEntry` by subclassing keeps one source of truth for the ~40
curated fields and the physicochemical / composition feature computation; this
module only extends it.

Consumes: nothing at import time.
Emits: `AmyloidRecord`.
Invariants:
  * `dedup_key` includes `record_level`, so a PROTEIN record is never merged
    with a REGION record (the two identity notions are incompatible).
  * `evidence_tier` is always set; it defaults to the weakest tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .curated.models import AmyloidEntry
from .provenance import (
    EvidenceTier,
    PrionClass,
    Provenance,
    RecordLevel,
    classify_prion,
    evidence_tier_from_curated,
    tier_weight,
)


@dataclass
class AmyloidRecord(AmyloidEntry):
    """A unified record spanning curated and mined origins.

    Inherits every curated field (identifiers, region, sequence, 11-level
    structural classification, pathogenicity, experimental evidence, references,
    computed features) from `AmyloidEntry`.
    """

    # --- cross-origin bookkeeping ---
    provenance: Provenance = Provenance.CURATED
    record_level: RecordLevel = RecordLevel.REGION
    prion_class: PrionClass = PrionClass.NOT_PRION
    evidence_tier: EvidenceTier = EvidenceTier.COMPUTATIONAL

    # --- literature-mined extensions (empty for most curated records) ---
    gene_name: str = ""
    reversibility: str = "unknown"        # reversible, irreversible, unknown
    assembly_mechanism: str = "unknown"   # autocatalytic, cofactor_dependent, unknown
    is_functional: Optional[bool] = None
    is_pathological: Optional[bool] = None
    biological_role: str = ""
    cofactors_assembly: List[str] = field(default_factory=list)
    cofactors_disassembly: List[str] = field(default_factory=list)
    string_partners: List[str] = field(default_factory=list)
    extraction_confidence: float = 0.0     # 0..1, meaningful only for mined records
    source_pmids: List[str] = field(default_factory=list)

    # --- model-based mining (NLI / structure NER) ---
    undergoes_llps: Optional[bool] = None  # True only on positive evidence; never inferred False
    claim_scores: Dict[str, float] = field(default_factory=dict)   # max NLI entailment per claim
    reversibility_votes: Dict[str, int] = field(default_factory=dict)  # per-paper decisions
    reported_methods: List[str] = field(default_factory=list)
    reported_method_tier: int = -1         # strongest method the source papers report (not verified)
    # Mined protein records only: how the aggregator role is supported --
    # "curated" (built-in list), "text" (a sentence says the protein itself
    # aggregates) or "model_only" (the NLI vote alone: a predicted amyloid,
    # kept and marked, about 1 in 4 confirmed on review). "" elsewhere.
    aggregation_evidence: str = ""
    # For aggregation_evidence "curator_review": "aggregation_text" or
    # "LCR_text"; "unresolved_attribution" beside "unresolved". "" elsewhere.
    evidence_basis: str = ""
    # Individual curator decisions behind this record, one dict per source row
    # (decision, basis, supporting PMIDs, region/construct scope, ...). Kept
    # through alias unification and consensus whichever label wins, so the
    # scope of a peptide or construct result is never lost (docs 9.16).
    evidence_items: List[Dict] = field(default_factory=list)

    # Redeclared from AmyloidEntry. is_amyloid is three-valued: None for a
    # candidate (neither confirmed nor refuted). evidence_weight None means
    # "not set": __post_init__ fills it from the tier, while an explicit 0
    # (an unresolved record that must not vote) is kept.
    is_amyloid: Optional[bool] = True
    evidence_weight: Optional[float] = None

    def __post_init__(self) -> None:
        # Keep the inherited scalar evidence_weight aligned with the unified tier
        # so the curated unifier's weighted voting keeps working unchanged.
        if self.evidence_weight is None:
            self.evidence_weight = tier_weight(self.evidence_tier)

    def dedup_key(self) -> Tuple:
        """Identity key that respects record level.

        REGION records key on (level, sequence, protein identity, coordinates).
        PROTEIN records key on (level, uniprot or protein name) -- never on a
        region, so they cannot collide with REGION records.
        """
        if self.record_level == RecordLevel.PROTEIN:
            anchor = self.uniprot_id or self.protein_name.lower()
            return (RecordLevel.PROTEIN.value, anchor)
        # region level: reuse the curated biological key, prefixed by level
        return (RecordLevel.REGION.value,) + tuple(super().get_dedup_key())

    @classmethod
    def from_curated(cls, entry: AmyloidEntry) -> "AmyloidRecord":
        """Lift a curated `AmyloidEntry` into a unified `AmyloidRecord`.

        Infers record level from whether a specific region is asserted, and maps
        the curated ``evidence_type`` onto the unified tier.
        """
        data = {k: v for k, v in vars(entry).items() if not k.startswith("_")}
        rec = cls(**data)
        rec.provenance = Provenance.CURATED
        has_region = entry.region_start is not None or entry.region_end is not None
        rec.record_level = RecordLevel.REGION if (entry.sequence and len(entry.sequence) < 60) or has_region else RecordLevel.PROTEIN
        rec.evidence_tier = evidence_tier_from_curated(entry.evidence_type)
        # Unified tier is the single source of truth for the voting weight, so
        # tier and weight can never drift apart during reconciliation.
        rec.evidence_weight = tier_weight(rec.evidence_tier)
        # Curated records are prion-classified too, so prion-like-domain RBPs are
        # tagged (and kept) rather than silently passing as ordinary amyloids.
        rec.prion_class = classify_prion(entry.protein_name, entry.notes, entry.disease)
        # carry over the private feature caches if already computed
        rec._physicochemical = getattr(entry, "_physicochemical", None)
        rec._composition = getattr(entry, "_composition", None)
        return rec
