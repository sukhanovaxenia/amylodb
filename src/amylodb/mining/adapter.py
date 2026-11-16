"""Adapter: literature-mined extractions -> unified `AmyloidRecord`.

This is the boundary between the mining stack (which produces protein-level
`ExtractedProtein` records from abstracts) and the reconciliation schema. It is
pure data mapping -- no network, no models -- so it is unit-tested here even
though the mining stages themselves run on the user's machine.

Consumes: a mined record as an `ExtractedProtein` instance or a plain dict with
the same fields.
Emits: an `AmyloidRecord` at PROTEIN level, provenance MINED, evidence tier
TEXT_MINED, prion class assigned by `classify_prion` (so prion-like-domain RBPs
are kept and tagged rather than dropped).
Invariant: a mined record never carries a sequence (sequences are fetched
downstream from UniProt), so its feature row is populated only after enrichment.
"""

from __future__ import annotations

from typing import Dict, List, Union

from ..provenance import (CURATOR_TIERS, EvidenceTier, Provenance, RecordLevel, aggregation_weight,
                          classify_prion, is_candidate_support, tier_weight)
from ..schema import AmyloidRecord

# aggregate_structure -> curated secondary_structure_class vocabulary
_STRUCTURE_MAP = {
    "beta-sheet": "cross_beta",
    "alpha-helical": "cross_alpha",
    "mixed": "unknown",
    "amorphous": "unknown",
    "unknown": "unknown",
}

# functional_class -> (is_functional, is_pathological)
_FUNCTIONAL_MAP = {
    "functional": (True, False),
    "pathological": (False, True),
    "both": (True, True),
    "unknown": (None, None),
}


def _get(obj: Union[Dict, object], key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def mined_to_record(mined: Union[Dict, object]) -> AmyloidRecord:
    """Map one mined extraction into an `AmyloidRecord`."""
    forms = bool(_get(mined, "forms_aggregates", True))
    func = (_get(mined, "functional_class", "unknown") or "unknown").lower()
    is_functional, is_pathological = _FUNCTIONAL_MAP.get(func, (None, None))
    pmid = _get(mined, "pmid", "") or ""
    # Accept a full pmid list from the exporter; fall back to the single pmid.
    pmids = list(_get(mined, "source_pmids", None) or _get(mined, "pmids", None) or
                 ([pmid] if pmid else []))
    name = _get(mined, "protein_name", "") or ""
    role = _get(mined, "biological_role", "") or ""
    disease = _get(mined, "associated_disease", "") or ""

    rec = AmyloidRecord(
        source_db="LLM-mined",
        provenance=Provenance.MINED,
        record_level=RecordLevel.PROTEIN,
        protein_name=name,
        gene_name=_get(mined, "gene_name", "") or "",
        uniprot_id=_get(mined, "uniprot_id", "") or "",
        organism=_get(mined, "organism", "") or "",
        sequence="",  # fetched downstream from UniProt
        is_amyloid=forms,
        experimental_label="amyloid" if forms else "non-amyloid",
        category="literature_mined",
        secondary_structure_class=_STRUCTURE_MAP.get(
            (_get(mined, "aggregate_structure", "unknown") or "unknown").lower(), "unknown"),
        reversibility=_get(mined, "reversibility", "unknown") or "unknown",
        assembly_mechanism=_get(mined, "assembly_mechanism", "unknown") or "unknown",
        is_functional=is_functional,
        is_pathological=is_pathological,
        biological_role=role,
        disease=disease,
        cofactors_assembly=list(_get(mined, "assembly_cofactors", []) or []),
        cofactors_disassembly=list(_get(mined, "disassembly_cofactors", []) or []),
        string_partners=list(_get(mined, "string_partners", []) or []),
        evidence_type="text_mined",
        evidence_tier=EvidenceTier.TEXT_MINED,
        evidence_weight=tier_weight(EvidenceTier.TEXT_MINED),
        extraction_confidence=float(_get(mined, "extraction_confidence", 0.0) or 0.0),
        pmid=pmid or (pmids[0] if pmids else ""),
        source_pmids=pmids,
    )
    rec.aggregation_evidence = _get(mined, "aggregation_evidence", "") or ""
    rec.evidence_basis = _get(mined, "evidence_basis", "") or ""
    rec.evidence_items = [dict(x) for x in (_get(mined, "evidence_items", None) or [])]
    # A predicted amyloid (model_only) or an LCR-based candidate votes with half
    # the weight of a text-supported record; an unresolved one does not vote.
    rec.evidence_weight = (tier_weight(EvidenceTier.TEXT_MINED)
                           * aggregation_weight(rec.aggregation_evidence, rec.evidence_basis))
    if rec.aggregation_evidence in CURATOR_TIERS:
        # Only the curator states decide the label from the support; the older
        # tiers keep the forms_aggregates label (a text record with
        # forms_aggregates False stays non-amyloid).
        if is_candidate_support(rec.aggregation_evidence, rec.evidence_basis):
            rec.is_amyloid, rec.experimental_label = None, "candidate"
        else:
            rec.is_amyloid, rec.experimental_label = True, "amyloid"
    llps = (_get(mined, "undergoes_llps", "unknown") or "unknown")
    rec.undergoes_llps = True if str(llps).lower() in ("yes", "true", "1") else None
    rec.claim_scores = dict(_get(mined, "claim_scores", {}) or {})
    rec.reversibility_votes = dict(_get(mined, "reversibility_votes", {}) or {})
    rec.reported_methods = list(_get(mined, "experimental_methods", []) or [])
    tier = _get(mined, "reported_method_tier", None)
    rec.reported_method_tier = int(tier) if tier not in (None, "") else -1
    if rec.reported_methods:
        rec.raw_method = "; ".join(rec.reported_methods)
    # Gene symbol decides the mammalian prion protein unambiguously.
    gene = (rec.gene_name or "").upper()
    rec.prion_class = classify_prion(name, role, disease)
    if gene == "PRNP" and rec.prion_class.value == "not_prion":
        from ..provenance import PrionClass
        rec.prion_class = PrionClass.TRANSMISSIBLE_PRION
    return rec


def is_modifier(mined: Union[Dict, object]) -> bool:
    """True for anything that is not an aggregator: modifiers (chaperones,
    inhibitors, regulators, co-deposits) and condensate candidates. Neither
    becomes an amyloid record; `split_mined` and the build keep them apart."""
    return (_get(mined, "protein_role", "aggregator") or "aggregator") != "aggregator"


def is_condensate(mined: Union[Dict, object]) -> bool:
    """True for a condensatedb candidate: the condensate role, or condensate
    evidence beside an amyloid role (TDP-43 in stress granules)."""
    return ((_get(mined, "protein_role", "") or "") == "condensate"
            or bool(_get(mined, "condensate_evidence", False)))


def has_modifier_role(mined: Union[Dict, object]) -> bool:
    """True for a modifier, or an aggregator that some paper shows acting on
    another protein's aggregation (gelsolin)."""
    return ((_get(mined, "protein_role", "") or "") == "modifier"
            or bool(_get(mined, "modifier_roles", None)))


def split_mined(mined_items: List[Union[Dict, object]]):
    """(aggregator items, side items).

    Aggregators become amyloid records. Side items feed modifiers.tsv and
    condensate_candidates.tsv: every protein with a non-aggregator primary
    role, plus dual-role aggregators and aggregators with condensate evidence,
    which therefore appear in both lists.
    """
    aggs = [m for m in mined_items if not is_modifier(m)]
    side = [m for m in mined_items if is_modifier(m) or has_modifier_role(m) or is_condensate(m)]
    return aggs, side


def mined_batch_to_records(mined_items: List[Union[Dict, object]]) -> List[AmyloidRecord]:
    """Map a batch of mined extractions into records, skipping modifiers.

    A chaperone that dissolves aggregates is evidence about the disassembly
    machinery, not an amyloid-forming protein, so it does not become an
    AmyloidRecord; see `split_mined` and the modifiers.tsv output.
    """
    return [mined_to_record(m) for m in mined_items if not is_modifier(m)]
