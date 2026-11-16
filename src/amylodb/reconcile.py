"""Reconcile curated and mined records into one consensus dataset.

This is the component the whole database hinges on, and the one most likely to
go wrong: curated records are region-level with a sequence, mined records are
protein-level and often carry no coordinates, so a naive merge either fails to
join them or collapses distinct biology. The level-aware `dedup_key` from
`schema` keeps the two identity notions apart; this module groups by that key,
resolves label conflicts by tier-weighted voting, and merges the surviving
annotations.

Consumes: a list of `AmyloidRecord` (curated already lifted via
`AmyloidRecord.from_curated`, mined built directly).
Emits: a `ReconcileResult` (consensus records, conflicts, stats).
Invariants:
  * Every input record lands in exactly one group; group count is reported.
  * A consensus record's `evidence_tier` is the max tier in its group.
  * No PROTEIN record is ever merged into a REGION group (guaranteed by the key).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .provenance import EvidenceTier, PrionClass, Provenance, aggregation_rank
from .schema import AmyloidRecord

logger = logging.getLogger(__name__)

# A dominant label must out-weigh the runner-up by at least this factor to win a
# conflicted group; otherwise the group is reported as an unresolved conflict.
CONSENSUS_MARGIN = 1.5


@dataclass
class ReconcileResult:
    consensus: List[AmyloidRecord]
    conflicts: List[Dict]
    stats: Dict[str, int] = field(default_factory=dict)


def _merge_group(records: List[AmyloidRecord]) -> AmyloidRecord:
    """Merge a non-conflicting group into one record.

    Picks the record with the strongest evidence tier as the structural base,
    then unions the additive annotations (pmids, cofactors, partners) and
    promotes prion-like tagging and provenance across the group.
    """
    base = max(records, key=lambda r: (int(r.evidence_tier), r.extraction_confidence))
    base.evidence_tier = max(r.evidence_tier for r in records)
    # Preserve the adapter's model-only discount; duplicate rows do not
    # strengthen the evidence of the merged record.
    base.evidence_weight = max(r.evidence_weight for r in records)

    provs = {r.provenance for r in records}
    # If a group mixes origins, mark it curated-backed when any curated record
    # is present (curated carries the structural fields), else mined.
    base.provenance = Provenance.CURATED if Provenance.CURATED in provs else (
        Provenance.MINED if Provenance.MINED in provs else Provenance.PREDICTED)

    # Prion-like tagging wins across the group (keep, never drop).
    if any(r.prion_class == PrionClass.PRION_LIKE_DOMAIN for r in records):
        base.prion_class = PrionClass.PRION_LIKE_DOMAIN
    elif any(r.prion_class == PrionClass.TRANSMISSIBLE_PRION for r in records):
        base.prion_class = PrionClass.TRANSMISSIBLE_PRION

    def _union(attr: str) -> List[str]:
        seen: List[str] = []
        for r in records:
            for v in getattr(r, attr, []) or []:
                if v and v not in seen:
                    seen.append(v)
        return seen

    base.source_pmids = _union("source_pmids")
    base.cofactors_assembly = _union("cofactors_assembly")
    base.cofactors_disassembly = _union("cofactors_disassembly")
    base.string_partners = _union("string_partners")
    base.extraction_confidence = max((r.extraction_confidence for r in records), default=0.0)
    base.reported_methods = _union("reported_methods")
    base.reported_method_tier = max((r.reported_method_tier for r in records), default=-1)
    if any(r.undergoes_llps for r in records):
        base.undergoes_llps = True
    votes: Dict[str, int] = defaultdict(int)
    scores: Dict[str, float] = {}
    for r in records:
        for k, v in (r.reversibility_votes or {}).items():
            votes[k] += v
        for k, v in (r.claim_scores or {}).items():
            scores[k] = max(scores.get(k, 0.0), v)
    base.reversibility_votes = dict(votes)
    base.claim_scores = scores
    # the strongest support any merged row has, with the basis of that row
    supports = [(r.aggregation_evidence, getattr(r, "evidence_basis", "")) for r in records
                if getattr(r, "aggregation_evidence", "")]
    tier, basis = max(supports, key=lambda tb: aggregation_rank(*tb)) if supports else ("", "")
    base.aggregation_evidence, base.evidence_basis = tier, basis
    base.evidence_items = collect_evidence_items(records)
    return base


def collect_evidence_items(records: List[AmyloidRecord]) -> List[Dict]:
    """Union of the curator decisions of ``records``, independent of any vote.

    Deduplicated on (source_row_id, decision) and sorted on it, so the result
    does not depend on the input order.
    """
    seen: Dict[Tuple, Dict] = {}
    for r in records:
        for item in getattr(r, "evidence_items", None) or []:
            key = (str(item.get("source_row_id", "")), str(item.get("decision", "")))
            seen.setdefault(key, dict(item))
    return [seen[k] for k in sorted(seen)]


def reconcile(records: List[AmyloidRecord],
              exclude_transmissible_prion: bool = False) -> ReconcileResult:
    """Unify records into a consensus dataset.

    If ``exclude_transmissible_prion`` is True, records classified as
    transmissible mammalian prion pathology are dropped *before* grouping;
    prion-like-domain records are always kept and tagged.
    """
    if exclude_transmissible_prion:
        kept = [r for r in records if r.prion_class != PrionClass.TRANSMISSIBLE_PRION]
        dropped = len(records) - len(kept)
        if dropped:
            logger.info("Dropped %d transmissible-prion records", dropped)
        records = kept

    groups: Dict[Tuple, List[AmyloidRecord]] = defaultdict(list)
    for r in records:
        groups[r.dedup_key()].append(r)

    consensus: List[AmyloidRecord] = []
    conflicts: List[Dict] = []

    for key, grp in groups.items():
        if len(grp) == 1:
            consensus.append(grp[0])
            continue

        labels = {r.experimental_label for r in grp if r.experimental_label}
        if len(labels) <= 1:
            consensus.append(_merge_group(grp))
            continue

        # Conflicting labels: tier-weighted vote.
        weight: Dict[str, float] = defaultdict(float)
        by_label: Dict[str, List[AmyloidRecord]] = defaultdict(list)
        for r in grp:
            if r.experimental_label:
                weight[r.experimental_label] += r.evidence_weight
                by_label[r.experimental_label].append(r)
        ranked = sorted(weight.items(), key=lambda x: x[1], reverse=True)

        if len(ranked) >= 2 and ranked[1][1] > 0 and ranked[0][1] < ranked[1][1] * CONSENSUS_MARGIN:
            # Too close to call: report, keep nothing silently.
            conflicts.append({
                "key": str(key),
                "labels": sorted(labels),
                "weights": {k: round(v, 3) for k, v in ranked},
                "sources": sorted({r.source_db for r in grp}),
                # curator decisions of the whole group survive an unresolved conflict
                "evidence_items": collect_evidence_items(grp),
            })
            continue

        winner = ranked[0][0]
        merged = _merge_group(by_label[winner])
        # the label is the winner's; the decision history is the whole group's
        merged.evidence_items = collect_evidence_items(grp)
        consensus.append(merged)

    stats = {
        "input_records": len(records),
        "groups": len(groups),
        "consensus": len(consensus),
        "conflicts": len(conflicts),
        "region_records": sum(1 for r in consensus if r.record_level.value == "region"),
        "protein_records": sum(1 for r in consensus if r.record_level.value == "protein"),
        "prion_like": sum(1 for r in consensus if r.prion_class == PrionClass.PRION_LIKE_DOMAIN),
    }
    logger.info("Reconcile: %s", stats)
    return ReconcileResult(consensus=consensus, conflicts=conflicts, stats=stats)
