"""Provenance, record level, evidence tiers and prion classification.

This module defines the shared vocabulary that lets records coming from two
very different origins -- curated databases (peptide/region centred) and
literature mining (protein centred) -- live in one dataset with a single,
ordered notion of how trustworthy each record is.

Consumes: raw evidence-type strings and free text (protein names, abstracts).
Emits: enums (`Provenance`, `RecordLevel`, `EvidenceTier`, `PrionClass`) and a
numeric `tier_weight`.
Invariants:
  * `EvidenceTier` is a total order; higher == stronger empirical support.
  * A text-mined record never outranks a structurally resolved one.
  * Prion classification uses word boundaries, never substring membership, so
    that PRPF8 / PRPS1 and "prion-like domain" are handled correctly.
"""

from __future__ import annotations

import re
from enum import Enum, IntEnum
from typing import Optional


class Provenance(str, Enum):
    """Where a record came from."""

    CURATED = "curated"      # parsed from a curated amyloid database
    MINED = "mined"          # extracted from literature (NER / NLI / LLM)
    PREDICTED = "predicted"  # computational prediction only


class RecordLevel(str, Enum):
    """Biological granularity of a record.

    Reconciliation must never deduplicate a protein-level record against a
    region-level one, because their identity keys mean different things.
    """

    REGION = "region"    # a peptide / APR with (optional) start-end coordinates
    PROTEIN = "protein"  # a whole protein, no specific region asserted


class EvidenceTier(IntEnum):
    """Ordered evidence strength. Higher is stronger.

    Unifies the curated pipeline's ``evidence_type`` weighting with a new,
    explicitly lower tier for literature-mined claims that carry no
    experimental verification of their own.
    """

    COMPUTATIONAL = 0      # predictor output only
    TEXT_MINED = 1         # asserted in literature, mined, not independently verified here
    LITERATURE_CURATED = 2  # curated from literature by a database
    STAINING_BINDING = 3   # ThT / Congo red / FTIR / CD
    KINETIC = 4            # aggregation kinetics, seeding
    STRUCTURAL = 5         # XRD / cryo-EM / ssNMR / crystallography


# Numeric weights for consensus voting. Kept compatible with the curated
# pipeline's EVIDENCE_WEIGHTS ordering, with TEXT_MINED slotted just above
# pure computation.
TIER_WEIGHT = {
    EvidenceTier.COMPUTATIONAL: 0.0,
    EvidenceTier.TEXT_MINED: 0.25,
    EvidenceTier.LITERATURE_CURATED: 0.5,
    EvidenceTier.STAINING_BINDING: 1.0,
    EvidenceTier.KINETIC: 2.0,
    EvidenceTier.STRUCTURAL: 3.0,
}

# Map the curated pipeline's evidence_type strings onto the unified tiers.
_CURATED_EVIDENCE_TO_TIER = {
    "structural": EvidenceTier.STRUCTURAL,
    "kinetic": EvidenceTier.KINETIC,
    "staining_binding": EvidenceTier.STAINING_BINDING,
    "literature_curated": EvidenceTier.LITERATURE_CURATED,
    "computational": EvidenceTier.COMPUTATIONAL,
}


def tier_weight(tier: EvidenceTier) -> float:
    """Return the consensus-voting weight for a tier."""
    return TIER_WEIGHT[tier]


# Support of a mined aggregator (`aggregation_evidence`, with `evidence_basis`
# for a curator decision) -> (rank, share of the TEXT_MINED weight). The rank
# only picks the support reported for a protein whose rows share a UniProt; it
# never replaces the EvidenceTier ordering of the consensus vote. Curator
# decisions of 2026-11-16 (docs/changes_from_source.md 9.16): a curator review
# of an aggregation description outranks the automatic text rule; an LCR basis
# makes a candidate, voting at half weight like a model-only prediction; an
# unresolved attribution does not vote.
_AGGREGATION_SUPPORT = {
    ("curated", ""): (5, 1.0),
    ("curator_review", "aggregation_text"): (4, 1.0),
    ("text", ""): (3, 1.0),
    ("model_only", ""): (2, 0.5),
    ("curator_review", "LCR_text"): (1, 0.5),
    ("unresolved", ""): (0, 0.0),
}
# Supports that make a candidate, not an amyloid label: is_amyloid stays None.
CANDIDATE_SUPPORT = {("curator_review", "LCR_text"), ("unresolved", "")}
# Supports whose label is decided by the support itself; the older tiers keep
# taking the label from forms_aggregates.
CURATOR_TIERS = {"curator_review", "unresolved"}


def _support_key(tier: str, basis: str = "") -> tuple:
    """Normalise (tier, basis): the basis distinguishes curator_review only.

    Raises ValueError for a curator_review whose basis is not a known one, so a
    malformed decision cannot silently take a rank or a weight.
    """
    tier = tier or ""
    key = (tier, basis or "") if tier == "curator_review" else (tier, "")
    if tier == "curator_review" and key not in _AGGREGATION_SUPPORT:
        raise ValueError(f"curator_review with unknown evidence_basis {basis!r}")
    return key


def aggregation_rank(tier: str, basis: str = "") -> int:
    """Rank of an aggregator support; -1 for an empty or unknown tier."""
    return _AGGREGATION_SUPPORT.get(_support_key(tier, basis), (-1, 1.0))[0]


def aggregation_weight(tier: str, basis: str = "") -> float:
    """Share of the TEXT_MINED weight a mined record of this support votes with.

    Empty and unknown tiers keep the full weight, as before this table existed.
    """
    return _AGGREGATION_SUPPORT.get(_support_key(tier, basis), (-1, 1.0))[1]


def is_candidate_support(tier: str, basis: str = "") -> bool:
    """True when the support makes a candidate (label 'candidate', is_amyloid None)."""
    return _support_key(tier, basis) in CANDIDATE_SUPPORT


def evidence_tier_from_curated(evidence_type: str) -> EvidenceTier:
    """Map a curated ``evidence_type`` string to an `EvidenceTier`.

    Unknown or empty values fall back to COMPUTATIONAL, the weakest tier, so an
    unmapped source can never silently inflate a record's standing.
    """
    return _CURATED_EVIDENCE_TO_TIER.get((evidence_type or "").strip().lower(),
                                         EvidenceTier.COMPUTATIONAL)


# ---------------------------------------------------------------------------
# Prion classification
# ---------------------------------------------------------------------------

class PrionClass(str, Enum):
    """How a record relates to prions.

    The dissertation concerns translation-coupled RNP condensates, which are
    organised by prion-LIKE-domain RBPs (FUS, hnRNPA1, TDP-43, PRRC2A/B).
    Those must be kept and tagged, not discarded. Only transmissible
    mammalian prion pathology (PrP^Sc and relatives) is a candidate for
    exclusion, and that choice is left to the caller.
    """

    NOT_PRION = "not_prion"
    PRION_LIKE_DOMAIN = "prion_like_domain"
    TRANSMISSIBLE_PRION = "transmissible_prion"


# Transmissible mammalian prion pathology. Word-boundary patterns only.
_TRANSMISSIBLE_PATTERNS = [
    r"\bPrP(?:Sc|C|res)?\b",
    r"\bscrapie\b",
    r"\btransmissible spongiform\b",
    r"\bcreutzfeldt\b",
    r"\bcreutzfeldt[-\s]?jakob\b",
    r"\bCJD\b",
    r"\bbovine spongiform\b",
    r"\bBSE\b",
    r"\bchronic wasting disease\b",
    r"\bfatal familial insomnia\b",
    r"\bGerstmann\b",
]

# Prion-like domain markers. These KEEP the record (tagged), never exclude it.
_PRION_LIKE_PATTERNS = [
    r"\bprion[-\s]?like\b",
    r"\bprion[-\s]?like domain\b",
    r"\bPLD\b",
    r"\bLCD\b",  # low-complexity domain, used interchangeably in the SG literature
]

# Known prion-like-domain RBPs relevant to RNP condensates. Matched as whole
# tokens so that e.g. "FUS" does not match inside "fusion".
_PRION_LIKE_PROTEINS = [
    r"\bFUS\b", r"\bhnRNPA1\b", r"\bhnRNPA2B1\b", r"\bTDP[-\s]?43\b",
    r"\bTARDBP\b", r"\bEWSR1\b", r"\bTAF15\b", r"\bPRRC2A\b", r"\bPRRC2B\b",
    r"\bTIA1\b", r"\bG3BP1\b",
]

_transmissible_re = re.compile("|".join(_TRANSMISSIBLE_PATTERNS), re.IGNORECASE)
_prion_like_re = re.compile("|".join(_PRION_LIKE_PATTERNS + _PRION_LIKE_PROTEINS),
                            re.IGNORECASE)


def classify_prion(*texts: Optional[str]) -> PrionClass:
    """Classify the prion relationship of a record from any number of text fields.

    Order of precedence: a prion-LIKE signal wins over a transmissible signal,
    because a paper on a prion-like domain often also mentions PrP for contrast,
    and the prion-like-domain record is one we want to keep. Only a
    transmissible signal with no prion-like signal yields TRANSMISSIBLE_PRION.

    Consumes: protein name, abstract, notes -- any subset, Nones ignored.
    Emits: a `PrionClass`.
    """
    blob = " ".join(t for t in texts if t)
    if not blob:
        return PrionClass.NOT_PRION
    if _prion_like_re.search(blob):
        return PrionClass.PRION_LIKE_DOMAIN
    if _transmissible_re.search(blob):
        return PrionClass.TRANSMISSIBLE_PRION
    return PrionClass.NOT_PRION
