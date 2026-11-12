"""Name resolution rules shared by the mining pipeline and scripts/revalidate_mining.py.

Order of operations for one candidate found in one text:

1. clean the written form; if it is a short form defined in the text
   ("cerebral amyloid angiopathy (CAA)"), replace it by its long form;
2. reject noise (methods, chemicals, disease names, category nouns, fragments)
   -- evaluated on the long form when one was found;
3. after UniProt lookup: require a match; a bare short symbol (<= 3
   characters) matched only through a gene symbol must be corroborated by a
   content word of the UniProt name in the same text; a hit reached only
   through the short form after the long form failed needs the same
   corroboration unless the short form is the entry's gene symbol;
4. assign the role: 'modifier' for chaperones / disaggregases (UniProt keyword
   or name, or a dominant NLI disaggregation claim), else 'aggregator'.

Consumes: candidate dicts (protein_name, surface_form, optional claim_scores) and text.
Emits: the dicts annotated in place, plus a drop reason or None.
Invariant: every rejection carries a reason string, so nothing is dropped silently.
"""

from __future__ import annotations

import re

from typing import Dict, Iterable, List, Optional, Tuple

from .validation import (ALIAS_TO_GENE, clean_name, corroborated, find_abbreviations,
                         genuine_mention, is_short_symbol, is_size_label, looks_like_noise,
                         resolve_abbreviation)

# Nouns that name a protein a modifier when they are *applied to it*.
_CHAPERONE_NOUN = r"(?:co-?chaperones?|chaperones?|chaperonins?|sequestrases?|holdases?|disaggregases?|foldases?)"


def lexical_modifier(sentences: Iterable[str], names: Iterable[str],
                     votes: Optional[Dict[str, int]] = None) -> Optional[str]:
    """'chaperone' when a sentence applies a chaperone-class noun to the protein, else None.

    Three constructions count, and nothing looser:
      apposition   "metal-binding chaperone metallothionein-3", "co-chaperone Sis1";
      copula       "Btn2 and Hsp42 are compartment-specific sequestrases";
      activity     "fibrinogen ... known to exhibit chaperone-like activities".
    Mere co-occurrence does not: "Hsp104 is a chaperone that dissolves Sup35
    fibrils" names Sup35 beside the noun, and the first version of this check,
    which accepted any shared sentence, turned Sup35 -- 15 aggregator votes --
    into a chaperone. For the same reason the override never acts against a
    majority of aggregator votes from the NLI re-scoring (``votes``).
    """
    if votes:
        decided = {k: v for k, v in votes.items() if k != "abstain"}
        if decided and 2 * decided.get("aggregator", 0) > sum(decided.values()):
            return None
    alts = sorted({re.escape(n.strip()) for n in names if n and len(n.strip()) >= 2}, key=len, reverse=True)
    if not alts:
        return None
    nm = r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w])"
    noun = _CHAPERONE_NOUN + r"(?:-like)?"
    patterns = (
        # apposition: the noun directly before the name, at most a head noun
        # between ("chaperone protein X"); "chaperone that dissolves Sup35" is not
        noun + r"(?:\s+(?:protein|proteins|molecule|homolog|called|named|termed))?[\s,]+" + nm,
        # copula: the protein is the subject of "is/are ... <noun>"
        nm + r"(?:\s*,?\s*(?:and|or)\s+[\w-]+)?\s+(?:is|are|was|were|acts? as|functions? as|"
             r"serves? as)\s+(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,3}?" + noun + r"\b",
        # activity: the protein has/exhibits chaperone(-like) activity;
        # "Sup35 fibrils require the chaperone activity of Hsp104" has no such verb
        nm + r"[^.;]{0,100}?\b(?:exhibits?|shows?|displays?|possess(?:es)?|has|have|had)\b"
             r"[^.;]{0,30}?\b" + _CHAPERONE_NOUN + r"(?:-like)?\s+(?:activit|function)",
    )
    for sent in sentences:
        if any(re.search(pt, sent, re.I) for pt in patterns):
            return "chaperone"
    return None


def prepare_candidates(proteins: Iterable[Dict], text: str,
                       abbreviations: Optional[dict] = None) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """Steps 1-2: abbreviation expansion and noise rejection, before any lookup."""
    abbrev = find_abbreviations(text) if abbreviations is None else abbreviations
    kept: List[Dict] = []
    dropped: List[Tuple[str, str]] = []
    for pd in proteins:
        surface = clean_name(pd.get("surface_form") or pd.get("protein_name"))
        pd["surface_form"] = surface
        if looks_like_noise(surface):
            # A listed non-protein acronym (CTAB, SDS, PTMs) stays noise even when
            # the abstract spells it out into a long form the lists do not cover.
            dropped.append((surface, "noise"))
            continue
        long_form = resolve_abbreviation(surface, abbrev)
        if long_form:
            pd["resolved_from"] = surface
            pd["protein_name"] = long_form
        else:
            pd["protein_name"] = clean_name(pd.get("protein_name")) or surface
        if looks_like_noise(pd["protein_name"]):
            dropped.append((surface, "noise_abbreviation" if long_form else "noise"))
            continue
        # a size or segment label the field reads one way ("p62" = SQSTM1,
        # "PHF6" = the tau hexapeptide) is searched as that gene; any other
        # size label is ambiguous and must be corroborated (gate_resolved)
        label = (pd["protein_name"] or "").strip().lower()
        # "human beta-secretase" is the alias without its species word
        label = re.sub(r"^human\s+", "", label) if re.sub(r"^human\s+", "", label) in ALIAS_TO_GENE else label
        if label in ALIAS_TO_GENE:
            pd.setdefault("resolved_from", surface)
            pd["protein_name"] = ALIAS_TO_GENE[label]
        elif is_size_label(label):
            pd["size_label"] = True
        kept.append(pd)
    return kept, dropped


def _symbol(x) -> str:
    """Loose symbol identity: "Bcl-2" == "BCL2"."""
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def gate_resolved(pd: Dict, text: str, require_uniprot: bool = True) -> Optional[str]:
    """Step 3: drop reason after UniProt enrichment, or None to keep."""
    if not pd.get("uniprot_id"):
        if pd.get("lookup_error"):
            return "lookup_error"         # UniProt unreachable: retried by revalidation
        return "no_uniprot_match" if require_uniprot else None
    # resolved through the short form after its long form failed: the short form
    # may be a different protein's symbol (FAP = functional amyloid in Pseudomonas
    # vs prolyl endopeptidase FAP), so the UniProt name must appear in the text
    if pd.get("match_type") == "known":           # curated canonical (Aβ42, tau): no further check
        return None
    # "p25" is TPPP, CDK5R1 or TMED9 depending on the paper: only the text decides
    if pd.get("size_label") and not corroborated(pd.get("uniprot_name"), text):
        return "size_label_uncorroborated"
    if (pd.get("short_form_fallback") and not corroborated(pd.get("uniprot_name"), text)
            and _symbol(pd.get("gene_name")) != _symbol(pd.get("resolved_from"))):
        return "short_form_fallback_uncorroborated"
    # a short symbol matched only as a gene needs the UniProt name in the text,
    # or -- for a non-human entry -- its organism named in the text (Hfq + E. coli)
    if (pd.get("match_type") == "gene" and pd.get("lookup_stage") != "known"
            and not pd.get("resolved_from") and is_short_symbol(pd.get("surface_form"))
            and not corroborated(pd.get("uniprot_name"), text)
            and not (pd.get("lookup_stage") == "reviewed_any" and pd.get("organism_in_text"))
            # a symbol with a digit equal to a gene symbol or synonym (YB-1 = YB1, PS1)
            # is specific enough; letter-only ones (Kap, PAc, lap) are not
            and not any(ch.isdigit() for ch in (pd.get("surface_form") or ""))):
        return "short_symbol_uncorroborated"
    return None


def assign_role(pd: Dict, margin: float = 0.2, from_claims: bool = True) -> str:
    """Step 4: 'modifier' or 'aggregator'; also writes pd['protein_role'] and
    pd['role_subtype'].

    A modifier role already set (UniProt Chaperone keyword, curated chaperone)
    stands, with subtype 'chaperone' -- or 'regulator' for a `...factor` name.

    ``from_claims`` lets the NLI role claims decide the rest, and must stay off
    while mining: a single paper's scores with fixed thresholds entail the broad
    modifier claims for almost anything named in an aggregation abstract. Only
    re-validation, which sees every abstract of a protein and the thresholds
    calibrated against review, turns it on.
    """
    from .models.nli import NLIThresholds, decide_role
    scores = pd.get("claim_scores") or {}
    if pd.get("protein_role") == "modifier":
        # "Heat shock factor protein 1" is a transcription factor, not a chaperone
        default = "regulator" if "factor" in (pd.get("uniprot_name") or "").lower() else "chaperone"
        pd["role_subtype"] = pd.get("role_subtype") or default
        return "modifier"
    role = decide_role(scores, NLIThresholds(margin=margin)) if (scores and from_claims) else None
    if role not in (None, "aggregator"):
        pd["protein_role"], pd["role_subtype"] = "modifier", role
    else:
        pd["protein_role"], pd["role_subtype"] = "aggregator", None
    return pd["protein_role"]
