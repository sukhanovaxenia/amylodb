"""Protein-structure NER (PDBEurope BiomedNLP-PubMedBERT-ProteinStructure-NER-v3.1).

The model tags 20 entity types in text, including ``experimental_method``,
``protein``, ``gene``, ``oligomeric_state`` and ``structure_element``. Here it is
used for two things the keyword patterns did poorly:

* **experimental methods** -- mapped through the curated method rules to a
  universal name and an evidence tier, so a mined record can say which kind of
  evidence its source paper reports (it still stays at tier TEXT_MINED: the
  evidence is reported, not verified here);
* **telling methods apart from proteins** -- span texts tagged as methods are
  removed from the protein candidates (on a real run "afm" was stored as a
  protein).

Consumes: text.
Emits: `Span` lists; methods + max tier; protein-like span texts.
Invariant: span offsets refer to the original text (chunk offsets corrected).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from ...curated.utils.sequence import map_method_to_universal
from ...provenance import EvidenceTier, evidence_tier_from_curated
from .registry import require, resolve, resolve_device

LABELS = (
    "bond_interaction", "chemical", "complex_assembly", "evidence", "experimental_method",
    "gene", "mutant", "oligomeric_state", "protein", "protein_state", "protein_type", "ptm",
    "residue_name", "residue_name_number", "residue_number", "residue_range", "site",
    "species", "structure_element", "taxonomy_domain",
)
PROTEIN_LABELS = {"protein", "gene"}
METHOD_LABELS = {"experimental_method"}
STRUCTURE_LABELS = {"oligomeric_state", "structure_element", "complex_assembly"}


@dataclass
class Span:
    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


def _char_chunks(text: str, max_chars: int = 1200) -> List[Tuple[int, str]]:
    """Fallback char-based chunking (used when no tokenizer is available)."""
    out, start, buf = [], 0, ""
    pos = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        idx = text.find(sent, pos)
        if buf and len(buf) + len(sent) + 1 > max_chars:
            out.append((start, buf))
            start, buf = idx, sent
        else:
            if not buf:
                start = idx
            buf = (buf + " " + sent) if buf else sent
        pos = idx + len(sent)
    if buf:
        out.append((start, buf))
    return out


class ProteinStructureNER:
    """Token classification with span aggregation; ``pipe`` injectable for tests.

    The PubMedBERT backbone has a 512-position limit and its config declares no
    tokenizer max length, so the pipeline does not truncate by default: a long
    chunk overflows the position embeddings (RuntimeError: tensor a (592) vs b
    (512)). Chunks are therefore budgeted in *tokens* against the model's own
    tokenizer, with a hard model_max_length guard on the tokenizer as a backstop.
    """

    def __init__(self, model: Optional[str] = None, models_dir: Optional[str] = None,
                 device: str = "auto", pipe: Optional[Callable] = None, min_score: float = 0.5,
                 max_tokens: int = 400):
        self.model = model or resolve("structner", models_dir)
        self.device = device
        self.min_score = min_score
        self.max_tokens = max_tokens
        self._pipe_obj = pipe

    def _pipe(self):
        if self._pipe_obj is None:
            transformers = require("transformers")
            tok = transformers.AutoTokenizer.from_pretrained(self.model, model_max_length=512)
            self._pipe_obj = transformers.pipeline(
                "token-classification", model=self.model, tokenizer=tok,
                aggregation_strategy="simple", device=resolve_device(self.device))
        return self._pipe_obj

    def _tokenizer(self):
        return getattr(self._pipe(), "tokenizer", None)

    def _split_long_sentence(self, sent: str, base: int, tok) -> List[Tuple[int, str]]:
        """Cut a single over-long sentence at token boundaries, preserving offsets."""
        try:
            enc = tok(sent, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
        except (TypeError, KeyError):
            # not a fast tokenizer: fall back to a conservative char split
            return _char_chunks(sent, max_chars=self.max_tokens * 2)
        out = []
        for i in range(0, len(offsets), self.max_tokens):
            window = offsets[i:i + self.max_tokens]
            cs, ce = window[0][0], window[-1][1]
            out.append((base + cs, sent[cs:ce]))
        return out

    def _chunks(self, text: str) -> List[Tuple[int, str]]:
        """Token-budgeted chunks with original-text offsets; char fallback if no tokenizer."""
        tok = self._tokenizer()
        if tok is None:
            return _char_chunks(text, max_chars=1200)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        out: List[Tuple[int, str]] = []
        cur_start = cur_end = None
        cur_tokens = 0
        pos = 0
        for sent in sentences:
            if not sent:
                continue
            idx = text.find(sent, pos)
            pos = idx + len(sent)
            n = len(tok.encode(sent, add_special_tokens=False))
            if n > self.max_tokens:
                if cur_start is not None:
                    out.append((cur_start, text[cur_start:cur_end]))
                    cur_start = cur_end = None
                    cur_tokens = 0
                out.extend(self._split_long_sentence(sent, idx, tok))
                continue
            if cur_start is not None and cur_tokens + n > self.max_tokens:
                out.append((cur_start, text[cur_start:cur_end]))
                cur_start = cur_end = None
                cur_tokens = 0
            if cur_start is None:
                cur_start = idx
            cur_end = idx + len(sent)
            cur_tokens += n
        if cur_start is not None:
            out.append((cur_start, text[cur_start:cur_end]))
        return out

    def extract(self, text: str) -> List[Span]:
        spans: List[Span] = []
        for offset, chunk in self._chunks(text or ""):
            for ent in self._pipe()(chunk):
                label = ent.get("entity_group") or ent.get("entity") or ""
                label = label.split("-", 1)[-1] if label[:2] in ("B-", "I-") else label
                score = float(ent.get("score", 1.0))
                if score < self.min_score:
                    continue
                start = int(ent.get("start", 0)) + offset
                end = int(ent.get("end", 0)) + offset
                surface = text[start:end] if 0 <= start < end <= len(text) else ent.get("word", "")
                spans.append(Span(surface.strip(), label, start, end, score))
        return spans


def methods_from_spans(spans: List[Span]) -> Tuple[List[str], int]:
    """Universal method names (rule-matched first) and the max evidence tier.

    Methods that no curated rule recognises are kept by their surface text but
    do not contribute a tier. Returns tier -1 when no method is recognised.
    """
    methods: List[str] = []
    best = -1
    for sp in spans:
        if sp.label not in METHOD_LABELS:
            continue
        universal, evidence_type, _w, _c = map_method_to_universal(sp.text)
        recognised = universal != sp.text.strip()
        name = universal if recognised else sp.text.strip()
        if name and name not in methods:
            methods.append(name)
        if recognised:
            best = max(best, int(evidence_tier_from_curated(evidence_type)))
    return methods, best


def protein_names_from_spans(spans: List[Span]) -> List[str]:
    """Distinct protein/gene span texts, in order of appearance."""
    seen, out = set(), []
    for sp in spans:
        if sp.label in PROTEIN_LABELS and sp.text and sp.text.lower() not in seen:
            seen.add(sp.text.lower())
            out.append(sp.text)
    return out


def method_texts(spans: List[Span]) -> set:
    """Lower-cased surface texts tagged as methods (used to purge protein candidates)."""
    return {sp.text.lower() for sp in spans if sp.label in METHOD_LABELS}


__all__ = ["Span", "ProteinStructureNER", "methods_from_spans", "protein_names_from_spans",
           "method_texts", "LABELS", "EvidenceTier"]
