"""Model-based extractor (``--mode models``).

Per abstract:

1. candidate proteins from a NER backend (OpenBioNER by default; scispaCy /
   PubMedBERT also accepted) plus protein/gene spans of the structure NER;
2. candidates are purged of noise names, of anything the structure NER tagged
   as an experimental method, and of names not actually present in the text;
3. experimental methods come from the structure NER and are mapped to a
   universal name and an evidence tier;
4. for each candidate, NLI scores fixed claims against the sentences that
   mention it, and `nli.decide` turns the scores into fields;
5. the LLM (optional) is asked only about fields NLI left undecided.

A candidate is kept only if NLI supports aggregation or LLPS for it; proteins
mentioned merely as tools or context do not become records.

Consumes: title, abstract, pmid.
Emits: `HybridExtractionResult` (same interface as the hybrid extractor), with
per-protein ``claim_scores``, ``undergoes_llps``, ``reported_method_tier``,
``surface_form`` and ``extraction_method``.
Invariant: every returned protein has a boundary-delimited mention in the text.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from .hybrid.extractor import HybridExtractionResult, llm_fill_gaps
from .models.nli import ClaimClassifier, NLIThresholds, decide, focus_premise
from .models.structner import (ProteinStructureNER, method_texts, methods_from_spans,
                               protein_names_from_spans)
from .validation import (clean_name, find_abbreviations, genuine_mention, looks_like_noise,
                         resolve_abbreviation)

logger = logging.getLogger(__name__)


class ModelHybridExtractor:
    def __init__(self, ner=None, ner_backend: str = "openbioner",
                 nli: Optional[ClaimClassifier] = None,
                 structner: Optional[ProteinStructureNER] = None,
                 use_structner: bool = True, llm=None,
                 thresholds: NLIThresholds = NLIThresholds(),
                 max_proteins_per_paper: int = 12, models_dir: Optional[str] = None,
                 device: str = "auto"):
        if ner is None:
            if ner_backend == "openbioner":
                from .models.openbioner import OpenBioNERExtractor
                ner = OpenBioNERExtractor(models_dir=models_dir, device="cpu")
            else:
                from .ner.extractor import create_ner_extractor
                ner = create_ner_extractor(model_type=ner_backend, use_gpu=False)
        self.ner = ner
        self.nli = nli or ClaimClassifier(models_dir=models_dir, device=device)
        self.structner = structner if structner is not None else (
            ProteinStructureNER(models_dir=models_dir, device=device) if use_structner else None)
        self.llm = llm
        self.thresholds = thresholds
        self.max_proteins = max_proteins_per_paper
        self.stats = {"papers_processed": 0, "candidates": 0, "dropped_noise": 0,
                      "dropped_method": 0, "dropped_no_mention": 0, "dropped_no_claim": 0,
                      "kept": 0, "llm_calls": 0}

    def _candidates(self, text: str, spans) -> List[tuple]:
        """(name used for NLI, written form) pairs; short forms defined in the text are expanded."""
        names: List[str] = []
        seen = set()
        for raw in list(self.ner.extract(text).protein_names()) + protein_names_from_spans(spans):
            n = clean_name(raw)
            key = n.lower()
            if key and key not in seen:
                seen.add(key)
                names.append(n)
        # deterministic order: first appearance in text
        names.sort(key=lambda n: (text.lower().find(n.lower()) if n.lower() in text.lower() else 10**9, n))
        self.stats["candidates"] += len(names)

        methods = method_texts(spans)
        abbreviations = find_abbreviations(text)
        kept: List[tuple] = []
        for n in names:
            target = resolve_abbreviation(n, abbreviations) or n
            if looks_like_noise(target):
                self.stats["dropped_noise"] += 1
            elif n.lower() in methods:
                self.stats["dropped_method"] += 1
            elif not genuine_mention(text, [n], min_len=2):
                self.stats["dropped_no_mention"] += 1
            else:
                kept.append((target, n))
        return kept[: self.max_proteins]

    def extract(self, title: str, abstract: str, pmid: str,
                force_llm: bool = False) -> HybridExtractionResult:
        self.stats["papers_processed"] += 1
        text = f"{title}. {abstract}"
        spans = self.structner.extract(text) if self.structner else []
        methods, method_tier = methods_from_spans(spans)
        names = self._candidates(text, spans)

        proteins = []
        llm_called = False
        for name, surface in names:
            premise = focus_premise(text, [name, surface])
            fields = decide(self.nli.score(premise, name), self.thresholds)
            # Chaperones / disaggregases are kept as modifiers, not dropped:
            # the disassembly machinery is part of what the project studies.
            if (not fields["forms_aggregates"] and fields["undergoes_llps"] != "yes"
                    and not fields.get("is_modifier")):
                self.stats["dropped_no_claim"] += 1
                continue
            undecided = fields.pop("undecided")
            # the broad modifier claims are recorded, not acted on: the role
            # decision needs every abstract of a protein and calibrated
            # thresholds, which only re-validation has
            role_scores = fields.pop("role_scores", None)
            data = {
                "role_scores": json.dumps(role_scores) if role_scores else None,
                "protein_name": name,
                "surface_form": surface,
                "protein_role": "modifier" if fields.get("is_modifier") else None,
                "pmid": pmid,
                "experimental_methods": list(methods),
                "reported_method_tier": method_tier,
                "extraction_method": "models",
                **fields,
            }
            if self.llm is not None and (undecided or force_llm):
                data = llm_fill_gaps(self.llm, data, abstract, only=set(undecided) or None)
                data["extraction_method"] = "models+llm"
                llm_called = True
                self.stats["llm_calls"] += 1
            proteins.append(data)
            self.stats["kept"] += 1

        n_agg = sum(1 for p in proteins if p.get("forms_aggregates"))
        relevance = "high" if n_agg > 1 else ("medium" if proteins else "low")
        return HybridExtractionResult(
            proteins=proteins, paper_relevance=relevance, ner_proteins_count=len(names),
            llm_called=llm_called, extraction_method="models+llm" if llm_called else "models")

    def get_stats(self):
        return dict(self.stats)
