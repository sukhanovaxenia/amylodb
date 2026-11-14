"""Relevance triage of abstracts (kotsonor/BiomedBERT-Amyloid).

Caution on scope: this classifier was trained on 1,939 PubMed articles from
amyloid x antibody/nanobody searches, labelling whether an article reports
experimental effects of antibodies on amyloid formation (9 % positive, no
metrics on the model card). Its notion of "relevant" is therefore narrower than
"this abstract reports protein aggregation". The default mode is ``annotate``:
every abstract gets a score that is stored, nothing is dropped. Switch to
``gate`` only after checking the scores against your own judgement on a sample
(see scripts/audit_mining.py).

Consumes: abstract texts.
Emits: P(positive) per text, and a keep/drop decision per mode.
Invariant: in ``off`` and ``annotate`` modes no abstract is ever dropped.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from .registry import require, resolve, resolve_device

logger = logging.getLogger(__name__)

MODES = ("off", "annotate", "gate")
_POSITIVE_NAMES = {"positive", "pos", "useful", "relevant", "yes", "true", "1", "label_1"}
_NEGATIVE_MARKERS = ("neg", "not", "irrelevant", "no", "false", "label_0")


def detect_positive_index(id2label: Dict) -> int:
    """Index of the positive class from a model's id2label mapping.

    Recognises explicit names (positive/useful/relevant), negations (not useful,
    negative) and the generic LABEL_0 / LABEL_1 convention; falls back to 1.
    """
    norm = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    for idx, lab in norm.items():
        if lab in _POSITIVE_NAMES and not any(lab.startswith(m) for m in ("not", "neg")):
            return idx
    negatives = [idx for idx, lab in norm.items() if any(lab.startswith(m) or f" {m}" in lab for m in _NEGATIVE_MARKERS)]
    if len(norm) == 2 and len(negatives) == 1:
        return next(i for i in norm if i != negatives[0])
    logger.warning("Could not identify the positive label in %s; assuming index 1", norm)
    return 1


class RelevanceTriage:
    """Scores abstracts; ``scorer`` may be injected (tests) as texts -> list[float]."""

    def __init__(self, model: Optional[str] = None, models_dir: Optional[str] = None,
                 device: str = "auto", mode: str = "annotate", threshold: float = 0.5,
                 scorer: Optional[Callable[[List[str]], List[float]]] = None, batch_size: int = 16):
        if mode not in MODES:
            raise ValueError(f"triage mode must be one of {MODES}, got {mode!r}")
        self.model = model or resolve("triage", models_dir)
        self.device = device
        self.mode = mode
        self.threshold = threshold
        self.batch_size = batch_size
        self._scorer = scorer
        self.positive_index: Optional[int] = None

    def _load(self):
        transformers = require("transformers")
        torch = require("torch")
        tok = transformers.AutoTokenizer.from_pretrained(self.model)
        mdl = transformers.AutoModelForSequenceClassification.from_pretrained(self.model)
        device = resolve_device(self.device)
        mdl.to(device).eval()
        self.positive_index = detect_positive_index(mdl.config.id2label)
        logger.info("Triage labels %s; positive index %d", mdl.config.id2label, self.positive_index)

        def scorer(texts: List[str]) -> List[float]:
            out: List[float] = []
            for i in range(0, len(texts), self.batch_size):
                batch = tok(texts[i:i + self.batch_size], truncation=True, max_length=512,
                            padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    probs = torch.softmax(mdl(**batch).logits, dim=-1)
                out.extend(probs[:, self.positive_index].tolist())
            return out

        self._scorer = scorer

    def score(self, texts: List[str]) -> List[float]:
        if self.mode == "off":
            return [float("nan")] * len(texts)
        if self._scorer is None:
            self._load()
        return [float(x) for x in self._scorer(texts)]

    def keep(self, score: float) -> bool:
        """Keep/drop decision. Never drops outside ``gate`` mode."""
        if self.mode != "gate":
            return True
        return score >= self.threshold
