"""Description-conditioned zero-shot NER (OpenBioNER-v2 via zshot).

Entity types are given as names with natural-language descriptions, so the
extractor can be told exactly what counts as a protein in this project and what
must be kept apart from it (methods, diseases, reagents, condensates). This
targets the noise observed with the generic scispaCy model, which returned
reagents, methods and disease abbreviations as proteins.

Consumes: text.
Emits: a `ModelNERResult` (drop-in for `NERResult`: proteins, chemicals,
diseases, plus methods and condensates).
Invariant: only spans labelled protein / peptide reach ``proteins``, and they
pass `looks_like_noise`; a chunk the model cannot process is skipped, never
raised, so one abstract cannot end a run.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..ner.extractor import BioEntity, NERResult
from ..validation import looks_like_noise
from .registry import require, resolve
from .structner import _char_chunks

logger = logging.getLogger(__name__)

# zshot 0.0.11 raises KeyError(-1) inside its SMXM span decoder
# (utils/models/smxm/utils.py: words_offset_mappings[input_id - 1]) whenever a
# predicted span starts at the first sub-token. The input that triggers it
# cannot be predicted from the text, so the call is made per chunk and guarded:
# a failing chunk costs its own spans, not the abstract and not the run.
_MAX_CHUNK_CHARS = 1000

# Zero-width and line/paragraph separators: invisible, but they shift the
# sub-token offsets the model's span decoder indexes into.
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2028\u2029\ufeff]")

DEFAULT_ENTITIES: Dict[str, str] = {
    "protein": ("A protein, enzyme or polypeptide chain named by its protein or gene name, "
                "for example alpha-synuclein, FUS, TDP-43, tau, Sup35 or CsgA."),
    "peptide": ("A short peptide or a defined fragment of a protein, for example amyloid-beta 1-42 "
                "or an amyloidogenic hexapeptide."),
    "experimental_method": ("An experimental technique used to study proteins, for example the "
                            "thioflavin T assay, cryo-EM, FTIR spectroscopy, AFM or FRAP."),
    "disease": "A disease or disorder, for example Alzheimer's disease or amyotrophic lateral sclerosis.",
    "chemical": "A small molecule, reagent, dye or drug, for example EGCG, Congo red, ATP or ABTS.",
    "condensate": ("A membraneless organelle or biomolecular condensate, for example a stress granule, "
                   "P-body, nucleolus or liquid droplet."),
}
PROTEIN_TYPES = {"protein", "peptide"}


@dataclass
class ModelNERResult(NERResult):
    methods: List[BioEntity] = field(default_factory=list)
    condensates: List[BioEntity] = field(default_factory=list)


class OpenBioNERExtractor:
    """zshot pipeline with an OpenBioNER linker; ``nlp`` injectable for tests."""

    def __init__(self, model: Optional[str] = None, models_dir: Optional[str] = None,
                 entities: Optional[Dict[str, str]] = None, device: str = "cpu",
                 nlp: Optional[Callable] = None):
        self.model = model or resolve("openbioner", models_dir)
        self.models_dir = models_dir
        self.entities = entities or DEFAULT_ENTITIES
        self.device = device
        self._nlp = nlp
        self.failed_chunks = 0          # chunks the model could not process

    def _load(self):
        # zshot reads MODELS_CACHE_PATH at import time and always loads the
        # bert-large-cased tokenizer into it; keep that cache inside models_dir
        # so scripts/fetch_models.py can pre-fill it for offline runs.
        if self.models_dir:
            os.environ.setdefault("MODELS_CACHE_PATH",
                                  os.path.join(os.path.abspath(self.models_dir), "zshot_cache") + os.sep)
        spacy = require("spacy")
        zshot = require("zshot")
        try:
            from zshot.linker import LinkerSMXM
        except ImportError:  # pragma: no cover - layout differs across zshot versions
            LinkerSMXM = getattr(zshot, "LinkerSMXM")
        try:
            from zshot.utils.data_models import Entity
        except ImportError:  # pragma: no cover
            Entity = getattr(zshot, "Entity")
        config = zshot.PipelineConfig(
            linker=LinkerSMXM(model_name=self.model),
            entities=[Entity(name=n, description=d) for n, d in self.entities.items()],
            device=self.device,
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("zshot", config=config, last=True)
        self._nlp = nlp

    @staticmethod
    def _clean(text: str) -> str:
        """Neutralise the characters that desynchronise the model's offset mapping.

        Control codes and zero-width marks occur in PubMed abstracts and shift
        the sub-token offsets zshot indexes into. Each becomes a space rather
        than being deleted: the cleaned text keeps the length and the word
        boundaries of the original (a tab between two words must not merge
        them), so the spans this extractor reports still index the input.
        """
        if not text:
            return ""
        chars = [" " if (c != "\n" and unicodedata.category(c) == "Cc") else c for c in text]
        return _ZERO_WIDTH.sub(" ", "".join(chars))

    def extract(self, text: str) -> ModelNERResult:
        """Entities over the whole text, chunk by chunk.

        Each chunk is passed to the model on its own and its span offsets are
        shifted back onto the original text. A chunk the model fails on is
        counted in ``self.failed_chunks`` and skipped, so the rest of the
        abstract is still annotated.
        """
        if self._nlp is None:
            self._load()
        clean = self._clean(text)
        res = ModelNERResult()
        if not clean.strip():
            return res
        for offset, chunk in _char_chunks(clean, max_chars=_MAX_CHUNK_CHARS):
            try:
                doc = self._nlp(chunk)
            except Exception as e:                       # zshot span-decoder bugs, OOM, ...
                self.failed_chunks += 1
                logger.warning(f"OpenBioNER skipped a chunk ({type(e).__name__}: {e}); "
                               f"{self.failed_chunks} skipped so far")
                continue
            for ent in getattr(doc, "ents", []):
                label = str(getattr(ent, "label_", "")).lower()
                surface = str(getattr(ent, "text", "")).strip()
                be = BioEntity(text=surface, label=label.upper(),
                               start=offset + int(getattr(ent, "start_char", 0)),
                               end=offset + int(getattr(ent, "end_char", 0)))
                if label in PROTEIN_TYPES:
                    if not looks_like_noise(surface):
                        res.proteins.append(be)
                elif label == "experimental_method":
                    res.methods.append(be)
                elif label == "disease":
                    res.diseases.append(be)
                elif label == "chemical":
                    res.chemicals.append(be)
                elif label == "condensate":
                    res.condensates.append(be)
        return res
