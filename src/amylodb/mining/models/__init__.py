"""Frozen pretrained models used by the mining stage (inference only).

* `nli.ClaimClassifier`            -- mDeBERTa-v3 zero-shot NLI: claims about a protein
* `structner.ProteinStructureNER`  -- PDBEurope protein-structure NER: methods, proteins
* `triage.RelevanceTriage`         -- BiomedBERT-Amyloid abstract relevance score
* `openbioner.OpenBioNERExtractor` -- OpenBioNER-v2 description-conditioned NER

Heavy dependencies (transformers, torch, zshot) are imported lazily, so importing
this package costs nothing until a model is actually used.
"""

from .registry import MODELS, resolve, resolve_device

__all__ = ["MODELS", "resolve", "resolve_device"]
