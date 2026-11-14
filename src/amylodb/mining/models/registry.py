"""Model registry: hub ids, local resolution, device selection.

All models are used for frozen inference only (no training). Their hub ids are
listed here once; `scripts/fetch_models.py` downloads them into a local
directory and records the resolved commit of each in ``models.lock.json`` so a
mining run can be reproduced offline with exactly the same weights.

Consumes: a model key and an optional local models directory.
Emits: a path or hub id for `transformers` / `zshot`, and a device string.
Invariant: if a lockfile entry and its directory exist, the local copy is used;
otherwise the hub id is returned (requires network on first use).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

MODELS: Dict[str, str] = {
    # zero-shot NLI for claims about a protein (aggregation, reversibility, structure, LLPS)
    "nli": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    # binary relevance triage of abstracts (trained on amyloid x antibody literature)
    "triage": "kotsonor/BiomedBERT-Amyloid",
    # token classification of protein-structure terms, incl. experimental_method
    "structner": "PDBEurope/BiomedNLP-PubMedBERT-ProteinStructure-NER-v3.1",
    # zero-shot, description-conditioned biomedical NER (via zshot)
    "openbioner": "disi-unibo-nlp/openbioner-base-v2",
}

LOCKFILE_NAME = "models.lock.json"


def read_lock(models_dir: Optional[str]) -> Dict[str, Dict]:
    """Return the lockfile content, or {} if absent."""
    if not models_dir:
        return {}
    path = os.path.join(models_dir, LOCKFILE_NAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve(key: str, models_dir: Optional[str] = None) -> str:
    """Local directory for ``key`` if it was fetched, else the hub id."""
    hub_id = MODELS[key]
    entry = read_lock(models_dir).get(key)
    if entry and entry.get("path") and os.path.isdir(entry["path"]):
        return entry["path"]
    return hub_id


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    if device != "auto":
        return device
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def require(module: str, extra: str = "mining"):
    """Import ``module`` or raise an ImportError naming the pip extra to install."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(
            f"'{module}' is required for this model stage. "
            f"Install with: pip install -e \".[{extra}]\"") from exc
