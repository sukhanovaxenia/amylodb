"""Fixed-width feature matrix for downstream ML (amylobert, llpscope).

The curated pipeline computes a per-record composition in which dipeptide
frequencies are the top-25 *of that record*, and its TSV export then fixes the
column set from the first record only (dropping every other record's
dipeptides). That makes the dipeptide block order-dependent and largely lost.

This module replaces that with a deterministic, record-order-independent matrix:
  * 21 physicochemical scalars (reused from the curated computation)
  * 20 single amino-acid frequencies (fixed alphabetical order)
  * 400 dipeptide frequencies (full AA x AA grid, fixed order)
  * 3 grouped-size fractions
The column list depends only on the amino-acid alphabet, never on the data, so
two builds of the same sequences yield byte-identical matrices.

Consumes: objects exposing a ``sequence`` attribute (e.g. `AmyloidRecord`).
Emits: (columns, rows) where columns is a fixed list[str] and rows is a
list[list[float]] aligned to it.
Invariant: ``len(columns)`` is constant (444) for a given alphabet, independent
of the input records.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from .curated.config import STANDARD_AA
from .curated.features.physicochemical import compute_physicochemical_features
from .curated.utils.sequence import clean_sequence

# Fixed alphabetical alphabet -> deterministic column order everywhere.
_AA = tuple(sorted(STANDARD_AA))                       # 20
_DIPEPTIDES = tuple(a + b for a in _AA for b in _AA)   # 400

# Physicochemical scalar fields, in a fixed order (mirrors PhysicochemicalFeatures,
# excluding the boolean pattern flags which we keep as 0/1 at the end).
_PHYS_SCALARS = [
    "length", "molecular_weight",
    "hydrophobicity_mean", "hydrophobicity_std", "hydrophobicity_max", "hydrophobicity_min",
    "net_charge", "positive_residues", "negative_residues", "charge_density",
    "beta_propensity_mean", "beta_propensity_max",
    "aggregation_propensity_mean", "aggregation_propensity_max",
    "aromatic_fraction", "aliphatic_fraction", "polar_fraction", "charged_fraction",
]
_PHYS_FLAGS = ["has_polyq", "has_glycine_rich", "has_proline"]

_GROUPED = {
    "grp_tiny": "AGS",
    "grp_small": "ACDGNPSTV",
    "grp_large": "FHKRWY",
}


def feature_columns() -> List[str]:
    """Return the fixed column order. Length is constant (444)."""
    cols: List[str] = []
    cols += [f"phys_{k}" for k in _PHYS_SCALARS]
    cols += [f"phys_{k}" for k in _PHYS_FLAGS]
    cols += [f"aa_{a}" for a in _AA]
    cols += [f"dp_{dp}" for dp in _DIPEPTIDES]
    cols += list(_GROUPED.keys())
    return cols


def _composition_row(seq: str) -> List[float]:
    """Single-AA (20) + full dipeptide (400) + grouped (3) frequencies."""
    n = len(seq)
    counts = Counter(seq)
    aa = [counts.get(a, 0) / n for a in _AA]

    if n >= 2:
        dp_counts = Counter(seq[i:i + 2] for i in range(n - 1))
        total = n - 1
        dp = [dp_counts.get(d, 0) / total for d in _DIPEPTIDES]
    else:
        dp = [0.0] * len(_DIPEPTIDES)

    grouped = [sum(counts.get(a, 0) for a in residues) / n for residues in _GROUPED.values()]
    return aa + dp + grouped


def feature_row(sequence: str) -> List[float]:
    """Compute one fixed-width feature row for a sequence.

    An empty / invalid sequence yields an all-zero row of the correct width, so
    the matrix stays rectangular.
    """
    width = len(feature_columns())
    seq = clean_sequence(sequence or "")
    if not seq:
        return [0.0] * width

    phys = compute_physicochemical_features(seq)
    row: List[float] = []
    for k in _PHYS_SCALARS:
        row.append(float(getattr(phys, k)))
    for k in _PHYS_FLAGS:
        row.append(1.0 if getattr(phys, k) else 0.0)
    row += _composition_row(seq)
    assert len(row) == width, (len(row), width)
    return row


def build_feature_matrix(records: Iterable) -> Tuple[List[str], List[List[float]]]:
    """Build a (columns, rows) matrix from records exposing ``.sequence``.

    The column list is data-independent; rows align to it one-per-record.
    """
    cols = feature_columns()
    rows = [feature_row(getattr(r, "sequence", "") or "") for r in records]
    return cols, rows
