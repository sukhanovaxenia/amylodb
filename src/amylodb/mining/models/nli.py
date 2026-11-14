"""Zero-shot NLI claims about a protein (mDeBERTa-v3-base-mnli-xnli).

For each protein mentioned in an abstract, the abstract sentences that mention
it form the premise, and a fixed set of hypotheses ("<protein> forms amyloid
fibrils...", "<protein> assemblies are reversible...") is scored for
entailment. The scores are then turned into categorical fields by explicit,
documented rules. This replaces keyword pattern counting as the source of
reversibility / structure / functional class -- the fields the pattern path
left empty for 97 % of proteins on a real run.

Consumes: a premise text and a protein name.
Emits: per-claim entailment probabilities and a decision dict.
Invariants:
  * NLI never asserts a negative it cannot support: low LLPS entailment yields
    'unknown', not 'no' (absence of a statement is not evidence of absence).
  * Every decided field is backed by a claim score >= the positive threshold;
    the confidence is the mean of the scores that back the decisions made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from ..validation import genuine_mention
from .registry import require, resolve, resolve_device

# Claim key -> one or more hypotheses. "{p}" is replaced by the protein name.
# Several phrasings per claim are scored and the maximum entailment is kept, so
# a claim is recognised whether the abstract says "resisted disaggregation",
# "remained stable" or "did not dissolve". This raised the recall of the fields
# (reversibility was decided for only 13 % of proteins with single phrasings).
CLAIMS: Dict[str, list] = {
    "aggregates": ["{p} forms amyloid fibrils or protein aggregates.",
                   "{p} aggregates into amyloid fibrils."],
    "reversible": ["{p} assemblies are reversible and can dissolve.",
                   "The aggregates of {p} dissolve and disassemble."],
    "irreversible": ["{p} forms stable, irreversible aggregates.",
                     "The aggregates of {p} are stable and resist disaggregation.",
                     "The aggregates of {p} do not dissolve."],
    "beta": ["{p} aggregates have a cross-beta sheet structure.",
             "{p} forms beta-sheet-rich amyloid fibrils."],
    "alpha": ["{p} forms alpha-helical aggregates."],
    "functional": ["{p} aggregation serves a normal physiological function."],
    "pathological": ["{p} aggregation is toxic or associated with disease."],
    "autocatalytic": ["{p} aggregation is self-templating, seeded or autocatalytic."],
    "cofactor_dependent": ["{p} aggregation requires a cofactor such as RNA, a lipid, or a metal ion."],
    "llps": ["{p} undergoes liquid-liquid phase separation and forms condensates.",
             "{p} forms liquid droplets by phase separation."],
    # A negative discriminator: a chaperone / disaggregase is found in an
    # aggregation abstract because it DISSOLVES aggregates, not because it forms
    # them (e.g. Hsp70, Hsp104, HtrA). Used to withhold the aggregator call.
    "disaggregase": ["{p} dissolves, clears or prevents protein aggregates as a chaperone "
                     "or disaggregase, rather than forming them."],
    # Further roles of a protein named in an aggregation abstract without forming
    # the aggregates itself (first manual review: artemin, TRIM11, SIRT1, Sel1L,
    # ROCK2, trypsin, serum amyloid P were recorded as aggregators).
    "inhibits": ["{p} inhibits, prevents or reverses the aggregation of another protein.",
                 "{p} degrades or removes amyloid fibrils formed by another protein."],
    "regulates": ["{p} regulates the expression, processing, modification or clearance of an "
                  "aggregating protein without aggregating itself.",
                  "{p} is an enzyme or signalling protein acting upstream or downstream of "
                  "protein aggregation."],
    "co_deposits": ["{p} binds to or co-deposits with amyloid formed by another protein."],
}

# Claim keys that decide the role of a protein; the rest describe its aggregates.
ROLE_KEYS = ("aggregates", "disaggregase", "inhibits", "regulates", "co_deposits")
# Modifier subtype per claim key.
ROLE_SUBTYPE = {"disaggregase": "chaperone", "inhibits": "inhibitor",
                "regulates": "regulator", "co_deposits": "associated"}


def decide_role(scores: Dict[str, float], th: "NLIThresholds" = None) -> Optional[str]:
    """Role of a protein in one premise: 'aggregator', a modifier subtype
    ('chaperone' | 'inhibitor' | 'regulator' | 'associated'), or None.

    A modifier subtype wins when its claim is entailed (>= positive) and leads
    the self-aggregation claim by the margin; otherwise an entailed
    self-aggregation claim makes an aggregator. Neither entailed -> None.
    """
    th = th or NLIThresholds()
    agg = float(scores.get("aggregates", 0.0))
    others = {k: float(scores.get(k, 0.0)) for k in ROLE_SUBTYPE}
    key, best = max(others.items(), key=lambda kv: kv[1])
    if best >= th.positive and best - agg >= th.margin:
        return ROLE_SUBTYPE[key]
    if agg >= th.positive:
        return "aggregator"
    return None

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class NLIThresholds:
    positive: float = 0.7   # a claim counts as entailed at or above this probability
    margin: float = 0.2     # winning claim must lead the opposing one by this much


def focus_premise(text: str, names: Iterable[str], max_chars: int = 1800) -> str:
    """Sentences mentioning any of ``names``, each with the sentence after it.

    The forward window matters because the evidence for reversibility or
    structure usually continues into the next sentence with an anaphor ("...
    alpha-synuclein fibrils. The fibrils resisted disaggregation and remained
    stable."), which a name-only filter would drop. If no sentence mentions the
    protein, the whole text is used.
    """
    names = [n for n in names if n]
    sentences = _SENT_SPLIT.split(text or "")
    keep = set()
    for i, s in enumerate(sentences):
        if genuine_mention(s, names):
            keep.add(i)
            if i + 1 < len(sentences):
                keep.add(i + 1)
    chosen = [sentences[i] for i in sorted(keep)] if keep else sentences
    return " ".join(chosen)[:max_chars]


def decide_protein_role(stats: Dict[str, float], thresholds: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Protein-level role from scores aggregated over its abstracts, using the
    thresholds fitted by scripts/calibrate_roles.py.

    ``stats`` is the chosen statistic (mean or max) of each claim key over the
    protein's abstracts. The protein is an aggregator when the self-aggregation
    score clears ``min_aggregates`` and leads the strongest modifier claim by at
    least ``min_lead`` (which may be negative); otherwise it is a modifier and
    the leading modifier claim gives the subtype. Returns (role, subtype).
    """
    agg = float(stats.get("aggregates", 0.0))
    mods = {k: float(stats.get(k, 0.0)) for k in ROLE_SUBTYPE}
    if agg >= thresholds.get("min_aggregates", 0.5) and \
            agg - max(mods.values(), default=0.0) >= thresholds.get("min_lead", -1.0):
        return "aggregator", None
    key = max(mods, key=mods.get)
    return "modifier", ROLE_SUBTYPE[key]


def decide(scores: Dict[str, float], th: NLIThresholds = NLIThresholds()) -> Dict:
    """Turn claim scores into categorical fields + confidence + undecided list."""
    pos, margin = th.positive, th.margin
    s = {k: float(scores.get(k, 0.0)) for k in CLAIMS}
    out: Dict = {}
    backing: List[float] = []

    # A protein whose dominant claim is disaggregation is a chaperone found in an
    # aggregation abstract, not an aggregator: withhold the aggregator call.
    #
    # Only the disaggregation claim may do so. The broader modifier claims
    # (inhibits / regulates / co_deposits) are scored and passed on, but never
    # acted upon here, because zero-shot NLI entails them for almost any protein
    # named in an aggregation abstract: deciding on them per paper with fixed
    # thresholds labelled 1220 of 1870 proteins modifiers and suppressed
    # forms_aggregates for 1202 of them, amyloid-beta included. That decision
    # needs every abstract of a protein at once and thresholds calibrated against
    # review, so it belongs to revalidate_mining.py --rescore-claims, which has
    # both. `role_scores` carries the evidence there.
    dis = s["disaggregase"]
    is_chaperone = dis >= pos and dis - s["aggregates"] >= margin
    forms = s["aggregates"] >= pos and not is_chaperone
    out["forms_aggregates"] = forms
    out["is_disaggregase"] = is_chaperone
    out["is_modifier"] = is_chaperone
    out["role_subtype"] = "chaperone" if is_chaperone else None
    out["role_scores"] = {k: round(s[k], 4) for k in ROLE_KEYS if k in s}
    if forms:
        backing.append(s["aggregates"])

    # reversibility
    r, i = s["reversible"], s["irreversible"]
    if r >= pos and i >= pos:
        # both statements are entailed (e.g. dynamic droplets that mature into
        # solid aggregates): reversible under some conditions only
        out["reversibility"] = "conditionally_reversible"
        backing.append(min(r, i))
    elif r >= pos and r - i >= margin:
        out["reversibility"] = "reversible"
        backing.append(r)
    elif i >= pos and i - r >= margin:
        out["reversibility"] = "irreversible"
        backing.append(i)
    else:
        out["reversibility"] = "unknown"

    # structure
    b, a = s["beta"], s["alpha"]
    if b >= pos and a >= pos:
        out["aggregate_structure"] = "mixed"
        backing.append(min(a, b))
    elif b >= pos and b - a >= margin:
        out["aggregate_structure"] = "beta-sheet"
        backing.append(b)
    elif a >= pos and a - b >= margin:
        out["aggregate_structure"] = "alpha-helical"
        backing.append(a)
    else:
        out["aggregate_structure"] = "unknown"

    # functional vs pathological
    f, p = s["functional"], s["pathological"]
    if f >= pos and p >= pos:
        out["functional_class"] = "both"
        backing.append(min(f, p))
    elif f >= pos and f - p >= margin:
        out["functional_class"] = "functional"
        backing.append(f)
    elif p >= pos and p - f >= margin:
        out["functional_class"] = "pathological"
        backing.append(p)
    else:
        out["functional_class"] = "unknown"

    # assembly mechanism
    au, co = s["autocatalytic"], s["cofactor_dependent"]
    if au >= pos and co >= pos:
        out["assembly_mechanism"] = "mixed"
        backing.append(min(au, co))
    elif au >= pos and au - co >= margin:
        out["assembly_mechanism"] = "autocatalytic"
        backing.append(au)
    elif co >= pos and co - au >= margin:
        out["assembly_mechanism"] = "cofactor_dependent"
        backing.append(co)
    else:
        out["assembly_mechanism"] = "unknown"

    # LLPS: positive evidence only
    if s["llps"] >= pos:
        out["undergoes_llps"] = "yes"
        backing.append(s["llps"])
    else:
        out["undergoes_llps"] = "unknown"

    out["extraction_confidence"] = round(sum(backing) / len(backing), 4) if backing else round(s["aggregates"] * 0.5, 4)
    out["undecided"] = [k for k in ("reversibility", "aggregate_structure", "functional_class",
                                    "assembly_mechanism")
                        if out[k] == "unknown"] if forms else []
    out["claim_scores"] = {k: round(v, 4) for k, v in s.items()}
    return out


class ClaimClassifier:
    """Scores the `CLAIMS` hypotheses against a premise with a zero-shot NLI model.

    ``pipe`` may be injected (tests); otherwise a transformers
    ``zero-shot-classification`` pipeline is created lazily.
    """

    def __init__(self, model: Optional[str] = None, models_dir: Optional[str] = None,
                 device: str = "auto", pipe: Optional[Callable] = None, batch_size: int = 8):
        self.model = model or resolve("nli", models_dir)
        self.device = device
        self.batch_size = batch_size
        self._pipe_obj = pipe

    def _pipe(self):
        if self._pipe_obj is None:
            transformers = require("transformers")
            self._pipe_obj = transformers.pipeline(
                "zero-shot-classification", model=self.model,
                device=resolve_device(self.device))
        return self._pipe_obj

    def score(self, premise: str, protein: str, keys: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """Max entailment per claim key; ``keys`` restricts the claims scored
        (re-scoring roles only needs ROLE_KEYS, a third of the labels)."""
        # Flatten every phrasing into one zero-shot call, then keep the maximum
        # entailment per claim key.
        wanted = set(keys) if keys else set(CLAIMS)
        label_to_key = {}
        labels = []
        for key, templates in CLAIMS.items():
            if key not in wanted:
                continue
            for t in templates:
                lbl = t.format(p=protein)
                label_to_key[lbl] = key
                labels.append(lbl)
        result = self._pipe()(premise, candidate_labels=labels,
                              hypothesis_template="{}", multi_label=True,
                              batch_size=self.batch_size)
        by_label = dict(zip(result["labels"], result["scores"]))
        out = {k: 0.0 for k in CLAIMS if k in wanted}
        for lbl, score in by_label.items():
            key = label_to_key.get(lbl)
            if key is not None:
                out[key] = max(out[key], float(score))
        return out
