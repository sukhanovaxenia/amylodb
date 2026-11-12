#!/usr/bin/env python3
"""Re-validate an existing mining database with the current resolution rules.

Re-mining thousands of abstracts is not needed to apply stricter name
resolution: the abstracts and the written forms of every protein are already
stored. This script copies the database and, for each protein,

1. expands it if its written form is a short form defined in a linked abstract
   (Schwartz-Hearst), e.g. CAA -> cerebral amyloid angiopathy;
2. rejects noise (chemicals, disease names, category nouns, fragments);
3. re-resolves it in UniProt with the strict rules (no substrate-in-enzyme-name
   matches, no gene-symbol-only matches in other organisms for lower-case
   words, short symbols must be corroborated by the abstract) and REPLACES the
   previously stored identifiers;
4. assigns a role: 'modifier' for chaperones / disaggregases, else 'aggregator'.

Nothing is deleted. Every protein gets ``status`` = 'kept' or
'dropped:<reason>', and the provenance columns (uniprot_name, match_type,
lookup_stage, resolved_from, protein_role). A JSON report lists every change.

Consumes: a mining SQLite database (any schema version since the first run).
Emits: a new database at --out and a report at --report.

Usage:
    python scripts/revalidate_mining.py --db data/mining_v2/amyloid_aggregation.db \
        --out data/mining_v2/amyloid_aggregation.revalidated.db --report revalidation.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple


# Modifier subtypes whose votes are recorded but not acted on until calibrated.
UNVERIFIED_SUBTYPES = {"regulator"}


def _call(lookup: Callable, name: str, surface: str, text: str):
    """Call a lookup with the abstract as context when it accepts one (stubs may not)."""
    try:
        return lookup(name, surface, context=text)
    except TypeError:
        return lookup(name, surface)


def _resolve(lookup: Callable, pd: Dict, name: str, surface: str, text: str,
             report: Dict, pid: int) -> Tuple[Optional[Dict], str]:
    """Lookup with the short-form and truncated-name fallbacks; (meta, name used).

    Propagates ``UniProtUnavailable``: an unreachable UniProt must not turn into
    a no_uniprot_match drop.
    """
    from amylodb.mining.ner.normalizer import ProteinNormalizer
    from amylodb.mining.validation import extend_truncated, looks_like_noise, paralog_extension
    meta = _call(lookup, name, surface, text)
    if not meta and pd.get("resolved_from"):
        # same fallback as enrich_proteins: the short form may resolve,
        # flagged so the gate requires the UniProt name in the text
        meta = _call(lookup, ProteinNormalizer.normalize(pd["resolved_from"]),
                     pd["resolved_from"], text)
        if meta:
            pd["short_form_fallback"] = True
    if not meta:
        # NER may cut a name before its head noun ("Prostatic Acid")
        longer = extend_truncated(surface, text)
        if longer and not looks_like_noise(longer):          # "aspartic" -> "aspartic protease" is a class
            meta = _call(lookup, ProteinNormalizer.normalize(longer), longer, text)
            if meta:
                name = ProteinNormalizer.normalize(longer)
                report["expanded"].append({"id": pid, "from": surface, "to": longer})
    if not meta:
        # a family head the text writes with one member mark only ("syntaxin 6")
        member = paralog_extension(surface, text) or paralog_extension(name, text)
        if member:
            meta = _call(lookup, ProteinNormalizer.normalize(member), member, text)
            if meta:
                name = ProteinNormalizer.normalize(member)
                report["expanded"].append({"id": pid, "from": surface, "to": member})
    return meta, name


def rescore_role(rescorer, texts: List[str], names: List[str], label: str,
                 max_papers: int = 25, seed: int = 7,
                 scores_out: Optional[List[Dict]] = None,
                 thresholds: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str], Dict]:
    """Re-decide one protein's role from the abstracts it is linked to.

    Each abstract that genuinely mentions the protein gives a premise (the
    mention sentences plus the following ones) scored over the NLI role claims.
    With ``thresholds`` (from scripts/calibrate_roles.py) the decision is made
    once, on the chosen statistic of the scores over the protein's abstracts,
    against the calibrated cut-offs. Without them it falls back to a per-abstract
    vote with `decide_role`, where ties go to 'aggregator'. At most
    ``max_papers`` abstracts are scored (seeded sample).

    Returns (role, subtype, votes): role is 'aggregator', a 'modifier', or None
    when no abstract mentions the protein. The raw per-abstract role scores are
    appended to ``scores_out`` when given.
    """
    from amylodb.mining.models.nli import (ROLE_KEYS, decide_protein_role, decide_role,
                                           focus_premise, _SENT_SPLIT)
    from amylodb.mining.validation import genuine_mention
    pool = list(texts)
    if len(pool) > max_papers:
        pool = random.Random(seed).sample(pool, max_papers)
    from amylodb.mining.resolution import lexical_modifier
    votes: Counter = Counter()
    rows: List[Dict] = []          # scores of abstracts not overridden lexically
    for t in pool:
        sents = _SENT_SPLIT.split(t)
        if not any(genuine_mention(sent, names) for sent in sents):
            continue
        sc = {k: float(v) for k, v in rescorer.score(focus_premise(t, names), label, keys=ROLE_KEYS).items()}
        if scores_out is not None:
            scores_out.append({k: round(v, 4) for k, v in sc.items()})
        # A paper that applies a chaperone noun to the protein ("the chaperone
        # metallothionein-3") votes chaperone whatever the model says: zero-shot
        # NLI entails self-aggregation for almost anything named in an amyloid
        # abstract, so its aggregator vote in such a paper is not evidence.
        if lexical_modifier(sents, names):
            votes["chaperone"] += 1
            continue
        rows.append(sc)
        votes[decide_role(sc) or "abstain"] += 1
    if not rows and not votes:
        return None, None, dict(votes)
    mods = {k: v for k, v in votes.items() if k not in ("aggregator", "abstain")}
    top_mod = max(mods.items(), key=lambda kv: kv[1])[0] if mods else None
    if thresholds:
        if not rows:                               # every paper names it a chaperone
            return "modifier", top_mod, dict(votes)
        stat = thresholds.get("statistic", "mean")
        agg = (mean if stat == "mean" else max)
        stats = {k: agg([r.get(k, 0.0) for r in rows]) for k in ROLE_KEYS}
        role, subtype = decide_protein_role(stats, thresholds)
        return role, subtype, dict(votes)
    # One paper suffices for each role (a protein may be both: gelsolin forms the
    # AGel amyloid in some papers and inhibits Abeta fibrillisation in others).
    # The primary role is aggregator when any paper supports self-aggregation;
    # the modifier subtypes any paper supports are returned in the votes and
    # stored as modifier_roles by the caller.
    if votes.get("aggregator", 0):
        return "aggregator", None, dict(votes)
    if top_mod:
        return "modifier", top_mod, dict(votes)
    return None, None, dict(votes)


def revalidate(db_path: str, out_path: str, lookup: Optional[Callable] = None,
               require_uniprot: bool = True, sleep: float = 0.0,
               rescorer=None, max_papers: int = 25, role_thresholds: Optional[Dict] = None) -> Dict:
    """Re-apply the resolution rules to a copy of a mining DB (see module docstring).

    ``rescorer``: an object with ``score(premise, protein, keys)`` (a
    `ClaimClassifier`); when given, the role of every kept, non-curated protein
    is re-decided from its abstracts. ``role_thresholds`` (from
    scripts/calibrate_roles.py) switches the decision to the calibrated
    protein-level rule; without it the per-abstract vote is used and a protein
    whose abstracts assert no role is dropped as 'no_role_claim'.
    """
    from amylodb.mining.database.storage import AmyloidDatabase
    from amylodb.mining.ner.normalizer import ProteinNormalizer, UniProtClient, UniProtUnavailable
    from amylodb.mining.resolution import (assign_role, gate_resolved, lexical_modifier,
                                            prepare_candidates)
    from amylodb.mining.validation import relevance
    _SENT = re.compile(r"(?<=[.!?])\s+")
    from amylodb.mining.validation import extend_truncated, find_abbreviations

    lookup = lookup or UniProtClient.lookup
    shutil.copyfile(db_path, out_path)
    AmyloidDatabase(out_path)                      # applies the additive migrations

    con = sqlite3.connect(out_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    has_claims = cur.execute("SELECT 1 FROM sqlite_master WHERE name='protein_claims'").fetchone()

    proteins = [dict(r) for r in cur.execute("SELECT * FROM proteins")]
    report = {"db": db_path, "out": out_path, "proteins": len(proteins),
              "status": Counter(), "dropped": [], "expanded": [], "reidentified": [],
              "modifiers": [], "role_changes": [], "role_subtypes": Counter(),
              "forms_aggregates_repaired": 0, "condensates": 0}

    for p in proteins:
        pid = p["id"]
        texts = [f"{r['title'] or ''}. {r['abstract'] or ''}" for r in cur.execute(
            "SELECT pa.title, pa.abstract FROM protein_papers pp JOIN papers pa ON pa.pmid = pp.pmid "
            "WHERE pp.protein_id = ?", (pid,))]
        text = " ".join(texts)

        surfaces: Counter = Counter()
        scores: Dict[str, float] = {}
        if has_claims:
            for r in cur.execute("SELECT surface_form, claim_scores FROM protein_claims WHERE protein_id = ?", (pid,)):
                if r["surface_form"]:
                    surfaces[r["surface_form"]] += 1
                for k, v in json.loads(r["claim_scores"] or "{}").items():
                    scores[k] = max(scores.get(k, 0.0), float(v))
        surface = surfaces.most_common(1)[0][0] if surfaces else p["protein_name"]

        abbreviations: Dict[str, str] = {}
        for t in texts:
            for k, v in find_abbreviations(t).items():
                abbreviations.setdefault(k, v)

        pd = {"protein_name": surface, "surface_form": surface, "claim_scores": scores}
        kept, dropped = prepare_candidates([pd], text, abbreviations)
        if pd.get("resolved_from"):
            report["expanded"].append({"id": pid, "from": pd["resolved_from"], "to": pd["protein_name"]})

        annotations = {"resolved_from": pd.get("resolved_from")}
        if dropped:
            status = f"dropped:{dropped[0][1]}"
        else:
            name = ProteinNormalizer.normalize(pd["protein_name"])
            try:
                meta, name = _resolve(lookup, pd, name, surface, text, report, pid)
            except UniProtUnavailable:
                # not "no match": the status says so and the next run retries
                pd["lookup_error"], meta = True, None
            if sleep:
                time.sleep(sleep)
            for key in ("uniprot_id", "gene_name", "organism", "uniprot_name",
                        "match_type", "lookup_stage", "protein_role", "organism_in_text"):
                pd[key] = (meta or {}).get(key)
            reason = gate_resolved(pd, text, require_uniprot=require_uniprot)
            if reason:
                status = f"dropped:{reason}"
            else:
                status = "kept"
                old_role = p.get("protein_role") or "aggregator"
                keyword_modifier = pd.get("protein_role") == "modifier"
                # Deterministic evidence only here (UniProt keyword, curated
                # chaperone). The stored per-paper claim scores are not acted on:
                # deciding on them with fixed thresholds is the error that made
                # GAPDH and 940 others "regulators"; a role from claims comes only
                # from rescore_role below, by vote or by calibrated thresholds.
                role = assign_role(pd, from_claims=False)
                votes = None
                curated = pd.get("match_type") == "known"
                names = [x for x in (surface, name, pd.get("resolved_from"), pd.get("uniprot_name"),
                                     pd.get("gene_name")) if x]
                if rescorer is not None and not curated and not keyword_modifier:
                    per_abstract: List[Dict] = []
                    new_role, subtype, votes = rescore_role(rescorer, texts, names, surface, max_papers,
                                                            scores_out=per_abstract,
                                                            thresholds=role_thresholds)
                    annotations["role_votes"] = json.dumps(votes)
                    annotations["role_scores"] = json.dumps(per_abstract)
                    # An abstention is not evidence of absence: half of the proteins
                    # dropped as no_role_claim on the expanded corpus were real
                    # (vitronectin, whose sentence says it forms amyloid fibrils;
                    # ubiquilin-2, nucleobindin-1). The role stands and the
                    # vocabulary gates below decide instead.
                    if new_role is not None:
                        role, pd["protein_role"], pd["role_subtype"] = new_role, new_role, subtype
                # Modifier subtypes supported by at least one paper; they may sit
                # beside an aggregator role (dual-role proteins such as gelsolin).
                mod_roles = {k for k, v in (votes or {}).items()
                             if k not in ("aggregator", "abstain") and v}
                if role == "modifier" and pd.get("role_subtype"):
                    mod_roles.add(pd["role_subtype"])
                if not curated:
                    rel = relevance(texts, names)
                    # without the model, a chaperone noun applied to the protein is
                    # the only per-paper role evidence (with it, rescore_role has
                    # already counted such papers as chaperone votes)
                    if votes is None:
                        sents = [s for t in texts for s in _SENT.split(t)]
                        if lexical_modifier(sents, names):
                            mod_roles.add("chaperone")
                            if role == "aggregator":
                                role, pd["protein_role"], pd["role_subtype"] = "modifier", "modifier", "chaperone"
                    # Literature that never mentions aggregation is not about amyloid:
                    # with condensate vocabulary (stress granules, P-bodies, the
                    # nucleolus, LLPS) the protein is a condensate candidate, input
                    # for condensatedb; otherwise it is off-topic.
                    if not rel["agg_abstract"]:
                        mod_roles.clear()
                        if rel["cond_abstract"]:
                            role, pd["protein_role"], pd["role_subtype"] = "condensate", "condensate", None
                        else:
                            status = "dropped:off_topic"
                    elif role == "aggregator" and not rel["agg_mention"]:
                        # the text never names it beside aggregation: not an
                        # aggregator, though a supported modifier role stands
                        if mod_roles:
                            top = sorted(mod_roles)[0] if not votes else max(
                                mod_roles, key=lambda k: votes.get(k, 0))
                            role, pd["protein_role"], pd["role_subtype"] = "modifier", "modifier", top
                        else:
                            status = "dropped:no_aggregation_mention"
                    # An unverified subtype (regulator: 0 of 16 right on the second
                    # v3 modifier sheet) neither makes a modifier nor tags an
                    # aggregator as dual-role. The votes keep it for a later,
                    # calibrated rule. A protein left with no role is a condensate
                    # candidate when the text puts it beside a condensate, else it
                    # is dropped under a status of its own.
                    unverified = mod_roles & UNVERIFIED_SUBTYPES
                    mod_roles -= UNVERIFIED_SUBTYPES
                    if status == "kept" and role == "modifier" and pd.get("role_subtype") in UNVERIFIED_SUBTYPES:
                        if mod_roles:
                            top = max(sorted(mod_roles), key=lambda k: (votes or {}).get(k, 0))
                            pd["role_subtype"] = top
                        elif rel["cond_mention"] or rel["cond_abstract"]:
                            role, pd["protein_role"] = "condensate", "condensate"
                            pd["role_subtype"] = None
                        else:
                            status = f"dropped:{sorted(unverified)[0] if unverified else 'regulator'}_only"
                    # condensate evidence for condensatedb, whatever the primary role:
                    # TDP-43 and FUS are aggregators and stress-granule proteins
                    annotations["condensate_evidence"] = int(rel["cond_mention"] or role == "condensate")
                annotations["modifier_roles"] = json.dumps(sorted(mod_roles)) if mod_roles else None
                if status == "kept" and role != old_role:
                    report["role_changes"].append({"id": pid, "name": name, "from": old_role,
                                                   "to": role, "subtype": pd.get("role_subtype")})
            if status == "kept":
                # Repair forms_aggregates when the role is re-decided. A mining
                # run that acted on the broad modifier claims suppressed this
                # field for proteins it mislabelled, so the stored value cannot
                # be trusted where the role itself was recomputed: an aggregator
                # forms the aggregates, a modifier acts on another protein's.
                if rescorer is not None or role != old_role:
                    cur.execute("UPDATE proteins SET forms_aggregates = ? WHERE id = ?",
                                (1 if role == "aggregator" else 0, pid))
                    report["forms_aggregates_repaired"] += 1
                report["role_subtypes"][pd.get("role_subtype") or role] += 1
                if role == "condensate":
                    report["condensates"] += 1
                if role == "modifier":
                    report["modifiers"].append({"id": pid, "name": name, "uniprot_id": pd["uniprot_id"],
                                                "subtype": pd.get("role_subtype")})
                if pd["uniprot_id"] != p["uniprot_id"]:
                    report["reidentified"].append({"id": pid, "name": name,
                                                   "old": p["uniprot_id"], "new": pd["uniprot_id"]})
                # protein_name / organism form a UNIQUE key in this table, so the
                # resolved identity goes to resolved_* columns; rows that now share
                # a UniProt id are merged by the amylodb build, not here.
                cur.execute("UPDATE proteins SET uniprot_id = ?, gene_name = ?, resolved_name = ?, "
                            "resolved_organism = ? WHERE id = ?",
                            (pd["uniprot_id"], pd["gene_name"], name, pd["organism"], pid))
            annotations.update({k: pd.get(k) for k in ("uniprot_name", "match_type",
                                                        "lookup_stage", "protein_role", "role_subtype")})
            if pd.get("organism_in_text") is not None:
                annotations["organism_in_text"] = int(bool(pd["organism_in_text"]))
        if status != "kept":
            # the stored accession came from the old, looser lookup; the source DB keeps it
            cur.execute("UPDATE proteins SET uniprot_id = NULL, gene_name = NULL WHERE id = ?", (pid,))
            report["dropped"].append({"id": pid, "name": p["protein_name"], "surface": surface,
                                      "reason": status.split(":", 1)[1],
                                      "long_form": pd.get("protein_name") if pd.get("resolved_from") else None})
        report["status"][status] += 1
        # The role fields are this run's decision and are written even when
        # empty: skipping None left the mining-time subtypes in place, so 425
        # condensate candidates and 370 aggregators of the v3 base still carried
        # "regulator". Other annotations only fill gaps.
        decided = ("protein_role", "role_subtype", "modifier_roles", "condensate_evidence")
        if status != "kept":
            annotations.update({k: None for k in decided})
        cols = {k: v for k, v in annotations.items() if v not in (None, "") or k in decided}
        sets = ", ".join(f"{c} = ?" for c in ["status", *cols])
        cur.execute(f"UPDATE proteins SET {sets} WHERE id = ?", [status, *cols.values(), pid])

    con.commit()
    kept_uni = Counter(r[0] for r in cur.execute(
        "SELECT uniprot_id FROM proteins WHERE status = 'kept' AND uniprot_id IS NOT NULL"))
    report["kept_distinct_uniprot"] = len(kept_uni)
    report["duplicate_uniprot_groups"] = sum(1 for n in kept_uni.values() if n > 1)
    con.close()
    report["status"] = dict(report["status"])
    report["role_subtypes"] = dict(report["role_subtypes"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-validate a mining database with the current rules")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="path of the re-validated copy")
    ap.add_argument("--report", default="revalidation.json")
    ap.add_argument("--no-require-uniprot", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.05, help="pause between UniProt queries (s)")
    ap.add_argument("--rescore-claims", action="store_true",
                    help="re-decide roles with the NLI model from the stored abstracts")
    ap.add_argument("--models-dir", default=None, help="local model directory (scripts/fetch_models.py)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--max-papers", type=int, default=25, help="abstracts scored per protein")
    ap.add_argument("--role-thresholds", default=None,
                    help="role_thresholds.json from scripts/calibrate_roles.py; applies the "
                         "calibrated protein-level rule instead of the per-abstract vote")
    args = ap.parse_args()

    rescorer, thresholds = None, None
    if args.rescore_claims:
        from amylodb.mining.models.nli import ClaimClassifier
        rescorer = ClaimClassifier(models_dir=args.models_dir, device=args.device)
    if args.role_thresholds:
        thresholds = json.load(open(args.role_thresholds, encoding="utf-8"))
    rep = revalidate(args.db, args.out, require_uniprot=not args.no_require_uniprot, sleep=args.sleep,
                     rescorer=rescorer, max_papers=args.max_papers, role_thresholds=thresholds)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"proteins {rep['proteins']}: {rep['status']}")
    print(f"kept distinct UniProt {rep['kept_distinct_uniprot']}, "
          f"groups sharing one UniProt {rep['duplicate_uniprot_groups']} (merged at amylodb build)")
    print(f"expanded abbreviations {len(rep['expanded'])}, re-identified {len(rep['reidentified'])}, "
          f"modifiers {len(rep['modifiers'])}")
    print(f"roles: {rep['role_subtypes']}; changed {len(rep['role_changes'])}; "
          f"routed to condensate candidates {rep['condensates']}")
    reasons = defaultdict(list)
    for d in rep["dropped"]:
        reasons[d["reason"]].append(d["surface"])
    for r, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"  dropped {r} ({len(names)}): {', '.join(sorted(set(names))[:25])}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
