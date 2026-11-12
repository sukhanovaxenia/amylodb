#!/usr/bin/env python3
"""Validate the mining models on three abstracts with answers fixed in advance.

The abstracts below were written for this check (they are not taken from any
paper), each to state its facts unambiguously, so the expected extraction is
known before the models run:

  A  positive control   alpha-synuclein: amyloid, cross-beta, irreversible,
                         pathological; cryo-EM reported (structural tier)
  B  negative control   hexokinase metabolism: no aggregation, no LLPS
                         -> no record at all
  C  condensate case    FUS: prion-like domain, LLPS, droplets that dissolve
                         but mature into irreversible fibrils
                         -> conditionally reversible, LLPS = yes

Runs the real models (no PubMed, no LLM) and prints PASS/FAIL per expectation,
the claim scores behind each decision, and the time per abstract. A report is
written next to the models (check_report.json).

Usage:
    python scripts/check_models.py --models-dir models [--ner openbioner|scispacy] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import os
import time

ABSTRACTS = {
    "A": ("Cryo-EM structure of alpha-synuclein fibrils from a Parkinson's disease brain",
          "Alpha-synuclein aggregates into amyloid fibrils that accumulate in Lewy bodies in "
          "Parkinson's disease. We determined the structure of ex vivo alpha-synuclein fibrils by "
          "cryo-EM at 3.1 A resolution, revealing a cross-beta fold with parallel in-register "
          "strands. The fibrils bound thioflavin T, resisted disaggregation by the Hsp70 chaperone "
          "system, remained stable in 4 M guanidine and seeded the aggregation of soluble "
          "alpha-synuclein in neurons, where they were toxic."),
    "B": ("Hexokinase activity in liver mitochondria of fasting mice",
          "We measured hexokinase activity in isolated liver mitochondria from fed and fasted mice "
          "using a coupled spectrophotometric assay. Fasting reduced hexokinase activity by 40 %, "
          "while glucokinase regulatory protein levels were unchanged. These data describe the "
          "metabolic adaptation of hepatic glucose phosphorylation to nutrient deprivation."),
    "C": ("Phase separation of FUS and its conversion into fibrils",
          "FUS, an RNA-binding protein with a prion-like low-complexity domain, undergoes "
          "liquid-liquid phase separation and forms dynamic droplets that dissolve when stress is "
          "relieved. Over time, and much faster for ALS-linked mutants, FUS droplets mature into "
          "stable, irreversible fibrillar aggregates that no longer dissolve. Fluorescence recovery "
          "after photobleaching showed the loss of FUS mobility during maturation."),
}


def _find(proteins, needle):
    for p in proteins:
        if needle in p["protein_name"].lower():
            return p
    return None


def main() -> int:
    from amylodb.mining.models.nli import ClaimClassifier
    from amylodb.mining.models.structner import ProteinStructureNER
    from amylodb.mining.models.triage import RelevanceTriage
    from amylodb.mining.models_extractor import ModelHybridExtractor
    from amylodb.mining.adapter import mined_to_record
    from amylodb.provenance import EvidenceTier

    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--ner", default="openbioner", choices=["openbioner", "scispacy", "pubmedbert"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ex = ModelHybridExtractor(ner_backend=args.ner, llm=None,
                              nli=ClaimClassifier(models_dir=args.models_dir, device=args.device),
                              structner=ProteinStructureNER(models_dir=args.models_dir, device=args.device))
    triage = RelevanceTriage(models_dir=args.models_dir, device=args.device, mode="annotate")

    results, checks = {}, []

    def check(case, label, ok, detail=""):
        checks.append({"case": case, "check": label, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))

    for case, (title, abstract) in ABSTRACTS.items():
        t0 = time.perf_counter()
        res = ex.extract(title, abstract, f"check-{case}")
        tri = triage.score([f"{title}. {abstract}"])[0]
        dt = time.perf_counter() - t0
        results[case] = {"proteins": res.proteins, "triage": tri, "seconds": round(dt, 2)}
        print(f"\n=== {case}: {title}  [{dt:.1f} s, triage={tri:.2f}]")
        for p in res.proteins:
            print(f"  -> {p['protein_name']}: agg={p['forms_aggregates']} rev={p['reversibility']} "
                  f"struct={p['aggregate_structure']} func={p['functional_class']} "
                  f"llps={p['undergoes_llps']} methods={p['experimental_methods']} "
                  f"tier={p['reported_method_tier']}\n     scores={p['claim_scores']}")

        if case == "A":
            a = _find(res.proteins, "synuclein")
            check(case, "alpha-synuclein extracted", a is not None)
            if a:
                check(case, "forms aggregates", a["forms_aggregates"])
                check(case, "structure = beta-sheet", a["aggregate_structure"] == "beta-sheet", a["aggregate_structure"])
                check(case, "reversibility = irreversible", a["reversibility"] == "irreversible", a["reversibility"])
                check(case, "functional class = pathological", a["functional_class"] == "pathological", a["functional_class"])
                check(case, "cryo-EM reported, structural tier",
                      a["reported_method_tier"] == int(EvidenceTier.STRUCTURAL), str(a["experimental_methods"]))
            h = _find(res.proteins, "hsp70")
            check(case, "(secondary) Hsp70 not stored as aggregating", h is None)
        elif case == "B":
            check(case, "no record from a non-aggregation abstract", len(res.proteins) == 0,
                  ", ".join(p["protein_name"] for p in res.proteins))
        elif case == "C":
            f = _find(res.proteins, "fus")
            check(case, "FUS extracted", f is not None)
            if f:
                check(case, "LLPS = yes", f["undergoes_llps"] == "yes", f["undergoes_llps"])
                check(case, "reversibility = conditionally_reversible",
                      f["reversibility"] == "conditionally_reversible", f["reversibility"])
                rec = mined_to_record({**f, "biological_role": "RNA-binding protein with a prion-like domain"})
                check(case, "prion-like domain kept and tagged", rec.prion_class.value == "prion_like_domain")

    passed = sum(c["pass"] for c in checks)
    print(f"\n{passed}/{len(checks)} checks passed")
    out_dir = args.models_dir or "."
    with open(os.path.join(out_dir, "check_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "results": results}, fh, indent=2, default=str)
    print(f"Report: {os.path.join(out_dir, 'check_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
