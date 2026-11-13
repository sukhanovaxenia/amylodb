"""amylodb orchestration: curated load -> lift -> reconcile -> features -> export.

The literature-mining stage runs on the user's machine (it needs PubMed /
UniProt / model access); here it is an *optional* injected input, so the whole
pipeline runs end to end on curated data alone for testing. When mining output
is available it is passed in as a list of `AmyloidRecord` and reconciled
alongside the curated records.

Consumes: a mapping of curated-source name -> file path, and optionally a list
of mined `AmyloidRecord`.
Emits: files under ``output_dir`` (records.tsv, features.tsv.gz, graph.json,
amyloid_sequences.fasta, conflicts.json, reconcile_stats.json) and returns the
`ReconcileResult`.
Invariant (boundary): records parsed in == records grouped in reconcile; every
consensus record gets exactly one feature row and one graph node.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from typing import Dict, List, Optional

from .curated.parsers import PARSERS
from .features_matrix import build_feature_matrix
from .graph_export import build_graph
from .reconcile import ReconcileResult, reconcile
from .schema import AmyloidRecord

logger = logging.getLogger(__name__)


def load_curated(input_files: Dict[str, str]) -> List[AmyloidRecord]:
    """Parse curated sources and lift each entry into an `AmyloidRecord`.

    Unknown source keys and missing files are logged and skipped, never fatal.
    """
    records: List[AmyloidRecord] = []
    for key, path in input_files.items():
        if not path or not os.path.exists(path):
            logger.warning("Skipping %s: file not found (%s)", key, path)
            continue
        norm = key.lower().replace("-", "_").replace(" ", "_")
        parser_cls = PARSERS.get(norm)
        if parser_cls is None:
            logger.warning("Unknown curated source: %s", key)
            continue
        entries = parser_cls(key, logger).parse(path)
        records.extend(AmyloidRecord.from_curated(e) for e in entries)
    logger.info("Loaded %d curated records from %d sources", len(records), len(input_files))
    return records


def _assign_record_ids(records: List[AmyloidRecord]) -> None:
    """Give every record a stable, unique record_id derived from its dedup key.

    Needed so feature rows and graph nodes can be keyed back to records.
    """
    seen: Dict[str, int] = {}
    for r in records:
        base = r.record_id or f"{r.source_db}:{'|'.join(map(str, r.dedup_key()))}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        r.record_id = base if n == 0 else f"{base}#{n}"


def run(input_files: Dict[str, str],
        output_dir: str,
        mined: Optional[List[AmyloidRecord]] = None,
        exclude_transmissible_prion: bool = False,
        compute_features: bool = True,
        modifiers: Optional[List[Dict]] = None) -> ReconcileResult:
    """Run the full amylodb build. See module docstring for I/O and invariants.

    ``modifiers`` are mined chaperones / disaggregases (dicts from
    scripts/export_mined.py). They are not reconciled as amyloid records; they
    are written to modifiers.tsv with the aggregators mentioned in the same papers.
    """
    os.makedirs(output_dir, exist_ok=True)

    curated = load_curated(input_files)
    records = curated + list(mined or [])
    logger.info("Total input: %d curated + %d mined", len(curated), len(mined or []))

    result = reconcile(records, exclude_transmissible_prion=exclude_transmissible_prion)
    _assign_record_ids(result.consensus)

    # --- records.tsv (metadata + unified fields) ---
    _write_records_tsv(result.consensus, os.path.join(output_dir, "records.tsv"))

    # --- features.tsv.gz (fixed 444-wide matrix) ---
    if compute_features:
        cols, rows = build_feature_matrix(result.consensus)
        feat_path = os.path.join(output_dir, "features.tsv.gz")
        with gzip.open(feat_path, "wt", encoding="utf-8") as fh:
            fh.write("record_id\t" + "\t".join(cols) + "\n")
            for rec, row in zip(result.consensus, rows):
                fh.write(rec.record_id + "\t" + "\t".join(f"{v:.6g}" for v in row) + "\n")
        logger.info("Wrote features: %d rows x %d cols", len(rows), len(cols))

    # --- graph.json (nodes + homology/interaction/co-membership edges) ---
    graph = build_graph(result.consensus)
    with open(os.path.join(output_dir, "graph.json"), "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)

    # --- conflicts + stats ---
    with open(os.path.join(output_dir, "conflicts.json"), "w", encoding="utf-8") as fh:
        json.dump(result.conflicts, fh, indent=2)

    if modifiers:
        # condensate candidates (stress-granule / P-body / LLPS literature without
        # aggregation) are input for condensatedb, kept apart from amyloid modifiers
        from amylodb.mining.adapter import has_modifier_role, is_condensate
        mods = [m for m in modifiers if has_modifier_role(m)]
        conds = [m for m in modifiers if is_condensate(m)]
        _write_modifiers_tsv(mods, os.path.join(output_dir, "modifiers.tsv"))
        _write_modifiers_tsv(conds, os.path.join(output_dir, "condensate_candidates.tsv"))
        result.stats["modifiers"] = len(mods)
        result.stats["dual_role"] = sum((m.get("protein_role") == "aggregator") for m in mods)
        result.stats["condensate_candidates"] = len(conds)

    # support of the mined aggregator records: curated | text | model_only (predicted)
    from collections import Counter as _Counter
    tiers = _Counter(r.aggregation_evidence for r in result.consensus
                     if getattr(r, "aggregation_evidence", ""))
    if tiers:
        result.stats["mined_aggregation_evidence"] = dict(tiers)

    with open(os.path.join(output_dir, "reconcile_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(result.stats, fh, indent=2)

    logger.info("amylodb build complete -> %s", output_dir)
    return result


def _write_modifiers_tsv(modifiers: List[Dict], path: str) -> None:
    """One row per protein in a side list (modifiers.tsv, condensate_candidates.tsv).

    ``primary_role`` says whether the protein is also an amyloid record
    (aggregator) -- a dual-role protein appears in both outputs.
    ``modifier_roles`` lists every modifier subtype some paper supports;
    ``condensate_evidence`` marks a condensate term beside the protein;
    ``condensate_role`` is "regulator" for candidates that reached the list
    through regulator votes only, "component" for the other condensate rows.
    """
    import csv
    cols = ["protein_name", "primary_role", "modifier_roles", "condensate_evidence", "condensate_role", "uniprot_id",
            "gene_name", "organism", "uniprot_name", "co_mentioned_aggregators",
            "disassembly_cofactors", "source_pmids"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for m in modifiers:
            roles = list(m.get("modifier_roles") or [])
            if not roles and m.get("role_subtype") and m.get("protein_role") == "modifier":
                roles = [m["role_subtype"]]
            crole = ("" if m.get("protein_role") != "condensate" else
                     "regulator" if m.get("role_subtype") == "regulator" else "component")
            w.writerow([m.get("protein_name", ""), m.get("protein_role", "") or "",
                        ";".join(roles), int(bool(m.get("condensate_evidence"))
                                             or m.get("protein_role") == "condensate"), crole,
                        m.get("uniprot_id", ""), m.get("gene_name", ""),
                        m.get("organism", ""), m.get("uniprot_name", ""),
                        ";".join(m.get("co_mentioned_aggregators", []) or []),
                        ";".join(m.get("disassembly_cofactors", []) or []),
                        ";".join(m.get("source_pmids", []) or [])])
    logger.info("Wrote %d rows to %s", len(modifiers), path)


# Metadata columns written to records.tsv, in fixed order.
_RECORD_COLUMNS = [
    "record_id", "provenance", "record_level", "prion_class", "evidence_tier",
    "source_db", "protein_name", "gene_name", "uniprot_id", "organism",
    "region_start", "region_end", "sequence", "is_amyloid", "experimental_label",
    "secondary_structure_class", "pathogenicity",
    "reversibility", "assembly_mechanism", "is_functional", "is_pathological",
    "extraction_confidence", "evidence_weight", "pmid", "source_pmids",
    "undergoes_llps", "reported_methods", "reported_method_tier",
    "reversibility_votes", "claim_scores", "aggregation_evidence",
]


def _write_records_tsv(records: List[AmyloidRecord], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(_RECORD_COLUMNS)
        for r in records:
            w.writerow([
                r.record_id, r.provenance.value, r.record_level.value,
                r.prion_class.value, int(r.evidence_tier),
                r.source_db, r.protein_name, r.gene_name, r.uniprot_id, r.organism,
                r.region_start, r.region_end, r.sequence, r.is_amyloid, r.experimental_label,
                r.secondary_structure_class, r.pathogenicity,
                r.reversibility, r.assembly_mechanism, r.is_functional, r.is_pathological,
                f"{r.extraction_confidence:.3f}", f"{r.evidence_weight:.3f}",
                r.pmid, ";".join(r.source_pmids),
                "" if r.undergoes_llps is None else r.undergoes_llps,
                ";".join(r.reported_methods), r.reported_method_tier,
                json.dumps(r.reversibility_votes, sort_keys=True) if r.reversibility_votes else "",
                json.dumps(r.claim_scores, sort_keys=True) if r.claim_scores else "",
                getattr(r, "aggregation_evidence", ""),
            ])
    logger.info("Wrote %d records to %s", len(records), path)
