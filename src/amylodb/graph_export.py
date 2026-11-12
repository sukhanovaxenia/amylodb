"""Graph export for the graduated amyloid classification work.

Emits a plain node/edge dictionary (JSON-serialisable, no hard NetworkX
dependency) that the downstream graph / graduated-classification project can
load directly. Three edge kinds are produced:

  * ``belongs_to``    -- a region record linked to its protein (UniProt) hub
  * ``interaction``   -- STRING partners asserted on a record
  * ``co_mention``    -- records sharing a source PMID

The region-protein relationship is modelled as a hub-and-spoke (each region
links to one protein hub), not as a clique over co-protein regions. A clique is
O(k^2) in the number of regions per protein and blows up on densely sampled
proteins; the hub is linear and is also the shape the graduated-classification
graph wants (protein nodes connecting their regions).

Consumes: a list of `AmyloidRecord` with record_id already assigned.
Emits: ``{"nodes": [...], "edges": [...], "stats": {...}}``.
Invariant: one node per record; every edge endpoint is an existing node id.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List

from .schema import AmyloidRecord


def build_graph(records: List[AmyloidRecord]) -> Dict:
    nodes = []
    protein_hub_for: Dict[str, str] = {}   # uniprot -> node id acting as its hub
    by_pmid: Dict[str, List[str]] = defaultdict(list)
    ids = set()

    for r in records:
        nodes.append({
            "id": r.record_id,
            "level": r.record_level.value,
            "provenance": r.provenance.value,
            "uniprot_id": r.uniprot_id,
            "protein_name": r.protein_name,
            "prion_class": r.prion_class.value,
            "evidence_tier": int(r.evidence_tier),
            "is_amyloid": bool(r.is_amyloid),
            "undergoes_llps": r.undergoes_llps,
            "reported_method_tier": r.reported_method_tier,
        })
        ids.add(r.record_id)
        # A protein-level record is the natural hub for its UniProt.
        if r.uniprot_id and r.record_level.value == "protein":
            protein_hub_for.setdefault(r.uniprot_id, r.record_id)
        for pmid in r.source_pmids:
            if pmid:
                by_pmid[pmid].append(r.record_id)

    edges: List[Dict] = []

    # belongs_to: link each region to its protein hub (linear in #regions).
    # If no protein-level record exists for the UniProt, synthesise a hub node.
    synthetic_hubs: Dict[str, Dict] = {}
    for r in records:
        if not r.uniprot_id or r.record_level.value != "region":
            continue
        hub = protein_hub_for.get(r.uniprot_id)
        if hub is None:
            hub = f"prot:{r.uniprot_id}"
            if r.uniprot_id not in synthetic_hubs:
                synthetic_hubs[r.uniprot_id] = {
                    "id": hub, "level": "protein", "provenance": "derived",
                    "uniprot_id": r.uniprot_id, "protein_name": r.protein_name,
                    "prion_class": r.prion_class.value, "evidence_tier": 0,
                    "is_amyloid": False,
                }
            protein_hub_for[r.uniprot_id] = hub
        edges.append({"source": r.record_id, "target": hub, "kind": "belongs_to",
                      "key": r.uniprot_id})
    nodes.extend(synthetic_hubs.values())
    ids.update(synthetic_hubs[u]["id"] for u in synthetic_hubs)

    # co_mention: records sharing a PMID
    for pmid, members in by_pmid.items():
        if len(members) > 1:
            for a, b in combinations(sorted(set(members)), 2):
                edges.append({"source": a, "target": b, "kind": "co_mention", "key": pmid})

    # interaction: STRING partners (kept even if the partner is not a node, as a
    # dangling external reference flagged by external=True)
    for r in records:
        for partner in r.string_partners:
            edges.append({
                "source": r.record_id, "target": partner, "kind": "interaction",
                "external": partner not in ids,
            })

    stats = {
        "nodes": len(nodes),
        "record_nodes": len(nodes) - len(synthetic_hubs),
        "synthetic_protein_hubs": len(synthetic_hubs),
        "edges": len(edges),
        "belongs_to_edges": sum(1 for e in edges if e["kind"] == "belongs_to"),
        "co_mention_edges": sum(1 for e in edges if e["kind"] == "co_mention"),
        "interaction_edges": sum(1 for e in edges if e["kind"] == "interaction"),
    }
    return {"nodes": nodes, "edges": edges, "stats": stats}
