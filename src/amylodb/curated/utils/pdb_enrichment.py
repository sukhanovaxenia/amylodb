"""
PDB/EMDB metadata enrichment for amyloid entries.

Tier 1: AmyloidAtlas scraping (739 curated entries)
Tier 2: RCSB PDB GraphQL API (helical parameters, symmetry, citations)

No LLM costs - fully free approach.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ===== Data Classes =====

@dataclass
class PDBMetadata:
    """Structured metadata from RCSB PDB."""
    pdb_id: str
    title: str = ""
    resolution: Optional[float] = None
    method: str = ""
    
    # Helical parameters (from em_helical_entity)
    helical_twist: Optional[float] = None  # degrees
    helical_rise: Optional[float] = None   # Angstroms
    axial_symmetry: str = ""               # C1, C2, etc.
    
    # Assembly info
    assembly_symmetry: str = ""
    stoichiometry: str = ""
    
    # Cross-references
    emdb_ids: List[str] = field(default_factory=list)
    doi: str = ""
    pmid: str = ""
    
    # Protein info
    protein_name: str = ""
    organism: str = ""
    
    # Inferred fields
    protofilament_count: int = 0
    twist_handedness: str = "unknown"


@dataclass 
class AtlasEntry:
    """Entry from UCLA AmyloidAtlas."""
    protein: str
    fibril_origins: str  # species, mutations, source
    residues_ordered: str
    method: str
    resolution: str
    pdb_id: str
    reference: str


# ===== AmyloidAtlas Scraper (Tier 1) =====

ATLAS_URL = "https://people.mbi.ucla.edu/sawaya/amyloidatlas/"

def scrape_amyloid_atlas(cache_path: Optional[str] = None, 
                          logger: Optional[logging.Logger] = None) -> List[AtlasEntry]:
    """
    Scrape UCLA AmyloidAtlas HTML table.
    
    Returns list of AtlasEntry with PDB IDs and curated metadata.
    """
    if pd is None:
        raise ImportError("pandas required: pip install pandas lxml")
    if requests is None:
        raise ImportError("requests required: pip install requests")
    
    log = logger or logging.getLogger(__name__)
    
    # Check cache
    if cache_path and Path(cache_path).exists():
        log.info(f"Loading AmyloidAtlas from cache: {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
        return [AtlasEntry(**e) for e in data]
    
    log.info(f"Scraping AmyloidAtlas from {ATLAS_URL}")
    
    try:
        tables = pd.read_html(ATLAS_URL)
        if not tables:
            log.warning("No tables found on AmyloidAtlas page")
            return []
        
        df = tables[0]  # Main data table
        log.info(f"Found {len(df)} entries in AmyloidAtlas")
        
        entries = []
        for _, row in df.iterrows():
            try:
                entry = AtlasEntry(
                    protein=str(row.get('Protein', '')),
                    fibril_origins=str(row.get('Fibril Origins', '')),
                    residues_ordered=str(row.get('Residues Ordered', '')),
                    method=str(row.get('Method', '')),
                    resolution=str(row.get('Resolution (Å)', '')),
                    pdb_id=_extract_pdb_id(str(row.get('PDB ID', ''))),
                    reference=str(row.get('Reference', ''))
                )
                if entry.pdb_id:
                    entries.append(entry)
            except Exception as e:
                log.debug(f"Skipping row: {e}")
        
        log.info(f"Parsed {len(entries)} entries with PDB IDs")
        
        # Cache results
        if cache_path:
            with open(cache_path, 'w') as f:
                json.dump([e.__dict__ for e in entries], f, indent=2)
            log.info(f"Cached to {cache_path}")
        
        return entries
        
    except Exception as e:
        log.error(f"Failed to scrape AmyloidAtlas: {e}")
        return []


def _extract_pdb_id(text: str) -> str:
    """Extract 4-character PDB ID from text."""
    match = re.search(r'\b([0-9][A-Za-z0-9]{3})\b', text)
    return match.group(1).upper() if match else ""


def parse_atlas_metadata(entry: AtlasEntry) -> Dict:
    """
    Parse AmyloidAtlas entry into structural classification fields.
    
    Extracts from 'Fibril Origins' field:
    - aggregate_type (ex_vivo, in_vitro, recombinant)
    - polymorph_name
    - mutation info
    """
    origins = entry.fibril_origins.lower()
    
    # Aggregate type
    aggregate_type = "unknown"
    if any(x in origins for x in ['patient', 'brain', 'autopsy', 'ex vivo', 'extracted']):
        aggregate_type = "ex_vivo"
    elif 'recombinant' in origins:
        aggregate_type = "recombinant"
    elif any(x in origins for x in ['synthetic', 'in vitro']):
        aggregate_type = "in_vitro"
    
    # Polymorph detection
    polymorph = ""
    polymorph_patterns = [
        (r'polymorph\s*(\w+)', lambda m: m.group(1)),
        (r'type\s*([IVX\d]+)', lambda m: f"type_{m.group(1)}"),
        (r'\b(PHF|SF|NPF|WPF)\b', lambda m: m.group(1)),
        (r'\b(rod|twister|narrow|wide)\b', lambda m: m.group(1)),
    ]
    for pattern, extractor in polymorph_patterns:
        match = re.search(pattern, origins, re.I)
        if match:
            polymorph = extractor(match)
            break
    
    # Method → structure_type
    method_lower = entry.method.lower()
    if 'cryo' in method_lower or 'em' in method_lower:
        structure_type = "fibril"
    elif 'nmr' in method_lower:
        structure_type = "fibril"
    elif 'crystal' in method_lower or 'x-ray' in method_lower:
        structure_type = "crystal"
    else:
        structure_type = "fibril"  # Atlas is fibril-focused
    
    return {
        'aggregate_type': aggregate_type,
        'polymorph_name': polymorph,
        'structure_type': structure_type,
        'secondary_structure_class': 'cross_beta',  # Atlas = cross-β fibrils
        'residues_ordered': entry.residues_ordered,
    }


# ===== RCSB PDB GraphQL API (Tier 2) =====

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

FIBRIL_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_primary_citation { 
      pdbx_database_id_doi 
      pdbx_database_id_pub_med 
    }
    rcsb_entry_container_identifiers { emdb_ids }
    em_helical_entity {
      angular_rotation_per_subunit
      axial_rise_per_subunit
      axial_symmetry
    }
    em_3d_reconstruction { 
      resolution 
      symmetry_type 
    }
    polymer_entities {
      rcsb_polymer_entity { pdbx_description }
      rcsb_entity_source_organism { scientific_name }
    }
    assemblies {
      rcsb_struct_symmetry { 
        symbol 
        kind 
        stoichiometry 
      }
    }
  }
}
"""


def fetch_pdb_metadata(pdb_ids: List[str], 
                       batch_size: int = 50,
                       delay: float = 0.5,
                       logger: Optional[logging.Logger] = None) -> Dict[str, PDBMetadata]:
    """
    Fetch metadata for PDB entries via RCSB GraphQL API.
    
    Args:
        pdb_ids: List of 4-character PDB IDs
        batch_size: Entries per request (max ~50 recommended)
        delay: Seconds between batches (be nice to API)
        
    Returns:
        Dict mapping PDB ID → PDBMetadata
    """
    if requests is None:
        raise ImportError("requests required: pip install requests")
    
    log = logger or logging.getLogger(__name__)
    results = {}
    
    # Normalize IDs
    pdb_ids = [pid.upper() for pid in pdb_ids if pid and len(pid) == 4]
    log.info(f"Fetching metadata for {len(pdb_ids)} PDB entries")
    
    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i:i + batch_size]
        log.debug(f"Batch {i//batch_size + 1}: {len(batch)} entries")
        
        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": FIBRIL_QUERY, "variables": {"ids": batch}},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            resp.raise_for_status()
            
            data = resp.json()
            if "errors" in data:
                log.warning(f"GraphQL errors: {data['errors']}")
            
            entries = data.get("data", {}).get("entries", [])
            for entry in entries:
                if entry:
                    meta = _parse_pdb_entry(entry)
                    results[meta.pdb_id] = meta
            
        except Exception as e:
            log.error(f"Batch failed: {e}")
        
        if i + batch_size < len(pdb_ids):
            time.sleep(delay)
    
    log.info(f"Retrieved metadata for {len(results)} entries")
    return results


def _parse_pdb_entry(entry: Dict) -> PDBMetadata:
    """Parse GraphQL response into PDBMetadata."""
    
    pdb_id = entry.get("rcsb_id", "")
    
    # Basic info
    struct = entry.get("struct", {}) or {}
    title = struct.get("title", "")
    
    exptl = entry.get("exptl", []) or []
    method = exptl[0].get("method", "") if exptl else ""
    
    info = entry.get("rcsb_entry_info", {}) or {}
    resolution = info.get("resolution_combined")
    
    # Citation
    citation = entry.get("rcsb_primary_citation", {}) or {}
    doi = citation.get("pdbx_database_id_doi", "") or ""
    pmid = citation.get("pdbx_database_id_pub_med", "") or ""
    
    # EMDB
    identifiers = entry.get("rcsb_entry_container_identifiers", {}) or {}
    emdb_ids = identifiers.get("emdb_ids", []) or []
    
    # Helical parameters
    helical = entry.get("em_helical_entity", []) or []
    helical_twist = None
    helical_rise = None
    axial_symmetry = ""
    
    if helical:
        h = helical[0]
        helical_twist = h.get("angular_rotation_per_subunit")
        helical_rise = h.get("axial_rise_per_subunit")
        axial_symmetry = h.get("axial_symmetry", "") or ""
    
    # Assembly symmetry
    assemblies = entry.get("assemblies", []) or []
    assembly_symmetry = ""
    stoichiometry = ""
    
    for asm in assemblies:
        sym = asm.get("rcsb_struct_symmetry", {}) or {}
        if sym.get("symbol"):
            assembly_symmetry = sym.get("symbol", "")
            stoichiometry = sym.get("stoichiometry", "") or ""
            break
    
    # Polymer entities
    polymers = entry.get("polymer_entities", []) or []
    protein_name = ""
    organism = ""
    
    if polymers:
        p = polymers[0]
        entity = p.get("rcsb_polymer_entity", {}) or {}
        protein_name = entity.get("pdbx_description", "") or ""
        
        sources = p.get("rcsb_entity_source_organism", []) or []
        if sources:
            organism = sources[0].get("scientific_name", "") or ""
    
    # Infer protofilament count from axial symmetry
    protofilament_count = _infer_protofilament_count(axial_symmetry, stoichiometry)
    
    # Infer twist handedness
    twist_handedness = "unknown"
    if helical_twist is not None:
        twist_handedness = "left" if helical_twist < 0 else "right"
    
    return PDBMetadata(
        pdb_id=pdb_id,
        title=title,
        resolution=resolution,
        method=method,
        helical_twist=helical_twist,
        helical_rise=helical_rise,
        axial_symmetry=axial_symmetry,
        assembly_symmetry=assembly_symmetry,
        stoichiometry=stoichiometry,
        emdb_ids=emdb_ids,
        doi=doi,
        pmid=str(pmid) if pmid else "",
        protein_name=protein_name,
        organism=organism,
        protofilament_count=protofilament_count,
        twist_handedness=twist_handedness,
    )


def _infer_protofilament_count(axial_symmetry: str, stoichiometry: str) -> int:
    """Infer protofilament count from symmetry info."""
    
    # From axial symmetry (C1=1, C2=2, C3=3, etc.)
    match = re.match(r'C(\d+)', axial_symmetry, re.I)
    if match:
        return int(match.group(1))
    
    # From stoichiometry ("Homo 2-mer" → 2)
    match = re.search(r'(\d+)-mer', stoichiometry, re.I)
    if match:
        n = int(match.group(1))
        # Heuristic: large n likely means monomers per fibril layer, not protofilaments
        if n <= 6:
            return n
    
    return 0


# ===== Enrichment Integration =====

def enrich_entry_from_pdb(entry, pdb_meta: PDBMetadata) -> None:
    """
    Apply PDB metadata to AmyloidEntry (in-place).
    
    Only updates fields if currently unknown/empty.
    """
    # Resolution
    if not entry.resolution and pdb_meta.resolution:
        entry.resolution = f"{pdb_meta.resolution:.2f}"
    
    # EMDB
    if not entry.emdb_id and pdb_meta.emdb_ids:
        entry.emdb_id = pdb_meta.emdb_ids[0]
    
    # References
    if not entry.doi and pdb_meta.doi:
        entry.doi = pdb_meta.doi
    if not entry.pmid and pdb_meta.pmid:
        entry.pmid = pdb_meta.pmid
    
    # Protein info
    if not entry.protein_name and pdb_meta.protein_name:
        entry.protein_name = pdb_meta.protein_name
    if not entry.organism and pdb_meta.organism:
        entry.organism = pdb_meta.organism
    
    # Structural classification
    if entry.protofilament_count == 0 and pdb_meta.protofilament_count > 0:
        entry.protofilament_count = pdb_meta.protofilament_count
    
    if entry.twist_handedness == "unknown" and pdb_meta.twist_handedness != "unknown":
        entry.twist_handedness = pdb_meta.twist_handedness
    
    if entry.protofilament_symmetry == "unknown" and pdb_meta.axial_symmetry:
        entry.protofilament_symmetry = pdb_meta.axial_symmetry
    
    # Method enrichment
    if pdb_meta.method and not entry.raw_method:
        entry.raw_method = pdb_meta.method


def enrich_entry_from_atlas(entry, atlas_entry: AtlasEntry) -> None:
    """
    Apply AmyloidAtlas metadata to AmyloidEntry (in-place).
    """
    parsed = parse_atlas_metadata(atlas_entry)
    
    if entry.aggregate_type == "unknown":
        entry.aggregate_type = parsed['aggregate_type']
    
    if not entry.polymorph_name:
        entry.polymorph_name = parsed['polymorph_name']
    
    if entry.structure_type == "unknown":
        entry.structure_type = parsed['structure_type']
    
    if entry.secondary_structure_class == "unknown":
        entry.secondary_structure_class = parsed['secondary_structure_class']


def enrich_entries(entries: List, 
                   use_atlas: bool = True,
                   use_pdb: bool = True,
                   atlas_cache: Optional[str] = None,
                   logger: Optional[logging.Logger] = None) -> Dict[str, int]:
    """
    Enrich AmyloidEntry list with PDB/Atlas metadata.
    
    Args:
        entries: List of AmyloidEntry objects
        use_atlas: Scrape AmyloidAtlas
        use_pdb: Query RCSB PDB API
        atlas_cache: Path to cache Atlas data
        
    Returns:
        Dict with enrichment stats
    """
    log = logger or logging.getLogger(__name__)
    stats = {'atlas_enriched': 0, 'pdb_enriched': 0, 'total': len(entries)}
    
    # Collect PDB IDs
    pdb_entries = [(i, e) for i, e in enumerate(entries) if e.pdb_id]
    pdb_ids = list(set(e.pdb_id.upper() for _, e in pdb_entries))
    log.info(f"Found {len(pdb_ids)} unique PDB IDs in {len(pdb_entries)} entries")
    
    # Tier 1: AmyloidAtlas
    atlas_map = {}
    if use_atlas and pdb_ids:
        try:
            atlas_entries = scrape_amyloid_atlas(atlas_cache, log)
            atlas_map = {e.pdb_id.upper(): e for e in atlas_entries}
            log.info(f"AmyloidAtlas: {len(atlas_map)} entries loaded")
        except Exception as e:
            log.warning(f"AmyloidAtlas scraping failed: {e}")
    
    # Tier 2: RCSB PDB
    pdb_map = {}
    if use_pdb and pdb_ids:
        try:
            pdb_map = fetch_pdb_metadata(pdb_ids, logger=log)
        except Exception as e:
            log.warning(f"PDB API query failed: {e}")
    
    # Apply enrichment
    for idx, entry in pdb_entries:
        pid = entry.pdb_id.upper()
        
        if pid in atlas_map:
            enrich_entry_from_atlas(entry, atlas_map[pid])
            stats['atlas_enriched'] += 1
        
        if pid in pdb_map:
            enrich_entry_from_pdb(entry, pdb_map[pid])
            stats['pdb_enriched'] += 1
    
    log.info(f"Enrichment complete: {stats['atlas_enriched']} from Atlas, "
             f"{stats['pdb_enriched']} from PDB API")
    
    return stats


# ===== CLI for standalone testing =====

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, 
                        format='[%(levelname)s] %(message)s')
    
    parser = argparse.ArgumentParser(description="Test PDB enrichment")
    parser.add_argument("--pdb", nargs="+", help="PDB IDs to query")
    parser.add_argument("--atlas", action="store_true", help="Scrape AmyloidAtlas")
    parser.add_argument("--cache", default="atlas_cache.json", help="Atlas cache file")
    
    args = parser.parse_args()
    
    if args.atlas:
        entries = scrape_amyloid_atlas(args.cache)
        print(f"\nAmyloidAtlas: {len(entries)} entries")
        for e in entries[:5]:
            print(f"  {e.pdb_id}: {e.protein} ({e.method})")
    
    if args.pdb:
        metadata = fetch_pdb_metadata(args.pdb)
        for pid, meta in metadata.items():
            print(f"\n{pid}:")
            print(f"  Title: {meta.title[:60]}...")
            print(f"  Resolution: {meta.resolution}")
            print(f"  Helical twist: {meta.helical_twist}")
            print(f"  Helical rise: {meta.helical_rise}")
            print(f"  Axial symmetry: {meta.axial_symmetry}")
            print(f"  Protofilaments: {meta.protofilament_count}")
            print(f"  Twist: {meta.twist_handedness}")
            print(f"  EMDB: {meta.emdb_ids}")
