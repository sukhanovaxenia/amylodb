"""Utility functions."""

from .sequence import (
    clean_value,
    clean_sequence,
    validate_sequence,
    extract_uniprot_id,
    parse_range,
    map_method_to_universal,
)

from .fetchers import SequenceCache, fetch_uniprot_info, fetch_sequences_and_info

from .pdb_enrichment import (
    scrape_amyloid_atlas,
    fetch_pdb_metadata,
    enrich_entries,
    PDBMetadata,
    AtlasEntry,
)

__all__ = [
    'clean_value',
    'clean_sequence', 
    'validate_sequence',
    'extract_uniprot_id',
    'parse_range',
    'map_method_to_universal',
    'SequenceCache',
    'fetch_uniprot_info',
    'fetch_sequences_and_info',
    # PDB enrichment
    'scrape_amyloid_atlas',
    'fetch_pdb_metadata',
    'enrich_entries',
    'PDBMetadata',
    'AtlasEntry',
]
