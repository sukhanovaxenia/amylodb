"""
Data models for amyloid entries and computed features.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional


@dataclass
class PhysicochemicalFeatures:
    """Computed physicochemical features for a sequence."""
    length: int = 0
    molecular_weight: float = 0.0
    
    hydrophobicity_mean: float = 0.0
    hydrophobicity_std: float = 0.0
    hydrophobicity_max: float = 0.0
    hydrophobicity_min: float = 0.0
    
    net_charge: float = 0.0
    positive_residues: int = 0
    negative_residues: int = 0
    charge_density: float = 0.0
    
    beta_propensity_mean: float = 0.0
    beta_propensity_max: float = 0.0
    
    aggregation_propensity_mean: float = 0.0
    aggregation_propensity_max: float = 0.0
    
    aromatic_fraction: float = 0.0
    aliphatic_fraction: float = 0.0
    polar_fraction: float = 0.0
    charged_fraction: float = 0.0
    
    has_polyq: bool = False
    has_glycine_rich: bool = False
    has_proline: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SequenceComposition:
    """Amino acid composition features."""
    aa_freq: Dict[str, float] = field(default_factory=dict)
    dipeptide_freq: Dict[str, float] = field(default_factory=dict)
    tiny_fraction: float = 0.0
    small_fraction: float = 0.0
    large_fraction: float = 0.0
    
    def to_dict(self) -> Dict:
        from .config import STANDARD_AA
        result = {}
        for aa in STANDARD_AA:
            result[f'aa_{aa}'] = self.aa_freq.get(aa, 0.0)
        for dp, freq in self.dipeptide_freq.items():
            result[f'dp_{dp}'] = freq
        result['tiny_fraction'] = self.tiny_fraction
        result['small_fraction'] = self.small_fraction
        result['large_fraction'] = self.large_fraction
        return result


@dataclass
class AmyloidEntry:
    """Unified amyloid entry with all metadata fields."""
    
    # Core identifiers
    record_id: str = ""
    source_db: str = ""
    
    # Protein info
    protein_name: str = ""
    uniprot_id: str = ""
    organism: str = ""
    protein_family: str = ""
    
    # Region coordinates
    region_start: Optional[int] = None
    region_end: Optional[int] = None
    
    # Sequence data
    sequence: str = ""
    full_protein_sequence: str = ""
    
    # Classification
    is_amyloid: bool = True
    experimental_label: str = ""
    category: str = ""
    
    # ===== Multi-dimensional Structural Classification =====
    # Level 1: Secondary structure class
    secondary_structure_class: str = "unknown"  # cross_beta, cross_alpha, unknown
    
    # Level 2: Interface type
    interface_type: str = "unknown"  # steric_zipper, LARKS, cross_alpha_sheet, unknown
    
    # Level 3: Strand/helix arrangement
    strand_arrangement: str = "unknown"  # parallel_in_register, antiparallel, out_of_register, mixed, unknown
    
    # Level 4: Fold topology
    fold_topology: str = "unknown"  # beta_arcade, greek_key, beta_solenoid, beta_helix, beta_sandwich, superpleated, other, unknown
    
    # Level 5: Fold shape (informal cryo-EM descriptors)
    fold_shape: str = "unknown"  # J_shaped, S_shaped, C_shaped, LS_shaped, kidney, elongated, compact, other, unknown
    
    # Level 6: Steric zipper class (Sawaya 2007: 1-8, extended: 1-15)
    zipper_class: str = ""  # "1", "2", ..., "8", or extended "1"-"15", or ""
    
    # Level 7: Protofilament symmetry
    protofilament_symmetry: str = "unknown"  # C1, C2, C3, screw_21, unknown
    
    # Level 8: Protofilament count
    protofilament_count: int = 0  # 0=unknown, 1, 2, 3, 4+
    
    # Level 9: Twist handedness
    twist_handedness: str = "unknown"  # left, right, unknown
    
    # Level 10: Source/origin type
    aggregate_type: str = "unknown"  # ex_vivo, in_vitro, synthetic, recombinant, unknown
    
    # Level 11: Polymorph designation (free text)
    polymorph_name: str = ""  # e.g., "PHF", "SF", "type_1A", "rod", "twister"
    
    # Pathogenicity classification
    pathogenicity: str = "unknown"  # pathogenic, functional, non_pathogenic, unknown
    
    # Legacy field (kept for backward compatibility, derived from above)
    structure_type: str = "unknown"  # fibril, oligomer, crystal, aggregate, monomer, unknown
    
    # Structural data
    pdb_id: str = ""
    emdb_id: str = ""
    
    # Experimental evidence
    raw_method: str = ""
    method_universal: str = ""
    evidence_type: str = ""
    evidence_weight: float = 0.0
    confidence: int = 0
    resolution: str = ""
    
    # Disease/tissue context
    disease: str = ""
    tissue: str = ""
    mutation: str = ""
    
    # References
    doi: str = ""
    pmid: str = ""
    reference: str = ""
    
    # Additional
    notes: str = ""
    
    # Computed features
    _physicochemical: Optional[PhysicochemicalFeatures] = field(default=None, repr=False)
    _composition: Optional[SequenceComposition] = field(default=None, repr=False)
    
    def get_dedup_key(self) -> Tuple:
        """Get biologically meaningful deduplication key."""
        if self.uniprot_id and (self.region_start is not None or self.region_end is not None):
            return (self.sequence, self.uniprot_id, self.region_start, self.region_end)
        elif self.protein_name and (self.region_start is not None or self.region_end is not None):
            return (self.sequence, self.protein_name, self.region_start, self.region_end)
        return (self.sequence, self.source_db)
    
    def is_valid(self) -> bool:
        # A record is valid if it carries a sequence or a structure, OR if it
        # locates a region on a named protein (UniProt + coordinates) even
        # without the residue string -- the latter recovers kinetic sources
        # such as Amylobase, whose sequences are fetched downstream.
        has_seq_or_struct = bool(self.sequence or self.pdb_id)
        has_located_region = bool(self.uniprot_id) and (
            self.region_start is not None or self.region_end is not None)
        return (has_seq_or_struct or has_located_region) and bool(self.source_db)
    
    def compute_features(self):
        """Compute physicochemical and composition features."""
        if self.sequence:
            from .features import compute_physicochemical_features, compute_sequence_composition
            self._physicochemical = compute_physicochemical_features(self.sequence)
            self._composition = compute_sequence_composition(self.sequence)
    
    def to_dict(self, include_features: bool = False) -> Dict:
        result = {
            'record_id': self.record_id,
            'source_db': self.source_db,
            'protein_name': self.protein_name,
            'uniprot_id': self.uniprot_id,
            'organism': self.organism,
            'protein_family': self.protein_family,
            'region_start': self.region_start,
            'region_end': self.region_end,
            'sequence': self.sequence,
            'is_amyloid': self.is_amyloid,
            'experimental_label': self.experimental_label,
            'category': self.category,
            # Multi-dimensional structural classification
            'secondary_structure_class': self.secondary_structure_class,
            'interface_type': self.interface_type,
            'strand_arrangement': self.strand_arrangement,
            'fold_topology': self.fold_topology,
            'fold_shape': self.fold_shape,
            'zipper_class': self.zipper_class,
            'protofilament_symmetry': self.protofilament_symmetry,
            'protofilament_count': self.protofilament_count,
            'twist_handedness': self.twist_handedness,
            'aggregate_type': self.aggregate_type,
            'polymorph_name': self.polymorph_name,
            'pathogenicity': self.pathogenicity,
            'structure_type': self.structure_type,
            'pdb_id': self.pdb_id,
            'emdb_id': self.emdb_id,
            'raw_method': self.raw_method,
            'method_universal': self.method_universal,
            'evidence_type': self.evidence_type,
            'evidence_weight': self.evidence_weight,
            'confidence': self.confidence,
            'resolution': self.resolution,
            'disease': self.disease,
            'tissue': self.tissue,
            'mutation': self.mutation,
            'doi': self.doi,
            'pmid': self.pmid,
            'reference': self.reference,
            'notes': self.notes,
        }
        
        if include_features and self._physicochemical:
            for k, v in self._physicochemical.to_dict().items():
                result[f'phys_{k}'] = v
        
        if include_features and self._composition:
            for k, v in self._composition.to_dict().items():
                result[f'comp_{k}'] = v
        
        return result
