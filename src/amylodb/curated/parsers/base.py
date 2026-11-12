"""Base parser class with comprehensive structural classification inference."""

import logging
import re
from typing import List, Dict, Tuple

from ..models import AmyloidEntry
from ..config import PROTEIN_FAMILIES


# ===== Structural Classification Constants =====

# Patterns for secondary structure class inference
CROSS_ALPHA_PROTEINS = ['psma', 'psmα', 'psmb', 'psmβ', 'psm alpha', 'psm beta', 
                        'phenol-soluble modulin', 'uperin', 'aurein']

# Fold topology patterns (from cryo-EM/ssNMR literature)
FOLD_TOPOLOGY_PATTERNS = {
    'greek_key': ['greek key', 'greek-key', 'greekkey'],
    'beta_arcade': ['beta arcade', 'β-arcade', 'beta-arcade', 'b-arcade'],
    'beta_solenoid': ['solenoid', 'beta-solenoid', 'β-solenoid', 'beta solenoid', 'left-handed β-helix'],
    'beta_helix': ['beta helix', 'β-helix', 'beta-helix', 'b-helix'],
    'beta_sandwich': ['sandwich', 'beta-sandwich', 'β-sandwich'],
    'superpleated': ['superpleated', 'super-pleated', 'serpentine'],
}

# Fold shape patterns (informal cryo-EM descriptors)
FOLD_SHAPE_PATTERNS = {
    'J_shaped': ['j-shaped', 'j shaped', 'j-fold'],
    'S_shaped': ['s-shaped', 's shaped', 's-fold'],
    'C_shaped': ['c-shaped', 'c shaped', 'c-fold', 'crescent'],
    'LS_shaped': ['ls-shaped', 'ls shaped', 'l/s'],
    'kidney': ['kidney', 'kidney-shaped'],
    'elongated': ['elongated', 'extended'],
    'compact': ['compact', 'globular-like'],
}

# Strand arrangement patterns
STRAND_ARRANGEMENT_PATTERNS = {
    'parallel_in_register': ['parallel in-register', 'parallel, in-register', 'in-register parallel',
                              'parallel β-sheet', 'parallel beta'],
    'antiparallel': ['antiparallel', 'anti-parallel'],
    'out_of_register': ['out-of-register', 'out of register', 'staggered'],
    'mixed': ['mixed parallel', 'mixed arrangement', 'parallel and antiparallel'],
}

# Polymorph name patterns (disease-specific)
POLYMORPH_PATTERNS = {
    'tau': {
        'PHF': ['phf', 'paired helical'],
        'SF': ['sf', 'straight filament'],
        'NPF': ['npf', 'narrow pick'],
        'WPF': ['wpf', 'wide pick'],
    },
    'alpha-synuclein': {
        'rod': ['rod', 'rod-like'],
        'twister': ['twister'],
        'type_1': ['type 1', 'type1', 'polymorph 1'],
        'type_2': ['type 2', 'type2', 'polymorph 2'],
    },
    'abeta': {
        'type_1': ['type i', 'type 1', 'type1'],
        'type_2': ['type ii', 'type 2', 'type2'],
    }
}

# Protofilament symmetry patterns
SYMMETRY_PATTERNS = {
    'C1': ['c1', 'asymmetric', 'single protofilament'],
    'C2': ['c2', '2-fold', 'twofold', 'c2 symmetry', 'pseudo-2'],
    'C3': ['c3', '3-fold', 'threefold'],
    'screw_21': ['21 screw', 'screw axis', '2₁'],
}

# Interface type patterns
INTERFACE_PATTERNS = {
    'steric_zipper': ['steric zipper', 'zipper', 'dry interface'],
    'LARKS': ['larks', 'low-complexity', 'reversible', 'kinked'],
    'cross_alpha_sheet': ['cross-α', 'cross-alpha', 'helical sheet'],
}


class BaseParser:
    """Base class for database parsers with comprehensive structural inference."""
    
    def __init__(self, db_name: str, logger=None):
        self.db_name = db_name
        self.logger = logger or logging.getLogger(__name__)
        self.entries: List[AmyloidEntry] = []
        self.errors: List[str] = []
    
    def log_result(self):
        valid = sum(1 for e in self.entries if e.is_valid())
        self.logger.info(f"{self.db_name}: {len(self.entries)} entries ({valid} valid), {len(self.errors)} errors")
    
    def parse(self, filepath: str, **kwargs) -> List[AmyloidEntry]:
        raise NotImplementedError
    
    # ===== Multi-dimensional Structural Classification =====
    
    def _infer_secondary_structure_class(self, protein_name: str, method: str, notes: str = "") -> str:
        """Level 1: cross_beta vs cross_alpha."""
        text = f"{protein_name} {method} {notes}".lower()
        
        # Cross-alpha is rare - only PSMs, uperin, etc.
        if any(p in text for p in CROSS_ALPHA_PROTEINS):
            if 'cross-α' in text or 'cross-alpha' in text or 'helical sheet' in text:
                return 'cross_alpha'
        
        # Default for amyloids is cross-beta
        if any(kw in text for kw in ['amyloid', 'fibril', 'cross-β', 'cross-beta', 'β-sheet', 'beta-sheet']):
            return 'cross_beta'
        
        return 'unknown'
    
    def _infer_interface_type(self, method: str, notes: str = "", sequence: str = "") -> str:
        """Level 2: steric_zipper vs LARKS vs cross_alpha_sheet."""
        text = f"{method} {notes}".lower()
        
        for itype, patterns in INTERFACE_PATTERNS.items():
            if any(p in text for p in patterns):
                return itype
        
        # Heuristics: short peptides (6-10 aa) in crystal structures → likely steric zipper
        if sequence and 6 <= len(sequence) <= 10 and 'crystal' in text:
            return 'steric_zipper'
        
        return 'unknown'
    
    def _infer_strand_arrangement(self, method: str, notes: str = "") -> str:
        """Level 3: parallel_in_register, antiparallel, out_of_register, mixed."""
        text = f"{method} {notes}".lower()
        
        for arrangement, patterns in STRAND_ARRANGEMENT_PATTERNS.items():
            if any(p in text for p in patterns):
                return arrangement
        
        # Heuristic: most mature fibrils are parallel in-register
        if 'fibril' in text and ('cryo' in text or 'ssnmr' in text or 'solid-state' in text):
            return 'parallel_in_register'
        
        return 'unknown'
    
    def _infer_fold_topology(self, method: str, notes: str = "", protein_name: str = "") -> str:
        """Level 4: beta_arcade, greek_key, beta_solenoid, etc."""
        text = f"{method} {notes} {protein_name}".lower()
        
        for topology, patterns in FOLD_TOPOLOGY_PATTERNS.items():
            if any(p in text for p in patterns):
                return topology
        
        # Known protein-specific defaults
        if 'het-s' in text or 'hets' in text:
            return 'beta_solenoid'
        
        return 'unknown'
    
    def _infer_fold_shape(self, method: str, notes: str = "", protein_name: str = "") -> str:
        """Level 5: J_shaped, S_shaped, C_shaped, etc."""
        text = f"{method} {notes} {protein_name}".lower()
        
        for shape, patterns in FOLD_SHAPE_PATTERNS.items():
            if any(p in text for p in patterns):
                return shape
        
        # Known disease-specific defaults
        if 'pick' in text:
            return 'J_shaped'
        if 'alzheimer' in text and 'tau' in text:
            return 'C_shaped'
        
        return 'unknown'
    
    def _infer_zipper_class(self, method: str, notes: str = "") -> str:
        """Level 6: Steric zipper class 1-8 (or extended 1-15)."""
        text = f"{method} {notes}".lower()
        
        # Look for explicit class mentions
        match = re.search(r'class\s*(\d+)', text)
        if match:
            cls = match.group(1)
            if cls in [str(i) for i in range(1, 16)]:
                return cls
        
        match = re.search(r'zipper\s*(\d+)', text)
        if match:
            cls = match.group(1)
            if cls in [str(i) for i in range(1, 16)]:
                return cls
        
        return ""
    
    def _infer_protofilament_symmetry(self, method: str, notes: str = "") -> str:
        """Level 7: C1, C2, C3, screw_21."""
        text = f"{method} {notes}".lower()
        
        for sym, patterns in SYMMETRY_PATTERNS.items():
            if any(p in text for p in patterns):
                return sym
        
        return 'unknown'
    
    def _infer_protofilament_count(self, method: str, notes: str = "") -> int:
        """Level 8: Number of protofilaments (0=unknown)."""
        text = f"{method} {notes}".lower()
        
        # Look for explicit mentions
        patterns = [
            (r'(\d+)\s*protofilament', 1),
            (r'(\d+)-protofilament', 1),
            (r'single\s*protofilament', 1),
            (r'double\s*protofilament', 2),
            (r'two\s*protofilament', 2),
            (r'three\s*protofilament', 3),
        ]
        
        for pattern, count in patterns:
            if isinstance(count, int):
                if re.search(pattern, text):
                    return count
            else:
                match = re.search(pattern, text)
                if match:
                    return int(match.group(1))
        
        return 0
    
    def _infer_twist_handedness(self, method: str, notes: str = "") -> str:
        """Level 9: left or right twist."""
        text = f"{method} {notes}".lower()
        
        if 'left-handed' in text or 'left handed' in text or 'left twist' in text:
            return 'left'
        if 'right-handed' in text or 'right handed' in text or 'right twist' in text:
            return 'right'
        
        return 'unknown'
    
    def _infer_polymorph_name(self, protein_name: str, notes: str = "") -> str:
        """Level 11: Polymorph designation (free text)."""
        text = f"{notes}".lower()
        protein_lower = protein_name.lower()
        
        # Check protein-specific polymorphs
        for protein_key, polymorphs in POLYMORPH_PATTERNS.items():
            if protein_key in protein_lower or protein_key.replace('-', '') in protein_lower:
                for poly_name, patterns in polymorphs.items():
                    if any(p in text for p in patterns):
                        return poly_name
        
        # Generic polymorph extraction
        match = re.search(r'polymorph\s*(\w+)', text)
        if match:
            return match.group(1)
        
        match = re.search(r'type\s*(\w+)', text)
        if match:
            return f"type_{match.group(1)}"
        
        return ""
    
    # ===== Legacy/Simple Inference Methods =====
    
    def _infer_structure_type(self, method: str, notes: str = "") -> str:
        """Legacy: fibril, oligomer, crystal, aggregate, monomer."""
        text = f"{method} {notes}".lower()
        if 'fibril' in text or 'fiber' in text:
            return 'fibril'
        if 'oligomer' in text:
            return 'oligomer'
        if 'crystal' in text:
            return 'crystal'
        if 'aggregate' in text:
            return 'aggregate'
        if 'monomer' in text:
            return 'monomer'
        return 'unknown'
    
    def _infer_aggregate_type(self, notes: str, patient: bool = False) -> str:
        """Level 10: ex_vivo, in_vitro, synthetic, recombinant."""
        if patient:
            return 'ex_vivo'
        text = notes.lower()
        if 'patient' in text or 'ex vivo' in text or 'brain extract' in text or 'autopsy' in text:
            return 'ex_vivo'
        if 'recombinant' in text:
            return 'recombinant'
        if 'synthetic' in text:
            return 'synthetic'
        if 'in vitro' in text:
            return 'in_vitro'
        return 'unknown'
    
    def _infer_pathogenicity(self, disease: str, category: str, notes: str = "") -> str:
        """Pathogenic, functional, or non_pathogenic."""
        text = f"{disease} {category} {notes}".lower()
        pathogenic_diseases = ['alzheimer', 'parkinson', 'huntington', 'als', 'prion', 
                               'diabetes', 'amyloidosis', 'creutzfeldt', 'pick', 'psp',
                               'cbd', 'msa', 'ftd', 'systemic', 'cardiac', 'renal']
        if any(d in text for d in pathogenic_diseases):
            return 'pathogenic'
        if 'functional' in text or 'biofilm' in text or 'curli' in text or 'secretory' in text:
            return 'functional'
        if 'non-pathogenic' in text or 'non pathogenic' in text or 'benign' in text:
            return 'non_pathogenic'
        if disease:
            return 'pathogenic'
        return 'unknown'
    
    def _infer_protein_family(self, protein_name: str, disease: str = "") -> str:
        """Protein family classification."""
        text = f"{protein_name} {disease}".lower()
        for family, keywords in PROTEIN_FAMILIES.items():
            if any(kw in text for kw in keywords):
                return family
        return ""
    
    # ===== Batch Inference Helper =====
    
    def _infer_all_structural_fields(self, entry: AmyloidEntry) -> AmyloidEntry:
        """Apply all structural inference to an entry."""
        method = entry.raw_method or ""
        notes = entry.notes or ""
        protein = entry.protein_name or ""
        disease = entry.disease or ""
        sequence = entry.sequence or ""
        
        entry.secondary_structure_class = self._infer_secondary_structure_class(protein, method, notes)
        entry.interface_type = self._infer_interface_type(method, notes, sequence)
        entry.strand_arrangement = self._infer_strand_arrangement(method, notes)
        entry.fold_topology = self._infer_fold_topology(method, notes, protein)
        entry.fold_shape = self._infer_fold_shape(method, notes, protein)
        entry.zipper_class = self._infer_zipper_class(method, notes)
        entry.protofilament_symmetry = self._infer_protofilament_symmetry(method, notes)
        entry.protofilament_count = self._infer_protofilament_count(method, notes)
        entry.twist_handedness = self._infer_twist_handedness(method, notes)
        entry.polymorph_name = self._infer_polymorph_name(protein, notes)
        entry.structure_type = self._infer_structure_type(method, notes)
        entry.aggregate_type = self._infer_aggregate_type(notes)
        entry.pathogenicity = self._infer_pathogenicity(disease, entry.category, notes)
        entry.protein_family = self._infer_protein_family(protein, disease)
        
        return entry
