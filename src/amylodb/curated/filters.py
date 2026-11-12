"""Filtering system for amyloid entries."""

from typing import List, Union, Callable

from .models import AmyloidEntry


class AmyloidFilter:
    """
    Chainable filter for amyloid entries.
    
    Usage:
        f = AmyloidFilter()
        f.evidence_type(['structural', 'kinetic'])
        f.organism('Homo sapiens')
        f.min_confidence(50)
        filtered = f.apply(entries)
    """
    
    def __init__(self):
        self._filters: List[Callable[[AmyloidEntry], bool]] = []
        self._description: List[str] = []
    
    def evidence_type(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(types, str):
            types = [types]
        types_set = set(types)
        self._filters.append(lambda e: e.evidence_type in types_set)
        self._description.append(f"evidence_type in {types}")
        return self
    
    def exclude_evidence_type(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(types, str):
            types = [types]
        types_set = set(types)
        self._filters.append(lambda e: e.evidence_type not in types_set)
        self._description.append(f"evidence_type not in {types}")
        return self
    
    def organism(self, organism: str, exact: bool = False) -> 'AmyloidFilter':
        if exact:
            self._filters.append(lambda e: e.organism.lower() == organism.lower())
        else:
            self._filters.append(lambda e: organism.lower() in e.organism.lower())
        self._description.append(f"organism {'==' if exact else 'contains'} '{organism}'")
        return self
    
    def min_confidence(self, threshold: int) -> 'AmyloidFilter':
        self._filters.append(lambda e: e.confidence >= threshold)
        self._description.append(f"confidence >= {threshold}")
        return self
    
    def min_evidence_weight(self, threshold: float) -> 'AmyloidFilter':
        self._filters.append(lambda e: e.evidence_weight >= threshold)
        self._description.append(f"evidence_weight >= {threshold}")
        return self
    
    def is_amyloid(self, value: bool) -> 'AmyloidFilter':
        self._filters.append(lambda e: e.is_amyloid == value)
        self._description.append(f"is_amyloid == {value}")
        return self
    
    def structure_type(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(types, str):
            types = [types]
        types_set = set(t.lower() for t in types)
        self._filters.append(lambda e: e.structure_type.lower() in types_set)
        self._description.append(f"structure_type in {types}")
        return self
    
    def secondary_structure_class(self, classes: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by secondary structure: cross_beta, cross_alpha."""
        if isinstance(classes, str):
            classes = [classes]
        classes_set = set(c.lower() for c in classes)
        self._filters.append(lambda e: e.secondary_structure_class.lower() in classes_set)
        self._description.append(f"secondary_structure_class in {classes}")
        return self
    
    def interface_type(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by interface type: steric_zipper, LARKS, cross_alpha_sheet."""
        if isinstance(types, str):
            types = [types]
        types_set = set(t.lower() for t in types)
        self._filters.append(lambda e: e.interface_type.lower() in types_set)
        self._description.append(f"interface_type in {types}")
        return self
    
    def strand_arrangement(self, arrangements: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by strand arrangement: parallel_in_register, antiparallel, etc."""
        if isinstance(arrangements, str):
            arrangements = [arrangements]
        arr_set = set(a.lower() for a in arrangements)
        self._filters.append(lambda e: e.strand_arrangement.lower() in arr_set)
        self._description.append(f"strand_arrangement in {arrangements}")
        return self
    
    def fold_topology(self, topologies: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by fold topology: greek_key, beta_arcade, beta_solenoid, etc."""
        if isinstance(topologies, str):
            topologies = [topologies]
        topo_set = set(t.lower() for t in topologies)
        self._filters.append(lambda e: e.fold_topology.lower() in topo_set)
        self._description.append(f"fold_topology in {topologies}")
        return self
    
    def fold_shape(self, shapes: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by fold shape: J_shaped, S_shaped, C_shaped, etc."""
        if isinstance(shapes, str):
            shapes = [shapes]
        shapes_set = set(s.lower() for s in shapes)
        self._filters.append(lambda e: e.fold_shape.lower() in shapes_set)
        self._description.append(f"fold_shape in {shapes}")
        return self
    
    def protofilament_symmetry(self, symmetries: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by protofilament symmetry: C1, C2, C3, screw_21."""
        if isinstance(symmetries, str):
            symmetries = [symmetries]
        sym_set = set(s.lower() for s in symmetries)
        self._filters.append(lambda e: e.protofilament_symmetry.lower() in sym_set)
        self._description.append(f"protofilament_symmetry in {symmetries}")
        return self
    
    def protofilament_count(self, counts: Union[int, List[int]]) -> 'AmyloidFilter':
        """Filter by protofilament count."""
        if isinstance(counts, int):
            counts = [counts]
        counts_set = set(counts)
        self._filters.append(lambda e: e.protofilament_count in counts_set)
        self._description.append(f"protofilament_count in {counts}")
        return self
    
    def twist_handedness(self, handedness: Union[str, List[str]]) -> 'AmyloidFilter':
        """Filter by twist handedness: left, right."""
        if isinstance(handedness, str):
            handedness = [handedness]
        hand_set = set(h.lower() for h in handedness)
        self._filters.append(lambda e: e.twist_handedness.lower() in hand_set)
        self._description.append(f"twist_handedness in {handedness}")
        return self
    
    def is_cross_beta(self) -> 'AmyloidFilter':
        """Filter for canonical cross-β amyloids."""
        self._filters.append(lambda e: e.secondary_structure_class.lower() == 'cross_beta')
        self._description.append("is cross-β")
        return self
    
    def is_cross_alpha(self) -> 'AmyloidFilter':
        """Filter for cross-α amyloid-like structures."""
        self._filters.append(lambda e: e.secondary_structure_class.lower() == 'cross_alpha')
        self._description.append("is cross-α")
        return self
    
    def is_ex_vivo(self) -> 'AmyloidFilter':
        """Filter for patient-derived/ex vivo structures."""
        self._filters.append(lambda e: e.aggregate_type.lower() in ('ex_vivo', 'patient_derived'))
        self._description.append("is ex vivo")
        return self
    
    def aggregate_type(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(types, str):
            types = [types]
        types_set = set(t.lower() for t in types)
        self._filters.append(lambda e: e.aggregate_type.lower() in types_set)
        self._description.append(f"aggregate_type in {types}")
        return self
    
    def pathogenicity(self, types: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(types, str):
            types = [types]
        types_set = set(t.lower() for t in types)
        self._filters.append(lambda e: e.pathogenicity.lower() in types_set)
        self._description.append(f"pathogenicity in {types}")
        return self
    
    def source_db(self, dbs: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(dbs, str):
            dbs = [dbs]
        dbs_set = set(db.lower() for db in dbs)
        self._filters.append(lambda e: e.source_db.lower() in dbs_set)
        self._description.append(f"source_db in {dbs}")
        return self
    
    def has_sequence(self, min_length: int = 1) -> 'AmyloidFilter':
        self._filters.append(lambda e: len(e.sequence) >= min_length)
        self._description.append(f"len(sequence) >= {min_length}")
        return self
    
    def has_coordinates(self) -> 'AmyloidFilter':
        self._filters.append(lambda e: e.region_start is not None and e.region_end is not None)
        self._description.append("has coordinates")
        return self
    
    def has_pdb(self) -> 'AmyloidFilter':
        self._filters.append(lambda e: bool(e.pdb_id))
        self._description.append("has PDB ID")
        return self
    
    def has_uniprot(self) -> 'AmyloidFilter':
        self._filters.append(lambda e: bool(e.uniprot_id))
        self._description.append("has UniProt ID")
        return self
    
    def protein_family(self, families: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(families, str):
            families = [families]
        families_lower = [f.lower() for f in families]
        self._filters.append(lambda e: any(f in e.protein_family.lower() for f in families_lower))
        self._description.append(f"protein_family contains {families}")
        return self
    
    def disease(self, diseases: Union[str, List[str]]) -> 'AmyloidFilter':
        if isinstance(diseases, str):
            diseases = [diseases]
        diseases_lower = [d.lower() for d in diseases]
        self._filters.append(lambda e: any(d in e.disease.lower() for d in diseases_lower))
        self._description.append(f"disease contains {diseases}")
        return self
    
    def custom(self, func: Callable[[AmyloidEntry], bool], description: str = "custom") -> 'AmyloidFilter':
        self._filters.append(func)
        self._description.append(description)
        return self
    
    def apply(self, entries: List[AmyloidEntry]) -> List[AmyloidEntry]:
        result = entries
        for f in self._filters:
            result = [e for e in result if f(e)]
        return result
    
    def __str__(self) -> str:
        return " AND ".join(self._description) if self._description else "no filters"
    
    def count(self, entries: List[AmyloidEntry]) -> int:
        return len(self.apply(entries))


def filter_entries(entries: List[AmyloidEntry], **kwargs) -> List[AmyloidEntry]:
    """Quick filter function."""
    f = AmyloidFilter()
    
    if 'evidence_type' in kwargs:
        f.evidence_type(kwargs['evidence_type'])
    if kwargs.get('exclude_computational'):
        f.exclude_evidence_type('computational')
    if 'organism' in kwargs:
        f.organism(kwargs['organism'])
    if 'min_confidence' in kwargs:
        f.min_confidence(kwargs['min_confidence'])
    if 'is_amyloid' in kwargs:
        f.is_amyloid(kwargs['is_amyloid'])
    if 'structure_type' in kwargs:
        f.structure_type(kwargs['structure_type'])
    if 'aggregate_type' in kwargs:
        f.aggregate_type(kwargs['aggregate_type'])
    if 'pathogenicity' in kwargs:
        f.pathogenicity(kwargs['pathogenicity'])
    if 'source_db' in kwargs:
        f.source_db(kwargs['source_db'])
    if 'min_seq_length' in kwargs:
        f.has_sequence(kwargs['min_seq_length'])
    if kwargs.get('has_pdb'):
        f.has_pdb()
    if kwargs.get('has_uniprot'):
        f.has_uniprot()
    # New structural classification filters
    if 'secondary_structure_class' in kwargs:
        f.secondary_structure_class(kwargs['secondary_structure_class'])
    if 'interface_type' in kwargs:
        f.interface_type(kwargs['interface_type'])
    if 'strand_arrangement' in kwargs:
        f.strand_arrangement(kwargs['strand_arrangement'])
    if 'fold_topology' in kwargs:
        f.fold_topology(kwargs['fold_topology'])
    if 'fold_shape' in kwargs:
        f.fold_shape(kwargs['fold_shape'])
    if 'protofilament_symmetry' in kwargs:
        f.protofilament_symmetry(kwargs['protofilament_symmetry'])
    if 'protofilament_count' in kwargs:
        f.protofilament_count(kwargs['protofilament_count'])
    if 'twist_handedness' in kwargs:
        f.twist_handedness(kwargs['twist_handedness'])
    if kwargs.get('is_cross_beta'):
        f.is_cross_beta()
    if kwargs.get('is_cross_alpha'):
        f.is_cross_alpha()
    if kwargs.get('is_ex_vivo'):
        f.is_ex_vivo()
    
    return f.apply(entries)
