"""
Configuration settings for Amyloid Aggregation Database Pipeline
"""

# =============================================================================
# SEARCH TERMS
# =============================================================================

# Primary aggregation terms (broad definition)
AGGREGATION_TERMS = [
    "amyloid",
    "amyloid-like", 
    "amyloidogenic",
    "protein aggregation",
    "protein aggregates",
    "fibril formation",
    "fibrillar aggregates",
    "cross-beta",
    "cross-beta structure",
    "beta-sheet aggregation",
    "alpha-helical amyloid",  # newer structural diversity
    "functional amyloid",
    "pathological amyloid",
    "self-assembly protein",
    "protein self-assembly",
]

# Targeted term groups for corpus expansion (step 2).
#
# AGGREGATION_TERMS alone saturates: of 2906 papers retrieved with it, only 1234
# yielded a protein, and the per-term retmax caps recall on the broad terms.
# These groups name classes the broad query under-covers -- functional amyloids
# outside human pathology, prion-like and low-complexity domains, and the
# translation-coupled RNP condensates this dissertation is about -- so each is
# searched on its own budget instead of competing for one.
TARGETED_TERM_GROUPS = {
    "bacterial_functional": [
        "curli", "CsgA", "Fap amyloid", "FapC", "TasA", "biofilm amyloid",
        "bacterial functional amyloid", "chaplin", "phenol-soluble modulin",
    ],
    "fungal_plant_functional": [
        "fungal amyloid", "hydrophobin", "HET-s", "yeast prion domain",
        "plant amyloid", "seed storage protein amyloid",
    ],
    "prion_like_domains": [
        "prion-like domain", "prion-like domains", "low-complexity domain",
        "low complexity aromatic-rich kinked segment", "Q/N-rich domain",
        "polyQ aggregation",
    ],
    "rnp_condensates": [
        "stress granule", "stress granules", "RNA-binding protein aggregation",
        "ribonucleoprotein granule", "P-body", "nucleolar condensate",
        "liquid-liquid phase separation amyloid", "liquid-to-solid transition",
    ],
    "structural_evidence": [
        "cryo-EM amyloid fibril", "amyloid fibril structure",
        "solid-state NMR fibril", "fibril polymorph", "amyloid fold",
    ],
    "material_assembly": [
        "amyloid hydrogel", "protein nanofibril", "amyloid nanomaterial",
    ],
}

# Corpus plan: the broad terms plus every targeted group, each sliced by
# publication-year bins so the per-query cap applies per bin rather than per
# term. 1998 is the floor: the first amyloid fibre-diffraction and ssNMR
# structural data date from then, and earlier records carry no structural tier.
CORPUS_PLAN = {
    "date_from": "1998/01/01",
    "bin_years": 2,
    "per_bin": 400,          # PMIDs per (group, year-bin)
    "max_total": 20000,      # hard cap on distinct PMIDs
}

# Exclusion terms: transmissible mammalian prion pathology ONLY.
#
# The standalone pipeline excluded "prion" and "prion-like" wholesale. For this
# dissertation that is wrong: translation-coupled RNP condensates are organised
# by prion-LIKE-domain RBPs (FUS, hnRNPA1, TDP-43, PRRC2A/B), which are the core
# of the sample. So the query-level exclusion is narrowed to transmissible PrP
# pathology; prion-like-domain records are kept downstream and tagged by
# amylodb.provenance.classify_prion (see adapter.py). Set
# exclude_transmissible_prion=True in reconcile() to drop the transmissible set.
EXCLUSION_TERMS = [
    "PrPSc",
    "PrPC",
    "scrapie",
    "transmissible spongiform",
    "Creutzfeldt-Jakob",
    "bovine spongiform",
    "chronic wasting disease",
]

# Experimental validation methods
EXPERIMENTAL_METHODS = {
    "biophysical": [
        "ThT", "Thioflavin T", "Thioflavin-T",
        "ThS", "Thioflavin S",
        "Congo red", "Congo Red",
        "circular dichroism", "CD spectroscopy",
        "FTIR", "infrared spectroscopy",
        "X-ray diffraction", "X-ray fiber diffraction",
        "solid-state NMR", "ssNMR",
    ],
    "microscopy": [
        "electron microscopy", "EM", "TEM", "transmission electron",
        "cryo-EM", "cryo-electron microscopy",
        "AFM", "atomic force microscopy",
        "super-resolution microscopy",
    ],
    "structural": [
        "crystal structure",
        "cryo-EM structure",
        "NMR structure",
        "fibril structure",
    ],
    "cellular": [
        "in vivo aggregation",
        "cell culture",
        "inclusion bodies",
        "puncta formation",
    ],
    "kinetic": [
        "aggregation kinetics",
        "seeding assay",
        "nucleation",
        "elongation rate",
        "lag phase",
    ]
}

# Assembly/disassembly indicators
ASSEMBLY_PATTERNS = {
    "reversible": [
        "reversible aggregation",
        "dynamic assembly",
        "regulated assembly",
        "disassembly",
        "disaggregation",
        "chaperone-mediated",
        "ATP-dependent",
    ],
    "irreversible": [
        "irreversible aggregation",
        "stable aggregates",
        "persistent aggregates",
        "insoluble aggregates",
        "resistant to",
    ],
    "autocatalytic": [
        "self-seeding",
        "autocatalytic",
        "template-assisted",
        "seeding-competent",
        "propagation",
    ],
    "cofactor_dependent": [
        "RNA-mediated",
        "chaperone-assisted",
        "cofactor",
        "metal-induced",
        "lipid-induced",
        "membrane-mediated",
    ]
}

# Functional vs Pathological context
FUNCTIONAL_CONTEXT = [
    "functional amyloid",
    "biofilm",
    "secretory granule",
    "hormone storage",
    "melanosome",
    "bacterial",
    "curli",
    "Pmel17",
    "CPEB",
]

PATHOLOGICAL_CONTEXT = [
    "neurodegenerative",
    "Alzheimer",
    "Parkinson",
    "Huntington",
    "ALS",
    "systemic amyloidosis",
    "AL amyloidosis",
    "AA amyloidosis",
    "ATTR",
    "pathological",
    "toxic",
    "disease-associated",
]

# =============================================================================
# API SETTINGS
# =============================================================================

PUBMED_API = {
    "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
    "db": "pubmed",
    "retmax": 1000,  # max results per query
    "retmode": "xml",
    "email": "your_email@example.com",  # required for NCBI API
    "api_key": None,  # optional, increases rate limit
    "rate_limit_delay": 0.34,  # seconds between requests (3/sec without key)
}

STRING_DB_API = {
    "base_url": "https://string-db.org/api",
    "version": "12.0",
    "output_format": "json",
    "species": 9606,  # human by default
    "required_score": 700,  # high confidence
    "network_type": "physical",  # physical subnetwork
}

# LLM Settings
LLM_CONFIG = {
    "provider": "anthropic",  # or "openai"
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 4096,
    "temperature": 0.1,  # low for structured extraction
}

# =============================================================================
# DATABASE SETTINGS
# =============================================================================

DATABASE = {
    "type": "sqlite",  # start simple, can migrate to PostgreSQL
    "path": "data/amyloid_aggregation.db",
}

# =============================================================================
# EXTRACTION SCHEMA
# =============================================================================

# What we want to extract from each paper
EXTRACTION_SCHEMA = {
    "protein": {
        "name": str,
        "uniprot_id": str,
        "gene_name": str,
        "organism": str,
    },
    "aggregation_properties": {
        "forms_aggregates": bool,
        "aggregate_structure": str,  # beta-sheet, alpha-helical, mixed, unknown
        "reversibility": str,  # reversible, irreversible, unknown
        "assembly_mechanism": str,  # autocatalytic, cofactor-dependent, unknown
    },
    "functional_classification": {
        "is_functional": bool,
        "is_pathological": bool,
        "biological_role": str,
    },
    "experimental_evidence": {
        "methods_used": list,
        "confidence_level": str,  # high, medium, low
    },
    "cofactors": {
        "assembly_helpers": list,  # proteins, RNAs, small molecules
        "disassembly_helpers": list,
    },
    "source": {
        "pmid": str,
        "doi": str,
        "title": str,
        "year": int,
    }
}
