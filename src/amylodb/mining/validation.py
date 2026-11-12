"""Validation helpers for mined extractions.

These guard the mining output against three failure modes measured on a real
mining run (444 abstracts, 690 stored proteins):

* **Template echoes** -- small LLMs return the format string itself, e.g.
  ``"beta-sheet|alpha-helical|mixed|unknown"``. `clean_enum` maps anything that
  is not exactly one allowed value to ``"unknown"``.
* **Spurious protein-paper links** -- 53.5 % of links for canonical amyloid
  proteins pointed to abstracts that never mention the protein (a generic word
  such as "amyloid" had been normalised to amyloid-beta). `genuine_mention`
  requires a boundary-delimited mention of the name or one of its synonyms.
* **Noise names** -- NER returned methods ("afm"), diseases ("ad"), reagents
  ("abts", "4-methyl coumarin") and generic phrases ("aggregating protein") as
  proteins. `looks_like_noise` rejects them.

Consumes: strings (field values, names, abstracts).
Emits: cleaned values / booleans.
Invariant: pure functions, no network, no model access.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import Dict, Iterable, List, Optional


def clean_name(raw: Optional[str]) -> str:
    """Normalise a candidate protein name.

    Strips surrounding punctuation, dangling hyphens and whitespace, a trailing
    possessive ('s), and an unbalanced parenthetical tail; collapses internal
    whitespace. NER backends return names glued to punctuation ("amyloid,",
    "aβ.", "apoa-i..", "protein's", "insulin-", "factor(s"); without this the
    same protein splits into several records and the mention gate misfires.
    """
    if not raw:
        return ""
    n = unicodedata.normalize("NFC", str(raw)).strip()
    n = re.sub(r"\s+", " ", n)
    # an opening parenthesis with no closing one starts a truncated tail: "factor(s"
    if "(" in n and ")" not in n[n.index("("):]:
        n = n[:n.index("(")]
    if ")" in n and "(" not in n[:n.index(")")]:
        n = n[n.index(")") + 1:]
    n = re.sub(r"['’]s$", "", n.strip())         # trailing possessive
    # a construct word after a protein name: "tau variant", "SOD1 mutants"
    n = re.sub(r"\s+(?:variants?|mutants?|isoforms?|constructs?)$", "", n, flags=re.I)
    n = n.strip(" \t\r\n.,;:!?()[]{}\"'`/\\|*-–")
    return n.strip()

# Allowed values for categorical fields of an extraction.
ENUMS = {
    "aggregate_structure": {"beta-sheet", "alpha-helical", "mixed", "amorphous", "unknown"},
    "reversibility": {"reversible", "irreversible", "conditionally_reversible", "unknown"},
    "assembly_mechanism": {"autocatalytic", "cofactor_dependent", "mixed", "unknown"},
    "functional_class": {"functional", "pathological", "both", "unknown"},
    "evidence_strength": {"high", "medium", "low", "unknown"},
    "undergoes_llps": {"yes", "no", "unknown"},
}


def clean_enum(field: str, value: Optional[str]) -> str:
    """Return ``value`` if it is exactly one allowed value for ``field``, else 'unknown'.

    A pipe-joined echo of the option list, an empty value, or anything outside
    the vocabulary is treated as no decision.
    """
    allowed = ENUMS.get(field)
    if allowed is None or value is None:
        return "unknown"
    v = str(value).strip().lower().replace(" ", "_") if field == "reversibility" else str(value).strip().lower()
    if "|" in v or v not in allowed:
        return "unknown"
    return v


def clean_extraction(protein: dict) -> dict:
    """Clean every categorical field of an extraction dict in place and return it."""
    for field in ENUMS:
        if field in protein:
            protein[field] = clean_enum(field, protein.get(field))
    return protein


# ---------------------------------------------------------------------------
# Genuine mention
# ---------------------------------------------------------------------------

def _boundary_pattern(term: str) -> re.Pattern:
    """Case-insensitive pattern for ``term`` delimited by non-alphanumerics.

    Uses look-arounds rather than ``\\b`` so terms containing Greek letters or
    hyphens (``amyloid-β``, ``TDP-43``) are matched correctly.
    """
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def genuine_mention(text: str, names: Iterable[str], min_len: int = 3) -> bool:
    """True if any of ``names`` (len >= min_len) occurs boundary-delimited in ``text``."""
    if not text:
        return False
    for name in names:
        if name and len(name.strip()) >= min_len and _boundary_pattern(name.strip()).search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Noise names
# ---------------------------------------------------------------------------

# Method abbreviations and names that NER returns as entities.
_METHOD_TOKENS = {
    "afm", "tem", "sem", "em", "cd", "ftir", "nmr", "ssnmr", "tht", "ths", "ms", "ms/ms",
    "saxs", "sans", "dls", "cryo-em", "cryoem", "xrd", "elisa", "frap", "fret", "sds-page",
    "western blot", "congo red", "thioflavin t", "thioflavin s", "md", "afm imaging",
}
# Disease abbreviations frequently mis-tagged as proteins.
_DISEASE_TOKENS = {"ad", "pd", "als", "hd", "ftd", "ftld", "cjd", "t2d", "msa", "dlb", "attr-cm"}
# Amino acids (full names and 3-letter codes): a residue is not a protein.
_AMINO_ACIDS = {
    "alanine", "arginine", "asparagine", "aspartate", "aspartic acid", "cysteine", "glutamine",
    "glutamate", "glutamic acid", "glycine", "histidine", "isoleucine", "leucine", "lysine",
    "methionine", "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine",
    "valine", "ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "ile", "leu",
    "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val",
}
_NUCLEIC = {"dna", "rna", "mrna", "cdna", "trna", "rrna", "ncrna", "sirna", "mirna", "nucleic acid"}
# Function / leading words that never begin a protein name.
_LEAD_STOP = {"a", "an", "the", "and", "or", "by", "of", "in", "with", "to", "for", "from", "via",
              "that", "this", "these", "those", "at", "on"}
# Generic nouns and structural descriptors that are not a specific protein identity.
_GENERIC_PHRASES = {
    "protein", "proteins", "peptide", "peptides", "polypeptide", "polypeptides", "polypeptide chain",
    "chain", "chains", "aggregate", "aggregates", "fibril", "fibrils", "oligomer", "oligomers",
    "monomer", "monomers", "amyloid", "amyloids", "amyloid-like", "amyloidogenic", "amyloid protein",
    "amyloid fibril", "amyloid fibrils", "amyloid peptide", "amyloid polypeptide", "amyloid-like protein",
    "abnormal protein", "aggregating protein", "aggregate-prone protein", "aggregation-prone protein",
    "aggregation domain", "misfolded protein", "amyloidogenic protein", "chimeric protein",
    "fragment", "fragments", "sequence", "sequences", "residue", "residues", "domain", "domains",
    "motif", "motifs", "repeat", "repeats", "repeat domain", "dipeptide repeats", "precursor",
    "precursor protein", "precursor proteins", "mutant", "mutants", "variant", "variants", "isoform",
    "isoforms", "factor", "cofactor", "cofactors", "hormone", "peptide hormone", "bioactive peptide",
    "element", "element binding", "element binding protein", "enhancer", "enhancing factor", "antigen",
    "antigen 5", "human", "light", "cross", "endogenous", "promoted", "truncation", "seed", "seeds",
    "metabolite", "idp", "idps", "terminus", "n terminus", "c terminus", "n-terminus", "c-terminus",
    "n-terminal", "c-terminal", "n-terminal peptide", "n-terminal domain", "c-terminal domain",
    "amino", "amino acid", "amino acids", "gene", "genes", "binding", "rna binding protein",
    "dna binding protein", "age", "aim", "enzyme", "enzymes", "receptor", "antibody", "antibodies",
    "cytoplasmic", "nuclear", "extracellular", "alpha-helical", "α-helical", "beta-barrel", "β-barrel",
    "beta-sheet", "β-sheet", "cross-beta", "synaptic", "heparin", "melanin", "phosphatidic",
    "phosphatidic acid", "phosphatidylethanolamine", "lipid", "s1 protein",
}
# Small-molecule / reagent shapes: locants ("4-methyl"), common reagent acronyms.
_REAGENT_RE = re.compile(r"^\d+[-,]|^(abts|dtt|edta|tris|hepes|dmso|tfe|hfip|gdn?hcl|sds|egcg|atp|gtp)$",
                         re.IGNORECASE)


# Reagents, glycans, lipids, dyes, tags and modifications seen as "proteins" in
# a real mining run. Some collide with gene symbols in other organisms
# (urea -> ureA, ctab -> ctaB), so they are rejected before UniProt lookup.
_CHEMICALS = {
    "urea", "guanidine", "dmso", "dimethyl sulfoxide", "ctab", "sds", "lps",
    "lipopolysaccharide", "heparin", "heparan sulfate", "chondroitin sulfate",
    "glycosaminoglycan", "glycosaminoglycans", "gm1", "ganglioside", "gangliosides",
    "d-glucose", "glucose", "trehalose", "glycerol", "dextran", "peg", "purpurin",
    "curcumin", "congo red", "thioflavin t", "egcg", "red pigment", "melanin",
    "cholesterol", "liposome", "liposomes", "cyclic peptide", "n-glycosylation",
    "glycosylation", "phosphorylation", "ptm", "ptms", "green fluorescent protein",
    "gfp", "yfp", "mcherry", "cu-zn", "nanoparticles",
}

# Head words of condition names: the last word decides ("cerebral amyloid angiopathy").
_DISEASE_HEADS = {
    "disease", "diseases", "syndrome", "angiopathy", "amyloidosis", "carcinoma", "cancer",
    "sclerosis", "dementia", "diabetes", "polyneuropathy", "neuropathy", "disorder",
    "disorders", "infection", "encephalopathy", "myopathy", "cardiomyopathy", "dystrophy",
    "degeneration", "proteinopathy", "tauopathy", "synucleinopathy", "lymphoma", "myeloma",
}

# Category nouns: they name a class or a part, not one protein.
_CLASS_PHRASES = {
    "alpha", "beta", "gamma", "delta", "kappa", "class i", "class ii", "associated proteins",
    "associated protein", "associated speck", "transcription factor", "transcription factors",
    "virulence factor", "virulence factors", "tumor suppressor", "molecular chaperone",
    "molecular chaperones", "chaperone", "chaperones", "antimicrobial peptide",
    "antimicrobial peptides", "amps", "signal peptide", "open reading frame", "orf",
    "serine protease", "protease", "proteases", "kinase", "a chain", "b chain", "a-chain",
    "aspartic", "aspartic protease", "aspartic proteases", "aspartyl protease", "cysteine protease",
    "metalloprotease", "heat shock", "heat-shock proteins", "heat-shock protein",
    "b-chain", "light chain", "heavy chain", "toll-like", "prion-like", "nac domain",
    "brichos domain", "heterogeneous nuclear ribonucleoproteins",
    "microtubule-associated protein", "tau variant", "egg", "w protein", "c protein", "apo",
    "apoa", "heat shock protein", "heat shock proteins", "rna-binding protein",
    "rna-binding proteins", "functional amyloid", "functional amyloids", "nucleoporins",
    "cyclic peptides", "amyloidogenic peptide", "amyloidogenic peptides", "amyloid-like fibril",
    "amyloid-like fibrils",
    # enzyme classes and transport families: a name of a family resolves to one
    # arbitrary member ("importin" -> Importin-4, "AAA+ ATPase" -> YTA7 in a
    # sentence about Hsp104, "PI3K" -> a Dictyostelium paralogue)
    "atpase", "atpases", "aaa+", "aaa", "aaa+ atpase", "aaa+ atpases", "gtpase", "gtpases", "kinases",
    "phosphatase", "phosphatases", "ligase", "ligases", "e3 ligase", "e3 ubiquitin ligase",
    "helicase", "helicases", "rna helicase", "importin", "importins", "exportin", "exportins",
    "karyopherin", "karyopherins", "phosphatidylinositol 3-kinase", "pi3k",
    # complexes named like genes (TORC1 -> CRTC1 via a synonym)
    "torc1", "torc2", "mtorc1", "mtorc2",
}

# Head words of non-protein molecules ("extracellular polysaccharide (EP)" -> ACBP).
_CHEMICAL_HEADS = {"polysaccharide", "polysaccharides", "exopolysaccharide", "exopolysaccharides",
                   "oligosaccharide", "oligosaccharides", "glycosaminoglycan", "glycosaminoglycans",
                   "lipid", "lipids", "liposome", "liposomes"}

# A sample description, not a protein: "LF-MF-treated PG (MPG)" resolved to the
# MPG glycosylase gene through its abbreviation.
_SAMPLE_WORDS = {"treated", "irradiated", "exposed", "loaded", "coated", "heated", "sonicated",
                 "incubated"}

# Labels in proteostasis papers that name a protein by apparent size or by a
# segment, mapped to the gene they mean there. Resolving "p97" literally gave
# Gem-associated protein 4, "p38" CRK, "p65" synaptotagmin-1, "PHF6" the PHD
# finger protein instead of the tau hexapeptide. Labels absent from this table
# ("p25": TPPP or CDK5R1 or TMED9; "p50": NFKB1 or YB-1) stay ambiguous and must
# be corroborated by the text (`is_size_label`).
ALIAS_TO_GENE = {
    "p62": "SQSTM1", "p97": "VCP", "p38": "MAPK14", "p65": "RELA", "p53": "TP53",
    "p300": "EP300", "p21": "CDKN1A", "p27": "CDKN1B", "p16": "CDKN2A", "p35": "CDK5R1",
    "phf6": "MAPT", "phf6*": "MAPT",
    # Names UniProt's name fields cannot reach (the recommended name differs, or
    # the family rule leaves several paralogs): all were no_uniprot_match drops
    # or lost family matches in the second v3 run.
    "hsp27": "HSPB1", "beta-secretase": "BACE1", "beta-secretase 1": "BACE1", "bace": "BACE1",
    "beta-site app-cleaving enzyme": "BACE1", "beta-site app-cleaving enzyme 1": "BACE1",
    "butyrylcholinesterase": "BCHE", "tnf-alpha": "TNF", "tnf-α": "TNF",
    "tumor necrosis factor-alpha": "TNF", "tumour necrosis factor-alpha": "TNF",
    "tumor necrosis factor alpha": "TNF", "fibrillarin": "FBL", "staufen1": "STAU1",
    "staufen 1": "STAU1", "protein kinase rna-like er kinase": "EIF2AK3",
    "pkr-like er kinase": "EIF2AK3", "ddx3": "DDX3X", "dead-box rna helicase 3": "DDX3X",
    "g3bp": "G3BP1", "ews": "EWSR1", "calmodulin": "CALM1",
    # lost or mis-resolved on the lookup-diff review (family rule or no exact name)
    "von hippel-lindau protein": "VHL", "von-hippel lindau protein": "VHL", "pvhl": "VHL",
    "yes-associated protein": "YAP1",
    "immunoglobulin binding protein": "HSPA5", "immunoglobulin heavy chain binding protein": "HSPA5",
    "muscle acylphosphatase": "ACYP2", "erythrocyte acylphosphatase": "ACYP1",
    "polypyrimidine tract-binding protein": "PTBP1", "polypyrimidine tract binding protein": "PTBP1",
    "vascular endothelial growth factor": "VEGFA", "vegf": "VEGFA",
    "peptidyl arginine deiminase 4": "PADI4", "arginine deiminase 4": "PADI4", "pad4": "PADI4",
}


def is_size_label(name: Optional[str]) -> bool:
    """True for an apparent-molecular-mass label such as "p25" or "p110"."""
    return bool(re.fullmatch(r"p\d{2,3}", (name or "").strip().lower()))

# A name whose head word denotes a part of a protein, not a protein.
_PART_HEADS = {"domain", "domains", "region", "regions", "motif", "motifs", "fragment",
               "fragments", "fibril", "fibrils", "residue", "residues", "repeat", "repeats",
               "complex", "complexes", "filament", "filaments"}

# Residue notation, not a protein: Glu8, Lys-1, Tyr39.
# Antibody complementarity-determining regions, not the CDR1/2/3 genes.
_CDR_RE = re.compile(r"^cdr-?[hl]?[123]$")

_RESIDUE_RE = re.compile(r"^(?:ala|arg|asn|asp|cys|gln|glu|gly|his|ile|leu|lys|met|phe|pro|ser|"
                         r"thr|trp|tyr|val)-?\d+$")

# Truncated pieces of longer names: "protein-43", "a-i", "b-a".
_FRAGMENT_RES = [
    re.compile(r"^(?:protein|peptide|chain|domain|factor)[- ]?\d+[a-z]?$"),
    re.compile(r"^[a-z]{1,2}-[a-z]{1,2}$"),     # letters only: DJ-1, YB-1 are real
]


def is_short_symbol(name: Optional[str]) -> bool:
    """True for bare symbols of at most three alphanumerics (OAT, PAH, CAA)."""
    return len(re.sub(r"[^a-z0-9]", "", clean_name(name).lower())) <= 3


def looks_like_noise(name: Optional[str]) -> bool:
    """True if ``name`` is not a usable protein identity.

    Rejects empties, punctuation-only and single characters, method / disease /
    reagent tokens, amino acids, nucleic acids, function-word-led fragments and
    generic nouns. It is deliberately conservative on short tokens (FUS, TTR,
    SOD1 survive); the long tail of generic English nouns that happen to pass is
    removed downstream by the UniProt exact-match check, since they do not
    resolve to a matching entry.
    """
    n = clean_name(name)
    if not n or not any(c.isalpha() for c in n):
        return True
    low = n.lower()
    if len(low) < 2:
        return True
    toks = low.split()
    if "," in n and len(toks) <= 3:    # a short sentence fragment slipped through NER;
        return True                    # long names may contain commas ("NOD-, LRR-, and ...")
    if toks[0] in _LEAD_STOP:
        return True
    if low in _METHOD_TOKENS or low in _DISEASE_TOKENS or low in _AMINO_ACIDS or low in _NUCLEIC:
        return True
    if low in _GENERIC_PHRASES or low in _CHEMICALS or low in _CLASS_PHRASES:
        return True
    # a class phrase qualified by place or source: "functional amyloid in Pseudomonas"
    m = re.match(r"^(.+?)\s+(?:in|of|from|for)\s+\S", low)
    if m and m.group(1) in _CLASS_PHRASES:
        return True
    if all(t in _GENERIC_PHRASES or t in _LEAD_STOP for t in toks):
        return True
    head = next((t for t in reversed(toks) if not re.fullmatch(r"\d+|[ivx]+", t)), toks[-1])
    if head in _DISEASE_HEADS or head in _PART_HEADS or head in _CHEMICAL_HEADS or "residue" in toks:
        return True
    if any(w in _SAMPLE_WORDS for t in toks for w in t.split("-")):
        return True
    if _RESIDUE_RE.match(low) or _CDR_RE.match(low) or any(rx.match(low) for rx in _FRAGMENT_RES):
        return True
    if re.match(r"^amyloid(?:[-\s](?:like|fibril|peptide|polypeptide|protein))?$", low):
        return True
    if _REAGENT_RE.search(low):
        return True
    if "coumarin" in low or low.endswith("ase inhibitor"):
        return True
    return False


# ---------------------------------------------------------------------------
# Abbreviations (Schwartz & Hearst, Pac Symp Biocomput 2003)
# ---------------------------------------------------------------------------

def _best_long_form(short: str, candidate: str) -> Optional[str]:
    """Shortest tail of ``candidate`` whose characters spell ``short`` right to left.

    The first character of the short form must start a word of the long form.
    Returns None when no such long form exists.
    """
    s, lf = short.lower(), candidate.lower()
    si, li = len(s) - 1, len(lf) - 1
    while si >= 0:
        c = s[si]
        if not c.isalnum():
            si -= 1
            continue
        while li >= 0 and (lf[li] != c or (si == 0 and li > 0 and lf[li - 1].isalnum())):
            li -= 1
        if li < 0:
            return None
        li -= 1
        si -= 1
    start = lf.rfind(" ", 0, li + 1) + 1
    result = candidate[start:].strip()
    if len(result) <= len(short) or short.lower() in result.lower().split():
        return None
    if "(" in result or ")" in result or not result[0].isalnum():
        return None                    # crossed into another parenthetical
    return result


def find_abbreviations(text: str) -> dict:
    """Map short forms defined as "long form (SF)" in ``text`` to their long forms.

    Keys are the short form as written and its lower-case copy.
    """
    found: dict = {}
    for m in re.finditer(r"\(([^()]{1,20})\)", text or ""):
        short = m.group(1).split(";")[0].split(",")[0].strip()
        if not (2 <= len(short) <= 10) or len(short.split()) > 2:
            continue
        if not short[0].isalnum() or not any(c.isalpha() for c in short):
            continue
        words = (text[:m.start()]).split()
        window = words[-min(len(short) + 5, len(short) * 2):]
        long_form = _best_long_form(short, " ".join(window))
        if long_form:
            found.setdefault(short, long_form)
            found.setdefault(short.lower(), long_form)
    # Reversed form "SF (long form)": "complex PRC1 (Polycomb repressive complex 1)".
    for m in re.finditer(r"([A-Za-z0-9][\w-]{1,9})\s+\(([^()]{5,80})\)", text or ""):
        short, inner = m.group(1), m.group(2).split(";")[0].strip()
        if len(inner.split()) < 2 or not any(c.isupper() for c in short) or short in found:
            continue
        long_form = _best_long_form(short, inner)
        if long_form and long_form.split()[0].lower() == inner.split()[0].lower():
            found.setdefault(short, long_form)
            found.setdefault(short.lower(), long_form)
    return found


# Common names that stand for an organism in abstracts.
_ORGANISM_ALIASES = {
    "saccharomyces": ["yeast", "yeasts", "budding yeast"],
    "schizosaccharomyces": ["fission yeast"],
    "podospora": ["filamentous fungus"],
    "caenorhabditis": ["c. elegans", "nematode", "worm", "worms"],
    "drosophila": ["fruit fly", "flies", "drosophila"],
    "mus": ["mouse", "mice", "murine"],
    "rattus": ["rat", "rats"],
    "danio": ["zebrafish"],
    "homo": ["human", "humans", "patients"],
    "bos": ["bovine", "cow"],
    "gallus": ["chicken", "hen"],
    "equus": ["equine", "horse"],
    "sus": ["porcine", "pig"],
    "arabidopsis": ["arabidopsis"],
    "oryctolagus": ["rabbit"],
    "triticum": ["wheat"],
    "glycine": ["soy", "soybean", "soya"],
    "vigna": ["mungbean", "mung bean"],
    "pisum": ["pea", "peas"],
    "oryza": ["rice"],
    "zea": ["maize", "corn"],
    "physeter": ["sperm whale"],
}


_SERUM_REAGENT_RE = re.compile(r"(?:fetal |foetal )?bovine serum(?: albumin)?|\bbsa\b")


def organism_mentioned(organism: Optional[str], text: Optional[str]) -> bool:
    """True if ``text`` names the organism of a UniProt entry.

    Accepted forms: the genus (>= 4 letters), the abbreviated binomial
    ("E. coli", "S. cerevisiae"), or a common name for model organisms
    ("yeast" for Saccharomyces, "mice" for Mus).
    """
    if not organism or not text:
        return False
    words = re.sub(r"\(.*?\)", " ", organism).split()
    if not words:
        return False
    genus = words[0].lower()
    forms = []
    if len(genus) >= 4:
        forms.append(genus)
    if len(words) >= 2:
        forms.append(f"{genus[0]}. {words[1].lower()}")
        forms.append(f"{genus[0]}.{words[1].lower()}")
    forms += _ORGANISM_ALIASES.get(genus, [])
    # culture and blocking reagents do not name the organism a protein comes from
    low = _SERUM_REAGENT_RE.sub(" ", text.lower())
    return any(re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", low) for f in forms)


def resolve_abbreviation(name: Optional[str], abbreviations: dict) -> Optional[str]:
    """Long form if ``name`` is itself a short form defined in the text, else None."""
    n = clean_name(name)
    if not n:
        return None
    return abbreviations.get(n) or abbreviations.get(n.lower())


_NAME_FILLER = {"protein", "proteins", "subunit", "chain", "family", "member", "putative",
                "probable", "isoform", "precursor", "domain", "containing", "like", "type",
                "homolog", "fragment", "mitochondrial", "cytoplasmic", "nuclear",
                # generic words that occur in any abstract: "alpha" corroborated
                # "HBA" as sodium channel SCN2A ("...subunit alpha") in a paper
                # on alpha-haemoglobin, "membrane" RIMS4 in a yeast meiosis paper
                "alpha", "beta", "gamma", "delta", "component", "complex", "small", "large",
                "major", "minor", "class", "group", "subfamily", "factor", "receptor", "kinase",
                "transmembrane", "channel", "regulatory", "specific", "dependent", "membrane",
                "brain", "cell", "cells"}


def corroborated(uniprot_name: Optional[str], text: str) -> bool:
    """True if a content word of the UniProt recommended name occurs in ``text``.

    Used for bare short symbols: "OAT" in an abstract is accepted as ornithine
    aminotransferase only if the abstract also says "ornithine" or
    "aminotransferase".
    """
    if not uniprot_name or not text:
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9]+", uniprot_name)
             if len(w) >= 4 and w.lower() not in _NAME_FILLER and not w.isdigit()
             # participles ("regulating", "associated", "activated") are generic
             and not re.search(r"(?:ing|ed)$", w.lower())]
    return bool(words) and genuine_mention(text, words, min_len=4)


# Organisms other than human that abstracts name in words, not binomials only.
_NONHUMAN_CUE = re.compile(
    r"\b(?:yeasts?|saccharomyces|cerevisiae|schizosaccharomyces|pombe|bacillus|escherichia|"
    r"e\. ?coli|pseudomonas|salmonella|streptococc\w*|staphylococc\w*|helicobacter|"
    r"fusobacterium|mycobacteri\w*|myxococcus|plasmodium|leishmania|trypanosoma|podospora|"
    r"arabidopsis|caenorhabditis|c\. ?elegans|drosophila|zebrafish|xenopus|rabbits?|"
    # "bovine" as a source, not the reagent "(fetal) bovine serum (albumin)"
    r"bovine(?!\s+serum)|equine|porcine|ovine|chicken|hen egg)\b", re.I)
_HUMAN_CUE = re.compile(r"\b(?:human|humans|patients?|homo sapiens)\b", re.I)
# Bacteria and archaea: a human primary gene symbol in such a paper is a
# collision (CalY of Bacillus cereus), never the human protein studied in a model.
_PROKARYOTE_CUE = re.compile(
    r"\b(?:bacteri\w*|bacillus|escherichia|e\. ?coli|pseudomonas|salmonella|streptococc\w*|"
    r"staphylococc\w*|helicobacter|fusobacterium|mycobacteri\w*|myxococcus|streptomyces|"
    r"caulobacter|klebsiella|vibrio|archae\w*|curli)\b", re.I)


# Model organisms an abstract can name -> NCBI taxon of their UniProt entries
# (yeast entries are filed under the S288c strain, 559292).
_MODEL_TAXA = [
    (re.compile(r"\b(?:drosophila|fruit ?fl(?:y|ies))\b", re.I), 7227),
    (re.compile(r"\b(?:caenorhabditis|c\. ?elegans|nematodes?)\b", re.I), 6239),
    (re.compile(r"\b(?:saccharomyces|s\. ?cerevisiae|budding yeast|yeast)\b", re.I), 559292),
    (re.compile(r"\b(?:schizosaccharomyces|s\. ?pombe|fission yeast)\b", re.I), 284812),
    (re.compile(r"\b(?:zebrafish|danio)\b", re.I), 7955),
    (re.compile(r"\barabidopsis\b", re.I), 3702),
    (re.compile(r"\bxenopus\b", re.I), 8355),
    (re.compile(r"\b(?:chicken|hen egg)\b", re.I), 9031),
    (re.compile(r"\bbovine(?!\s+serum)\b", re.I), 9913),
    (re.compile(r"\bporcine\b", re.I), 9823),
]


def named_model_taxa(text: Optional[str], limit: int = 2) -> List[int]:
    """Taxa of the model organisms ``text`` names, most mentioned first (at most ``limit``)."""
    if not text:
        return []
    counts = [(len(rx.findall(text)), taxon) for rx, taxon in _MODEL_TAXA]
    return [t for n, t in sorted(counts, key=lambda x: -x[0]) if n][:limit]


_PARALOG_MARK = r"(?:[A-Z]|\d{1,2}[a-z]?|I{1,3}|IV|VI{0,3}|alpha|beta|gamma|delta|α|β|γ|δ)"


def paralog_extension(name: Optional[str], text: Optional[str]) -> Optional[str]:
    """Complete a family name the text uses for one member only.

    NER often keeps the family head ("syntaxin") and drops the member mark the
    abstract writes after it ("syntaxin 6-positive membranes"). If every
    marked occurrence of the name carries the same mark, return "<name> <mark>";
    if the text names several members (cathepsin B, D, L, S) or none, None.
    """
    if not name or not text or len(name) < 4:
        return None
    rx = re.compile(r"(?<![\w-])" + re.escape(name) + r"[\s-]?(" + _PARALOG_MARK + r")(?![A-Za-z])", re.I)
    marks = set()
    for m in rx.finditer(text):
        mark = m.group(1)
        if len(mark) == 1 and mark.isalpha() and not mark.isupper():
            continue                               # "cathepsin a ..." is prose, not a mark
        marks.add(mark.lower())
    return f"{name} {marks.pop()}" if len(marks) == 1 else None


def names_prokaryote(text: Optional[str]) -> bool:
    """True if the text names a bacterium or archaeon."""
    return bool(text) and bool(_PROKARYOTE_CUE.search(text))


def names_nonhuman_organism(text: Optional[str]) -> bool:
    """True if the text names a non-human organism (yeast, a bacterial genus, ...)."""
    return bool(text) and bool(_NONHUMAN_CUE.search(text))


def nonhuman_only(text: Optional[str]) -> bool:
    """True if the text names a non-human organism and nothing human.

    A human gene-symbol hit in such an abstract needs corroboration: CalY in a
    Bacillus cereus paper is not the human calcyon gene CALY.
    """
    return names_nonhuman_organism(text) and not _HUMAN_CUE.search(text)


_HEAD_RE = re.compile(r"^(?:\w+ases?|proteins?|factors?|receptors?|subunits?|chains?|antigens?)$", re.I)


def extend_truncated(surface: Optional[str], text: Optional[str], max_extra: int = 3) -> Optional[str]:
    """Complete a name that NER cut before its head noun, from the text itself.

    "Prostatic Acid" + "... Prostatic Acid Phosphatase (PAP) ..." -> "Prostatic Acid Phosphatase";
    "acyl-coenzyme A" -> "acyl-coenzyme A thioesterase-7"; "Tar-DNA" -> "Tar-DNA Binding Protein-43".
    Up to ``max_extra`` following words are taken, ending at the first head noun
    (an -ase word, protein, factor, receptor, subunit, chain, antigen) with an
    optional number. Returns None when no such continuation exists.
    """
    if not surface or not text:
        return None
    for m in re.finditer(r"(?<![\w-])" + re.escape(surface.strip()) + r"(?![\w])", text, re.I):
        words = re.findall(r"[\w-]+", text[m.end():m.end() + 80])[:max_extra]
        for i, w in enumerate(words):
            core = re.sub(r"-?\d+$", "", w)
            if _HEAD_RE.match(core):
                return text[m.start():m.end()] + " " + " ".join(words[:i + 1])
    return None


# ---------------------------------------------------------------------------
# Relevance: is a protein's literature about aggregation at all?
# ---------------------------------------------------------------------------

_AGG_TERMS = re.compile(
    r"\b(?:amyloid\w*|fibril\w*|protofibril\w*|aggregat\w*|disaggregat\w*|oligomer\w*|prion\w*|"
    r"misfold\w*|plaques?|deposits?|inclusions?|proteinopath\w*|seeding|fibrillogenesis|polyq|"
    r"cross-?β|cross-?beta|β-sheets?|beta-sheets?|curli)\b", re.I)
_COND_TERMS = re.compile(
    r"\b(?:stress granules?|p-bod(?:y|ies)|processing bod(?:y|ies)|condensat\w*|"
    r"phase[- ]separat\w*|llps|ribonucleoprotein granules?|rnp granules?|rna granules?|"
    r"paraspeckles?|nuclear speckles?|nucleol\w*|droplets?|coacervat\w*|cajal bod(?:y|ies)|"
    r"pml (?:nuclear )?bod(?:y|ies)|germ granules?|p granules?|nuage|balbiani bod(?:y|ies)|"
    r"membraneless organelles?|biomolecular condensat\w*|polar granules?|germ plasm)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
          "κ": "kappa", "λ": "lambda", "ζ": "zeta", "θ": "theta"}
_GREEK_RE = re.compile("|".join(_GREEK))


def _greek_to_latin(s: str) -> str:
    return _GREEK_RE.sub(lambda m: _GREEK[m.group(0)], s)


@functools.lru_cache(maxsize=20000)
def _flex_pattern(name: str) -> Optional[re.Pattern]:
    """Spelling-tolerant pattern for a protein name.

    Matches the name's letter and digit chunks in order, with optional spaces,
    hyphens or parentheses between them, Greek letters spelled out and a
    plural ending: "beta(2) microglobulin" matches "β2-microglobulin",
    "class I hydrophobin" matches "Class I hydrophobins", "TDP43" matches "TDP-43".
    Two letter chunks still need a separator ("class i" never matches "classic").
    """
    chunks = re.findall(r"[a-z]+|\d+", _greek_to_latin(name.lower()))
    if not chunks:
        return None
    parts = [re.escape(chunks[0])]
    for prev, cur in zip(chunks, chunks[1:]):
        sep = r"[\s\-\u2010\u2013()]+" if prev.isalpha() and cur.isalpha() else r"[\s\-\u2010\u2013()]*"
        parts.append(sep + re.escape(cur))
    return re.compile(r"(?<![a-z0-9])" + "".join(parts) + r"(?:e?s)?(?![a-z0-9])")


def _name_parts(names: Iterable[str]) -> List[str]:
    """Names plus the pieces of NER run-ons ("human stefin B.. Oligomers",
    "apoA-II/apolipoprotein A-I")."""
    out = []
    for n in names:
        if not n:
            continue
        out.append(n)
        for x in re.split(r"\.\.|/", n):
            x = x.strip()
            # a generic piece ("amyloid proteins") is not a name of this protein
            if x and x != n and x.lower() not in _GENERIC_PHRASES and x.lower()[:-1] not in _GENERIC_PHRASES:
                out.append(x)
    return out


def flex_mention(text: str, names: Iterable[str], min_len: int = 2) -> bool:
    """True if any name occurs in ``text`` up to spelling variation (see `_flex_pattern`)."""
    if not text:
        return False
    low = _greek_to_latin(text.lower())
    for n in _name_parts(names):
        if len(n.strip()) < min_len:
            continue
        pat = _flex_pattern(n.strip())
        if pat is not None and pat.search(low):
            return True
    return False


def relevance(texts: Iterable[str], names: Iterable[str]) -> Dict[str, bool]:
    """Aggregation and condensate vocabulary in a protein's literature.

    ``agg_abstract``: an aggregation term occurs in any linked abstract.
    ``cond_abstract``: a condensate term (stress granule, P-body, LLPS, ...) does.
    ``agg_mention``: an aggregation term shares a sentence with the protein.

    On the 12873-paper corpus, 471 of 1180 proteins labelled modifiers had no
    aggregation term in any abstract and 457 of those had condensate terms:
    stress-granule and P-body proteins brought in by the RNP queries, valid
    input for the condensate database but not modifiers of amyloid. Among
    aggregators, a mention sentence without an aggregation term marked markers
    and off-topic co-mentions (CD20 in an immunostaining list, 4E-T).
    """
    names = [n for n in names if n]
    texts = [t for t in texts if t]
    joined = " ".join(texts)
    # spelling-tolerant: "beta(2) microglobulin" is written "beta(2)-microglobulin",
    # "class I hydrophobin" as "Class I hydrophobins"; both were dropped as
    # no_aggregation_mention in the second v3 run
    # (in addition to exact matching, which keeps "IRE1" in "IRE1α" and "3'-kinase")
    mention = [s for t in texts for s in _SENT_SPLIT.split(t)
               if genuine_mention(s, names, min_len=2) or flex_mention(s, names, min_len=2)]
    return {"agg_abstract": bool(_AGG_TERMS.search(joined)),
            "cond_abstract": bool(_COND_TERMS.search(joined)),
            "agg_mention": any(_AGG_TERMS.search(s) for s in mention),
            # a condensate term beside the protein: evidence for condensatedb
            # whatever its amyloid role (TDP-43 in stress granules)
            "cond_mention": any(_COND_TERMS.search(s) for s in mention)}
