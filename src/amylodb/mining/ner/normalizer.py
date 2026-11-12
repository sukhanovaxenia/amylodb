"""
Protein name normalization and UniProt enrichment
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import requests
from functools import lru_cache

logger = logging.getLogger(__name__)


class UniProtUnavailable(RuntimeError):
    """UniProt could not be queried; the name is unresolved, not unmatched."""

# Words that may follow a gene symbol without changing which protein is meant.
_GENERIC_SUFFIX = {
    "protein", "proteins", "fibril", "fibrils", "aggregate", "aggregates", "aggregation",
    "oligomer", "oligomers", "monomer", "monomers", "domain", "peptide", "filament",
    "filaments", "inclusion", "inclusions", "lc", "lcd", "pld", "condensates", "droplets",
}


def _strict_token(term: str) -> "re.Pattern":
    """Match ``term`` delimited by anything except alphanumerics and hyphens."""
    return re.compile(r"(?<![A-Za-z0-9\-])" + re.escape(term) + r"(?![A-Za-z0-9\-])")


class ProteinNormalizer:
    """Normalize protein names and resolve synonyms"""
    
    # Canonical names for common amyloid proteins
    CANONICAL_NAMES = {
        # Amyloid-beta variants
        'amyloid-beta': ['amyloid beta', 'amyloid-β', 'amyloid β', 'abeta', 'aβ', 
                         'a-beta', 'a-β', 'beta-amyloid', 'β-amyloid', 'beta amyloid',
                         'amyloid beta peptide', 'amyloid-beta peptide', 'abeta peptide',
                         'amyloid precursor protein', 'app',
                         # hyphenated and plural forms the forward partial match
                         # cannot reach (hyphens count as word characters there);
                         # on the expanded corpus these were dropped as
                         # no_uniprot_match, or resolved to non-human entries
                         'amyloid-beta-peptide', 'amyloid-beta-peptides', 'amyloid beta-peptide',
                         'amyloid beta-peptides', 'amyloid beta peptides', 'amyloid-beta peptides',
                         'amyloid-β peptide', 'amyloid-β peptides', 'amyloid β-peptide',
                         'aβ peptide', 'aβ peptides', 'abeta peptides', 'beta-amyloid peptide',
                         'beta-amyloid peptides', 'β-amyloid peptide', 'amyloid beta-protein',
                         'amyloid-beta protein', 'amyloid beta protein', 'amyloid β-protein',
                         'abeta-protein precursor', 'abeta protein precursor',
                         'amyloid beta-protein precursor', 'amyloid beta precursor protein'],
        'aβ40': ['abeta40', 'aβ-40', 'abeta-40', 'amyloid-beta-40', 'amyloid-beta(1-40)',
                 'aβ(1-40)', 'abeta(1-40)', 'aβ 40', 'abeta 40'],
        'aβ42': ['abeta42', 'aβ-42', 'abeta-42', 'amyloid-beta-42', 'amyloid-beta(1-42)',
                 'aβ(1-42)', 'abeta(1-42)', 'aβ 42', 'abeta 42'],
        
        # Alpha-synuclein
        'alpha-synuclein': ['α-synuclein', 'α synuclein', 'alpha synuclein', 'a-synuclein',
                           'αsyn', 'a-syn', 'asyn', 'snca', 'α-syn'],
        
        # Tau
        'tau': ['tau protein', 'mapt', 'tau-protein', 'microtubule-associated protein tau'],
        
        # Huntingtin
        'huntingtin': ['htt', 'mhtt', 'mutant huntingtin', 'huntingtin protein'],
        
        # Prion
        'prion': ['prp', 'prion protein', 'prpsc', 'prpc', 'prnp'],
        
        # TDP-43
        'tdp-43': ['tdp43', 'tardbp', 'tar dna-binding protein 43', 'tdp 43'],
        
        # SOD1
        'sod1': ['superoxide dismutase 1', 'cu/zn superoxide dismutase', 'sod-1'],
        
        # FUS
        'fus': ['fused in sarcoma', 'fus protein', 'tls'],
        
        # Transthyretin
        'transthyretin': ['ttr', 'prealbumin', 'attr', 'transthyretin amyloid'],
        
        # IAPP/Amylin
        'iapp': ['amylin', 'islet amyloid polypeptide', 'islet amyloid'],
        
        # Serum amyloid A
        'saa': ['serum amyloid a', 'saa1', 'saa2'],
        
        # Heat shock proteins
        'hsp70': ['heat shock protein 70', 'hspa1a', 'hsp 70', 'hsp-70'],
        'hsp90': ['heat shock protein 90', 'hsp 90', 'hsp-90'],
        
        # Other
        'lysozyme': ['lysozyme c', 'lyz'],
        'insulin': ['human insulin', 'ins'],
        'β2-microglobulin': ['b2m', 'beta-2-microglobulin', 'beta2-microglobulin', 'β2m'],
    }
    
    # Build reverse lookup
    _SYNONYM_TO_CANONICAL = {}
    for canonical, synonyms in CANONICAL_NAMES.items():
        _SYNONYM_TO_CANONICAL[canonical.lower()] = canonical
        for syn in synonyms:
            _SYNONYM_TO_CANONICAL[syn.lower()] = canonical
    
    @classmethod
    def normalize(cls, name: str) -> str:
        """Normalize protein name to canonical form"""
        if not name:
            return name

        # Strip surrounding punctuation / possessives first, so "amyloid-beta,"
        # and "Abeta." normalise to the same canonical as "amyloid-beta".
        from amylodb.mining.validation import clean_name
        cleaned = clean_name(name) or ' '.join(name.strip().split())
        lookup = cleaned.lower()
        
        # Check direct match
        if lookup in cls._SYNONYM_TO_CANONICAL:
            return cls._SYNONYM_TO_CANONICAL[lookup]

        # Partial matching, forward direction only (a known synonym inside a
        # longer name). The previous bidirectional substring rule also matched a
        # name *inside* a synonym, which mapped generic words ("amyloid") onto
        # amyloid-beta and short tokens onto whichever synonym contained them;
        # on a real mining run 53.5 % of canonical protein-paper links were
        # spurious. Hyphens count as word characters here, so
        # "insulin-degrading enzyme" does not become insulin.
        best = None
        for syn, canonical in cls._SYNONYM_TO_CANONICAL.items():
            if len(syn) >= 4 and _strict_token(syn).search(lookup):
                if best is None or len(syn) > len(best[0]):
                    best = (syn, canonical)
        if best:
            return best[1]

        # Three-letter synonyms (FUS, TAU, TTR, APP...) only as a standalone
        # gene-symbol token: upper-case anywhere ("FUS LC domain"), or the head
        # word followed only by generic words ("Tau fibrils").
        tokens = cleaned.split()
        for syn, canonical in cls._SYNONYM_TO_CANONICAL.items():
            if len(syn) != 3:
                continue
            for i, tok in enumerate(tokens):
                if tok.lower() != syn:
                    continue
                rest = [t.lower() for t in tokens[:i] + tokens[i + 1:]]
                if tok.isupper() or (i == 0 and all(t in _GENERIC_SUFFIX for t in rest)):
                    return canonical

        # Return cleaned lowercase if no match
        return cleaned.lower()
    
    @classmethod
    def are_same_protein(cls, name1: str, name2: str) -> bool:
        """Check if two names refer to the same protein"""
        return cls.normalize(name1) == cls.normalize(name2)
    
    @classmethod
    def deduplicate(cls, proteins: List[Dict]) -> List[Dict]:
        """Remove duplicate proteins, keeping best data"""
        seen = {}  # canonical_name -> best protein data
        
        for protein in proteins:
            name = protein.get('protein_name', '')
            canonical = cls.normalize(name)
            
            if canonical in seen:
                # Merge: keep higher confidence, combine data
                existing = seen[canonical]
                if protein.get('extraction_confidence', 0) > existing.get('extraction_confidence', 0):
                    # Update with better extraction but keep metadata
                    for key, val in protein.items():
                        if val and not existing.get(key):
                            existing[key] = val
                    existing['extraction_confidence'] = protein['extraction_confidence']
            else:
                protein['protein_name'] = canonical  # Use canonical name
                seen[canonical] = protein.copy()
        
        return list(seen.values())


class UniProtClient:
    """Fetch protein metadata from UniProt"""
    
    BASE_URL = "https://rest.uniprot.org"
    
    # Pre-defined mappings for common amyloid proteins (human)
    KNOWN_PROTEINS = {
        'amyloid-beta': {'uniprot_id': 'P05067', 'gene_name': 'APP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'aβ40': {'uniprot_id': 'P05067', 'gene_name': 'APP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'aβ42': {'uniprot_id': 'P05067', 'gene_name': 'APP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'alpha-synuclein': {'uniprot_id': 'P37840', 'gene_name': 'SNCA', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'tau': {'uniprot_id': 'P10636', 'gene_name': 'MAPT', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'huntingtin': {'uniprot_id': 'P42858', 'gene_name': 'HTT', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'prion': {'uniprot_id': 'P04156', 'gene_name': 'PRNP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'tdp-43': {'uniprot_id': 'Q13148', 'gene_name': 'TARDBP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'sod1': {'uniprot_id': 'P00441', 'gene_name': 'SOD1', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'fus': {'uniprot_id': 'P35637', 'gene_name': 'FUS', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'transthyretin': {'uniprot_id': 'P02766', 'gene_name': 'TTR', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'iapp': {'uniprot_id': 'P10997', 'gene_name': 'IAPP', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'saa': {'uniprot_id': 'P0DJI8', 'gene_name': 'SAA1', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'hsp70': {'uniprot_id': 'P0DMV8', 'gene_name': 'HSPA1A', 'organism': 'Homo sapiens', 'forms_aggregates': False, 'functional_class': 'chaperone'},
        'hsp90': {'uniprot_id': 'P07900', 'gene_name': 'HSP90AA1', 'organism': 'Homo sapiens', 'forms_aggregates': False, 'functional_class': 'chaperone'},
        'lysozyme': {'uniprot_id': 'P61626', 'gene_name': 'LYZ', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
        'insulin': {'uniprot_id': 'P01308', 'gene_name': 'INS', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'functional'},
        'β2-microglobulin': {'uniprot_id': 'P61769', 'gene_name': 'B2M', 'organism': 'Homo sapiens', 'forms_aggregates': True, 'functional_class': 'pathological'},
    }
    
    # Tokens that may differ between a query and a UniProt name without changing
    # which protein is meant (numbers, roman numerals, greek letters, isoform and
    # tissue qualifiers). A new head noun such as "reductase" is not one of them.
    _QUALIFIERS = {
        "protein", "proteins", "subunit", "chain", "isoform", "precursor", "domain", "family",
        "member", "like", "type", "form", "variant", "homolog", "homologue", "mitochondrial",
        "cytoplasmic", "nuclear", "cardiac", "skeletal", "muscle", "small", "large", "light",
        "heavy", "major", "minor", "putative", "probable", "alpha", "beta", "gamma", "delta",
        "epsilon", "kappa", "lambda", "human", "mouse", "yeast", "recombinant", "wild",
        "mutant", "fibril", "fibrils", "aggregate", "aggregates", "oligomer", "oligomers",
        "monomer", "monomers", "peptide", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii",
        "ix", "x", "serum", "plasma", "bovine", "equine", "porcine", "chicken", "hen", "egg",
        "white", "full", "length", "native", "polypeptide", "rat", "murine", "rabbit",
    }

    # Generic heads that must not match a longer name on their own ("factor" vs
    # "Factor VIII", "antigen" vs "Antigen KI-67").
    _GENERIC_HEADS = {"factor", "protein", "peptide", "chain", "subunit", "antigen", "receptor",
                      "enzyme", "kinase", "protease", "domain", "complex", "component", "toxin"}

    @classmethod
    def _query_variants(cls, protein_name: str) -> List[Tuple[str, str]]:
        """(search string, kind) for one name: as given ('given'), without a
        leading "protein" ('unprefixed'), and without qualifier words ('core',
        "human serum albumin" -> "albumin"). A 'core' variant may match names
        only, never a gene symbol: "gla protein" -> "gla" is not the GLA gene."""
        out = [(protein_name, "given")]
        toks = (protein_name or "").split()
        if len(toks) >= 2 and toks[0].lower() == "protein" and len(toks[1]) >= 3:
            out.append((" ".join(toks[1:]), "unprefixed"))       # "protein tau" -> "tau"
        core = [t for t in toks if cls._norm_token(t) not in cls._QUALIFIERS]
        if core and len(core) < len(toks) and len(" ".join(core)) >= 3:
            out.append((" ".join(core), "core"))
        # a final Roman numeral as UniProt writes it: "semenogelin I" -> "Semenogelin-1"
        roman = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
        for q, kind in list(out):
            t = q.split()
            if len(t) >= 2 and t[-1].lower() in roman and len(t[-2]) >= 4:
                out.append((" ".join(t[:-1] + [roman[t[-1].lower()]]), kind))
        seen, uniq = set(), []
        for q, kind in out:
            if q.lower() not in seen:
                seen.add(q.lower())
                uniq.append((q, kind))
        return uniq

    @staticmethod
    def _parse_hit(result: Dict) -> Dict:
        """Accession, organism, keywords, recommended name, every name and every gene symbol."""
        desc = result.get("proteinDescription", {}) or {}

        def _full(entry):
            return ((entry or {}).get("fullName") or {}).get("value")

        def _shorts(entry):
            return [x.get("value") for x in (entry or {}).get("shortNames", []) or [] if x.get("value")]

        rec = desc.get("recommendedName") or {}
        # short names ("PAC", "HSP27") behave like symbols, so they join the gene
        # list and fall under the same gene-only gates
        names, shorts = [_full(rec)], _shorts(rec)
        for alt in desc.get("alternativeNames", []) or []:
            names.append(_full(alt))
            shorts += _shorts(alt)
        for part in desc.get("contains", []) or []:             # processed products: Glucagon in Pro-glucagon
            names.append(_full(part.get("recommendedName")))
        primary, genes = None, []
        for g in result.get("genes", []) or []:
            v = (g.get("geneName") or {}).get("value")
            if v:
                primary = primary or v
                genes.append(v)
            genes += [x.get("value") for x in g.get("synonyms", []) or [] if x.get("value")]
        genes += [x for x in shorts if x]
        return {
            "uniprot_id": result.get("primaryAccession"),
            "gene_name": primary,
            "genes": genes,
            "organism": (result.get("organism") or {}).get("scientificName"),
            "uniprot_name": _full(rec),
            "names": [n for n in names if n],
            "keywords": [k.get("name") for k in result.get("keywords", []) or [] if k.get("name")],
        }

    @classmethod
    def _match_hit(cls, query: str, canonical: str, hit: Dict) -> Optional[str]:
        """Best match type of ``query`` against any name / gene symbol of a parsed hit."""
        best = None
        rank = {"gene": 0, "name": 1, "run": 2}
        for i, name in enumerate(hit["names"] or [None]):
            mt = cls._match_type(query, canonical, name, hit["genes"] if i == 0 else [])
            # alternative and product names count only when equal to the query:
            # partial runs over them reach unrelated entries ("fatty acid-binding
            # protein" inside an aminotransferase's alternative name)
            if i > 0 and mt == "run":
                mt = None
            if mt and (best is None or rank[mt] < rank[best]):
                best = mt
        return best

    _SPECIES_WORDS = {"bovine", "equine", "porcine", "chicken", "hen", "rat", "murine", "mouse",
                      "yeast", "rabbit", "horse", "sperm", "whale"}

    @classmethod
    def _species_in(cls, name: str) -> bool:
        """True if the written name carries a non-human species word ("hen egg-white lysozyme")."""
        return any(t in cls._SPECIES_WORDS for t in cls._tokens(name))

    # Worm and fly genes are written "akt-1", "lin-28": the UniProt gene field does
    # not equate "akt1" with "akt-1", so letters+digits forms also search the hyphenated symbol.
    _HYPHENABLE = re.compile(r"^([A-Za-z]{2,5})(\d{1,3})$")

    @classmethod
    def _gene_alternatives(cls, q: str) -> List[str]:
        out = [q]
        m = cls._HYPHENABLE.match(q.strip())
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}")
        return out

    RETRIES = 4
    BACKOFF = 2.0          # seconds, doubled per attempt (max 30)

    @classmethod
    def _get(cls, url: str, params: Dict) -> Dict:
        """GET with retry and backoff on 429 / 5xx / network errors.

        Raises ``UniProtUnavailable`` when every attempt fails. The caller must not
        read that as "no hits": a cached empty result turned AKT1 and Cdc42 into
        no_uniprot_match drops. ``lru_cache`` does not cache an exception, so a
        later call retries.
        """
        import time
        last = None
        for attempt in range(cls.RETRIES):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (400, 404):            # malformed query: genuinely nothing
                    return {"results": []}
                last = f"HTTP {r.status_code}"
            except Exception as e:                       # timeout, connection reset, bad JSON
                last = repr(e)
            if attempt < cls.RETRIES - 1:
                time.sleep(min(30.0, cls.BACKOFF * 2 ** attempt))
        raise UniProtUnavailable(f"UniProt query failed after {cls.RETRIES} attempts: {last}")

    @classmethod
    @lru_cache(maxsize=8000)
    def candidates(cls, protein_name: str, exhaustive: bool = False) -> Tuple[Dict, ...]:
        """Every UniProt entry that genuinely corresponds to ``protein_name``.

        Stage 'human': reviewed (Swiss-Prot) human entries -- without the
        reviewed filter unreviewed fragments crowd the real entry out of the top
        hits ("fibronectin"). Stage 'reviewed_any': Swiss-Prot, any organism.
        Each stage tries the query variants in order and keeps the first that
        yields matches. Result order: human first, then by UniProt rank.
        Raises ``UniProtUnavailable`` if UniProt cannot be reached.
        """
        canonical = ProteinNormalizer.normalize(protein_name)
        url = f"{cls.BASE_URL}/uniprotkb/search"
        out: List[Dict] = []
        for stage, scope in (("human", "(organism_id:9606) AND (reviewed:true)"),
                             ("reviewed_any", "(reviewed:true)")):
            for q, kind in cls._query_variants(protein_name):
                genes = " OR ".join(f'gene:"{g}"' for g in cls._gene_alternatives(q))
                query = f'(protein_name:"{q}" OR {genes}) AND {scope}'
                data = cls._get(url, {
                    "query": query, "format": "json", "size": 25,
                    "fields": "accession,gene_names,gene_synonym,protein_name,organism_name,keyword",
                })
                found = []
                for result in data.get("results", []):
                    hit = cls._parse_hit(result)
                    mt = cls._match_hit(q, ProteinNormalizer.normalize(q), hit)
                    if mt == "gene" and kind == "core":
                        mt = None
                    mt = mt or cls._match_hit(protein_name, canonical, hit)
                    if mt:
                        hit.update(match_type=mt, lookup_stage=stage, query_variant=q,
                                   added=cls._added_tokens(q, hit["uniprot_name"]) if mt == "run" else [])
                        found.append(hit)
                if found:
                    out += found
                    break
            # `exhaustive` also searches other organisms when human entries exist:
            # needed when the abstract names another organism (Rim4 in a yeast
            # paper) or the human match is a gene synonym only
            if out and stage == "human" and not exhaustive and not cls._species_in(protein_name):
                break
        return tuple(out)

    @classmethod
    @lru_cache(maxsize=4000)
    def in_taxon(cls, protein_name: str, taxon: int) -> Optional[Dict]:
        """One entry of ``taxon`` for the name: Swiss-Prot first, then TrEMBL.

        For papers about one model organism. The general search returns 25 hits
        across all organisms, so the organism's own entry can be missing from it
        (yeast actin among a hundred actins); fly Drosha has no Swiss-Prot entry
        at all (TrEMBL Q7KNF1). Only exact recommended-name or primary-gene
        matches count. Several accessions are accepted only if they share one
        gene symbol (TrEMBL keeps fragments and duplicates of a gene); the first
        by UniProt rank is taken. Raises ``UniProtUnavailable``.
        """
        url = f"{cls.BASE_URL}/uniprotkb/search"
        for reviewed, stage in (("true", "taxon_reviewed"), ("false", "unreviewed_named")):
            for q, kind in cls._query_variants(protein_name):
                if kind == "core":
                    continue
                genes = " OR ".join(f'gene_exact:"{g}"' for g in cls._gene_alternatives(q))
                data = cls._get(url, {
                    "query": f'(protein_name:"{q}" OR {genes}) AND (organism_id:{taxon}) AND (reviewed:{reviewed})',
                    "format": "json", "size": 25,
                    "fields": "accession,gene_names,gene_synonym,protein_name,organism_name,keyword",
                })
                found = []
                for result in data.get("results", []):
                    hit = cls._parse_hit(result)
                    mt = cls._match_hit(q, ProteinNormalizer.normalize(q), hit)
                    if mt == "name" or (mt == "gene" and cls._sym(hit.get("gene_name")) == cls._sym(q)):
                        hit.update(match_type=mt, lookup_stage=stage, query_variant=q, added=[])
                        found.append(hit)
                if found:
                    genes_seen = {cls._sym(h.get("gene_name")) for h in found}
                    return found[0] if len(genes_seen) == 1 else None
        return None

    # Words a UniProt name adds to a family query that make it another protein
    # ("Ataxin-3-like protein" for ataxin-3, "von Hippel-Lindau-like" for VHL).
    _OTHER_PROTEIN = {"like", "homolog", "homologue", "related"}
    # Added words that do not tell paralogs apart.
    _NEUTRAL_ADDED = {"protein", "proteins", "polypeptide", "precursor", "peptide", "chain", "subunit"}

    @classmethod
    def _added_tokens(cls, query: str, rec_name: Optional[str]) -> List[str]:
        """Tokens of the recommended name that the query lacks."""
        qt = set(cls._tokens(query))
        return [t for t in cls._tokens(rec_name) if t not in qt]

    @classmethod
    def _filter_runs(cls, hits: List[Dict]) -> List[Dict]:
        """Drop run matches that name another protein or pick one paralog of a family.

        * a run adding "like" / "homolog" is a different protein (ATXN3L for ataxin-3);
        * a run adding a paralog mark (number, letter, Greek letter, "mitochondrial")
          is kept only if it is the only such entry of its organism: "carbonic
          anhydrase" -> CA1..CA14, "tubulin" -> the delta chain, "apolipoprotein" ->
          ApoC-I, "transportin" -> transportin-3 were all wrong; "perfringolysin" ->
          perfringolysin O and "synphilin" -> synphilin-1 are unique and right.
        On the second review sheets run matches were right in 3 of 19 cases.
        """
        from collections import Counter
        marked = lambda h: h["match_type"] == "run" and any(t not in cls._NEUTRAL_ADDED for t in h.get("added", []))
        # grouped by species, not by organism string: PSM alpha-2 and alpha-3 of two
        # S. aureus strains are still two paralogs
        species = lambda h: " ".join((h.get("organism") or "").split()[:2])
        per_org = Counter()
        seen = set()
        for h in hits:
            if marked(h) and h["uniprot_id"] not in seen:
                seen.add(h["uniprot_id"])
                per_org[species(h)] += 1
        out = []
        for h in hits:
            if h["match_type"] == "run" and cls._OTHER_PROTEIN & set(h.get("added", [])):
                continue
            if marked(h) and per_org[species(h)] > 1:
                continue
            out.append(h)
        return out

    _KIND_RANK = {"name": 0, "primary": 1, "run": 2, "synonym": 3}

    @classmethod
    def _ranked(cls, hits: List[Dict], qs: set) -> List[Dict]:
        """Strongest identity first, UniProt rank within a kind: the exact name
        "Ataxin-3" must win over a run match listed before it."""
        return sorted(hits, key=lambda h: cls._KIND_RANK.get(cls._kind(h, qs), 9))

    @classmethod
    def lookup(cls, protein_name: str, surface: Optional[str] = None,
               context: Optional[str] = None) -> Optional[Dict]:
        """Resolve a protein name to one UniProt entry, or None.

        Returns uniprot_id, gene_name, organism, uniprot_name, match_type
        ('known' | 'gene' | 'name' | 'run'), lookup_stage ('known' | 'human' |
        'reviewed_any'), organism_in_text, keywords and protein_role.

        A human Swiss-Prot entry is taken first. Otherwise, among reviewed
        entries of other organisms, one whose organism the abstract names
        (``context``) is preferred. A match on a gene symbol alone in another
        organism is accepted only if that organism is named in the abstract:
        without it, any lower-case word or 3-letter symbol collides with a gene
        somewhere (urea -> ureA, Glu8 -> rice GLU8, FadA -> E. coli FadA when the
        paper is about Fusobacterium). With no context, gene-like written forms
        (CsgA, Sup35, Rnq1) are accepted as before.
        """
        from amylodb.mining.validation import organism_mentioned
        canonical = ProteinNormalizer.normalize(protein_name)

        if canonical in cls.KNOWN_PROTEINS:
            meta = cls.KNOWN_PROTEINS[canonical].copy()
            role = "modifier" if meta.get("functional_class") == "chaperone" else "aggregator"
            if meta.get("functional_class") == "chaperone":
                meta["functional_class"] = "unknown"
            meta.update(match_type="known", lookup_stage="known", keywords=[],
                        protein_role=role, uniprot_name=None, organism_in_text=None)
            return meta

        written = surface or protein_name
        hits = cls._filter_runs(list(cls.candidates(protein_name)))
        if cls._species_in(protein_name):
            # "equine skeletal myoglobin" is the horse protein, not the human one
            same = [h for h in hits if organism_mentioned(h["organism"], protein_name)]
            if same:
                return cls._result(same[0], True)

        from amylodb.mining.validation import corroborated, names_nonhuman_organism, nonhuman_only
        ctx = context or ""
        qs = {cls._sym(protein_name), cls._sym(written)}
        human = [h for h in hits if h["lookup_stage"] == "human"]
        if context and (names_nonhuman_organism(ctx) or
                        (human and all(cls._kind(h, qs) == "synonym" for h in human))):
            hits = cls._filter_runs(list(cls.candidates(protein_name, exhaustive=True)))
        hits = cls._ranked(hits, qs)
        human = [h for h in hits if h["lookup_stage"] == "human"]
        other = [h for h in hits if h["lookup_stage"] != "human"]
        only_other = bool(context) and nonhuman_only(ctx)
        in_text = lambda h: organism_mentioned(h["organism"], ctx) if context else None

        # an entry of an organism the abstract names, matched by name or primary gene
        named = [h for h in other if cls._kind(h, qs) != "synonym" and in_text(h)]

        def ok_human(h):
            # in an abstract naming only another organism, a human gene-symbol hit
            # needs corroboration: CalY of Bacillus cereus is not calcyon (CALY),
            # and Drosha in a fly paper is the fly protein, not human DROSHA
            return not (only_other and h["match_type"] == "gene") or corroborated(h["uniprot_name"], ctx)
        # 1. the abstract names only that organism: Rim4 / Pat1 in a yeast paper
        if only_other and named:
            return cls._result(named[0], True)
        # 2. human, strongest identity first: recommended name, qualifier run, primary gene
        for h in human:
            if cls._kind(h, qs) != "synonym" and ok_human(h):
                return cls._result(h, in_text(h))
        # 2a. the model organism the abstract names had no entry among the hits:
        #     search that organism alone, Swiss-Prot then TrEMBL, by name or primary
        #     gene only (Drosha -> fly Q7KNF1)
        if only_other and not named:
            from amylodb.mining.validation import named_model_taxa
            for taxon in named_model_taxa(ctx):
                hit = cls.in_taxon(protein_name, taxon)
                if hit:
                    return cls._result(hit, True)
        # 3. a human match by synonym only loses to a named organism's own entry
        if named:
            return cls._result(named[0], True)
        # 4. a human synonym is accepted when one entry carries it (Slug -> SNAI2,
        #    HRD1 -> synoviolin); several entries sharing it make it ambiguous, and
        #    then only the one the abstract corroborates is taken (HBA -> haemoglobin
        #    alpha, not the sodium channel SCN2A; p25 -> TPPP only if "tubulin" occurs)
        syn = cls._distinct([h for h in human if cls._kind(h, qs) == "synonym"])
        if len(syn) == 1 and ok_human(syn[0]):
            return cls._result(syn[0], in_text(syn[0]))
        if len(syn) > 1 and context:
            corr = [h for h in syn if corroborated(h["uniprot_name"], ctx)]
            if len(corr) == 1:
                return cls._result(corr[0], in_text(corr[0]))
        # 5. other organisms: a gene-symbol match needs its organism in the abstract
        #    (urea -> ureA, Glu8 -> rice GLU8); with no context, a gene-like form
        chosen = None
        for hit in other:
            it = in_text(hit)
            if hit["match_type"] == "gene":
                if context is not None and not it:
                    continue
                if context is None and not cls._gene_like(written):
                    continue
            if it or context is None:
                return cls._result(hit, it)
            chosen = chosen or (hit, it)                 # name match, organism unnamed
        return cls._result(*chosen) if chosen else None

    @staticmethod
    def _sym(x: Optional[str]) -> str:
        """Loose symbol identity: "Bcl-2" == "BCL2", "protein tau" != "tau"."""
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())

    @classmethod
    def _kind(cls, hit: Dict, qs: set) -> str:
        """'name' | 'run' | 'primary' (query is the primary gene) | 'synonym'."""
        if hit["match_type"] != "gene":
            return hit["match_type"]
        q = qs | {cls._sym(hit.get("query_variant"))}
        return "primary" if cls._sym(hit.get("gene_name")) in q else "synonym"

    @staticmethod
    def _distinct(hits: List[Dict]) -> List[Dict]:
        seen, out = set(), []
        for h in hits:
            if h["uniprot_id"] not in seen:
                seen.add(h["uniprot_id"])
                out.append(h)
        return out

    @classmethod
    def _result(cls, h: Dict, in_text) -> Dict:
        return {
            "uniprot_id": h["uniprot_id"], "gene_name": h["gene_name"],
            "organism": h["organism"], "uniprot_name": h["uniprot_name"],
            "match_type": h["match_type"], "lookup_stage": h["lookup_stage"],
            "organism_in_text": in_text, "keywords": h["keywords"],
            "protein_role": cls.role_from(h["keywords"], h["uniprot_name"]),
        }

    @staticmethod
    def role_from(keywords: list, rec_name: Optional[str]) -> str:
        """'modifier' for chaperones / disaggregases (UniProt keyword or name), else 'aggregator'."""
        kw = {(k or "").lower() for k in keywords or []}
        name = (rec_name or "").lower()
        if "chaperone" in kw or any(t in name for t in ("chaperon", "heat shock", "dnaj")):
            return "modifier"
        return "aggregator"

    @staticmethod
    def _gene_like(written: Optional[str]) -> bool:
        """Gene-symbol-like written form: >= 4 alphanumerics with an upper-case letter or a digit."""
        import re as _re
        w = (written or "").strip()
        core = _re.sub(r"[^A-Za-z0-9]", "", w)
        return len(core) >= 4 and (any(c.isupper() for c in w) or any(c.isdigit() for c in w))

    @staticmethod
    def _norm_token(s: str) -> str:
        """Lowercase and strip everything but alphanumerics, for loose equality."""
        import re as _re
        return _re.sub(r"[^a-z0-9]", "", (s or "").lower())

    @staticmethod
    def _tokens(s: str) -> list:
        """Lowercase alphanumeric tokens of a name."""
        import re as _re
        return [t for t in _re.split(r"[^a-z0-9]+", (s or "").lower()) if t]

    @staticmethod
    def _is_sublist(small: list, big: list) -> bool:
        """True if ``small`` occurs as a contiguous run inside ``big`` (len>=2 only)."""
        if len(small) < 2 or len(small) > len(big):
            return False
        return any(big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1))

    @classmethod
    def _extra_are_qualifiers(cls, small: list, big: list) -> bool:
        """Tokens of ``big`` outside its first run of ``small`` are all qualifiers."""
        for i in range(len(big) - len(small) + 1):
            if big[i:i + len(small)] == small:
                extra = big[:i] + big[i + len(small):]
                return all(t in cls._QUALIFIERS or t.isdigit() or len(t) == 1 for t in extra)
        return False

    @classmethod
    def _match_type(cls, query: str, canonical: str, rec_name: Optional[str],
                    gene_names: list) -> Optional[str]:
        """How the query corresponds to a UniProt entry, or None.

        * 'gene' -- the query (or its canonical form) equals a gene symbol;
        * 'name' -- the query equals the recommended name;
        * 'run'  -- one is a contiguous multi-word run of the other and every
          extra word is a qualifier ("carbonic anhydrase" vs "Carbonic
          anhydrase 2"), never a new head noun ("dimethyl sulfoxide" vs
          "Dimethyl sulfoxide reductase").
        """
        q_forms = {cls._norm_token(query), cls._norm_token(canonical)}
        q_forms.discard("")
        if not q_forms:
            return None
        if any(cls._norm_token(g) in q_forms for g in gene_names or []):
            return "gene"
        if rec_name and cls._norm_token(rec_name) in q_forms:
            return "name"
        name_tokens = cls._tokens(rec_name)
        for q in (query, canonical):
            qt = cls._tokens(q)
            # a run needs one specific word: "protein c" (= protein + letter) never
            # matches "Protein C-1" or similar by extension
            if not any(len(t) >= 4 and t not in cls._GENERIC_HEADS and t not in cls._QUALIFIERS
                       for t in qt):
                continue
            # single-word name inside a qualified query: "equine skeletal myoglobin"
            # -> "Myoglobin", "human serum albumin" -> "Albumin"
            if len(name_tokens) == 1 and name_tokens[0] in qt and len(qt) > 1:
                rest = [t for t in qt if t != name_tokens[0]]
                if all(t in cls._QUALIFIERS or t.isdigit() for t in rest):
                    return "run"
            # single-word query, longer name of qualifiers: "calmodulin" -> "Calmodulin-1",
            # "fibrinogen" -> "Fibrinogen alpha chain"; never for generic heads
            if (len(qt) == 1 and len(name_tokens) > 1 and name_tokens[0] == qt[0]
                    and qt[0] not in cls._GENERIC_HEADS and len(qt[0]) >= 4
                    and all(t in cls._QUALIFIERS or t.isdigit() or len(t) == 1 for t in name_tokens[1:])):
                return "run"
            if cls._is_sublist(qt, name_tokens) and cls._extra_are_qualifiers(qt, name_tokens):
                return "run"
            if cls._is_sublist(name_tokens, qt) and cls._extra_are_qualifiers(name_tokens, qt):
                return "run"
        return None

    @classmethod
    def _hit_matches(cls, query: str, canonical: str, rec_name: Optional[str],
                     gene_names: list) -> bool:
        """Boolean form of `_match_type`."""
        return cls._match_type(query, canonical, rec_name, gene_names) is not None

    @classmethod
    def enrich_proteins(cls, proteins: List[Dict]) -> List[Dict]:
        """Fill UniProt metadata (only missing fields), passing the written surface form."""
        enriched = []
        for protein in proteins:
            name = protein.get('protein_name', '')
            context = protein.get('_context')
            try:
                metadata = cls._enrich_one(protein, name, context)
            except UniProtUnavailable as e:
                # kept unresolved: scripts/revalidate_mining.py retries the lookup
                logger.warning(f"{name}: {e}")
                protein['lookup_error'] = True
                metadata = None
            if metadata:
                for key, val in metadata.items():
                    if val and not protein.get(key):
                        protein[key] = val
            enriched.append(protein)
        return enriched

    @classmethod
    def _enrich_one(cls, protein: Dict, name: str, context: Optional[str]) -> Optional[Dict]:
        """Lookup with the short-form and truncated-name fallbacks."""
        metadata = cls.lookup(name, protein.get('surface_form'), context=context)
        # an expanded long form ("B-cell lymphoma 2") may not match UniProt
        # where its short form ("Bcl-2") does: fall back to the short form,
        # flagged so the gate demands corroboration of the UniProt name
        if not metadata and protein.get('resolved_from'):
            metadata = cls.lookup(ProteinNormalizer.normalize(protein['resolved_from']),
                                  protein.get('resolved_from'), context=context)
            if metadata:
                protein['short_form_fallback'] = True
        if not metadata and context:
            from amylodb.mining.validation import extend_truncated, looks_like_noise
            longer = extend_truncated(protein.get('surface_form') or name, context)
            if longer and not looks_like_noise(longer):
                metadata = cls.lookup(ProteinNormalizer.normalize(longer), longer, context=context)
                if metadata:
                    protein['protein_name'] = ProteinNormalizer.normalize(longer)
        if not metadata and context:
            # a family head the text writes with one member mark only ("syntaxin 6")
            from amylodb.mining.validation import paralog_extension
            member = paralog_extension(protein.get('surface_form') or name, context)
            if member:
                metadata = cls.lookup(ProteinNormalizer.normalize(member), member, context=context)
                if metadata:
                    protein['protein_name'] = ProteinNormalizer.normalize(member)
        return metadata


def normalize_and_enrich(proteins: List[Dict], use_uniprot: bool = True) -> List[Dict]:
    """Full normalization pipeline: normalize names, deduplicate, enrich"""
    # 1. Normalize and deduplicate
    deduped = ProteinNormalizer.deduplicate(proteins)
    
    # 2. Enrich with UniProt (optional)
    if use_uniprot:
        deduped = UniProtClient.enrich_proteins(deduped)
    
    return deduped


if __name__ == "__main__":
    # Test
    test_proteins = [
        {'protein_name': 'Amyloid-Beta', 'extraction_confidence': 0.8},
        {'protein_name': 'amyloid beta', 'extraction_confidence': 0.7},
        {'protein_name': 'Aβ42', 'extraction_confidence': 0.9},
        {'protein_name': 'Alpha-synuclein', 'extraction_confidence': 0.85},
        {'protein_name': 'α-synuclein', 'extraction_confidence': 0.75},
        {'protein_name': 'tau', 'extraction_confidence': 0.9},
        {'protein_name': 'HSP70', 'extraction_confidence': 0.6},
    ]
    
    print("Before normalization:")
    for p in test_proteins:
        print(f"  {p['protein_name']}")
    
    result = normalize_and_enrich(test_proteins, use_uniprot=True)
    
    print(f"\nAfter normalization ({len(result)} unique):")
    for p in result:
        print(f"  {p['protein_name']}: gene={p.get('gene_name')}, uniprot={p.get('uniprot_id')}")
