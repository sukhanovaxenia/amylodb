"""Stronger noise filter and UniProt exact-match verification (measured on real names)."""

from amylodb.mining.ner.normalizer import ProteinNormalizer as N
from amylodb.mining.ner.normalizer import UniProtClient as U
from amylodb.mining.validation import clean_name, looks_like_noise


def test_clean_name_strips_punctuation_and_possessive():
    assert clean_name("amyloid,") == "amyloid"
    assert clean_name("apoa-i..") == "apoa-i"
    assert clean_name(". amyloid") == "amyloid"
    assert clean_name("protein's") == "protein"
    assert clean_name("  TDP-43 ") == "TDP-43"
    assert clean_name("α-synuclein.") == "α-synuclein"
    assert clean_name("aβ.") == "aβ"            # Aβ = amyloid-beta, kept (not noise)


def test_noise_drops_fragments_generics_amino_acids_nucleic():
    for junk in (". amyloid", "] amyloidogenic", "a c", "a gene", "amino acid", "and", "by α",
                 "chain", "binding", "factor", "fragment", "sequence", "mutant", "human", "that",
                 "amyloid", "amyloid-like", "amyloid fibril", "residues", "precursor", "domains",
                 "asp", "lys", "val", "histidine", "dna", "rna", "mrna", "n-terminus", "amyloid,",
                 "the β", "of amyloid", "with β", "polypeptide chain", "seeds", "melanin"):
        assert looks_like_noise(junk), junk


def test_noise_keeps_real_proteins():
    for name in ("alpha-synuclein", "amyloid-beta", "FUS", "tau", "TDP-43", "SOD1", "insulin",
                 "lysozyme", "huntingtin", "transthyretin", "clusterin", "prion", "Sup35", "CsgA",
                 "hnRNPA1", "p53", "IAPP", "titin", "SAA", "S100A9", "BRI2", "CPEB", "FXR1",
                 "carbonic anhydrase", "myosin-binding protein c", "serum albumin", "β-2-microglobulin"):
        assert not looks_like_noise(name), name


def test_uniprot_hit_requires_real_correspondence():
    # accepted: gene symbol, exact name, contiguous multiword run
    assert U._hit_matches("FUS", "fus", "RNA-binding protein FUS", ["FUS", "TLS"])
    assert U._hit_matches("alpha-synuclein", "alpha-synuclein", "Alpha-synuclein", ["SNCA"])
    assert U._hit_matches("CsgA", "csga", "Major curlin subunit CsgA", ["csgA"])
    assert U._hit_matches("carbonic anhydrase", "carbonic anhydrase", "Carbonic anhydrase 2", ["CA2"])
    assert U._hit_matches("serum albumin", "serum albumin", "Serum albumin", ["ALB"])
    # rejected: generic words that are mere substrings of a real name
    assert not U._hit_matches("factor", "factor", "Coagulation factor VIII", ["F8"])
    assert not U._hit_matches("binding", "binding", "Histone-binding protein RBBP4", ["RBBP4"])
    assert not U._hit_matches("that", "that", "Thrombin", ["F2"])
    assert not U._hit_matches("chain", "chain", "Myosin light chain 3", ["MYL3"])


def test_normalize_cleans_before_matching():
    assert N.normalize("amyloid-beta,") == N.normalize("amyloid-beta")
    assert N.normalize("TDP-43.") == "tdp-43"
