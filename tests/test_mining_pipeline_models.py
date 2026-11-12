"""End-to-end `--mode models` mining run with stubbed network and models.

PubMed, UniProt, STRING and all models are replaced by deterministic stubs, so
this exercises the orchestration itself: triage scores stored per paper,
model-based extraction, normalisation, the UniProt requirement and the mention
gate, per-paper claims written to SQLite, then export -> adapter -> amylodb
build. Expected answer, fixed in advance:

  paper 1 (alpha-synuclein + "amyloid"): alpha-synuclein stored, irreversible,
          cryo-EM reported (tier 5); generic "amyloid" never becomes a record
  paper 2 (FUS LLPS):                     FUS stored, LLPS yes,
          conditionally reversible, prion-like domain tagged after build
  paper 3 (hexokinase):                   nothing stored
"""

import json
import sqlite3

import pytest

pytestmark = pytest.mark.integration

ARTICLES = [
    ("1", "Alpha-synuclein amyloid", "Alpha-synuclein forms amyloid fibrils solved by cryo-EM. "
     "The amyloid fibrils were stable."),
    ("2", "FUS condensates", "FUS, an RNA-binding protein with a prion-like domain, undergoes phase "
     "separation; FUS droplets dissolve but later harden into fibrils."),
    ("3", "Hexokinase", "Hexokinase activity was measured in liver mitochondria."),
]


def _scores(**kw):
    from amylodb.mining.models.nli import CLAIMS
    base = {k: 0.05 for k in CLAIMS}
    base.update(kw)
    return base


class _NLI:
    def score(self, premise, protein):
        p = protein.lower()
        if "synuclein" in p:
            return _scores(aggregates=0.96, irreversible=0.9, beta=0.9, pathological=0.9)
        if p == "fus":
            return _scores(aggregates=0.8, llps=0.93, reversible=0.85, irreversible=0.8,
                           beta=0.8, functional=0.75, pathological=0.8)
        if p == "amyloid":
            return _scores(aggregates=0.99)   # would become a record if the noise filter failed
        return _scores()


class _NER:
    def extract(self, text):
        from amylodb.mining.ner.extractor import BioEntity, NERResult
        names = [n for n in ("Alpha-synuclein", "amyloid", "FUS", "Hexokinase") if n.lower() in text.lower()]
        return NERResult(proteins=[BioEntity(n, "PROTEIN", 0, 0) for n in names])


def _structner():
    from amylodb.mining.models.structner import ProteinStructureNER

    def pipe(chunk):
        i = chunk.find("cryo-EM")
        return [{"entity_group": "experimental_method", "word": "cryo-EM", "score": 0.99,
                 "start": i, "end": i + 7}] if i >= 0 else []
    return ProteinStructureNER(pipe=pipe)


@pytest.fixture
def mined_db(tmp_path, monkeypatch):
    from amylodb.mining import pipeline as mp
    from amylodb.mining.ner.normalizer import UniProtClient
    from amylodb.mining.pubmed.searcher import Article

    # no network: UniProt resolves only names in its built-in table
    monkeypatch.setattr(UniProtClient, "lookup",
                        classmethod(lambda cls, name, surface=None, context=None: UniProtClient.KNOWN_PROTEINS.get(
                            mp.ProteinNormalizer.normalize(name), None) and
                            dict(UniProtClient.KNOWN_PROTEINS[mp.ProteinNormalizer.normalize(name)])))
    db = str(tmp_path / "mining.db")
    cfg = mp.PipelineConfig(extraction_mode="models", db_path=db, use_llm_fallback=False,
                            triage_mode="annotate", use_prefilter=False)
    pipe = mp.AmyloidPipeline(cfg)
    pipe.extractor.ner = _NER()
    pipe.extractor.nli = _NLI()
    pipe.extractor.structner = _structner()
    pipe.triage._scorer = lambda texts: [0.9 if "amyloid" in t.lower() else 0.1 for t in texts]
    monkeypatch.setattr(pipe, "_search_pubmed", lambda q=None: iter(
        [Article(pmid=p, title=t, abstract=a, authors=[], journal="J", year=2025) for p, t, a in ARTICLES]))
    monkeypatch.setattr(pipe, "_enrich_with_string", lambda proteins: None)
    stats = pipe.run_full_pipeline()
    return db, stats


def test_stats_and_gates(mined_db):
    _, stats = mined_db
    assert stats["papers_processed"] == 3
    assert stats["triage_scored"] == 3 and stats["triage_dropped"] == 0   # annotate drops nothing
    assert stats["proteins_extracted"] == 2


def test_database_contents(mined_db):
    db, _ = mined_db
    c = sqlite3.connect(db)
    names = sorted(r[0] for r in c.execute("SELECT protein_name FROM proteins"))
    assert names == ["alpha-synuclein", "fus"]
    triage = dict(c.execute("SELECT pmid, triage_score FROM papers"))
    assert triage == {"1": 0.9, "2": 0.1, "3": 0.1}
    claims = c.execute("SELECT count(*) FROM protein_claims").fetchone()[0]
    assert claims == 2
    rev = dict(c.execute("SELECT protein_name, reversibility FROM proteins"))
    assert rev == {"alpha-synuclein": "irreversible", "fus": "conditionally_reversible"}
    tier = c.execute("SELECT reported_method_tier FROM proteins WHERE protein_name='alpha-synuclein'").fetchone()[0]
    assert tier == 5


def test_export_and_build(mined_db, tmp_path):
    import importlib.util
    import os

    from amylodb.mining.adapter import mined_batch_to_records
    from amylodb.pipeline import run

    db, _ = mined_db
    spec = importlib.util.spec_from_file_location(
        "export_mined", os.path.join(os.path.dirname(__file__), "..", "scripts", "export_mined.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    items = em.export(db)
    out = tmp_path / "build"
    res = run({}, str(out), mined=mined_batch_to_records(items))
    rows = {r["uniprot_id"]: r for r in __import__("csv").DictReader(open(out / "records.tsv"), delimiter="\t")}
    assert set(rows) == {"P37840", "P35637"}
    fus = rows["P35637"]
    assert fus["undergoes_llps"] == "True"
    assert fus["prion_class"] == "prion_like_domain"
    assert json.loads(fus["reversibility_votes"]) == {"conditionally_reversible": 1}
    assert rows["P37840"]["reported_method_tier"] == "5"
    assert res.stats["protein_records"] == 2
