"""A failing model must cost one chunk or one paper, never the run.

zshot 0.0.11 raises KeyError(-1) in its SMXM span decoder on inputs that cannot
be predicted from the text, and on the first expanded corpus this ended a whole
run from a single abstract. Two guards are tested here: OpenBioNER skips the
chunk it cannot process, and the pipeline skips the paper whose extraction
raises, recording both.
"""

import pytest

from amylodb.mining.models.openbioner import OpenBioNERExtractor
from amylodb.mining.pubmed.searcher import Article


class _Raises:
    """Stands in for the zshot pipeline, failing the way the library does."""

    def __init__(self, on: str = None):
        self.on, self.calls = on, 0

    def __call__(self, chunk):
        self.calls += 1
        if self.on is None or self.on in chunk:
            raise KeyError(-1)
        return _Doc([])


class _Ent:
    def __init__(self, text, label, start, end):
        self.text, self.label_, self.start_char, self.end_char = text, label, start, end


class _Doc:
    def __init__(self, ents):
        self.ents = ents


def test_clean_neutralises_offset_breaking_characters():
    raw = "Tau​ forms\x01 fibrils.﻿"
    clean = OpenBioNERExtractor._clean(raw)
    assert "​" not in clean and "\x01" not in clean and "﻿" not in clean
    assert clean.split() == ["Tau", "forms", "fibrils."]
    # length-preserving, so the spans this extractor reports still index the input
    assert len(clean) == len(raw)
    # a tab between two words must not merge them
    assert OpenBioNERExtractor._clean("a\t\tb").split() == ["a", "b"]
    assert OpenBioNERExtractor._clean(None) == ""
    assert OpenBioNERExtractor._clean("   ").strip() == ""


def test_failing_chunk_is_skipped_not_raised():
    ner = OpenBioNERExtractor(nlp=_Raises(), model="x")
    res = ner.extract("Tau forms amyloid fibrils. SOD1 aggregates too.")
    assert res.proteins == [] and ner.failed_chunks == 1


def test_only_the_failing_chunk_is_lost():
    """A long text is chunked; one bad chunk leaves the others annotated."""
    class Mixed:
        def __init__(self):
            self.failed = 0

        def __call__(self, chunk):
            if "poison" in chunk:
                self.failed += 1
                raise KeyError(-1)
            i = chunk.find("SOD1")
            return _Doc([_Ent("SOD1", "protein", i, i + 4)] if i >= 0 else [])

    text = ("The poison sentence is here. " + "Filler text about fibrils. " * 50
            + "SOD1 aggregates in ALS.")
    ner = OpenBioNERExtractor(nlp=Mixed(), model="x")
    res = ner.extract(text)
    assert ner.failed_chunks == 1
    assert [p.text for p in res.proteins] == ["SOD1"]
    # the span still points at the protein in the cleaned text
    clean = OpenBioNERExtractor._clean(text)
    p = res.proteins[0]
    assert clean[p.start:p.end] == "SOD1"


def test_extraction_failure_skips_one_paper_only(tmp_path, monkeypatch):
    from amylodb.mining import pipeline as mp

    articles = [Article(pmid=str(i), title=f"T{i}", abstract=f"Tau forms fibrils {i}.",
                        authors=[], journal="J", year=2025) for i in (1, 2, 3)]
    cfg = mp.PipelineConfig(extraction_mode="models", db_path=str(tmp_path / "m.db"),
                            use_llm_fallback=False, use_prefilter=False, triage_mode="off")
    pipe = mp.AmyloidPipeline(cfg)

    class Extractor:
        """Fails on PMID 2 only."""
        ner = _Raises()

        def extract(self, title, abstract, pmid):
            if pmid == "2":
                raise KeyError(-1)
            return mp.HybridResult(proteins=[], paper_relevance="relevant") \
                if hasattr(mp, "HybridResult") else _Result()

    class _Result:
        proteins = []
        paper_relevance = "relevant"

    pipe.extractor = Extractor()
    pipe.use_hybrid = True
    monkeypatch.setattr(pipe, "_search_pubmed", lambda q=None: iter(articles))
    monkeypatch.setattr(pipe, "_enrich_with_string", lambda proteins: None)

    stats = pipe.run_full_pipeline()
    assert stats["extraction_failed"] == 1
    assert stats["failed_pmids"] == ["2"]
    assert stats["papers_processed"] == 2           # the other two still went through
