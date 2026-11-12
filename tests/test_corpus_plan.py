"""Year-sliced corpus search: bins, group selection, counting and de-duplication.

Offline — PubMed is replaced by a fake `requests.get` that reports a fixed hit
count and returns PMIDs derived from the query's date range, so the de-duplication
across groups and bins and the `max_total` cut are exercised deterministically.
"""

import re

import pytest

from amylodb.mining.pubmed.searcher import PubMedSearcher
from amylodb.mining.settings import CORPUS_PLAN, TARGETED_TERM_GROUPS


def test_year_bins_cover_the_range_inclusively():
    bins = PubMedSearcher.year_bins("1998/01/01", "2003/12/31", 2)
    assert bins == [("1998/01/01", "1999/12/31"), ("2000/01/01", "2001/12/31"),
                    ("2002/01/01", "2003/12/31")]
    # a trailing partial bin is kept, not dropped
    assert PubMedSearcher.year_bins("1998/01/01", "2002/06/01", 2)[-1] == ("2002/01/01", "2002/12/31")
    assert PubMedSearcher.year_bins("2020/01/01", "2020/12/31", 5) == [("2020/01/01", "2020/12/31")]


def test_term_groups_selection():
    s = PubMedSearcher(email="x@y.z")
    assert "broad" in s.term_groups() and set(TARGETED_TERM_GROUPS) <= set(s.term_groups())
    picked = s.term_groups(["broad", "rnp_condensates"])
    assert set(picked) == {"broad", "rnp_condensates"}


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_pubmed(monkeypatch):
    """PMIDs keyed by the query's start year, so bins differ and groups overlap."""
    calls = []

    def get(url, params=None, **kw):
        calls.append(params)
        year = re.search(r'"(\d{4})/01/01"\[Date', params["term"])
        year = year.group(1) if year else "0000"
        if int(params.get("retmax", 0)) == 0:
            return _Resp({"esearchresult": {"count": "137", "idlist": []}})
        n = int(params["retmax"])
        ids = [f"{year}{i:04d}" for i in range(n)]
        return _Resp({"esearchresult": {"count": str(len(ids)), "idlist": ids}})

    import amylodb.mining.pubmed.searcher as sm
    monkeypatch.setattr(sm.requests, "get", get)
    monkeypatch.setattr(PubMedSearcher, "_rate_limit", lambda self: None)
    return calls


def test_count_does_not_retrieve_pmids(fake_pubmed):
    s = PubMedSearcher(email="x@y.z")
    assert s.count('"amyloid"[Title/Abstract]') == 137
    assert int(fake_pubmed[-1]["retmax"]) == 0


def test_plan_corpus_rows_per_group_and_bin(fake_pubmed):
    s = PubMedSearcher(email="x@y.z")
    rows = s.plan_corpus(groups=["broad", "rnp_condensates"], date_from="2020/01/01",
                         date_to="2023/12/31", bin_years=2)
    assert len(rows) == 4                                  # 2 groups x 2 bins
    assert {r["group"] for r in rows} == {"broad", "rnp_condensates"}
    assert all(r["count"] == 137 for r in rows)
    assert {(r["from"], r["to"]) for r in rows} == {("2020", "2021"), ("2022", "2023")}


def test_search_corpus_dedups_across_groups(fake_pubmed):
    s = PubMedSearcher(email="x@y.z")
    pmids = s.search_corpus(groups=["broad", "rnp_condensates"], date_from="2020/01/01",
                            date_to="2023/12/31", bin_years=2, per_bin=5, max_total=1000)
    # both groups return the same ids for a given bin, so only the 2 bins x 5 survive
    assert len(pmids) == 10 and len(set(pmids)) == 10
    assert sorted(pmids) == sorted([f"2020{i:04d}" for i in range(5)] +
                                   [f"2022{i:04d}" for i in range(5)])


def test_search_corpus_respects_max_total(fake_pubmed):
    s = PubMedSearcher(email="x@y.z")
    pmids = s.search_corpus(groups=["broad"], date_from="2000/01/01", date_to="2009/12/31",
                            bin_years=2, per_bin=10, max_total=25)
    assert len(pmids) == 25 and len(set(pmids)) == 25


def test_corpus_plan_defaults_are_sane():
    assert CORPUS_PLAN["date_from"].startswith("1998")
    assert CORPUS_PLAN["bin_years"] >= 1 and CORPUS_PLAN["per_bin"] > 0
    assert CORPUS_PLAN["max_total"] >= CORPUS_PLAN["per_bin"]
