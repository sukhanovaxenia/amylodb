"""Offline regression controls for text and record boundaries in PubMed XML."""
from dataclasses import asdict

import pytest

from amylodb.mining.pubmed.searcher import PubMedSearcher


def _read(body):
    xml = ("<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID>"
           f"<Article>{body}</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")
    return PubMedSearcher.__new__(PubMedSearcher)._parse_pubmed_xml(xml)[0]


@pytest.mark.parametrize(("markup", "expected"), [
    ("Role of β<sub>2</sub>-microglobulin", "Role of β2-microglobulin"),
    ("<i>Rim4</i> controls translation", "Rim4 controls translation"),
    ("<i>Ca<sup>2+</sup></i> &amp; aggregation", "Ca2+ & aggregation"),
])
def test_title_keeps_inline_text_and_tails(markup, expected):
    assert _read(f"<ArticleTitle>{markup}</ArticleTitle>").title == expected


def test_structured_abstract_preserves_sections_nested_text_and_tails():
    article = _read("""<ArticleTitle>Example</ArticleTitle><Abstract>
      <AbstractText Label="BACKGROUND">Deposits contain β<sub>2</sub>-microglobulin.</AbstractText>
      <AbstractText Label="RESULTS"><i>Rim<sup>4</sup></i> &amp; RNA were detected <i>in vitro</i>.</AbstractText>
      <AbstractText>Terminal text.</AbstractText>
    </Abstract>""")
    assert article.abstract == ("BACKGROUND: Deposits contain β2-microglobulin. "
                                "RESULTS: Rim4 & RNA were detected in vitro. Terminal text.")


def test_plain_text_and_other_metadata_are_unchanged():
    article = _read("""<ArticleTitle>Plain title</ArticleTitle>
      <Abstract><AbstractText Label="RESULTS">Plain text, with punctuation.</AbstractText></Abstract>
      <AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author></AuthorList>
      <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
      <ArticleIdList><ArticleId IdType="doi">10.0/example</ArticleId></ArticleIdList>
      <KeywordList><Keyword>amyloid</Keyword></KeywordList>
      <MeshHeadingList><MeshHeading><DescriptorName>Proteins</DescriptorName></MeshHeading></MeshHeadingList>""")
    assert asdict(article) == {
        "pmid": "42", "title": "Plain title", "abstract": "RESULTS: Plain text, with punctuation.",
        "authors": ["Doe Jane"], "journal": "Example Journal", "year": 2024,
        "doi": "10.0/example", "keywords": ["amyloid"], "mesh_terms": ["Proteins"],
        "full_text_available": False,
    }


@pytest.mark.parametrize("body", ["", "<ArticleTitle/>",
                                  "<ArticleTitle/><Abstract><AbstractText/></Abstract>"])
def test_missing_or_empty_text_fields_are_empty_strings(body):
    article = _read(body)
    assert article.title == article.abstract == ""


def test_record_without_pmid_is_skipped_without_losing_valid_record():
    xml = """<PubmedArticleSet>
      <PubmedArticle><MedlineCitation><Article><ArticleTitle>No PMID</ArticleTitle></Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>7</PMID><Article><ArticleTitle><i>Valid</i> title</ArticleTitle>
      <Abstract><AbstractText>A<sub>2</sub>B</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    articles = PubMedSearcher.__new__(PubMedSearcher)._parse_pubmed_xml(xml)
    assert [(a.pmid, a.title, a.abstract) for a in articles] == [("7", "Valid title", "A2B")]


def test_malformed_xml_does_not_return_a_partial_success():
    assert PubMedSearcher.__new__(PubMedSearcher)._parse_pubmed_xml("<PubmedArticleSet>") == []
