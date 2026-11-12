"""End-to-end build on the mini fixture, with a known expected answer.

Expected (hand-checked from amyload_mini.csv):
  * 4 input rows, all with distinct sequences -> 4 consensus records
  * 3 amyloid (AMY1, AMY3, AMY4), 1 non-amyloid (AMY2)
  * AMY4 ("FUS prion-like domain") is tagged PRION_LIKE_DOMAIN, not dropped
  * feature matrix is 444 wide, one row per record
  * graph has one record node per consensus record
"""

import csv
import gzip
import json

import pytest

from amylodb.pipeline import run


@pytest.fixture
def built(tmp_path, amyload_mini_path):
    out = tmp_path / "out"
    res = run({"amyload": amyload_mini_path}, str(out), compute_features=True)
    return res, out


def test_consensus_counts(built):
    res, _ = built
    assert res.stats["consensus"] == 4
    assert res.stats["conflicts"] == 0


def test_label_split(built):
    _, out = built
    rows = list(csv.DictReader(open(out / "records.tsv"), delimiter="\t"))
    amyloid = [r for r in rows if r["is_amyloid"] == "True"]
    assert len(amyloid) == 3
    assert len(rows) - len(amyloid) == 1


def test_prion_like_kept_and_tagged(built):
    _, out = built
    rows = list(csv.DictReader(open(out / "records.tsv"), delimiter="\t"))
    fus = [r for r in rows if "FUS" in r["protein_name"]]
    assert len(fus) == 1
    assert fus[0]["prion_class"] == "prion_like_domain"


def test_feature_matrix_width_and_alignment(built):
    res, out = built
    with gzip.open(out / "features.tsv.gz", "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        n = sum(1 for _ in fh)
    assert len(header) - 1 == 444
    assert n == res.stats["consensus"]


def test_graph_one_record_node_per_record(built):
    res, out = built
    graph = json.load(open(out / "graph.json"))
    assert graph["stats"]["record_nodes"] == res.stats["consensus"]
