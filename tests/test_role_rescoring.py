"""Role decision from NLI claims and its re-scoring over a stored mining DB.

Second manual review: regulators, inhibitors, degrading enzymes and co-deposited
proteins (TRIM11, Sel1L, ROCK2, trypsin, serum amyloid P) were recorded as
aggregators. The fake scorer below entails the claims a reader would.
"""

import importlib.util
import json
import os
import sqlite3

import pytest

from amylodb.mining.models.nli import ROLE_KEYS, NLIThresholds, decide, decide_role
from amylodb.mining.resolution import assign_role

_HERE = os.path.dirname(__file__)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("scores,expected", [
    ({"aggregates": 0.92, "inhibits": 0.1}, "aggregator"),
    ({"aggregates": 0.40, "inhibits": 0.95}, "inhibitor"),
    ({"aggregates": 0.30, "regulates": 0.85}, "regulator"),
    ({"aggregates": 0.20, "co_deposits": 0.90}, "associated"),
    ({"aggregates": 0.30, "disaggregase": 0.90}, "chaperone"),
    ({"aggregates": 0.85, "inhibits": 0.90}, "aggregator"),   # lead below the margin
    ({"aggregates": 0.30, "inhibits": 0.40}, None),           # nothing entailed
])
def test_decide_role(scores, expected):
    assert decide_role(scores) == expected


def test_decide_acts_only_on_the_disaggregation_claim():
    """Regression: the broad modifier claims must not reach a mining decision.

    Acting on them per paper with fixed thresholds labelled 1220 of 1870
    proteins modifiers on the expanded corpus and suppressed forms_aggregates
    for 1202, amyloid-beta included. `decide` may now withhold the aggregator
    call only for a dominant disaggregation claim; the rest is recorded in
    role_scores for re-validation to weigh.
    """
    out = decide({"aggregates": 0.6, "regulates": 0.95})
    assert out["forms_aggregates"] is False      # below the 0.7 positive threshold
    assert out["is_modifier"] is False and out["role_subtype"] is None
    # a canonical aggregator keeps its call however high the broad claims score
    out = decide({"aggregates": 0.95, "regulates": 0.99, "co_deposits": 0.98, "inhibits": 0.97})
    assert out["forms_aggregates"] is True
    assert out["is_modifier"] is False and out["role_subtype"] is None
    assert out["role_scores"]["regulates"] == 0.99      # recorded, not acted on
    # a dominant disaggregation claim still withholds the aggregator call
    out = decide({"aggregates": 0.4, "disaggregase": 0.95})
    assert out["forms_aggregates"] is False and out["is_modifier"] and out["role_subtype"] == "chaperone"


def test_mining_role_assignment_ignores_broad_claims():
    """The mining path keeps deterministic modifier evidence and nothing else."""
    pd = {"claim_scores": {"aggregates": 0.3, "regulates": 0.95}}
    assert assign_role(dict(pd), from_claims=False) == "aggregator"
    assert assign_role(dict(pd), from_claims=True) == "modifier"       # re-validation may
    # a UniProt Chaperone keyword still wins while mining
    kw = {"protein_role": "modifier", "claim_scores": {"aggregates": 0.99}}
    assert assign_role(kw, from_claims=False) == "modifier"


def test_assign_role_keeps_keyword_modifier():
    pd = {"protein_role": "modifier", "claim_scores": {"aggregates": 0.99}}
    assert assign_role(pd) == "modifier" and pd["role_subtype"] == "chaperone"
    pd = {"claim_scores": {"aggregates": 0.2, "inhibits": 0.9}}
    assert assign_role(pd) == "modifier" and pd["role_subtype"] == "inhibitor"


class FakeScorer:
    """Entails claims from cue words; records which keys were requested."""

    def __init__(self):
        self.calls = []

    def score(self, premise, protein, keys=None):
        self.calls.append((protein, tuple(keys or ())))
        low = premise.lower()
        s = {k: 0.05 for k in (keys or ROLE_KEYS)}
        if "prevents" in low or "reverses" in low:
            s.update(inhibits=0.93, aggregates=0.35)
        elif "forms amyloid" in low:
            s.update(aggregates=0.94)
        elif "upregulation" in low:
            s.update(regulates=0.88, aggregates=0.30)
        return s


PAPERS = [
    ("1", "Tau fibrils", "Microtubule-associated protein tau (tau) forms amyloid fibrils."),
    ("2", "TRIM11", "TRIM11 prevents and reverses protein aggregation in a Parkinson's disease model."),
    ("3", "RAD51", "Human RAD51 protein forms amyloid-like aggregates in vitro."),
    ("4", "Dialysis", "After dialysis the serum levels of matrix Gla protein were reduced."),
    ("5", "ROCK2", "ROCK2 upregulation and associated protein aggregation in glaucoma."),
]
PROTEINS = [("tau", "1"), ("trim11", "2"), ("rad51", "3"), ("matrix gla protein", "4"), ("rock2", "5")]
UNIPROT = {"tau": ("P10636", "MAPT", "Microtubule-associated protein tau"),
           "trim11": ("Q96F44", "TRIM11", "E3 ubiquitin-protein ligase TRIM11"),
           "rad51": ("Q06609", "RAD51", "DNA repair protein RAD51 homolog 1"),
           "matrix gla protein": ("P08493", "MGP", "Matrix Gla protein"),
           "rock2": ("O75116", "ROCK2", "Rho-associated protein kinase 2")}


def _lookup(name, surface=None, context=None):
    hit = UNIPROT.get(name.lower())
    if not hit:
        return None
    known = name.lower() == "tau"
    return {"uniprot_id": hit[0], "gene_name": hit[1], "organism": "Homo sapiens",
            "uniprot_name": None if known else hit[2], "match_type": "known" if known else "gene",
            "lookup_stage": "known" if known else "human", "protein_role": "aggregator",
            "organism_in_text": None}


@pytest.fixture
def mining_db(tmp_path):
    from amylodb.mining.database.storage import AmyloidDatabase
    path = str(tmp_path / "m.db")
    AmyloidDatabase(path)
    c = sqlite3.connect(path)
    c.executemany("INSERT INTO papers(pmid,title,abstract) VALUES(?,?,?)", PAPERS)
    for i, (name, pmid) in enumerate(PROTEINS, 1):
        c.execute("INSERT INTO proteins(id,protein_name) VALUES(?,?)", (i, name))
        c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(?,?)", (i, pmid))
    c.commit()
    c.close()
    return path


def test_revalidate_rescores_roles(mining_db, tmp_path):
    reval = _script("revalidate_mining")
    out = str(tmp_path / "rv.db")
    scorer = FakeScorer()
    rep = reval.revalidate(mining_db, out, lookup=_lookup, rescorer=scorer)
    rows = {r[0]: r[1:] for r in sqlite3.connect(out).execute(
        "SELECT protein_name, status, protein_role, role_subtype, role_votes FROM proteins")}
    assert rows["trim11"][:3] == ("kept", "modifier", "inhibitor")
    # a regulator vote alone no longer makes a modifier (0/16 on review)
    assert rows["rock2"][:3] == ("dropped:regulator_only", None, None)
    assert rows["rad51"][:2] == ("kept", "aggregator")
    # the dialysis abstract never mentions aggregation: off-topic, not a role question
    assert rows["matrix gla protein"][0] == "dropped:off_topic"
    assert json.loads(rows["rad51"][3]) == {"aggregator": 1}
    # curated canonical proteins are not re-scored; only role claims are requested
    assert "tau" not in {p for p, _ in scorer.calls}
    assert all(set(k) == set(ROLE_KEYS) for _, k in scorer.calls)
    assert rep["role_subtypes"] == {"aggregator": 2, "inhibitor": 1}


def test_without_rescorer_roles_unchanged(mining_db, tmp_path):
    reval = _script("revalidate_mining")
    out = str(tmp_path / "rv.db")
    reval.revalidate(mining_db, out, lookup=_lookup)
    status = dict(sqlite3.connect(out).execute("SELECT protein_name, status FROM proteins"))
    # without a model nothing is dropped on a role claim; only the relevance gate acts
    assert status.pop("matrix gla protein") == "dropped:off_topic"
    assert set(status.values()) == {"kept"}


def test_rescore_role_caps_papers():
    reval = _script("revalidate_mining")
    scorer = FakeScorer()
    texts = [f"RAD51 forms amyloid fibrils in study {i}." for i in range(40)]
    role, subtype, votes = reval.rescore_role(scorer, texts, ["RAD51"], "RAD51", max_papers=10)
    assert role == "aggregator" and votes == {"aggregator": 10} and len(scorer.calls) == 10


def test_rescore_role_uses_calibrated_thresholds():
    reval = _script("revalidate_mining")
    scorer = FakeScorer()
    # "upregulation" cue -> regulates 0.88, aggregates 0.30; lead -0.58 < -0.05 -> modifier
    texts = ["ROCK2 upregulation and associated protein aggregation in glaucoma."]
    th = {"statistic": "mean", "min_aggregates": 0.5, "min_lead": -0.05}
    role, subtype, votes = reval.rescore_role(scorer, texts, ["ROCK2"], "ROCK2", thresholds=th)
    assert role == "modifier" and subtype == "regulator"
    # a clear aggregator stays aggregator and is never dropped under thresholds
    texts = ["RAD51 forms amyloid-like aggregates in vitro."]
    role, _, _ = reval.rescore_role(scorer, texts, ["RAD51"], "RAD51", thresholds=th)
    assert role == "aggregator"


def test_thresholds_disable_no_role_drop(mining_db, tmp_path):
    reval = _script("revalidate_mining")
    out = str(tmp_path / "rv.db")
    th = {"statistic": "mean", "min_aggregates": 0.5, "min_lead": -0.05}
    reval.revalidate(mining_db, out, lookup=_lookup, rescorer=FakeScorer(), role_thresholds=th)
    status = dict(__import__("sqlite3").connect(out).execute("SELECT protein_name, status FROM proteins"))
    assert "dropped:no_role_claim" not in status.values()


def test_lexical_modifier_chaperone_noun():
    from amylodb.mining.resolution import lexical_modifier
    assert lexical_modifier(["a metal-binding chaperone metallothionein-3 binds fibrils"],
                            ["metallothionein-3"]) == "chaperone"
    assert lexical_modifier(["Btn2 and Hsp42 are compartment-specific sequestrases"], ["Btn2"]) == "chaperone"
    # a true aggregator is never flipped (no chaperone-class noun)
    assert lexical_modifier(["RAD51 forms amyloid-like aggregates"], ["RAD51"]) is None
    # the noun must share a sentence with the protein
    assert lexical_modifier(["Chaperones are abundant.", "Tau forms fibrils."], ["tau"]) is None


def test_revalidate_lexical_override(tmp_path):
    reval = _script("revalidate_mining")
    from amylodb.mining.database.storage import AmyloidDatabase
    src, out = str(tmp_path / "m.db"), str(tmp_path / "rv.db")
    AmyloidDatabase(src)
    c = sqlite3.connect(src)
    c.execute("INSERT INTO papers(pmid,title,abstract) VALUES('1','MT3',"
              "'A metal-binding chaperone metallothionein-3 modulates fibril formation.')")
    c.execute("INSERT INTO proteins(id,protein_name) VALUES(1,'metallothionein-3')")
    c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(1,'1')")
    c.commit(); c.close()

    def lk(name, surface=None, context=None):
        return {"uniprot_id": "P25713", "gene_name": "MT3", "organism": "Homo sapiens",
                "uniprot_name": "Metallothionein-3", "match_type": "name", "lookup_stage": "human",
                "protein_role": "aggregator", "organism_in_text": None}
    reval.revalidate(src, out, lookup=lk)            # no rescorer: lexical override still fires
    row = sqlite3.connect(out).execute(
        "SELECT status, protein_role, role_subtype FROM proteins WHERE id=1").fetchone()
    assert row == ("kept", "modifier", "chaperone")


def test_revalidate_repairs_forms_aggregates(mining_db, tmp_path):
    """A mining run that mislabelled roles also suppressed forms_aggregates;
    re-scoring must restore it, not inherit the stored value."""
    out = str(tmp_path / "rv.db")
    src = sqlite3.connect(mining_db)
    # simulate the damage: every protein stored as a non-aggregating modifier
    src.execute("UPDATE proteins SET forms_aggregates = 0, protein_role = 'modifier', "
                "role_subtype = 'regulator'")
    src.commit(); src.close()

    reval = _script("revalidate_mining")
    rep = reval.revalidate(mining_db, out, lookup=_lookup, rescorer=FakeScorer())
    rows = dict((r[0], r[1:]) for r in sqlite3.connect(out).execute(
        "SELECT protein_name, protein_role, forms_aggregates FROM proteins WHERE status='kept'"))
    assert rows["rad51"] == ("aggregator", 1)      # restored
    assert rows["trim11"] == ("modifier", 0)       # genuinely a modifier
    assert rep["forms_aggregates_repaired"] >= 2


# ------------------------------------------------ roles from any paper
def _two_paper_db(tmp_path, name, abstracts, extra_sql=None):
    from amylodb.mining.database.storage import AmyloidDatabase
    path = str(tmp_path / "m.db")
    AmyloidDatabase(path)
    c = sqlite3.connect(path)
    c.execute("INSERT INTO proteins(id,protein_name) VALUES(1,?)", (name,))
    for i, a in enumerate(abstracts, 1):
        c.execute("INSERT INTO papers(pmid,title,abstract) VALUES(?,?,?)", (str(i), "T", a))
        c.execute("INSERT INTO protein_papers(protein_id,pmid) VALUES(1,?)", (str(i),))
    if extra_sql:
        c.execute(extra_sql)
    c.commit(); c.close()
    return path


def _lk_one(acc):
    def lookup(name, surface=None, context=None):
        return {"uniprot_id": acc, "gene_name": name.upper(), "organism": "Homo sapiens",
                "uniprot_name": name, "match_type": "name", "lookup_stage": "human",
                "protein_role": "aggregator", "organism_in_text": None}
    return lookup


def test_dual_role_from_different_papers(tmp_path):
    """Gelsolin forms the AGel amyloid in one paper and inhibits Abeta in another."""
    src = _two_paper_db(tmp_path, "gelsolin", [
        "Gelsolin forms amyloid fibrils in familial amyloidosis of the Finnish type.",
        "Gelsolin prevents the fibrillization of amyloid-beta and dissolves its fibrils."])
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk_one("P06396"), rescorer=FakeScorer())
    row = sqlite3.connect(out).execute(
        "SELECT status, protein_role, modifier_roles FROM proteins").fetchone()
    assert row[0] == "kept" and row[1] == "aggregator"
    assert json.loads(row[2]) == ["inhibitor"]

    from amylodb.mining.adapter import split_mined
    exp = _script("export_mined").export(out)
    aggs, side = split_mined(exp)
    assert [a["protein_name"] for a in aggs] == ["gelsolin"]       # an amyloid record ...
    assert [s["protein_name"] for s in side] == ["gelsolin"]       # ... and a modifier
    sheet = _script("audit_mining").review_sample(out, 10, which="modifiers")
    assert [r["name"] for r in sheet] == ["gelsolin"] and sheet[0]["modifier_roles"] == "inhibitor"
    assert sheet[0]["supporting_papers"] == 1 and "prevents" in sheet[0]["evidence_1"]


def test_chaperone_noun_overrides_that_papers_vote(tmp_path):
    """The model votes aggregator; the paper names the protein a chaperone."""
    src = _two_paper_db(tmp_path, "metallothionein-3", [
        "The chaperone metallothionein-3 forms amyloid fibril complexes with Abeta."])
    reval = _script("revalidate_mining")
    role, subtype, votes = reval.rescore_role(
        FakeScorer(), ["The chaperone metallothionein-3 forms amyloid fibril complexes with Abeta."],
        ["metallothionein-3"], "metallothionein-3")
    assert (role, subtype, votes) == ("modifier", "chaperone", {"chaperone": 1})


def test_role_fields_overwritten_not_inherited(tmp_path):
    """Mining-time subtypes must not survive a re-decision (425 condensates kept 'regulator')."""
    src = _two_paper_db(tmp_path, "rad51", ["Human RAD51 protein forms amyloid-like aggregates in vitro."],
                        extra_sql="UPDATE proteins SET protein_role='modifier', role_subtype='regulator', "
                                  "modifier_roles='[\"regulator\"]'")
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk_one("Q06609"), rescorer=FakeScorer())
    assert sqlite3.connect(out).execute(
        "SELECT protein_role, role_subtype, modifier_roles FROM proteins").fetchone() == ("aggregator", None, None)


def test_inhibitor_paper_and_aggregation_paper_give_dual_role(tmp_path):
    """One paper shows the protein inhibiting, another aggregating: both roles stand."""
    src = _two_paper_db(tmp_path, "rock2", ["ROCK2 prevents protein aggregation in glaucoma.",
                                            "ROCK2 forms amyloid deposits in vitro."])
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk_one("O75116"), rescorer=FakeScorer())
    status, role, mods = sqlite3.connect(out).execute(
        "SELECT status, protein_role, modifier_roles FROM proteins").fetchone()
    assert status == "kept" and role == "aggregator" and json.loads(mods) == ["inhibitor"]


def test_regulator_vote_does_not_tag_an_aggregator(tmp_path):
    """A regulator vote beside an aggregation paper leaves a plain aggregator."""
    src = _two_paper_db(tmp_path, "rock2", ["ROCK2 upregulation and associated protein aggregation in glaucoma.",
                                            "ROCK2 forms amyloid deposits in vitro."])
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk_one("O75116"), rescorer=FakeScorer())
    status, role, mods, votes = sqlite3.connect(out).execute(
        "SELECT status, protein_role, modifier_roles, role_votes FROM proteins").fetchone()
    assert status == "kept" and role == "aggregator" and mods is None
    assert json.loads(votes).get("regulator") == 1          # the vote is kept for calibration


def test_regulator_only_with_condensate_text_is_a_condensate(tmp_path):
    src = _two_paper_db(tmp_path, "rock2", ["ROCK2 upregulation promotes stress granules and protein aggregation."])
    out = str(tmp_path / "rv.db")
    _script("revalidate_mining").revalidate(src, out, lookup=_lk_one("O75116"), rescorer=FakeScorer())
    status, role, ev = sqlite3.connect(out).execute(
        "SELECT status, protein_role, condensate_evidence FROM proteins").fetchone()
    assert (status, role, ev) == ("kept", "condensate", 1)


def test_side_lists_hold_dual_and_condensate_evidence(tmp_path):
    import csv
    from amylodb.pipeline import run
    side = [{"protein_name": "gelsolin", "protein_role": "aggregator", "modifier_roles": ["inhibitor"],
             "uniprot_id": "P06396"},
            {"protein_name": "tdp-43", "protein_role": "aggregator", "condensate_evidence": True,
             "uniprot_id": "Q13148"},
            {"protein_name": "g3bp1", "protein_role": "condensate", "uniprot_id": "Q13283"},
            {"protein_name": "hsp70", "protein_role": "modifier", "role_subtype": "chaperone",
             "uniprot_id": "P0DMV8"}]
    fixture = os.path.join(_HERE, "fixtures", "amyload_mini.csv")
    res = run({"amyload": fixture}, str(tmp_path), compute_features=False, modifiers=side)
    rows = lambda f: {r["uniprot_id"]: r for r in csv.DictReader(open(tmp_path / f), delimiter="\t")}
    mods, conds = rows("modifiers.tsv"), rows("condensate_candidates.tsv")
    assert set(mods) == {"P06396", "P0DMV8"} and mods["P06396"]["primary_role"] == "aggregator"
    assert mods["P0DMV8"]["modifier_roles"] == "chaperone"
    assert set(conds) == {"Q13148", "Q13283"}
    assert res.stats["dual_role"] == 1 and res.stats["condensate_candidates"] == 2
