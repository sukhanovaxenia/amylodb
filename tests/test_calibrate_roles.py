"""Threshold calibration of the role decision against manual marks."""

import csv
import importlib.util
import json
import os
import random
import sqlite3

_HERE = os.path.dirname(__file__)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture(tmp_path, n=40, seed=3):
    """Aggregators: high 'aggregates'; non-aggregators: same 'aggregates', higher 'regulates'."""
    rnd = random.Random(seed)
    db = str(tmp_path / "rv.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE proteins (id INTEGER, uniprot_id TEXT, role_scores TEXT)")
    sheet = tmp_path / "review.tsv"
    with open(sheet, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["uniprot_id", "role", "correct"])
        for i in range(n):
            is_agg = i % 4 != 0
            s = {"aggregates": 0.85 + rnd.random() * 0.1, "disaggregase": 0.1, "inhibits": 0.2,
                 "co_deposits": 0.1, "regulates": 0.3 if is_agg else 0.9 + rnd.random() * 0.05}
            con.execute("INSERT INTO proteins VALUES (?,?,?)", (i, f"P{i:05d}", json.dumps([s])))
            w.writerow([f"P{i:05d}", "aggregator", "y" if is_agg else "n"])
        w.writerow(["Q99999", "modifier", "n"])                 # modifier rows are ignored
    con.commit()
    con.close()
    return db, str(sheet)


def test_labels_and_features(tmp_path):
    cal = _script("calibrate_roles")
    db, sheet = _fixture(tmp_path)
    labels = cal.load_labels([sheet])
    assert len(labels) == 40 and "Q99999" not in labels
    feats = cal.load_features(db)
    assert set(feats["P00000"]) == {"mean", "max"}


def test_fitted_rule_separates_and_beats_current(tmp_path):
    cal = _script("calibrate_roles")
    db, sheet = _fixture(tmp_path)
    items = [(y, cal.load_features(db)[a]) for a, y in cal.load_labels([sheet]).items()]
    current = cal.metrics([(y, cal.predict(f, "max", 0.7, -1.0)) for y, f in items])
    assert current["precision"] == 0.75
    params = cal.best_params(items, 0.9)
    fit = cal.metrics([(y, cal.predict(f, *params)) for y, f in items])
    assert fit["precision"] == 1.0 and fit["recall"] == 1.0


def test_cli_writes_thresholds(tmp_path, monkeypatch, capsys):
    cal = _script("calibrate_roles")
    db, sheet = _fixture(tmp_path)
    out = str(tmp_path / "th.json")
    monkeypatch.setattr("sys.argv", ["x", "--db", db, "--labels", sheet, "--out", out])
    assert cal.main() == 0
    th = json.load(open(out))
    assert th["leave_one_out"]["precision"] >= 0.9 and th["n"] == 40
