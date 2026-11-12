"""Fixed-width feature matrix (finding 4)."""

from amylodb.features_matrix import build_feature_matrix, feature_columns, feature_row


class _Rec:
    def __init__(self, seq):
        self.sequence = seq


def test_width_is_constant_444():
    cols = feature_columns()
    assert len(cols) == 444
    assert len(set(cols)) == 444
    assert sum(1 for c in cols if c.startswith("dp_")) == 400  # full dipeptide grid


def test_matrix_is_order_independent_and_rectangular():
    recs = [_Rec("STVIIE"), _Rec("GNNQQNY"), _Rec(""), _Rec("MKV")]
    cols, rows = build_feature_matrix(recs)
    assert all(len(r) == len(cols) for r in rows)
    cols2, rows2 = build_feature_matrix(list(reversed(recs)))
    assert cols == cols2
    assert rows2[0] == rows[-1]  # same sequence -> same row regardless of position


def test_empty_sequence_is_all_zero_row():
    assert all(v == 0.0 for v in feature_row(""))


def test_aa_frequencies_sum_to_one():
    cols = feature_columns()
    row = feature_row("STVIIE")
    aa_idx = [i for i, c in enumerate(cols) if c.startswith("aa_")]
    assert abs(sum(row[i] for i in aa_idx) - 1.0) < 1e-9
