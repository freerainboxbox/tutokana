"""Detection metrics, the Spearman ceiling, and snapping to the label grid.

These exist because the reported rank correlation was being read as a model failure when
most of it is a property of the labels. On the test split a *perfect* continuous predictor
caps at SCC 0.528 on word accuracy, simply because 90% of the labels are tied at 10 and a
continuous readout never ties. Meanwhile perfect ceiling-vs-not detection with random
ordering below reaches 99% of the achievable rank correlation, and perfect ordering with no
detection reaches 1% — so Spearman here is a detection metric in disguise.
"""

from __future__ import annotations

import numpy as np
import pytest

from tutokana.metrics import (
    detection_auc,
    detection_metrics,
    field_metrics,
    snap_to_support,
    spearman,
    spearman_ceiling,
)


# --- detection ---------------------------------------------------------------------------


def test_auc_is_one_for_a_perfect_separator():
    gold = np.array([2.0] * 8 + [1.0, 0.5])
    pred = np.array([1.0] * 8 + [0.0, 0.0])  # imperfect labels score strictly lower
    assert detection_auc(pred, gold) == pytest.approx(1.0)


def test_auc_is_half_for_a_useless_score():
    rng = np.random.default_rng(0)
    gold = np.array([2.0] * 800 + [1.0] * 200)
    assert detection_auc(rng.normal(size=1000), gold) == pytest.approx(0.5, abs=0.05)


def test_auc_is_zero_when_the_score_is_inverted():
    gold = np.array([2.0] * 8 + [1.0, 0.5])
    pred = np.array([0.0] * 8 + [1.0, 1.0])
    assert detection_auc(pred, gold) == pytest.approx(0.0)


def test_auc_is_nan_when_one_class_is_absent():
    """A field with nothing below the ceiling has no detection problem to measure."""
    gold = np.array([2.0] * 10)
    assert detection_auc(np.arange(10.0), gold) != detection_auc(np.arange(10.0), gold)


def test_detection_reports_the_base_rate():
    gold = np.array([10.0] * 90 + [3.0] * 10)
    d = detection_metrics(np.linspace(0, 1, 100), gold)
    assert d.positives == 10
    assert d.share == pytest.approx(0.10)


def test_best_f1_threshold_beats_an_arbitrary_one():
    """Base rates run from 0.8% to 19%, so a fixed midpoint would understate a model that
    ranks well but is badly calibrated."""
    gold = np.array([2.0] * 95 + [1.0] * 5)
    pred = np.concatenate([np.full(95, 0.9), np.full(5, 0.6)])  # separable, not around 0.5
    d = detection_metrics(pred, gold)
    assert d.f1 == pytest.approx(1.0)
    assert d.precision == pytest.approx(1.0) and d.recall == pytest.approx(1.0)


def test_degenerate_field_yields_nan_not_zero(recwarn):
    gold = np.array([10.0] * 50)
    d = detection_metrics(np.linspace(0, 1, 50), gold)
    assert d.positives == 0
    assert d.auc != d.auc and d.f1 != d.f1


# --- the ceiling -------------------------------------------------------------------------


def test_ceiling_is_one_when_nothing_is_tied():
    gold = np.arange(100.0)
    assert spearman_ceiling(gold) == pytest.approx(1.0, abs=1e-6)


def test_heavier_ties_lower_the_ceiling():
    """This is the whole point: word accuracy caps lower than phone accuracy because it is
    more tied, not because word scores are harder to predict."""
    light = np.array([2.0] * 500 + list(np.linspace(0, 1.8, 500)))
    heavy = np.array([2.0] * 900 + list(np.linspace(0, 1.8, 100)))
    assert spearman_ceiling(heavy) < spearman_ceiling(light) < 1.0


def test_a_model_can_exceed_nothing_above_its_ceiling():
    rng = np.random.default_rng(0)
    gold = np.array([2.0] * 800 + list(rng.choice(np.arange(0, 2.0, 0.2), 200)))
    ceiling = spearman_ceiling(gold)
    for sd in (0.05, 0.2, 0.5):
        noisy = gold + rng.normal(0, sd, gold.size)
        assert spearman(noisy, gold) <= ceiling + 1e-6


def test_detection_alone_reaches_almost_all_of_the_ceiling():
    """The finding that says a differentiable ranking loss is the wrong tool."""
    rng = np.random.default_rng(0)
    gold = np.array([2.0] * 813 + list(rng.choice(np.arange(0, 2.0, 0.2), 187)))
    ceiling = spearman_ceiling(gold)

    detector = (gold == gold.max()).astype(float)          # knows only ceiling-vs-not
    ordering = gold.copy()
    ordering[gold == gold.max()] = np.median(gold[gold < gold.max()])  # orders, cannot detect

    assert spearman(detector, gold) / ceiling > 0.95
    assert spearman(ordering, gold) / ceiling < 0.25


# --- snapping ----------------------------------------------------------------------------


def test_snapping_lands_on_the_grid_and_nowhere_else():
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out = snap_to_support(np.array([0.09, 0.11, 0.51, 0.99]), grid)
    assert set(np.unique(out)) <= set(grid)
    np.testing.assert_allclose(out, [0.0, 0.2, 0.6, 1.0])


def test_snapping_is_monotone():
    """It cannot invent ordering the predictions did not already have."""
    grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    raw = np.array([0.1, 0.4, 0.9, 1.2, 1.9])
    out = snap_to_support(raw, grid)
    assert np.all(np.diff(out) >= 0)


def test_snapping_restores_the_ties_gold_has():
    """A continuous readout never ties; gold is quantised. That mismatch is what a rank
    correlation is penalising."""
    grid = [0.0, 1.0, 2.0]
    raw = np.array([1.91, 1.94, 1.97, 2.03])
    assert len(np.unique(raw)) == 4
    assert len(np.unique(snap_to_support(raw, grid))) == 1


def test_snapping_helps_a_quantised_target():
    rng = np.random.default_rng(0)
    grid = list(np.arange(0, 2.01, 0.2))
    gold = rng.choice(grid, 4000)
    noisy = gold + rng.normal(0, 0.25, gold.size)
    raw = field_metrics(noisy, gold, bootstrap=False)
    snapped = field_metrics(snap_to_support(noisy, grid), gold, bootstrap=False)
    assert snapped.pearson > raw.pearson


def test_snapping_an_empty_grid_is_a_no_op():
    """Runs predating support in target_stats.json must be left alone, not guessed at."""
    raw = np.array([0.3, 0.7])
    np.testing.assert_allclose(snap_to_support(raw, []), raw)


# --- reporting ---------------------------------------------------------------------------


def test_detection_table_omits_fields_with_no_dominant_ceiling():
    """Utterance scores run 2-10 with no dominant value, so 95-99% are 'below the maximum'
    and F1 hits 0.999 by calling everything positive. That row is noise, not a result."""
    from tutokana.reporting import render_detection

    rng = np.random.default_rng(0)
    phone_gold = np.array([2.0] * 813 + list(rng.choice(np.arange(0, 2.0, 0.2), 187)))
    utt_gold = rng.choice(np.arange(2.0, 10.0), 500)  # 10 never occurs -> ceiling is 9
    results = {
        "phone.accuracy": detection_metrics(phone_gold + rng.normal(0, .3, 1000), phone_gold),
        "utterance.total": detection_metrics(utt_gold + rng.normal(0, 1, 500), utt_gold),
    }
    table = render_detection(results)
    assert "phone.accuracy" in table
    assert "no dominant ceiling" in table
    assert table.count("utterance.total") == 1  # only in the skipped note, not as a row


def test_detection_metrics_serialise_for_the_out_file():
    """--out has to round-trip through JSON for the decomposition to be analysable later."""
    import json
    from dataclasses import asdict

    gold = np.array([2.0] * 90 + [1.0] * 10)
    blob = asdict(detection_metrics(np.linspace(0, 1, 100), gold))
    assert json.loads(json.dumps(blob))["positives"] == 10


def test_snapping_table_shows_per_field_deltas():
    """Snapping is a per-field call, not a global one: it helps a fine grid and hurts a
    coarse one, so the table has to show both directions rather than a single verdict."""
    from tutokana.reporting import render_snapping

    rng = np.random.default_rng(0)
    grid = list(np.arange(0, 2.01, 0.2))
    gold = rng.choice(grid, 2000)
    noisy = gold + rng.normal(0, 0.25, gold.size)
    raw = {"phone.accuracy": field_metrics(noisy, gold, bootstrap=False)}
    snapped = {"phone.accuracy": field_metrics(snap_to_support(noisy, grid), gold,
                                               bootstrap=False)}
    table = render_snapping(raw, snapped)
    assert "PCC snap" in table and "+" in table


def test_target_stats_round_trip_the_label_grid(tmp_path):
    """Snapping must use the training grid; deriving it from the split under evaluation
    would be reading that split's labels."""
    from tutokana.data import TargetStats

    stats = TargetStats(mean={"phone.accuracy": 1.9}, std={"phone.accuracy": 0.37},
                        support={"phone.accuracy": [0.0, 0.2, 2.0]})
    path = tmp_path / "target_stats.json"
    stats.save(path)
    assert TargetStats.load(path).support["phone.accuracy"] == [0.0, 0.2, 2.0]


def test_target_stats_without_a_grid_load_cleanly(tmp_path):
    """Runs trained before support was recorded must still load — and snap nothing."""
    import json
    from tutokana.data import TargetStats

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"mean": {"phone.accuracy": 1.9},
                                "std": {"phone.accuracy": 0.37}}))
    assert TargetStats.load(path).support == {}


def test_nan_formats_in_a_signed_column():
    """A constant prediction gives a NaN delta, and ">+9" is not a valid string format.
    Snapping word.stress collapses it to one value, so this is reachable, not theoretical."""
    from tutokana.reporting import _fmt

    assert _fmt(float("nan"), "+9.3f").strip() == "nan"
    assert len(_fmt(float("nan"), "+9.3f")) == 9
    assert _fmt(0.5, "+9.3f").strip() == "+0.500"


def test_snapping_table_survives_a_field_that_collapses():
    from tutokana.reporting import render_snapping

    # The real failure mode: at a 99.2% base rate the model never predicts far from the
    # ceiling, so every prediction snaps to the same grid point and the correlation is NaN.
    gold = np.array([10.0] * 995 + [5.0] * 5)
    pred = 9.9 + np.random.default_rng(0).normal(0, 0.02, gold.size)
    raw = {"word.stress": field_metrics(pred, gold, bootstrap=False)}
    snapped = {"word.stress": field_metrics(snap_to_support(pred, [5.0, 10.0]), gold,
                                            bootstrap=False)}
    assert "nan" in render_snapping(raw, snapped)  # renders rather than raising
