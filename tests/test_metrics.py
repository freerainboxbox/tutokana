"""Metrics, including the columns that make a collapsed prediction visible."""

from __future__ import annotations

import numpy as np
import pytest

from tutokana.metrics import (
    field_metrics,
    levenshtein,
    pearson,
    spearman,
    strip_stress,
    transcription_metrics,
)


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert pearson(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_pearson_is_nan_for_a_constant_prediction():
    """This is how the predecessor's collapsed fields showed up. Do not paper over it."""
    assert np.isnan(pearson(np.full(50, 10.0), np.arange(50, dtype=float)))


def test_spearman_is_one_for_a_monotone_transform():
    x = np.linspace(0.1, 5, 100)
    assert spearman(np.exp(x), x) == pytest.approx(1.0, abs=1e-9)
    assert pearson(np.exp(x), x) < 0.95  # which Pearson would miss


def test_sigma_ratio_reports_shrinkage():
    gold = np.random.default_rng(1).normal(size=500)
    metrics = field_metrics(0.5 * gold, gold, bootstrap=False)
    assert metrics.sigma_ratio == pytest.approx(0.5, abs=0.02)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(2)
    gold = rng.normal(size=400)
    pred = gold + rng.normal(size=400)
    metrics = field_metrics(pred, gold)
    assert metrics.pearson_lo < metrics.pearson < metrics.pearson_hi


def test_levenshtein_basics():
    assert levenshtein(["A", "B"], ["A", "B"]) == 0
    assert levenshtein(["A", "B"], ["A"]) == 1
    assert levenshtein([], ["A", "B"]) == 2
    assert levenshtein(["A", "X", "C"], ["A", "B", "C"]) == 1


def test_strip_stress():
    assert strip_stress(["AH0", "B", "IY1"]) == ["AH", "B", "IY"]


def test_transcription_metrics_are_exact_for_a_perfect_prediction():
    gold = [[("WE", ["W", "IY0"]), ("CALL", ["K", "AO0", "L"])]]
    metrics = transcription_metrics(gold, gold)
    assert metrics.phone_error_rate == 0.0
    assert metrics.word_exact_match == 1.0
    assert metrics.n_words == 2 and metrics.n_phones == 5


def test_missing_words_count_as_wrong_rather_than_being_skipped():
    """Dropping unaligned words is how a transcription number gets flattered."""
    gold = [[("WE", ["W", "IY0"]), ("CALL", ["K", "AO0", "L"])]]
    predicted = [[("WE", ["W", "IY0"])]]
    metrics = transcription_metrics(predicted, gold)
    assert metrics.n_words == 2
    assert metrics.word_exact_match == pytest.approx(0.5)
    assert metrics.phone_error_rate == pytest.approx(3 / 5)


def test_stress_digits_only_affect_the_stressed_rate():
    gold = [[("WE", ["W", "IY0"])]]
    predicted = [[("WE", ["W", "IY1"])]]
    metrics = transcription_metrics(predicted, gold)
    assert metrics.phone_error_rate == pytest.approx(0.5)
    assert metrics.phone_error_rate_no_stress == 0.0
