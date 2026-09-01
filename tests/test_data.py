"""Data handling: splits, normalisation, the mix, and the label facts they rest on."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tutokana.data import (
    STRESS_RATERS,
    TargetStats,
    compute_target_stats,
    phone_vocabulary,
    speaker_split,
    stratified_indices,
    stress_median,
)
from tutokana.mix import build_mix, compute_multiplicities, phone_class_counts


def test_stress_median_thresholds_the_panel():
    """The released label is a majority vote, which is why it has only two values."""
    assert [stress_median(k) for k in range(STRESS_RATERS + 1)] == [
        5.0, 5.0, 5.0, 10.0, 10.0, 10.0
    ]


def test_validation_split_is_speaker_disjoint(utterances):
    train, val = speaker_split(utterances, val_speakers=2, seed=0)
    assert {u.speaker for u in train} & {u.speaker for u in val} == set()
    assert len(train) + len(val) == len(utterances)


def test_split_is_deterministic(utterances):
    first = speaker_split(utterances, 2, seed=7)[1]
    second = speaker_split(utterances, 2, seed=7)[1]
    assert [u.index for u in first] == [u.index for u in second]


def test_split_rejects_impossible_sizes(utterances):
    with pytest.raises(ValueError):
        speaker_split(utterances, val_speakers=99, seed=0)


def test_target_stats_roundtrip(tmp_path, utterances):
    stats = compute_target_stats(utterances)
    path = tmp_path / "stats.json"
    stats.save(path)
    assert TargetStats.load(path).mean == stats.mean


def test_normalisation_inverts(utterances):
    stats = compute_target_stats(utterances)
    value = 7.0
    roundtrip = stats.denormalize(
        "utterance", "accuracy", stats.normalize("utterance", "accuracy", value)
    )
    assert roundtrip == pytest.approx(value)


def test_degenerate_field_gets_a_floored_deviation(utterances):
    """Completeness is constant in the fixture, as it nearly is in the real corpus."""
    stats = compute_target_stats(utterances)
    assert stats.std["utterance.completeness"] >= 1e-3


def test_stress_statistics_are_on_the_published_median_scale(utterances):
    """Normalisation follows the reported label, not the vote count the head trains on."""
    stats = compute_target_stats(utterances)
    assert 5.0 <= stats.mean["word.stress"] <= 10.0
    assert stats.support["word.stress"] == [5.0, 10.0]


def test_phone_vocabulary_is_sorted_and_deduplicated(utterances):
    phones = phone_vocabulary(utterances)
    assert phones == tuple(sorted(set(phones)))
    assert "AO0" in phones


def test_stratified_subset_is_deterministic_and_bounded(utterances):
    first = stratified_indices(utterances, 6)
    assert first == stratified_indices(utterances, 6)
    assert len(first) == 6 and len(set(first)) == 6
    assert stratified_indices(utterances, 999) == list(range(len(utterances)))


def test_multiplicities_stay_within_bounds(utterances):
    mult = compute_multiplicities(utterances, k=5)
    assert mult.min() >= 1 and mult.max() <= 5


def test_oversampling_raises_phone_class_entropy(utterances):
    _, stats = build_mix(utterances, k=5, n_negatives=0, seed=0)
    assert stats["phone_entropy_mixed"] >= stats["phone_entropy_base"]


def test_negatives_take_transcript_from_a_and_audio_from_b(utterances):
    mixed, stats = build_mix(utterances, k=1, n_negatives=4, seed=0)
    assert stats["n_negatives"] == 4
    negatives = [u for u in mixed if u.index < 0]
    assert len(negatives) == 4
    for negative in negatives:
        # Nothing was attempted, so completeness is 0 and every phone scores 0.
        assert negative.completeness == 0.0
        assert all(a == 0.0 for w in negative.words for a in w.phone_accuracy)
        # Delivery really is B's, which teaches that the dimensions are separable.
        assert negative.accuracy <= 2.0 and negative.total <= 2.0


def test_mix_is_reproducible_from_the_seed(utterances):
    a, _ = build_mix(utterances, k=3, n_negatives=3, seed=11)
    b, _ = build_mix(utterances, k=3, n_negatives=3, seed=11)
    assert [u.index for u in a] == [u.index for u in b]
    assert [u.text for u in a] == [u.text for u in b]


def test_phone_class_counts_round_the_continuous_label(utterances):
    counts = phone_class_counts(utterances)
    assert counts.sum() == sum(len(w.phones) for u in utterances for w in u.words)
    assert counts.shape[1] == 3


def test_degenerate_field_is_floored_by_span_and_warns(utterances):
    """A near-constant field must not turn an ordinary label into a huge z-score.

    Flooring at a fixed 1e-3 made a 20-utterance probe split produce targets of magnitude
    7000 and a loss to match, with nothing in the output saying why.
    """
    import warnings

    from tutokana.data import FIELD_RANGE, MIN_STD_FRACTION

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = compute_target_stats(utterances)

    low, high = FIELD_RANGE[("utterance", "completeness")]
    assert stats.std["utterance.completeness"] == pytest.approx(
        MIN_STD_FRACTION * (high - low)
    )
    assert any("completeness" in str(w.message) for w in caught)


def test_worst_case_z_score_is_bounded(utterances):
    """The floor exists to bound this; the preflight refuses anything beyond it."""
    from tutokana.data import FIELD_RANGE

    stats = compute_target_stats(utterances)
    for (level, field), (low, high) in FIELD_RANGE.items():
        key = TargetStats.key(level, field)
        if key not in stats.mean:
            continue
        extreme = max(
            abs(stats.normalize(level, field, low)),
            abs(stats.normalize(level, field, high)),
        )
        assert extreme < 60.0, (key, extreme)
