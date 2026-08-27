"""The objective's anti-collapse machinery."""

from __future__ import annotations

import math

import pytest
import torch

from tutokana.losses import (
    ALWAYS_REWEIGHTED,
    CorrelationBuffer,
    LabelReweighter,
    LossConfig,
    build_reweighter,
    concordance,
    log_cosh,
)
from tutokana.metrics import pearson


def test_log_cosh_matches_the_naive_form_in_range():
    x = torch.linspace(-5, 5, 101)
    expected = torch.log(torch.cosh(x))
    assert torch.allclose(log_cosh(x, torch.zeros_like(x)), expected, atol=1e-5)


def test_log_cosh_is_finite_where_the_naive_form_overflows():
    x = torch.tensor([200.0, -200.0])
    assert torch.isfinite(log_cosh(x, torch.zeros_like(x))).all()
    assert torch.isinf(torch.log(torch.cosh(x))).all()  # the reason the identity is used


def test_log_cosh_gradient_is_bounded():
    x = torch.tensor([50.0], requires_grad=True)
    log_cosh(x, torch.zeros_like(x)).backward()
    assert x.grad.abs().item() <= 1.0 + 1e-6


def test_concordance_is_one_for_an_exact_match():
    x = torch.randn(64)
    assert concordance(x, x).item() == pytest.approx(1.0, abs=1e-4)


def test_concordance_punishes_variance_collapse_where_pearson_does_not():
    """The whole reason CCC is the correlation term rather than Pearson."""
    gold = torch.randn(512)
    shrunk = 0.2 * gold  # perfectly correlated, five times too flat

    assert pearson(shrunk.numpy(), gold.numpy()) == pytest.approx(1.0, abs=1e-6)
    assert concordance(shrunk, gold).item() < 0.6


def test_concordance_punishes_a_mean_shift():
    gold = torch.randn(512)
    assert concordance(gold + 2.0, gold).item() < concordance(gold, gold).item()


def test_buffer_widens_the_population_used_for_the_statistic():
    buffer = CorrelationBuffer(capacity=256)
    gold = torch.randn(256)
    buffer.extend("k", gold, gold)

    live_pred = torch.randn(4, requires_grad=True)
    pred, target = buffer.augmented("k", live_pred, torch.randn(4))
    assert pred.shape[0] == 260 and target.shape[0] == 260
    # Only the live slice carries gradient; the history is detached by construction.
    assert not buffer._predictions["k"].requires_grad
    concordance(pred, target).backward()
    assert live_pred.grad is not None


def test_buffer_is_a_ring():
    buffer = CorrelationBuffer(capacity=10)
    for _ in range(5):
        buffer.extend("k", torch.randn(4), torch.randn(4))
    pred, _ = buffer.augmented("k", torch.zeros(1), torch.zeros(1))
    assert pred.shape[0] == 11


def test_buffer_disabled_at_zero_capacity():
    buffer = CorrelationBuffer(capacity=0)
    buffer.extend("k", torch.randn(4), torch.randn(4))
    pred, _ = buffer.augmented("k", torch.zeros(2), torch.zeros(2))
    assert pred.shape[0] == 2


def test_reweighter_has_mean_one_under_the_training_marginal():
    """Turning reweighting on must not implicitly change the effective learning rate."""
    histogram = {"phone.accuracy": {2.0: 800, 1.0: 150, 0.0: 50}}
    reweighter = LabelReweighter(histogram, strength=1.0)
    labels = torch.tensor([2.0] * 800 + [1.0] * 150 + [0.0] * 50)
    weights = reweighter.weights_for("phone.accuracy", labels)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-4)


def test_reweighter_favours_the_rare_label():
    histogram = {"phone.accuracy": {2.0: 800, 0.0: 50}}
    reweighter = LabelReweighter(histogram, strength=1.0)
    weights = reweighter.weights_for("phone.accuracy", torch.tensor([2.0, 0.0]))
    assert weights[1] > weights[0]


def test_zero_strength_is_uniform():
    histogram = {"phone.accuracy": {2.0: 800, 0.0: 50}}
    reweighter = LabelReweighter(histogram, strength=0.0)
    weights = reweighter.weights_for("phone.accuracy", torch.tensor([2.0, 0.0]))
    assert weights[0].item() == pytest.approx(weights[1].item())


def test_stress_is_reweighted_even_when_its_level_is_not(utterances):
    """At 99:1 there is no configuration in which an unbalanced stress head is intended."""
    config = LossConfig(reweight_levels=())
    reweighter = build_reweighter(utterances, config)
    assert "word.stress" in ALWAYS_REWEIGHTED
    assert reweighter.weights_for("word.stress", torch.tensor([0.0, 1.0])) is not None


#: The real word-accuracy histogram of the training split, whose tail is the whole problem:
#: accuracy 9 occurs 3 times in 15849 words.
WORD_ACCURACY_HISTOGRAM = {"word.accuracy": {10.0: 13907, 3.0: 1109, 5.0: 358, 9.0: 3}}


def _dynamic_range(reweighter, key):
    weights = reweighter.tables[key][1]
    return (weights.max() / weights.min()).item()


def test_uncapped_inverse_frequency_has_a_ruinous_dynamic_range():
    """The bug this cap exists for: one rare label outweighing hundreds of ordinary ones."""
    uncapped = LabelReweighter(WORD_ACCURACY_HISTOGRAM, strength=1.0, maximum=0)
    assert _dynamic_range(uncapped, "word.accuracy") > 1000


def test_cap_bounds_the_dynamic_range():
    capped = LabelReweighter(WORD_ACCURACY_HISTOGRAM, strength=1.0, maximum=10.0)
    assert _dynamic_range(capped, "word.accuracy") == pytest.approx(10.0, rel=1e-4)


def test_capped_weights_are_still_mean_one():
    """The cap must not shift the effective learning rate."""
    reweighter = LabelReweighter(WORD_ACCURACY_HISTOGRAM, strength=1.0, maximum=10.0)
    histogram = WORD_ACCURACY_HISTOGRAM["word.accuracy"]
    labels = torch.tensor([v for v, n in histogram.items() for _ in range(n)])
    weights = reweighter.weights_for("word.accuracy", labels)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-3)


def test_cap_preserves_the_ordering_of_rarity():
    """Weakly, not strictly: labels rarer than the ceiling all tie there, by design."""
    histogram = {"phone.accuracy": {2.0: 37707, 1.6: 2416, 0.2: 44}}
    reweighter = LabelReweighter(histogram, strength=1.0, maximum=10.0)
    values, weights = reweighter.tables["phone.accuracy"]
    by_value = dict(zip(values, weights.tolist()))

    assert by_value[0.2] >= by_value[1.6] > by_value[2.0]
    assert by_value[0.2] == by_value[1.6]  # both rarer than the ceiling allows

    # A cap wide enough to hold them apart keeps the ordering strict.
    wide = LabelReweighter(histogram, strength=1.0, maximum=1000.0)
    by_value = dict(zip(*[wide.tables["phone.accuracy"][0], wide.tables["phone.accuracy"][1].tolist()]))
    assert by_value[0.2] > by_value[1.6] > by_value[2.0]


def test_build_reweighter_passes_the_cap_through(utterances):
    reweighter = build_reweighter(utterances, LossConfig(reweight_max=3.0))
    for key in reweighter.tables:
        assert _dynamic_range(reweighter, key) <= 3.0 + 1e-4


def test_only_the_phone_head_is_reweighted_by_default(utterances):
    """Reweighting is a classification device; only the phone head is a classifier.

    On a regression head a label weight scales a log-cosh residual, which rebalances nothing.
    `word.stress` is the exception — a binary head at 99:1 — and is always included.
    """
    reweighter = build_reweighter(utterances, LossConfig())
    assert set(reweighter.tables) == {"phone.accuracy", "word.stress"}


def test_word_level_reweighting_is_available_as_an_ablation(utterances):
    reweighter = build_reweighter(utterances, LossConfig(reweight_levels=("phone", "word")))
    assert {"word.accuracy", "word.total"} <= set(reweighter.tables)


def test_disabling_every_level_still_reweights_stress(utterances):
    reweighter = build_reweighter(utterances, LossConfig(reweight_levels=()))
    assert set(reweighter.tables) == {"word.stress"}
