"""Heads: layer mixing, the three output modes, and FiLM conditioning."""

from __future__ import annotations

import pytest
import torch

from tutokana.data import FIELD_SUPPORT, TargetStats
from tutokana.heads import HeadBank, LayerMixture, ScoreHead, build_head_specs


def test_layer_mixture_is_convex():
    mixture = LayerMixture(n_layers=8, bias_toward=1.0)
    weights = torch.tensor(mixture.distribution())
    assert weights.sum().item() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_layer_mixture_bias_points_where_asked():
    """Phones read low, utterances read high — HMamba's hierarchy, as an initialisation."""
    low = LayerMixture(n_layers=8, bias_toward=0.0).distribution()
    high = LayerMixture(n_layers=8, bias_toward=1.0).distribution()
    assert low[0] == max(low)
    assert high[-1] == max(high)


def test_layer_mixture_selects_the_window():
    mixture = LayerMixture(n_layers=2, bias_toward=1.0)
    hidden = tuple(torch.full((1, 3, 4), float(i)) for i in range(5))
    out = mixture(hidden)
    assert out.shape == (1, 3, 4)
    assert 3.0 <= out.mean().item() <= 4.0  # only the last two layers contribute


def test_regression_head_predicts_on_both_scales():
    head = ScoreHead(hidden_size=16, mode="regression", mean=7.0, std=2.0)
    logits = torch.zeros(5, 1)
    assert torch.allclose(head.predict_native(logits), torch.full((5,), 7.0))
    assert torch.allclose(head.predict_normalized(logits), torch.zeros(5))


def test_soft_class_readout_is_continuous():
    """The point of soft_class: an expectation over a saturated support, not an argmax."""
    support = FIELD_SUPPORT[("phone", "accuracy")]
    head = ScoreHead(hidden_size=16, mode="soft_class", support=support, mean=1.9, std=0.36)
    logits = torch.zeros(1, len(support))  # uniform over 0.0 .. 2.0
    assert head.predict_native(logits).item() == pytest.approx(1.0, abs=1e-5)


def test_soft_class_targets_snap_to_the_nearest_support_value():
    support = FIELD_SUPPORT[("phone", "accuracy")]
    head = ScoreHead(hidden_size=8, mode="soft_class", support=support)
    indices = head.class_targets(torch.tensor([0.0, 1.0, 1.8, 2.0]))
    assert torch.equal(indices, torch.tensor([0, 5, 9, 10]))


def test_binary_head_returns_a_probability():
    head = ScoreHead(hidden_size=8, mode="binary", mean=0.99, std=0.1)
    value = head.predict_native(torch.zeros(3, 1))
    assert torch.allclose(value, torch.full((3,), 0.5))


def test_soft_class_requires_a_support():
    with pytest.raises(ValueError):
        ScoreHead(hidden_size=8, mode="soft_class")


def test_film_starts_as_the_identity():
    """Conditioning must earn its keep, so it begins as a no-op."""
    plain = ScoreHead(hidden_size=8, mode="regression").eval()
    conditioned = ScoreHead(hidden_size=8, mode="regression", n_conditions=4).eval()
    conditioned.body.load_state_dict(plain.body.state_dict())
    states = torch.randn(6, 8)
    assert torch.allclose(
        plain(states), conditioned(states, torch.randint(0, 4, (6,))), atol=1e-6
    )


def test_film_head_demands_a_condition():
    head = ScoreHead(hidden_size=8, mode="regression", n_conditions=4)
    with pytest.raises(ValueError):
        head(torch.randn(2, 8))


def test_build_head_specs_forces_binary_stress(utterances):
    from tutokana.data import compute_target_stats

    specs = build_head_specs(
        levels=("phone", "word", "utterance"),
        head_modes={"phone": "soft_class", "word": "regression", "utterance": "regression"},
        n_phone_conditions=7,
        phone_conditioning="film",
        stats=compute_target_stats(utterances),
    )
    assert specs["word.stress"]["mode"] == "binary"
    assert specs["phone.accuracy"]["n_conditions"] == 7
    assert "utterance.completeness" not in specs  # measured, never trained


def test_head_bank_reads_the_requested_positions(utterances):
    from tutokana.collate import HeadBatch
    from tutokana.data import compute_target_stats

    stats = compute_target_stats(utterances)
    specs = build_head_specs(
        ("utterance",),
        {"utterance": "regression"},
        n_phone_conditions=0,
        phone_conditioning="none",
        stats=stats,
    )
    bank = HeadBank(hidden_size=12, specs=specs, levels=("utterance",), n_layers=2)
    hidden = tuple(torch.randn(2, 9, 12) for _ in range(3))
    batch = {
        "utterance.accuracy": HeadBatch(
            positions=torch.tensor([[0, 4], [1, 7]]),
            target=torch.zeros(2),
            raw=torch.zeros(2),
        )
    }
    out = bank.read(hidden, batch)
    assert out["utterance.accuracy"].shape == (2, 1)
