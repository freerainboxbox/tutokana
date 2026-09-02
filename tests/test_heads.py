"""Heads: layer mixing, the four output modes, and FiLM conditioning."""

from __future__ import annotations

import pytest
import torch

from tutokana.collate import HeadBatch
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
    conditioned = ScoreHead(
        hidden_size=8, mode="regression", n_conditions=4, conditioning="film"
    ).eval()
    conditioned.body.load_state_dict(plain.body.state_dict())
    states = torch.randn(6, 8)
    assert torch.allclose(
        plain(states), conditioned(states, torch.randint(0, 4, (6,))), atol=1e-6
    )


def test_conditioned_head_demands_a_condition():
    for mode in ("film", "concat"):
        head = ScoreHead(hidden_size=8, mode="regression", n_conditions=4, conditioning=mode)
        with pytest.raises(ValueError, match="condition"):
            head(torch.randn(2, 8))


def test_concat_widens_the_input_and_starts_neutral():
    """A learned per-symbol embedding on the input, zero-init so it begins as a no-op."""
    head = ScoreHead(
        hidden_size=8, mode="regression", n_conditions=4, conditioning="concat",
        condition_dim=3,
    ).eval()
    assert head.body[1].in_features == 11
    states = torch.randn(6, 8)
    a = head(states, torch.zeros(6, dtype=torch.long))
    b = head(states, torch.full((6,), 3, dtype=torch.long))
    assert torch.allclose(a, b)  # every symbol embedding is zero at initialisation


def test_unknown_conditioning_is_rejected():
    """Silently degrading to `none` would make an ablation arm quietly measure nothing."""
    with pytest.raises(ValueError, match="conditioning"):
        ScoreHead(hidden_size=8, mode="regression", n_conditions=4, conditioning="per_phone")


def test_build_head_specs_rejects_unknown_conditioning(utterances):
    from tutokana.data import compute_target_stats

    with pytest.raises(ValueError, match="phone_conditioning"):
        build_head_specs(
            ("phone",), {"phone": "soft_class"}, 7, "per_phone",
            compute_target_stats(utterances),
        )


def test_build_head_specs_forces_binomial_stress(utterances):
    from tutokana.data import compute_target_stats

    specs = build_head_specs(
        levels=("phone", "word", "utterance"),
        head_modes={"phone": "soft_class", "word": "regression", "utterance": "regression"},
        n_phone_conditions=7,
        phone_conditioning="film",
        stats=compute_target_stats(utterances),
    )
    assert specs["word.stress"]["mode"] == "binomial"
    assert specs["word.stress"]["n_raters"] == 5
    assert specs["word.stress"]["support"] == (5.0, 10.0)
    assert specs["phone.accuracy"]["n_conditions"] == 7
    assert specs["phone.accuracy"]["conditioning"] == "film"
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
            native=torch.zeros(2),
        )
    }
    out = bank.read(hidden, batch)
    assert out["utterance.accuracy"].shape == (2, 1)


# --- the binomial (word stress) head --------------------------------------------------


def _stress_head(concentration: float = 8.6) -> ScoreHead:
    return ScoreHead(
        hidden_size=8, mode="binomial", support=(5.0, 10.0),
        n_raters=5, concentration=concentration,
    )


def test_binomial_head_needs_a_panel_and_a_concentration():
    with pytest.raises(ValueError, match="two-valued median scale"):
        ScoreHead(hidden_size=8, mode="binomial", n_raters=5, concentration=8.6)
    with pytest.raises(ValueError, match="n_raters"):
        ScoreHead(hidden_size=8, mode="binomial", support=(5.0, 10.0), concentration=8.6)


def test_vote_distribution_is_a_distribution():
    log_pmf = _stress_head().vote_log_pmf(torch.tensor([[-2.0], [0.0], [3.0]]))
    assert log_pmf.shape == (3, 6)  # a panel of five has six possible verdicts
    assert torch.allclose(log_pmf.logsumexp(dim=-1), torch.zeros(3), atol=1e-5)


def test_vote_distribution_matches_the_closed_form():
    """The lgamma expansion is easy to get subtly wrong, so pin it to the definition."""
    head = _stress_head(concentration=4.0)
    logits = torch.tensor([[-1.5], [0.5]])
    p = torch.sigmoid(logits.squeeze(-1))
    k = torch.arange(6.0)
    alpha, beta = 4.0 * p.unsqueeze(-1), 4.0 * (1.0 - p).unsqueeze(-1)
    expected = (
        torch.lgamma(torch.tensor(6.0)) - torch.lgamma(k + 1) - torch.lgamma(6.0 - k)
        + torch.lgamma(k + alpha) + torch.lgamma(5.0 - k + beta)
        - torch.lgamma(5.0 + alpha + beta)
        + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    )
    assert torch.allclose(head.vote_log_pmf(logits), expected, atol=1e-4)


def test_stress_readout_is_the_probability_of_a_majority():
    """The corpus publishes the panel's median, so that is what the head must report."""
    head = _stress_head()
    logits = torch.tensor([[-3.0], [0.0], [3.0]])
    native = head.predict_native(logits)
    assert ((native >= 5.0) & (native <= 10.0)).all()
    assert (native.diff() > 0).all()                      # monotone in the logit
    assert float(native[1]) == pytest.approx(7.5)         # p = 0.5 splits the panel evenly

    majority = head.vote_log_pmf(logits)[:, 3:].logsumexp(dim=-1).exp()
    assert torch.allclose(native, 5.0 + 5.0 * majority, atol=1e-5)


def test_vote_likelihood_selects_the_observed_count():
    head = _stress_head()
    logits = torch.tensor([[1.0], [1.0]])
    log_pmf = head.vote_log_pmf(logits)
    likelihood = head.vote_log_likelihood(logits, torch.tensor([5.0, 2.0]))
    assert torch.allclose(likelihood, torch.stack([log_pmf[0, 5], log_pmf[1, 2]]))


def test_a_split_panel_is_cheaper_than_a_binomial_would_make_it():
    """Why beta-binomial and not binomial.

    A unanimous dissent against a confident head is a routine event here — 0.1% of words —
    and a binomial calls it a one-in-a-billion surprise. The overdispersed likelihood keeps
    that cost finite, which is what stops a single word from owning the step.
    """
    import math

    head = _stress_head()
    logits = torch.tensor([[6.0]])
    beta_binomial = -float(head.vote_log_likelihood(logits, torch.tensor([0.0])))
    binomial = -5.0 * math.log(1.0 - torch.sigmoid(torch.tensor(6.0)).item())
    assert beta_binomial < binomial / 2


def test_the_gradient_survives_a_confidently_wrong_head():
    """A saturated logit is exactly where the loss still has to pull.

    Bounding p by clamping it to [eps, 1-eps] looks harmless and silently zeroes the
    gradient here, leaving the head stuck at whatever it was confidently wrong about.
    """
    for z in (-20.0, 0.0, 6.0, 20.0, 40.0):
        logits = torch.tensor([[z]], requires_grad=True)
        loss = -_stress_head().vote_log_likelihood(logits, torch.tensor([0.0])).sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
        if z > 0.0:
            assert float(logits.grad) > 0.5


def test_flagged_logit_reads_dissent_off_the_same_distribution():
    head = _stress_head()
    logits = torch.tensor([[-2.0], [2.0], [6.0]])
    log_pmf = head.vote_log_pmf(logits)
    expected = log_pmf[:, :-1].logsumexp(dim=-1) - log_pmf[:, -1]
    assert torch.allclose(head.flagged_logit(logits), expected, atol=1e-4)
    # More confident stress means less expected dissent.
    assert (head.flagged_logit(logits).diff() < 0).all()


# --- sibling context (the --stress-siblings control) ----------------------------------


def _word_batch(word_index, n=3):
    return HeadBatch(
        positions=torch.zeros(n, 2, dtype=torch.long),
        target=torch.zeros(n), raw=torch.zeros(n), native=torch.zeros(n),
        word_index=torch.tensor(word_index, dtype=torch.long),
    )


def _sibling_bank(utterances):
    from tutokana.data import compute_target_stats

    specs = build_head_specs(
        levels=("word",),
        head_modes={"word": "regression"},
        n_phone_conditions=0,
        phone_conditioning="none",
        stats=compute_target_stats(utterances),
        stress_siblings=True,
    )
    return HeadBank(hidden_size=16, specs=specs, levels=("word",), n_layers=1)


def test_stress_siblings_is_off_by_default(utterances):
    from tutokana.data import compute_target_stats

    specs = build_head_specs(
        levels=("word",), head_modes={"word": "regression"}, n_phone_conditions=0,
        phone_conditioning="none", stats=compute_target_stats(utterances),
    )
    assert "context_from" not in specs["word.stress"]


def test_stress_siblings_declares_its_sources(utterances):
    from tutokana.data import compute_target_stats

    specs = build_head_specs(
        levels=("word",), head_modes={"word": "regression"}, n_phone_conditions=0,
        phone_conditioning="none", stats=compute_target_stats(utterances),
        stress_siblings=True,
    )
    assert specs["word.stress"]["context_from"] == ("word.accuracy", "word.total")
    assert "context_from" not in specs["word.accuracy"]


def test_sibling_context_widens_only_the_stress_head(utterances):
    """Two extra scalars, so the control costs 2 x proj_size weights and nothing else."""
    bank = _sibling_bank(utterances)
    assert bank.head("word.stress").body[1].in_features == 16 + 2
    assert bank.head("word.accuracy").body[1].in_features == 16
    assert bank.head("word.total").body[1].in_features == 16


def test_stress_head_reads_the_sibling_readouts(utterances):
    bank = _sibling_bank(utterances)
    hidden = (torch.randn(1, 12, 16),)
    heads = {k: _word_batch([0, 1, 2]) for k in ("word.accuracy", "word.stress", "word.total")}
    out = bank.read(hidden, heads)
    assert set(out) == {"word.accuracy", "word.stress", "word.total"}
    assert out["word.stress"].shape == (3, 1)


def test_sibling_context_does_not_backpropagate_into_its_sources(utterances):
    """One-way on purpose: the stress loss must not perturb the two heads that work."""
    bank = _sibling_bank(utterances)
    hidden = (torch.randn(1, 12, 16),)
    heads = {k: _word_batch([0, 1, 2]) for k in ("word.accuracy", "word.stress", "word.total")}
    bank.read(hidden, heads)["word.stress"].sum().backward()

    assert bank.head("word.stress").body[1].weight.grad is not None
    assert bank.head("word.accuracy").body[1].weight.grad is None
    assert bank.head("word.total").body[1].weight.grad is None


def test_sibling_rows_pair_by_word_not_by_position(utterances):
    """The stress batch may omit unrated words; its context must still be those words' own."""
    bank = _sibling_bank(utterances)
    hidden = (torch.randn(1, 12, 16),)
    heads = {k: _word_batch([0, 1, 2]) for k in ("word.accuracy", "word.total")}
    heads["word.stress"] = _word_batch([2, 0], n=2)
    out = bank.read(hidden, heads)
    assert out["word.stress"].shape == (2, 1)

    context = bank.sibling_context("word.stress", heads["word.stress"], heads, out)
    for column, source in enumerate(("word.accuracy", "word.total")):
        expected = bank.head(source).predict_normalized(out[source])[[2, 0]]
        assert torch.allclose(context[:, column], expected)


def test_words_missing_from_a_source_are_refused_rather_than_paired_by_row(utterances):
    """A reader rated on a word its source is not is an error, not a silent skip."""
    bank = _sibling_bank(utterances)
    hidden = (torch.randn(1, 12, 16),)
    heads = {k: _word_batch([0, 1, 2]) for k in ("word.accuracy", "word.total")}
    heads["word.stress"] = _word_batch([0, 1, 9])
    with pytest.raises(RuntimeError, match="words that word.accuracy does not"):
        bank.read(hidden, heads)

