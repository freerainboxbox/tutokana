"""End-to-end wiring, on a tiny randomly-initialised Gemma 4 Unified.

The architecture is taken from the real config with the layer count and widths shrunk, so
this exercises the actual model class — the multimodal audio scatter, the hidden-state
stack, PEFT's rewrapping — rather than a stand-in. No weights are downloaded and nothing
here measures quality; the correlations a random four-sample model produces are noise. What
it checks is that the parts are connected, which is exactly the class of bug that is
otherwise discovered thirty hours into a run.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tutokana.collate import Collator
from tutokana.data import FIELD_RANGE
from tutokana.data import compute_target_stats, phone_vocabulary
from tutokana.engine import emitted_row, move_batch, preflight, score, score_emitted
from tutokana.heads import HeadBank, build_head_specs
from tutokana.losses import CorrelationBuffer, LossConfig, build_reweighter, composite_loss
from tutokana.model import LORA_TARGET_MODULES, ModelConfig, RegisterDelta, TutokanaModel
from tutokana.prompting import TargetSpec
from tutokana.tokens import register_grid_tokens, register_tokens

HIDDEN = 64
CPU = torch.device("cpu")


def _snapshot() -> Path | None:
    pattern = Path.home() / ".cache/huggingface/hub/models--*gemma-4-12B-it*/snapshots/*"
    for path in sorted(glob.glob(str(pattern))):
        if (Path(path) / "config.json").exists() and (Path(path) / "tokenizer.json").exists():
            return Path(path)
    return None


@pytest.fixture
def tiny_base():
    """The real architecture at toy scale, built from the cached config.

    Function-scoped on purpose: PEFT warns (rightly) when the same module is wrapped twice,
    and a shared base would let one test's optimizer steps leak into the next.
    """
    snapshot = _snapshot()
    if snapshot is None:
        pytest.skip("no local Gemma 4 config snapshot")
    from transformers.models.gemma4_unified import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedForConditionalGeneration,
    )

    blob = json.loads((snapshot / "config.json").read_text())
    blob["text_config"].update(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    blob["vision_config"].update(mm_embed_dim=HIDDEN, output_proj_dims=HIDDEN)
    blob.pop("architectures", None)
    blob.pop("unsloth_fixed", None)
    torch.manual_seed(0)
    return Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(**blob)).to(torch.float32)


@pytest.fixture(scope="module")
def processor():
    snapshot = _snapshot()
    if snapshot is None:
        pytest.skip("no local Gemma 4 processor snapshot")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(snapshot))
    register_tokens(processor.tokenizer)
    return processor


@pytest.fixture
def wired(tiny_base, processor, utterances):
    from peft import LoraConfig, get_peft_model

    sample = utterances[:4]
    stats = compute_target_stats(sample)
    phones = phone_vocabulary(sample)
    spec = TargetSpec()
    specs = build_head_specs(
        spec.levels,
        {"phone": "soft_class", "word": "regression", "utterance": "regression"},
        len(phones) + 1,
        "film",
        stats,
    )
    base = get_peft_model(
        tiny_base,
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=list(LORA_TARGET_MODULES),
        ),
    )
    model = TutokanaModel(
        base,
        HeadBank(HIDDEN, specs, spec.levels, n_layers=3),
        RegisterDelta(register_tokens(processor.tokenizer), HIDDEN),
        ModelConfig(head_layers=3),
    )
    collator = Collator(processor, spec, stats, phones, max_length=4096)
    return model, collator, sample


def test_preflight_passes_on_a_wired_model(wired):
    model, collator, sample = wired
    preflight(model, collator, sample[:2], CPU, LossConfig(), None)


def test_register_delta_reaches_the_forward_pass(wired):
    """If the hook were not attached the delta would train forever with no effect."""
    model, collator, sample = wired
    batch = move_batch(collator(sample[:2]), CPU)
    model.eval()
    with torch.no_grad():
        before, _ = model(batch)
        model.register_delta.delta.add_(1.0)
        after, _ = model(batch)
        model.register_delta.delta.sub_(1.0)
    assert before.keys() == after.keys()
    for key in before:
        assert (before[key] - after[key]).abs().max() > 1e-6, key


def test_every_head_is_populated_and_the_loss_descends(wired):
    model, collator, sample = wired
    config = LossConfig(lambda_ccc=0.5, lambda_lm=0.5)
    reweighter = build_reweighter(sample, config)
    buffer = CorrelationBuffer(64)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    model.train()

    trajectory = []
    for _ in range(3):
        batch = move_batch(collator(sample), CPU)
        head_outputs, lm_loss = model(batch)
        assert set(head_outputs) == set(batch["heads"])
        loss, parts = composite_loss(
            model.heads, head_outputs, batch["heads"], config, buffer, reweighter, lm_loss
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trajectory.append(parts["loss/total"])

    assert trajectory[-1] < trajectory[0]
    assert "loss/lm" in parts and "ccc/phone.accuracy" in parts


def test_scoring_produces_every_reported_field(wired):
    model, collator, sample = wired
    result = score(model, collator, sample, CPU, batch_size=2, bootstrap=False)
    assert set(result.metrics) == {
        "phone.accuracy", "word.accuracy", "word.stress", "word.total",
        "utterance.accuracy", "utterance.prosodic", "utterance.fluency", "utterance.total",
    }
    # Stress is reported on the corpus's 5-10 scale, not the binary one it is trained on.
    assert result.gold["word.stress"].min() >= 5.0
    assert result.metrics["phone.accuracy"].n == sum(
        len(w.phones) for u in sample for w in u.words
    )


def test_trained_parts_round_trip(wired, tmp_path):
    model, _, _ = wired
    with torch.no_grad():
        model.register_delta.delta.normal_()
    model.save(tmp_path)
    assert (tmp_path / "heads.safetensors").exists() and (tmp_path / "adapter").is_dir()

    from copy import deepcopy

    reloaded = deepcopy(model)
    with torch.no_grad():
        reloaded.register_delta.delta.zero_()
        for parameter in reloaded.heads.parameters():
            parameter.zero_()
    reloaded.load_trained(tmp_path)

    assert torch.allclose(reloaded.register_delta.delta, model.register_delta.delta)
    original = model.heads.state_dict()
    for key, value in reloaded.heads.state_dict().items():
        assert torch.allclose(value, original[key]), key


def test_scoring_skips_the_language_model_loss(wired):
    """The memory fix. Asking for `labels` materialises logits over the whole vocabulary —
    5.5 GB in fp32 at batch 8, sequence 700, scaling linearly with both — and scoring throws
    that loss away. It was the largest allocation in an eval pass and existed to be discarded.
    """
    model, collator, sample = wired
    batch = move_batch(collator(sample[:2]), CPU)
    model.eval()
    with torch.no_grad():
        trained_heads, lm_loss = model(batch, with_lm_loss=True)
        scored_heads, no_loss = model(batch, with_lm_loss=False)

    assert lm_loss is not None, "training still needs the text objective"
    assert no_loss is None, "scoring must not compute a loss it discards"

    # Skipping the loss must not change a single prediction: the heads read hidden states,
    # which do not depend on `labels`.
    assert trained_heads.keys() == scored_heads.keys()
    for key in trained_heads:
        torch.testing.assert_close(trained_heads[key], scored_heads[key])


def test_score_reports_how_much_it_scored(wired):
    """`scored`/`partial` let an interrupted eval label its own table honestly."""
    model, collator, sample = wired
    result = score(model, collator, sample, CPU, batch_size=2, bootstrap=False)
    assert result.scored == len(sample)
    assert result.partial is False


def test_score_stops_early_and_returns_what_it_has(wired):
    """Ctrl-C during a long eval should yield a partial table, not lose everything."""
    model, collator, sample = wired
    tripped = {"requested": True}  # as if the signal arrived before the first batch finished
    result = score(model, collator, sample, CPU, batch_size=1, bootstrap=False,
                   interrupt=tripped)
    assert result.partial is True
    assert 0 < result.scored < len(sample)
    assert result.metrics, "a partial pass must still produce metrics"


# --- emitted delta: reaching the logits under tied weights -----------------------------


def _emitted_model(tiny_base, processor):
    """The same tiny base with the output-side delta switched on."""
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)
    delta = RegisterDelta({**register_ids, **grid_ids}, HIDDEN)
    model = TutokanaModel(
        tiny_base,
        HeadBank(HIDDEN, {}, ("phone",), n_layers=1),
        delta,
        ModelConfig(head_layers=1, emitted_delta=True),
    )
    return model, delta, grid_ids


def test_emitted_delta_reaches_the_logits(tiny_base, processor):
    """Tied weights mean an input-only delta leaves the output row stale.

    The model emits score tokens under RAIL/RAFT, so `h . (W[i] + delta[i])` is the logit it
    should get. Without the output hook the delta trains and never moves a logit.
    """
    model, delta, grid_ids = _emitted_model(tiny_base, processor)
    emb = model.base.get_input_embeddings()
    lm = model.base.get_output_embeddings()

    target = grid_ids["2.0"]
    with torch.no_grad():
        delta.delta.zero_()
        delta.delta[-1] = 0.0
        # Put the delta on the row that owns `target`, then compare against the reference.
        row = delta.token_ids.index(target)
        delta.delta[row] = torch.ones(HIDDEN)
        tok = torch.tensor([[2, target]])
        hidden = emb(tok).detach()
        logits = lm(emb(tok)).detach()

        weight = emb.weight.detach().clone()
        weight[target] = weight[target] + delta.delta[row].detach()
        assert torch.allclose(logits, hidden @ weight.T, atol=1e-5)


def test_emitted_delta_leaves_other_columns_alone(tiny_base, processor):
    model, delta, grid_ids = _emitted_model(tiny_base, processor)
    emb = model.base.get_input_embeddings()
    lm = model.base.get_output_embeddings()
    # Columns outside the delta set, so the hook must not touch them. Registers occupy the
    # low ids, so picking 7 or 8 here would assert the opposite of what is true.
    ours = set(delta.token_ids)
    outside = [i for i in (0, 1, 2, 3, 4, 5, 40, 41) if i not in ours][:3]
    assert outside, "no vocabulary id outside the delta set to test against"
    with torch.no_grad():
        delta.delta.normal_()
        tok = torch.tensor([[2, grid_ids["1.0"], 40, 41]])
        logits = lm(emb(tok)).detach()
        plain = emb(tok).detach() @ emb.weight.detach().T
        for column in outside:
            assert torch.allclose(logits[..., column], plain[..., column], atol=1e-6)


def test_emitted_delta_gives_gradient(tiny_base, processor):
    """A delta that reaches the logits must also receive gradient through them."""
    model, delta, grid_ids = _emitted_model(tiny_base, processor)
    emb = model.base.get_input_embeddings()
    lm = model.base.get_output_embeddings()
    tok = torch.tensor([[2, grid_ids["0.2"]]])
    lm(emb(tok)).sum().backward()
    assert delta.delta.grad is not None
    assert float(delta.delta.grad.abs().sum()) > 0.0


def test_head_path_leaves_logits_untouched(tiny_base, processor):
    """Default config keeps the regression-head path exactly as it was."""
    register_ids = register_tokens(processor.tokenizer)
    delta = RegisterDelta(register_ids, HIDDEN)
    model = TutokanaModel(
        tiny_base,
        HeadBank(HIDDEN, {}, ("phone",), n_layers=1),
        delta,
        ModelConfig(head_layers=1),
    )
    assert model.config.emitted_delta is False
    assert model._delta_output_handle is None
    emb = model.base.get_input_embeddings()
    lm = model.base.get_output_embeddings()
    with torch.no_grad():
        delta.delta.normal_()
        tok = torch.tensor([[2, register_ids["word_stress"]]])
        logits = lm(emb(tok))
        assert torch.allclose(logits, emb(tok).detach() @ emb.weight.detach().T, atol=1e-6)


def test_digit_seed_makes_the_row_read_as_a_number(processor):
    """The delta is added to the slot's own base row, so the seed must cancel that row."""
    from tutokana.model import digit_seed_rows

    weight = torch.randn(262144, 16)
    grid_ids = register_grid_tokens(processor.tokenizer)
    seeds = digit_seed_rows(processor.tokenizer, weight, grid_ids)

    assert set(seeds) == set(grid_ids.values())
    for text, token_id in grid_ids.items():
        parts = processor.tokenizer.encode(text, add_special_tokens=False)
        want = weight[torch.tensor(parts)].mean(dim=0)
        assert torch.allclose(weight[token_id] + seeds[token_id], want, atol=1e-6)


def test_digit_seed_excludes_registers(processor):
    """A register has no textual meaning to inherit; zero is the right start."""
    from tutokana.model import digit_seed_rows

    weight = torch.randn(262144, 16)
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)
    seeds = digit_seed_rows(processor.tokenizer, weight, grid_ids)
    assert not (set(seeds) & set(register_ids.values()))


def test_init_rows_follow_sorted_ids(processor):
    """Rows are ordered by id, not by dict insertion: registers and grid interleave."""
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)
    ordered = sorted(set(register_ids.values()) | set(grid_ids.values()))
    init = torch.stack([torch.full((HIDDEN,), float(i)) for i in range(len(ordered))])
    delta = RegisterDelta({**register_ids, **grid_ids}, HIDDEN, init=init)
    assert delta.token_ids == ordered
    for row, token_id in enumerate(ordered):
        assert torch.allclose(delta.delta[row], init[row])


# --- delta checkpoints are keyed by vocabulary id, not by row index ---------------------


def test_delta_rows_are_id_keyed_across_token_sets(processor):
    """A regression-head checkpoint (11 rows) must warm-start a variant (11 + 22 rows)."""
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)

    source = RegisterDelta(register_ids, HIDDEN)
    with torch.no_grad():
        source.delta.normal_()
    state = source.state_dict()
    assert state["_token_ids"].tolist() == sorted(register_ids.values())

    target = RegisterDelta({**register_ids, **grid_ids}, HIDDEN)
    with torch.no_grad():  # the digit seed is what the grid rows must keep
        target.delta.normal_()
        seeded = target.delta.clone()

    target.merge_state(state["delta"], state["_token_ids"])
    for name, token_id in register_ids.items():
        row = target.token_ids.index(token_id)
        src = source.token_ids.index(token_id)
        assert torch.allclose(target.delta[row], source.delta[src])
    for token_id in grid_ids.values():
        row = target.token_ids.index(token_id)
        assert torch.allclose(target.delta[row], seeded[row]), "grid rows kept their seed"


def test_legacy_delta_without_id_list_still_loads(processor):
    """Files written before `token_ids` was persisted held exactly the 11 registers."""
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)

    source = RegisterDelta(register_ids, HIDDEN)
    with torch.no_grad():
        source.delta.normal_()
    legacy = {"delta": source.delta.detach().clone()}  # no token_ids key

    target = RegisterDelta({**register_ids, **grid_ids}, HIDDEN)
    target.merge_state(legacy["delta"], legacy.get("_token_ids"))
    for token_id in register_ids.values():
        row = target.token_ids.index(token_id)
        src = source.token_ids.index(token_id)
        assert torch.allclose(target.delta[row], source.delta[src])


def test_legacy_delta_with_unknown_rows_is_rejected(processor):
    """Guessing which 13 rows meant what would silently misplace trained weights."""
    register_ids = register_tokens(processor.tokenizer)
    delta = RegisterDelta({**register_ids, **register_grid_tokens(processor.tokenizer)}, HIDDEN)
    with pytest.raises(RuntimeError, match="cannot be attributed"):
        delta.merge_state(torch.randn(13, HIDDEN), None)


def test_save_and_load_round_trip_preserves_ids(tiny_base, processor, tmp_path):
    """The persisted id list is what makes the two token sets interoperable."""
    register_ids = register_tokens(processor.tokenizer)
    grid_ids = register_grid_tokens(processor.tokenizer)
    delta = RegisterDelta({**register_ids, **grid_ids}, HIDDEN)
    with torch.no_grad():
        delta.delta.normal_()

    from safetensors.torch import save_file, load_file

    blob = {f"register_delta.{k}": v.detach().cpu().contiguous()
            for k, v in delta.state_dict().items()}
    save_file(blob, str(tmp_path / "heads.safetensors"))
    reloaded = load_file(str(tmp_path / "heads.safetensors"))

    assert reloaded["register_delta._token_ids"].tolist() == delta.token_ids
    other = RegisterDelta(register_ids, HIDDEN)
    other.merge_state(reloaded["register_delta.delta"], reloaded["register_delta._token_ids"])
    for token_id in register_ids.values():
        assert torch.allclose(
            other.delta[other.token_ids.index(token_id)],
            delta.delta[delta.token_ids.index(token_id)],
        )


# --- the emitted path: read the grid, feed the token back -------------------------------


@pytest.fixture
def emitted(tiny_base, processor, utterances):
    """The same tiny base with the scores in the text and no regression heads."""
    from peft import LoraConfig, get_peft_model

    from tutokana.tokens import register_grid_tokens

    sample = utterances[:4]
    stats = compute_target_stats(sample)
    phones = phone_vocabulary(sample)
    spec = TargetSpec(emit_scores=True)
    tokenizer = processor.tokenizer
    register_ids = register_tokens(tokenizer)
    grid_ids = register_grid_tokens(tokenizer)
    base = get_peft_model(
        tiny_base,
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=list(LORA_TARGET_MODULES),
        ),
    )
    model = TutokanaModel(
        base,
        HeadBank(HIDDEN, {}, spec.levels, n_layers=3),
        RegisterDelta({**register_ids, **grid_ids}, HIDDEN),
        ModelConfig(head_layers=3, emitted_delta=True),
    )
    collator = Collator(
        processor, spec, stats, phones, max_length=4096,
        score_ce=True, grid_ids=grid_ids,
    )
    return model, collator, sample


def test_emitted_scoring_returns_every_field(emitted, utterances):
    """Same keys, same gold, same ScoreResult: the reporting side must not notice a change."""
    model, collator, sample = emitted
    result = score_emitted(model, collator, sample, CPU, bootstrap=False)
    assert set(result.metrics) == set(result.predictions)
    assert result.predictions, "the emitted path produced nothing"
    for key, prediction in result.predictions.items():
        assert len(prediction) == len(result.gold[key]), key
        assert np.isfinite(prediction).all(), key


def test_emitted_gold_is_the_collators_native_scale(emitted):
    """The reported gold must be the published scale, not the grid the model was trained on.

    For stress those two differ (5/10 median against 0-5 votes), and a table that quietly
    compared a vote-count prediction to a median label would be unreadable rather than wrong.
    """
    model, collator, sample = emitted
    result = score_emitted(model, collator, sample, CPU, bootstrap=False)
    batch = move_batch(collator(sample), CPU)
    for key, gold in result.gold.items():
        assert np.allclose(gold, batch["heads"][key].native.numpy()), key
    stress = result.gold["word.stress"]
    assert set(np.unique(stress)) <= {5.0, 10.0}, "stress gold must be the median scale"


VOCAB = 1024  # the tiny model's vocab; only the grid columns are read


def test_predictions_land_on_the_reported_scale(emitted):
    """A phone score read off the decimal grid must sit in 0.0-2.0, not in votes."""
    model, collator, sample = emitted
    result = score_emitted(model, collator, sample, CPU, bootstrap=False)
    for key, prediction in result.predictions.items():
        low, high = FIELD_RANGE[tuple(key.split(".", 1))]
        assert prediction.min() >= low - 1e-4, (key, prediction.min())
        assert prediction.max() <= high + 1e-4, (key, prediction.max())


def test_reinjection_changes_the_readout(emitted):
    """The whole point of the loop.

    If feeding the snapped token back did nothing, reading every score from one teacher-forced
    pass would be equivalent, cheaper, and the model would still be a feature extractor with a
    regressor bolted on. So: bias one phone score token, and every score read *after* it must
    move, because the token written back into the stream changed.
    """
    from tutokana.rail import grid_supports

    model, collator, sample = emitted
    batch = move_batch(collator(sample[:1]), CPU)
    supports = grid_supports(collator.grid_ids)
    injected = emitted_row(model, batch, supports)

    support = supports["phone.accuracy"]
    row = model.register_delta.token_ids.index(int(support.token_ids[0]))
    original = model.register_delta.delta.clone()
    with torch.no_grad():
        # A delta of zero is the default here, so *zeroing* a row changes nothing. Push it.
        model.register_delta.delta[row] = 40.0
    changed = emitted_row(model, batch, supports)
    with torch.no_grad():
        model.register_delta.delta.copy_(original)

    moved = [
        key for key in injected
        if not torch.allclose(injected[key], changed[key], atol=1e-5)
    ]
    assert moved, "changing a score\'s logits changed no later readout: nothing was reinjected"
    # The score read before the perturbed one is untouched, which is what "causal" means.
    assert "utterance.total" in moved, "the last score did not see the reinjected token"


def test_emitted_row_is_deterministic(emitted):
    from tutokana.rail import grid_supports

    model, collator, sample = emitted
    batch = move_batch(collator(sample[:1]), CPU)
    supports = grid_supports(collator.grid_ids)
    first = emitted_row(model, batch, supports)
    second = emitted_row(model, batch, supports)
    for key in first:
        assert torch.allclose(first[key], second[key]), key


def test_expectation_is_not_the_sampled_token(emitted):
    """The reported number is the expectation; the reinjected token is its nearest rendering."""
    from tutokana.rail import grid_supports

    support = grid_supports(emitted[1].grid_ids)["phone.accuracy"]
    torch.manual_seed(0)
    logits = torch.randn(1, VOCAB) * 3.0
    expectation = support.expectation(logits)
    snapped = int(support.snap(expectation)[0])
    assert snapped in support.token_ids.tolist()
    # A skewed distribution puts the expectation between two grid points, so the token the
    # loop writes back is a rendering of the number, not the number the metrics report.
    skewed = torch.full((1, VOCAB), -20.0)
    for token, value in zip(support.token_ids.tolist(), support.values.tolist()):
        skewed[0, token] = -abs(value - 1.3)
    assert float(support.expectation(skewed)) != float(
        support.values[support.token_ids.tolist().index(int(support.snap(support.expectation(skewed))))]
    )


def test_stress_reports_the_median_scale_not_the_vote_count(emitted):
    """Grid: 0-5 votes. Reported: the panel's 5/10 median, mapped through P(votes >= 3)."""
    from tutokana.rail import grid_supports

    supports = grid_supports(emitted[1].grid_ids)
    support = supports["word.stress"]
    assert support.values.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    all_wrong = torch.full((1, VOCAB), -60.0)
    all_wrong[0, int(support.token_ids[0])] = 20.0
    assert float(support.report(all_wrong)) == pytest.approx(5.0, abs=1e-4)

    all_right = torch.full((1, VOCAB), -60.0)
    all_right[0, int(support.token_ids[-1])] = 20.0
    assert float(support.report(all_right)) == pytest.approx(10.0, abs=1e-4)

    # The expectation, by contrast, is a vote count.
    assert float(support.expectation(all_wrong)) == pytest.approx(0.0, abs=1e-4)
    assert float(support.expectation(all_right)) == pytest.approx(5.0, abs=1e-4)


def test_probabilities_are_renormalised_onto_the_grid(emitted):
    """Mass sitting on the rest of the vocabulary is not part of the score."""
    from tutokana.rail import grid_supports

    supports = grid_supports(emitted[1].grid_ids)
    support = supports["word.total"]
    logits = torch.full((1, VOCAB), -60.0)
    logits[0, support.token_ids.tolist()] = 2.0
    assert float(support.probabilities(logits).sum()) == pytest.approx(1.0, abs=1e-6)


def test_preflight_passes_on_the_emitted_path(emitted):
    """The preflight is the last thing between a flag and thirty hours of nothing."""
    model, collator, sample = emitted
    preflight(model, collator, sample[:2], CPU, LossConfig(), None)


def test_ce_rail_loss_is_the_text_loss_alone(emitted):
    """No heads, so the composite objective is cross-entropy on the emitted text."""
    model, collator, sample = emitted
    batch = move_batch(collator(sample[:2]), CPU)
    head_outputs, lm_loss = model(batch)
    assert head_outputs == {}, "the emitted path must not build heads"
    total, parts = composite_loss(
        model.heads, head_outputs, batch["heads"], LossConfig(lambda_lm=0.5), None, None, lm_loss
    )
    assert set(parts) == {"loss/lm", "loss/total"}
    assert float(total.detach()) == pytest.approx(
        0.5 * float(lm_loss.detach()), rel=1e-5
    )


def test_ce_rail_score_positions_carry_the_gradient(emitted):
    """If the mask dropped them, the run would train for hours and never learn a score."""
    model, collator, sample = emitted
    batch = move_batch(collator(sample[:2]), CPU)
    grid = torch.tensor(sorted(collator.grid_ids.values()))
    supervised = torch.isin(batch["input_ids"], grid) & (batch["labels"] != -100)
    assert int(supervised.sum()) > 0, "no score token is supervised"

    _, lm_loss = model(batch)
    lm_loss.backward()
    # The delta rows for the grid are the only trainable thing holding the score tokens'
    # output embeddings, so they must be the ones moving.
    assert model.register_delta.delta.grad is not None
    assert float(model.register_delta.delta.grad.abs().sum()) > 0.0
