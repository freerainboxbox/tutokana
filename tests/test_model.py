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
from tutokana.data import compute_target_stats, phone_vocabulary
from tutokana.engine import move_batch, preflight, score
from tutokana.heads import HeadBank, build_head_specs
from tutokana.losses import CorrelationBuffer, LossConfig, build_reweighter, composite_loss
from tutokana.model import LORA_TARGET_MODULES, ModelConfig, RegisterDelta, TutokanaModel
from tutokana.prompting import TargetSpec
from tutokana.tokens import register_tokens

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
