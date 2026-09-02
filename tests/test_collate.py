"""Collation: register positions, per-head targets, and the loss mask.

These use a stub processor that wraps the real Gemma 4 tokenizer and expands `<|audio|>`
the way the real one does. That keeps the test on the parts that can actually go wrong —
locating registers in a padded batch and masking the language-model loss — without needing
the feature extractor or any model weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tutokana.collate import IGNORE_INDEX, Collator, head_key
from tutokana.data import compute_target_stats, phone_vocabulary
from tutokana.prompting import TargetSpec

AUDIO_TOKENS_PER_SAMPLE = 4


class StubProcessor:
    """Tokenizer-only stand-in: expands the audio placeholder, pads, returns tensors."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text, audio=None, sampling_rate=None, return_tensors=None,
                 padding=True, **kwargs):
        expanded = [
            t.replace("<|audio|>", "<|audio>" + "<|audio|>" * AUDIO_TOKENS_PER_SAMPLE + "<audio|>")
            for t in text
        ]
        batch = self.tokenizer(
            expanded, add_special_tokens=False, padding=padding, return_tensors="pt"
        )
        batch["input_features"] = torch.zeros(len(text), AUDIO_TOKENS_PER_SAMPLE, 640)
        batch["input_features_mask"] = torch.ones(
            len(text), AUDIO_TOKENS_PER_SAMPLE, dtype=torch.bool
        )
        return batch


@pytest.fixture
def collator(tokenizer, utterances):
    from tutokana.tokens import register_tokens

    register_tokens(tokenizer)
    tokenizer.padding_side = "right"
    return Collator(
        StubProcessor(tokenizer),
        TargetSpec(),
        compute_target_stats(utterances),
        phone_vocabulary(utterances),
        max_length=4096,
    )


def test_every_recorded_position_holds_its_own_register(collator, utterances):
    batch = collator(utterances[:3])
    ids = batch["input_ids"]
    for key, head_batch in batch["heads"].items():
        expected = collator.register_ids[key.replace(".", "_")]
        for b, position in head_batch.positions.tolist():
            assert int(ids[b, position]) == expected


def test_label_counts_match_the_corpus(collator, utterances):
    chunk = utterances[:3]
    batch = collator(chunk)
    n_words = sum(len(u.words) for u in chunk)
    n_phones = sum(len(w.phones) for u in chunk for w in u.words)

    assert len(batch["heads"][head_key("phone", "accuracy")]) == n_phones
    assert len(batch["heads"][head_key("word", "accuracy")]) == n_words
    assert len(batch["heads"][head_key("utterance", "total")]) == len(chunk)


def test_targets_carry_the_right_values(collator, utterance):
    batch = collator([utterance])
    phone = batch["heads"][head_key("phone", "accuracy")]
    expected = [a for w in utterance.words for a in w.phone_accuracy]
    assert phone.raw.tolist() == pytest.approx(expected)

    stress = batch["heads"][head_key("word", "stress")]
    # `raw` is what the head is trained on, `native` what the metrics report.
    assert stress.raw.tolist() == [float(w.stress_votes) for w in utterance.words]
    assert stress.native.tolist() == [w.stress for w in utterance.words]

    accuracy = batch["heads"][head_key("utterance", "accuracy")]
    assert accuracy.raw.tolist() == [utterance.accuracy]


def test_targets_are_z_scored(collator, utterance):
    batch = collator([utterance])
    key = head_key("phone", "accuracy")
    head_batch = batch["heads"][key]
    expected = (head_batch.native - collator.stats.mean[key]) / collator.stats.std[key]
    assert torch.allclose(head_batch.target, expected)


def test_phone_conditioning_indices_follow_the_phones(collator, utterance):
    batch = collator([utterance])
    phone = batch["heads"][head_key("phone", "accuracy")]
    expected = [
        collator.phone_to_id.get(p, 0) for w in utterance.words for p in w.phones
    ]
    assert phone.phone_id.tolist() == expected


def test_unknown_phones_fall_back_to_the_shared_row(tokenizer, utterances):
    from tutokana.tokens import register_tokens

    register_tokens(tokenizer)
    collator = Collator(
        StubProcessor(tokenizer),
        TargetSpec(),
        compute_target_stats(utterances),
        ("W",),  # a deliberately tiny vocabulary
        max_length=4096,
    )
    batch = collator(utterances[:1])
    assert 0 in batch["heads"][head_key("phone", "accuracy")].phone_id.tolist()


def test_registers_are_excluded_from_the_language_loss(collator, utterances):
    batch = collator(utterances[:2])
    labels, ids = batch["labels"], batch["input_ids"]
    register_ids = torch.tensor(sorted(collator.register_ids.values()))
    assert (labels[torch.isin(ids, register_ids)] == IGNORE_INDEX).all()


def test_audio_and_prompt_are_excluded_from_the_language_loss(collator, utterances):
    """Supervising audio placeholder positions produces a model that cannot hear."""
    batch = collator(utterances[:2])
    labels, ids = batch["labels"], batch["input_ids"]
    audio_id = collator.tokenizer.convert_tokens_to_ids("<|audio|>")
    assert (labels[ids == audio_id] == IGNORE_INDEX).all()

    boundary = (ids[0] == collator.end_of_channel_id).nonzero()[0, 0]
    assert (labels[0, : boundary + 1] == IGNORE_INDEX).all()
    assert (labels[0, boundary + 1 :] != IGNORE_INDEX).any()


def test_padding_is_excluded_from_the_language_loss(collator, utterances):
    from dataclasses import replace

    from tutokana.data import Word

    short = replace(utterances[0], words=utterances[0].words[:1], text="WE")
    long = replace(
        utterances[1],
        words=utterances[1].words
        + (Word("BIRD", 4.0, 10.0, 5, 4.0, ("B", "ER1", "D"), (2.0, 0.4, 0.0)),),
        text="WE CALL BIRD",
    )
    batch = collator([short, long])
    labels, mask = batch["labels"], batch["attention_mask"]
    assert (mask == 0).any(), "fixture did not actually produce ragged lengths"
    assert (labels[mask == 0] == IGNORE_INDEX).all()


def test_over_length_batches_are_refused_not_truncated(tokenizer, utterances):
    from tutokana.tokens import register_tokens

    register_tokens(tokenizer)
    collator = Collator(
        StubProcessor(tokenizer),
        TargetSpec(),
        compute_target_stats(utterances),
        phone_vocabulary(utterances),
        max_length=8,
    )
    with pytest.raises(ValueError, match="max_length"):
        collator(utterances[:1])


def test_generation_mode_omits_the_target(tokenizer, utterances):
    from tutokana.tokens import register_tokens

    register_tokens(tokenizer)
    collator = Collator(
        StubProcessor(tokenizer),
        TargetSpec(),
        compute_target_stats(utterances),
        phone_vocabulary(utterances),
        max_length=4096,
        with_target=False,
    )
    batch = collator(utterances[:1])
    assert "heads" not in batch and "labels" not in batch
    register_ids = torch.tensor(sorted(collator.register_ids.values()))
    assert not torch.isin(batch["input_ids"], register_ids).any()


def test_levels_restrict_which_heads_are_supervised(tokenizer, utterances):
    from tutokana.tokens import register_tokens

    register_tokens(tokenizer)
    collator = Collator(
        StubProcessor(tokenizer),
        TargetSpec(levels=("utterance",), emit_phones=False),
        compute_target_stats(utterances),
        phone_vocabulary(utterances),
        max_length=4096,
    )
    batch = collator(utterances[:2])
    assert {k.split(".")[0] for k in batch["heads"]} == {"utterance"}


def test_word_index_pairs_the_sibling_registers(collator, utterances):
    """The three word heads must land in the same order, and it must be checkable.

    Sibling coupling reads accuracy and total for the same word off matching rows. That
    lockstep is a consequence of `render_target`'s slot order, not something it promises, so
    `word_index` records the pairing instead of leaving it to be assumed.
    """
    batch = collator(utterances)
    word_keys = [head_key("word", f) for f in ("accuracy", "stress", "total")]
    indices = [batch["heads"][k].word_index for k in word_keys]

    assert all(i is not None for i in indices)
    for other in indices[1:]:
        assert torch.equal(indices[0], other)
    # Batch-unique and dense, so a row identifies exactly one word of one utterance.
    n_words = sum(len(u.words) for u in utterances)
    assert sorted(indices[0].tolist()) == list(range(n_words))


def test_only_word_heads_carry_a_word_index(collator, utterances):
    batch = collator(utterances)
    assert batch["heads"][head_key("phone", "accuracy")].word_index is None
    assert batch["heads"][head_key("utterance", "accuracy")].word_index is None



def test_unrated_stress_registers_get_no_row(collator, utterances):
    """A negative's stress register is emitted but unsupervised: no row, hence no gradient."""
    import random

    from tutokana.mix import make_negative

    real, negative = utterances[0], make_negative(utterances[0], utterances[1], random.Random(0))
    batch = collator([real, negative])
    stress = batch["heads"][head_key("word", "stress")]
    accuracy = batch["heads"][head_key("word", "accuracy")]

    assert len(stress) == len(real.words)
    assert len(accuracy) == len(real.words) + len(negative.words)
    assert stress.positions[:, 0].tolist() == [0] * len(real.words)
    assert stress.raw.tolist() == [float(w.stress_votes) for w in real.words]
    # The stress rows are a strict subset of the accuracy rows, paired by word index.
    assert set(stress.word_index.tolist()) < set(accuracy.word_index.tolist())


def test_a_batch_of_only_negatives_has_no_stress_batch(collator, utterances):
    import random

    from tutokana.mix import make_negative

    negative = make_negative(utterances[0], utterances[1], random.Random(0))
    batch = collator([negative])
    assert head_key("word", "stress") not in batch["heads"]
    assert len(batch["heads"][head_key("word", "accuracy")]) == len(negative.words)
