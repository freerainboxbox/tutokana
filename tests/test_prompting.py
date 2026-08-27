"""The prompt contract: training text must be the eval prompt plus the target, exactly.

The predecessor's most expensive bug was a drift between these two. Training rendered
assistant turns one way while evaluation decoded from the generation prompt, so the empty
thought block the chat template appends sat immediately before the first generated score —
a prefix the model had never seen. That is now a test rather than a convention.
"""

from __future__ import annotations

import pytest

from tutokana.prompting import (
    TargetSpec,
    parse_generated,
    render_prompt,
    render_target,
    render_training_text,
)
from tutokana.tokens import PHONE_REGISTERS, TASK, UTTERANCE_REGISTERS, WORD_REGISTERS


def test_training_text_is_prompt_plus_target(tokenizer, utterance):
    class _Shim:  # apply_chat_template lives on the tokenizer for the bare-tokenizer case
        tokenizer = None

    spec = TargetSpec()
    prompt = render_prompt(tokenizer, utterance.text)
    full = render_training_text(tokenizer, utterance, spec)
    target = render_target(utterance, spec)

    assert full.text.startswith(prompt)
    assert full.text == prompt + target.text
    assert full.slots == target.slots


def test_prompt_ends_with_the_empty_thought_block(tokenizer, utterance):
    assert render_prompt(tokenizer, utterance.text).endswith("<|channel>thought\n<channel|>")


def test_slot_count_matches_the_labels(utterance):
    target = render_target(utterance, TargetSpec())
    n_phones = sum(len(w.phones) for w in utterance.words)
    by_level = {}
    for slot in target.slots:
        by_level.setdefault(slot.level, []).append(slot)

    assert len(by_level["phone"]) == n_phones * len(PHONE_REGISTERS)
    assert len(by_level["word"]) == len(utterance.words) * len(WORD_REGISTERS)
    assert len(by_level["utterance"]) == len(UTTERANCE_REGISTERS)


def test_hierarchy_ordering(utterance):
    """Phones, then that word's registers, then the utterance registers last."""
    target = render_target(utterance, TargetSpec())
    levels = [slot.level for slot in target.slots]

    # Every utterance slot comes after every word slot, which comes after its own phones.
    assert levels[-len(UTTERANCE_REGISTERS) :] == ["utterance"] * len(UTTERANCE_REGISTERS)
    first_word = levels.index("word")
    assert set(levels[:first_word]) == {"phone"}

    text = target.text
    for word_index, word in enumerate(utterance.words):
        word_register = text.index(WORD_REGISTERS[0].token, text.index(word.text))
        for phone in word.phones:
            assert text.index(phone, text.index(word.text)) < word_register


def test_completeness_is_not_a_register(utterance):
    target = render_target(utterance, TargetSpec())
    assert "completeness" not in {slot.field for slot in target.slots}


def test_levels_control_which_registers_appear(utterance):
    target = render_target(utterance, TargetSpec(levels=("utterance",), emit_phones=False))
    assert {slot.level for slot in target.slots} == {"utterance"}
    assert PHONE_REGISTERS[0].token not in target.text
    assert WORD_REGISTERS[0].token not in target.text
    # The word transcript survives; only the scoring registers are dropped.
    assert "WE" in target.text and "CALL" in target.text


def test_phone_registers_require_phones():
    with pytest.raises(ValueError):
        TargetSpec(levels=("phone",), emit_phones=False)


def test_parse_generated_recovers_the_transcript(utterance):
    target = render_target(utterance, TargetSpec())
    parsed = parse_generated(target.text)
    assert parsed == [(w.text, list(w.phones)) for w in utterance.words]


def test_parse_generated_ignores_the_utterance_register_line(utterance):
    target = render_target(utterance, TargetSpec())
    assert all(word not in {r.token for r in UTTERANCE_REGISTERS} for word, _ in
               parse_generated(target.text))
    assert TASK.token not in target.text.split("\n", 1)[1]
