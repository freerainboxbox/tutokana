"""The register vocabulary must be atomic, stable, and free."""

from __future__ import annotations

from tutokana.tokens import (
    ALL_REGISTERS,
    REGISTERS_BY_LEVEL,
    SCORE_REGISTERS,
    register_by_name,
    register_tokens,
    score_register_ids,
)


def test_registers_are_distinct():
    tokens = [r.token for r in ALL_REGISTERS]
    names = [r.name for r in ALL_REGISTERS]
    assert len(set(tokens)) == len(tokens)
    assert len(set(names)) == len(names)


def test_unused8_is_avoided():
    """<unused8> is token id 14, a known llama.cpp degenerate-loop trigger for Gemma 4."""
    assert "<unused8>" not in {r.token for r in ALL_REGISTERS}


def test_utterance_total_is_last():
    """The total register must follow the others so it can attend to them."""
    assert REGISTERS_BY_LEVEL["utterance"][-1].field == "total"


def test_register_by_name_roundtrip():
    for reg in ALL_REGISTERS:
        assert register_by_name(reg.name) is reg


def test_registration_is_atomic_and_free(tokenizer):
    before = len(tokenizer)
    ids = register_tokens(tokenizer)

    # The whole design rests on these two: the rows already exist, so no embedding resize.
    assert len(tokenizer) == before
    assert len(set(ids.values())) == len(ids)

    for reg in ALL_REGISTERS:
        assert tokenizer.encode(reg.token, add_special_tokens=False) == [ids[reg.name]]


def test_registration_is_idempotent(tokenizer):
    assert register_tokens(tokenizer) == register_tokens(tokenizer)


def test_registers_survive_surrounding_text(tokenizer):
    ids = register_tokens(tokenizer)
    text = "CALL<unused10>K<unused1> AO0<unused1><unused11><unused2>"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert encoded.count(ids["phone_accuracy"]) == 2
    assert encoded.count(ids["sep_phones"]) == 1
    assert encoded.count(ids["word_accuracy"]) == 1


def test_score_register_ids_filters_levels(tokenizer):
    ids = register_tokens(tokenizer)
    only_phone = score_register_ids(ids, levels=("phone",))
    assert {r.level for r in only_phone.values()} == {"phone"}
    assert len(score_register_ids(ids)) == len(SCORE_REGISTERS)
