"""The register vocabulary must be atomic, stable, and free.

The score grids that RAIL/RAFT emit as text must be derived from the spacing, disjoint
across rendering classes, and registered without growing the vocabulary.
"""

from __future__ import annotations

import pytest

from tutokana.tokens import (
    ALL_REGISTERS,
    GRID_STRINGS,
    REGISTERS_BY_LEVEL,
    SCORE_GRID,
    SCORE_REGISTERS,
    ScoreGrid,
    grid_for,
    register_by_name,
    register_grid_tokens,
    register_tokens,
    score_register_ids,
    slots_for,
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


# --- score grids: derived from the spacing, never listed ------------------------------


def test_spacing_decides_the_rendering():
    """A step of 1 renders integers; a step of 0.2 renders one decimal place."""
    assert ScoreGrid(spacing=1.0, low=0.0, high=2.0).decimals == 0
    assert ScoreGrid(spacing=1.0, low=0.0, high=2.0).values == (0.0, 1.0, 2.0)
    assert ScoreGrid(spacing=0.2, low=0.0, high=1.0).decimals == 1
    assert ScoreGrid(spacing=0.05, low=0.0, high=0.2).decimals == 2
    assert ScoreGrid(spacing=0.2, low=0.0, high=1.0).values == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert ScoreGrid(spacing=1.0, low=0.0, high=2.0).cls == "integer"
    assert ScoreGrid(spacing=0.2, low=0.0, high=1.0).cls == "decimal"


def test_grid_spans_the_target_space_not_the_observed_support():
    """word accuracy 4 is absent from the training split but must still be on the grid."""
    assert 4.0 in grid_for("word", "accuracy").values
    assert 0.0 in grid_for("utterance", "accuracy").values
    assert grid_for("word", "stress").high == 5.0


def test_render_is_keyed_by_score_type_not_by_float():
    """The corpus writes `2` for a phone score of 2.0; it must still render as `2.0`.

    A formatter that inspected the float would put 80.1% of phone labels on the integer
    tokens and silently break the decimal grid.
    """
    assert grid_for("phone", "accuracy").render(2.0) == "2.0"
    assert grid_for("word", "accuracy").render(2.0) == "2"
    assert grid_for("word", "stress").render(2.0) == "2"
    assert grid_for("phone", "accuracy").render(1.0) == "1.0"
    assert grid_for("word", "accuracy").render(1.0) == "1"


def test_render_snaps_and_rejects_out_of_range():
    assert grid_for("phone", "accuracy").render(1.79) == "1.8"
    assert grid_for("word", "stress").render(3.4) == "3"
    with pytest.raises(ValueError):
        grid_for("word", "stress").render(6.0)


def test_integer_and_decimal_grids_are_disjoint():
    """`1` and `1.0` are different tokens, so the two classes must never collide."""
    integers = {s for s in GRID_STRINGS if "." not in s}
    decimals = {s for s in GRID_STRINGS if "." in s}
    assert not (integers & decimals)
    assert len(set(GRID_STRINGS)) == len(GRID_STRINGS)
    assert "1" in integers and "1.0" in decimals


def test_grid_strings_are_the_derived_union():
    expected = {grid.render(v) for grid in SCORE_GRID.values() for v in grid.values}
    assert set(GRID_STRINGS) == expected


def test_every_scored_field_is_on_its_own_grid():
    """Every released label must land on its grid, so snapping is lossless on gold."""
    from conftest import make_utterance

    for u in (make_utterance(i) for i in range(6)):
        for w in u.words:
            for a in w.phone_accuracy:
                assert grid_for("phone", "accuracy").render(a) in GRID_STRINGS
            for field in ("accuracy", "total"):
                assert grid_for("word", field).render(getattr(w, field)) in GRID_STRINGS
            assert grid_for("word", "stress").render(w.stress_votes) in GRID_STRINGS
        for field in ("accuracy", "prosodic", "fluency", "total"):
            assert grid_for("utterance", field).render(getattr(u, field)) in GRID_STRINGS


def test_completeness_is_deliberately_gridless():
    """Untrained, unregistered, and off any step of 1 or 0.2 — so it gets no grid."""
    with pytest.raises(KeyError):
        grid_for("utterance", "completeness")


def test_slots_avoid_the_hazards(tokenizer):
    taken = {r.token for r in ALL_REGISTERS}
    slots = slots_for(tokenizer, len(GRID_STRINGS), taken=taken)
    assert "<unused8>" not in slots
    assert not (set(slots) & taken)
    ids = [tokenizer.convert_tokens_to_ids(s) for s in slots]
    assert all(i <= 1024 for i in ids), "RegisterDelta would size its lookup to the max id"


# --- grid registration: atomic, non-special, and free --------------------------------


def test_grid_registration_is_atomic_and_free(tokenizer):
    """The map is keyed by *value*, so the id is the only atomic handle.

    Encoding the value string `"0"` yields the ordinary digit token, not the grid slot —
    the score channel is addressed by id, never by the digit text.
    """
    before = len(tokenizer)
    register_tokens(tokenizer)
    ids = register_grid_tokens(tokenizer)

    assert len(tokenizer) == before, "the slots already exist; no embedding resize"
    assert len(set(ids.values())) == len(ids)
    assert set(ids) == set(GRID_STRINGS)
    for token_id in ids.values():
        slot = tokenizer.convert_ids_to_tokens(token_id)
        assert tokenizer.encode(slot, add_special_tokens=False) == [token_id]


def test_grid_registration_is_idempotent(tokenizer):
    register_tokens(tokenizer)
    assert register_grid_tokens(tokenizer) == register_grid_tokens(tokenizer)


def test_grid_ids_do_not_collide_with_registers(tokenizer):
    register_ids = set(register_tokens(tokenizer).values())
    assert not (set(register_grid_tokens(tokenizer).values()) & register_ids)


def test_grid_tokens_survive_skip_special_tokens(tokenizer):
    """Load-bearing: the score channel lives in the text, so decode must not delete it."""
    register_tokens(tokenizer)
    ids = register_grid_tokens(tokenizer)

    assert not (set(ids.values()) & set(tokenizer.all_special_ids))
    for token_id in ids.values():
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        assert tokenizer.encode(decoded, add_special_tokens=False) == [token_id]


# --- uniform special=False ------------------------------------------------------------


def test_every_added_token_is_non_special(tokenizer):
    """One policy for every token this package adds.

    `special` does not gate emission — generate() suppresses only `suppress_tokens` — it only
    gates deletion at decode. A split policy makes the *default* decode strip the anchors and
    keep the scores, which is a string that looks clean but cannot be parsed.
    """
    register_ids = register_tokens(tokenizer)
    grid_ids = register_grid_tokens(tokenizer)

    special = set(tokenizer.all_special_ids)
    for token_id in list(register_ids.values()) + list(grid_ids.values()):
        assert token_id not in special
        assert tokenizer.added_tokens_decoder[token_id].special is False


def test_default_decode_is_parseable(tokenizer):
    """Our tokens must survive the default decode, or a call site gets the anchors-stripped
    half by accident.

    `<turn|>` is a base-vocab token this package does not register, and it still strips by
    default. That is correct and is why the assertion is about our tokens, not string equality.
    """
    register_ids = register_tokens(tokenizer)
    grid_ids = register_grid_tokens(tokenizer)

    seq = [
        register_ids["task"],
        register_ids["sep_phones"],
        grid_ids["1.8"],
        register_ids["sep_word"],
        grid_ids["10"],
        *tokenizer.encode("<turn|>", add_special_tokens=False),
    ]
    placed = seq[: -len(tokenizer.encode("<turn|>", add_special_tokens=False))]
    default = tokenizer.decode(seq, skip_special_tokens=True)
    full = tokenizer.decode(seq, skip_special_tokens=False)

    for token_id in placed:
        assert tokenizer.convert_ids_to_tokens(token_id) in default
    # Only the model's own turn terminator differs.
    assert full.replace("<turn|>", "") == default


def test_real_specials_still_strip(tokenizer):
    """Uniform non-special must not start leaking the model's own control tokens."""
    register_tokens(tokenizer)
    register_grid_tokens(tokenizer)

    ids = tokenizer.encode("<bos><pad><eos><|audio|> WE CALL", add_special_tokens=False)
    stripped = tokenizer.decode(ids, skip_special_tokens=True)
    assert stripped.strip() == "WE CALL"


def test_strip_anchors_is_for_display_only(tokenizer):
    from tutokana.tokens import strip_anchors

    register_ids = register_tokens(tokenizer)
    grid_ids = register_grid_tokens(tokenizer)
    text = tokenizer.decode(
        [register_ids["task"], *tokenizer.encode("WE", add_special_tokens=False), grid_ids["10"]],
        skip_special_tokens=False,
    )
    assert "<unused" in text
    assert "unused" not in strip_anchors(text)
    assert "WE" in strip_anchors(text)
