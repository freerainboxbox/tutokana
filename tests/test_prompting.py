"""The prompt contract: what training renders must prefix-match what evaluation sends.

A drift between the two is silent — the model trains on one format and is asked for another —
so it is asserted rather than left as a convention.
"""

from __future__ import annotations

import pytest

from tutokana.prompting import (
    slot_value,
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


def test_parse_generated_survives_scored_turn():
    """A RAIL/RAFT turn puts a score token after every register.

    Stripping only the register list would leave `W<unused32>` in the phone slot, because
    score tokens are not registers. Anchors are stripped uniformly instead.
    """
    scored = (
        "<unused0>\n"
        "WE<unused10>W<unused1><unused32> IY0<unused1><unused28>"
        "<unused11><unused22><unused17><unused22>\n"
        "CALL<unused10>K<unused1><unused30> AO0<unused1><unused26> L<unused1><unused34>"
        "<unused11><unused22><unused15><unused22>\n"
        "<unused20><unused20><unused21><unused20><turn|>"
    )
    assert parse_generated(scored) == [("WE", ["W", "IY0"]), ("CALL", ["K", "AO0", "L"])]


def test_parse_generated_leaves_no_anchor_residue():
    from tutokana.tokens import ANCHOR

    scored = (
        "<unused0>\nWE<unused10>W<unused1><unused32><unused11><unused22><unused17><unused22>"
        "<turn|>"
    )
    for word, phones in parse_generated(scored):
        assert not ANCHOR.search(word)
        assert not any(ANCHOR.search(p) for p in phones)


# --- emitted scores: the grid token as text --------------------------------------------

#: Grid string -> registered token, built from the real tokenizer.
def _grid_tokens(tokenizer):
    from tutokana.tokens import grid_token_map, register_grid_tokens, register_tokens

    register_tokens(tokenizer)
    return grid_token_map(tokenizer, register_grid_tokens(tokenizer))


def _emitted(tokenizer, utterance):
    """(token ids, register ids, grid ids) for the emitted target."""
    from tutokana.tokens import register_grid_tokens, register_tokens

    ids = tokenizer.encode(
        render_target(utterance, TargetSpec(emit_scores=True),
                      grid_tokens=_grid_tokens(tokenizer)).text,
        add_special_tokens=False,
    )
    return ids, set(register_tokens(tokenizer).values()), set(
        register_grid_tokens(tokenizer).values()
    )


def test_emitting_scores_needs_the_token_map(tokenizer, utterance):
    """A score is one registered token, so the renderer cannot write the digits itself."""
    with pytest.raises(ValueError, match="grid_tokens"):
        render_target(utterance, TargetSpec(emit_scores=True))


def test_every_score_register_is_followed_by_one_grid_token(tokenizer, utterance):
    """One token per score, never a digit string: `1.8` is three tokens and off the grid."""
    from tutokana.tokens import SCORE_REGISTERS

    ids, registers, grid = _emitted(tokenizer, utterance)
    scored = {tokenizer.convert_tokens_to_ids(reg.token) for reg in SCORE_REGISTERS}
    following = [ids[i + 1] for i in range(len(ids) - 1) if ids[i] in scored]
    assert following, "no score registers in the rendered turn"
    assert all(t in grid for t in following), "a score register was not followed by a grid token"
    # Every score register is followed by exactly one token, so the counts must match.
    assert len(following) == sum(1 for t in ids if t in scored)
    # The structural anchors are registers too, and nothing is emitted after them: they are
    # the reason the register set is larger than the scored set.
    assert registers - scored, "the structural anchors should not be score registers"


def test_the_emitted_token_names_the_slot_value(tokenizer, utterance):
    """The token must name the number the head target uses, or the two channels disagree.

    Walks the token stream the way the collator does: registers in order of appearance,
    zipped against the slots the renderer declared.
    """
    from tutokana.tokens import grid_for, register_grid_tokens, register_tokens

    gt = _grid_tokens(tokenizer)
    target = render_target(utterance, TargetSpec(emit_scores=True), grid_tokens=gt)
    ids = tokenizer.encode(target.text, add_special_tokens=False)

    name_of = {tid: name for name, tid in register_tokens(tokenizer).items()}
    grid_of = {tid: text for text, tid in register_grid_tokens(tokenizer).items()}
    # Slots in emission order per register name, matched against registers in token order.
    queue: dict[str, list] = {}
    for slot in target.slots:
        queue.setdefault(slot.name, []).append(slot)
    seen: dict[str, int] = {}
    checked = 0
    for pos, tid in enumerate(ids):
        name = name_of.get(tid)
        if name not in queue:
            continue
        slot = queue[name][seen.get(name, 0)]
        seen[name] = seen.get(name, 0) + 1
        value = slot_value(utterance, slot)
        if value is None:
            continue
        grid = grid_for(slot.level, slot.field)
        assert grid_of.get(ids[pos + 1]) == grid.render(value), (
            f"{slot.name}: emitted {grid_of.get(ids[pos + 1])!r} for value {value}"
        )
        checked += 1
    assert checked == len(target.slots) - _unrated(utterance)


def _unrated(utterance) -> int:
    """How many slots carry no label, which is the only reason a slot emits no token."""
    return sum(1 for w in utterance.words if not w.stress_rated)


def test_slot_value_is_the_vote_count_for_stress(utterance):
    """The stress grid is the vote grid; the released 5/10 median is a reporting scale."""
    from tutokana.prompting import RegisterSlot

    slot = RegisterSlot("word_stress", "word", "stress", 0, None)
    assert slot_value(utterance, slot) == float(utterance.words[0].stress_votes)
    assert utterance.words[0].stress != slot_value(utterance, slot)


def test_slot_value_is_the_reported_label_elsewhere(utterance):
    from tutokana.prompting import RegisterSlot

    assert slot_value(
        utterance, RegisterSlot("word_accuracy", "word", "accuracy", 0, None)
    ) == utterance.words[0].accuracy
    assert slot_value(
        utterance, RegisterSlot("phone_accuracy", "phone", "accuracy", 0, 0)
    ) == utterance.words[0].phone_accuracy[0]
    assert slot_value(
        utterance, RegisterSlot("utterance_fluency", "utterance", "fluency")
    ) == utterance.fluency


def test_slot_value_is_none_only_for_unrated_stress(utterance):
    from dataclasses import replace

    from tutokana.prompting import RegisterSlot

    u = replace(
        utterance,
        words=(replace(utterance.words[0], stress=None, stress_votes=None),)
        + utterance.words[1:],
    )
    stress = RegisterSlot("word_stress", "word", "stress", 0, None)
    assert slot_value(u, stress) is None
    assert slot_value(
        u, RegisterSlot("word_accuracy", "word", "accuracy", 0, None)
    ) is not None


def test_unrated_stress_emits_no_score_token(tokenizer, utterance):
    """Inventing a score to keep the shape uniform would put a lie in every later context."""
    from dataclasses import replace

    unrated = replace(utterance.words[0], stress=None, stress_votes=None)
    mixed = replace(utterance, words=(unrated,) + utterance.words[1:])
    gt = _grid_tokens(tokenizer)
    with_gap = render_target(mixed, TargetSpec(emit_scores=True), grid_tokens=gt)
    full = render_target(utterance, TargetSpec(emit_scores=True), grid_tokens=gt)

    assert len(with_gap.slots) == len(full.slots), "the register layout must not change"
    assert len(with_gap.text) < len(full.text), "the unrated slot still emitted a score"


def test_slot_value_is_the_vote_count_for_stress(tokenizer, utterance):
    """The stress grid is the vote grid; the released 5/10 median is a reporting scale."""
    from tutokana.prompting import RegisterSlot

    slot = RegisterSlot("word_stress", "word", "stress", 0, None)
    assert slot_value(utterance, slot) == float(utterance.words[0].stress_votes)
    assert utterance.words[0].stress != slot_value(utterance, slot)


def test_emitting_scores_keeps_every_register(tokenizer, utterance):
    """The turn still carries one register per scored field, as `_head_batches` requires."""
    plain = render_target(utterance, TargetSpec())
    emitted = render_target(
        utterance, TargetSpec(emit_scores=True), grid_tokens=_grid_tokens(tokenizer)
    )
    assert [s.name for s in emitted.slots] == [s.name for s in plain.slots]
