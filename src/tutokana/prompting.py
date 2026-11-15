"""Rendering the prompt and the assistant turn.

One unified turn carries the phone transcript with a register after each phone, three
registers after each word, and four utterance registers at the end. Ordering is bottom-up
under causal masking, so word registers see their phones and utterance registers see
everything.

`render_training_text` is exactly `render_prompt` + `render_target`; a test asserts the
prefix is byte-identical to what evaluation sends, since a train/eval prompt mismatch is
silent and expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Utterance
from .tokens import (
    ANCHOR,
    LEVELS,
    PHONE_REGISTERS,
    SEP_PHONES,
    SEP_WORD,
    TASK,
    UTTERANCE_REGISTERS,
    WORD_REGISTERS,
    grid_for,
)

#: Short by design: the register layout *is* the output format, so no format specification
#: is needed in prose.
INSTRUCTION = (
    "Assess English pronunciation under the speechocean762 protocol. The speakers are "
    "non-native learners whose first language is Mandarin; about half are children. "
    "For each transcript word, emit the word followed by its ARPAbet phone sequence "
    "(vowels carry a stress digit, consonants do not), then the scoring registers."
)

USER_TEMPLATE = "<|audio|>Transcript: {transcript}"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """What the assistant turn contains.

    `levels` selects which score registers are emitted. `emit_phones` controls the phone
    *transcript* independently, so the joint-prediction ablation is a flag rather than a
    rewrite: published work on Qwen2-Audio and GPT-4o both found that asking for every
    granularity at once costs the coarser levels several points of correlation, and this is
    how that is measured here.
    """

    levels: tuple[str, ...] = LEVELS
    emit_phones: bool = True
    #: Under a RAIL/RAFT variant the score is text: one grid token follows every score
    #: register. Off keeps the regression-head turn, where a register is the end of the field
    #: and the score lives only in a head readout.
    emit_scores: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.levels) - set(LEVELS)
        if unknown:
            raise ValueError(f"unknown levels {sorted(unknown)}; known: {LEVELS}")
        if not self.levels:
            raise ValueError("at least one level must be selected")
        if "phone" in self.levels and not self.emit_phones:
            raise ValueError("phone-level registers require emit_phones=True")


@dataclass(frozen=True, slots=True)
class RegisterSlot:
    """One expected register occurrence, in emission order.

    `word` and `phone` are indices into the utterance; both are None for utterance-level
    slots, and `phone` is None for word-level slots. `collate` zips the actual register
    positions found in `input_ids` against this list, so a mismatch in either count or order
    is caught immediately rather than silently mislabelling scores.
    """

    name: str
    level: str
    field: str
    word: int | None = None
    phone: int | None = None


@dataclass(frozen=True, slots=True)
class RenderedTarget:
    text: str
    slots: tuple[RegisterSlot, ...] = ()


def slot_value(utterance: Utterance, slot: RegisterSlot) -> float | None:
    """The label one slot carries, on the scale its own grid is built from.

    This is the single definition both channels read: the renderer turns it into a grid token,
    and the collator turns it into a head target. If they each had their own copy, the token
    in the text and the number in the head could disagree without any test noticing.

    `word.stress` is the annotator vote count, not the released 5/10 median, because the stress
    grid is the vote grid.

    None means the slot is emitted but unsupervised. Only word stress can be: the synthetic
    negatives in the mix pair a transcript with unrelated audio, so their words have no stress
    pattern to judge.
    """
    if slot.level == "utterance":
        return utterance.utterance_targets()[slot.field]
    word = utterance.words[slot.word]
    if slot.level == "word":
        if slot.field == "stress":
            return None if word.stress_votes is None else float(word.stress_votes)
        return getattr(word, slot.field)
    return word.phone_accuracy[slot.phone]


def render_prompt(processor, transcript: str) -> str:
    """The exact prefix generation starts from — the single source both sides read.

    `<|audio|>` stays a bare placeholder here; the processor expands it to
    `<|audio>` + N x `<|audio|>` + `<audio|>` at tokenization time, where N is
    ceil(samples / 640), i.e. one soft token per 40 ms of waveform.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {
            "role": "user",
            "content": [{"type": "text", "text": USER_TEMPLATE.format(transcript=transcript)}],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def render_target(
    utterance: Utterance,
    spec: TargetSpec,
    eos: str = "<turn|>",
    grid_tokens: dict[str, str] | None = None,
) -> RenderedTarget:
    """The supervised assistant turn plus the register layout it implies.

    With `spec.emit_scores`, every score register is followed by the one grid token naming its
    value, and `grid_tokens` (grid string -> registered token) is required. A slot with no
    label gets no token at all: the trigger is still there, but the next thing after it is the
    next register, and nothing at that position is supervised. Inventing a score to keep the
    shape uniform would put a made-up number in the context every later token attends to.
    """
    if spec.emit_scores and grid_tokens is None:
        raise ValueError(
            "emit_scores needs grid_tokens: a score is emitted as one registered token, so "
            "the renderer cannot write the digits itself. Build the map with "
            "tokens.grid_token_map(tokenizer, register_grid_tokens(tokenizer))."
        )
    parts: list[str] = [TASK.token, "\n"]
    slots: list[RegisterSlot] = []
    want_phone = "phone" in spec.levels
    want_word = "word" in spec.levels
    want_utterance = "utterance" in spec.levels

    def emit(reg, word_idx=None, phone_idx=None) -> None:
        """Append one register, and under emit_scores the grid token naming its value."""
        parts.append(reg.token)
        slot = RegisterSlot(reg.name, reg.level, reg.field, word_idx, phone_idx)
        if spec.emit_scores:
            value = slot_value(utterance, slot)
            if value is not None:
                grid = grid_for(slot.level, slot.field)
                parts.append(grid_tokens[grid.render(value)])
        slots.append(slot)

    for w_idx, word in enumerate(utterance.words):
        parts.append(word.text)
        if spec.emit_phones:
            parts.append(SEP_PHONES.token)
            for p_idx, phone in enumerate(word.phones):
                if p_idx:
                    parts.append(" ")
                parts.append(phone)
                if want_phone:
                    for reg in PHONE_REGISTERS:
                        emit(reg, w_idx, p_idx)
        if want_word:
            parts.append(SEP_WORD.token)
            for reg in WORD_REGISTERS:
                emit(reg, w_idx, None)
        parts.append("\n")

    if want_utterance:
        for reg in UTTERANCE_REGISTERS:
            emit(reg, None, None)

    parts.append(eos)
    return RenderedTarget(text="".join(parts), slots=tuple(slots))


def render_training_text(
    processor,
    utterance: Utterance,
    spec: TargetSpec,
    grid_tokens: dict[str, str] | None = None,
) -> RenderedTarget:
    """Eval prompt + supervised target. Nothing else may build training text."""
    rendered = render_target(utterance, spec, grid_tokens=grid_tokens)
    return RenderedTarget(
        text=render_prompt(processor, utterance.text) + rendered.text,
        slots=rendered.slots,
    )


def parse_generated(text: str) -> list[tuple[str, list[str]]]:
    """Recover [(word, [phones])] from a generated assistant turn.

    Used by the generative-mode evaluation, which measures phone error rate and word
    exact-match. The scored evaluation never goes through here — it teacher-forces the
    canonical phones so every register aligns one-to-one with gold, which is the only way
    the phone correlation stays comparable to the published baselines.

    Every token `tokens.py` adds is an **anchor**, and the anchors are what make the stream
    parseable: they mark the task, separate phones from a word, separate a word from its
    scores, and carry the scores themselves. So the parser treats them uniformly — drop any
    `<unusedN>`, keep everything else — instead of stripping a hard-coded register list.
    That matters now that score tokens exist too: a fixed register list would leave
    `W<unused32>` in the phone slot, since score tokens are not registers.

    Callers must decode with `skip_special_tokens=False`. With the default, the anchors that
    separate the fields are gone and word/phone boundaries cannot be recovered.
    """
    body = text.split(TASK.token, 1)[-1]
    for cut in ("<turn|>", "<eos>"):
        body = body.split(cut, 1)[0]

    parsed: list[tuple[str, list[str]]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.split(SEP_WORD.token, 1)[0]
        word, sep, phone_blob = line.partition(SEP_PHONES.token)
        # The trailing utterance-register line carries no word; strip the anchors so it falls
        # out as empty rather than being read as a one-word transcript.
        word = ANCHOR.sub("", word).strip()
        if not word:
            continue
        phones: list[str] = []
        if sep:
            # Anchors sit *after* each phone, so splitting on whitespace alone would glue a
            # score onto its phone. Strip first, then split.
            phones = ANCHOR.sub("", phone_blob).split()
        parsed.append((word, phones))
    return parsed
