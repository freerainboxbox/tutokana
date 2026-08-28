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
    ALL_REGISTERS,
    LEVELS,
    PHONE_REGISTERS,
    SEP_PHONES,
    SEP_WORD,
    TASK,
    UTTERANCE_REGISTERS,
    WORD_REGISTERS,
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


def render_target(utterance: Utterance, spec: TargetSpec, eos: str = "<turn|>") -> RenderedTarget:
    """The supervised assistant turn plus the register layout it implies."""
    parts: list[str] = [TASK.token, "\n"]
    slots: list[RegisterSlot] = []
    want_phone = "phone" in spec.levels
    want_word = "word" in spec.levels
    want_utterance = "utterance" in spec.levels

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
                        parts.append(reg.token)
                        slots.append(
                            RegisterSlot(reg.name, reg.level, reg.field, w_idx, p_idx)
                        )
        if want_word:
            parts.append(SEP_WORD.token)
            for reg in WORD_REGISTERS:
                parts.append(reg.token)
                slots.append(RegisterSlot(reg.name, reg.level, reg.field, w_idx, None))
        parts.append("\n")

    if want_utterance:
        for reg in UTTERANCE_REGISTERS:
            parts.append(reg.token)
            slots.append(RegisterSlot(reg.name, reg.level, reg.field, None, None))

    parts.append(eos)
    return RenderedTarget(text="".join(parts), slots=tuple(slots))


def render_training_text(processor, utterance: Utterance, spec: TargetSpec) -> RenderedTarget:
    """Eval prompt + supervised target. Nothing else may build training text."""
    rendered = render_target(utterance, spec)
    return RenderedTarget(
        text=render_prompt(processor, utterance.text) + rendered.text,
        slots=rendered.slots,
    )


def parse_generated(text: str) -> list[tuple[str, list[str]]]:
    """Recover [(word, [phones])] from a generated assistant turn.

    Used only by the generative-mode evaluation, which measures phone error rate and word
    exact-match. The scored evaluation never goes through here — it teacher-forces the
    canonical phones so every register aligns one-to-one with gold, which is the only way
    the phone correlation stays comparable to the published baselines.
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
        # The trailing utterance-register line carries no word; strip every register token
        # so it falls out as empty rather than being read as a one-word transcript.
        for reg in ALL_REGISTERS:
            word = word.replace(reg.token, "")
        word = word.strip()
        if not word:
            continue
        phones: list[str] = []
        if sep:
            for reg in PHONE_REGISTERS:
                phone_blob = phone_blob.replace(reg.token, " ")
            phones = phone_blob.split()
        parsed.append((word, phones))
    return parsed
