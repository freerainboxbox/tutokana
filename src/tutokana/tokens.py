"""The register-token table — the single source of truth for tutokana's vocabulary.

Scores do not appear in the text the model emits. Every score position is instead a
*register*: a token whose only job is to exist at a known index so a regression head can
read the hidden state there. Gemma 4 ships 6227 `<unusedN>` entries (2.4% of a 262144
vocabulary), so registers cost no embedding growth at all.

Two facts make this work, both verified against the real tokenizer rather than assumed:

  1. `<unusedN>` strings are present in `model.vocab` but NOT in `added_tokens`, so out of
     the box `"<unused0>"` tokenizes as ['<', 'unused', '0', '>'] — four tokens, not one.
     `register_tokens()` fixes that with `add_tokens(..., special_tokens=True)`.
  2. Doing so does NOT grow the vocabulary: `len(tokenizer)` stays 262144 and the ids stay
     put, because the rows already exist. No `resize_token_embeddings`, ever. The unused
     rows are not zero either (norm ~1.00 vs ~1.03-1.14 for ordinary ids), so they are
     usable as-is and a zero-initialised delta on top starts training from the pretrained
     point (see model.RegisterDelta).

AVOID `<unused8>` (id 14): it is a known llama.cpp degenerate-loop trigger for Gemma 4
(ggml-org/llama.cpp#21321, #21516). The table below skips it, which is why the utterance
registers are not a contiguous `<unused5..8>` run.

Ordering is load-bearing. The assistant turn is bottom-up under causal masking — phones,
then that word's registers, then, after every word, the utterance registers with
UTT_TOTAL last so it can attend to the other three.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Levels, coarsest last. `LEVELS` is also the canonical ordering for report tables.
LEVELS = ("phone", "word", "utterance")


@dataclass(frozen=True, slots=True)
class Register:
    """One register token and the field it scores.

    `name`   short key used in configs, labels and metric tables
    `token`  the literal `<unusedN>` string
    `level`  one of LEVELS
    `field`  the speechocean762 field this register predicts (None for structural tokens)
    """

    name: str
    token: str
    level: str | None
    field: str | None


#: Structural markers — no head reads these; they only shape the text.
TASK = Register("task", "<unused0>", None, None)
SEP_PHONES = Register("sep_phones", "<unused10>", None, None)
SEP_WORD = Register("sep_word", "<unused11>", None, None)

#: Score registers, in emission order within their level.
PHONE_REGISTERS = (Register("phone_accuracy", "<unused1>", "phone", "accuracy"),)

WORD_REGISTERS = (
    Register("word_accuracy", "<unused2>", "word", "accuracy"),
    Register("word_stress", "<unused3>", "word", "stress"),
    Register("word_total", "<unused4>", "word", "total"),
)

# <unused8> (id 14) deliberately skipped — see module docstring.
UTTERANCE_REGISTERS = (
    Register("utterance_accuracy", "<unused5>", "utterance", "accuracy"),
    Register("utterance_prosodic", "<unused6>", "utterance", "prosodic"),
    Register("utterance_fluency", "<unused7>", "utterance", "fluency"),
    Register("utterance_total", "<unused9>", "utterance", "total"),
)

SCORE_REGISTERS = PHONE_REGISTERS + WORD_REGISTERS + UTTERANCE_REGISTERS
STRUCTURAL_REGISTERS = (TASK, SEP_PHONES, SEP_WORD)
ALL_REGISTERS = STRUCTURAL_REGISTERS + SCORE_REGISTERS

#: Registers grouped by level, in emission order.
REGISTERS_BY_LEVEL: dict[str, tuple[Register, ...]] = {
    "phone": PHONE_REGISTERS,
    "word": WORD_REGISTERS,
    "utterance": UTTERANCE_REGISTERS,
}

#: `completeness` is deliberately absent from UTTERANCE_REGISTERS: it is 10.0 for 99.6% of
#: the train split (sigma 0.11), so its correlation is undefined for any model that does not
#: happen to reproduce the handful of outliers. Every published LMM result reports NaN for
#: it. It is measured and reported, never trained.
UNTRAINED_FIELDS = {("utterance", "completeness")}


def register_by_name(name: str) -> Register:
    for reg in ALL_REGISTERS:
        if reg.name == name:
            return reg
    raise KeyError(f"unknown register {name!r}; known: {[r.name for r in ALL_REGISTERS]}")


def register_tokens(tokenizer) -> dict[str, int]:
    """Make every register token atomic and return {register name: token id}.

    Idempotent, and asserts the two invariants the whole design rests on: the vocabulary
    does not grow, and each register is exactly one token. Call this immediately after
    loading a tokenizer, before rendering anything.
    """
    from transformers import AddedToken

    size_before = len(tokenizer)
    tokenizer.add_tokens(
        [AddedToken(reg.token, special=True, normalized=False) for reg in ALL_REGISTERS],
        special_tokens=True,
    )
    size_after = len(tokenizer)
    if size_after != size_before:
        raise RuntimeError(
            f"registering tokens grew the vocabulary {size_before} -> {size_after}. "
            f"Every register must already exist in the base vocab; a grown vocab would "
            f"require resize_token_embeddings and silently invalidate the checkpoint."
        )

    ids: dict[str, int] = {}
    for reg in ALL_REGISTERS:
        encoded = tokenizer.encode(reg.token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"register {reg.name} ({reg.token}) tokenized to {len(encoded)} tokens "
                f"{encoded}; it must be atomic."
            )
        ids[reg.name] = encoded[0]

    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"register ids collide: {ids}")
    return ids


def score_register_ids(token_ids: dict[str, int], levels=LEVELS) -> dict[int, Register]:
    """{token id: Register} for the score registers of the requested levels."""
    return {
        token_ids[reg.name]: reg
        for level in levels
        for reg in REGISTERS_BY_LEVEL[level]
    }
