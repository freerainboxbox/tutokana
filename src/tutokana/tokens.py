"""Score-register tokens, and the score grids the model emits as text.

Eleven unused Gemma 4 vocabulary entries are registered as atomic tokens and used as
structural landmarks in the assistant turn: one per phone, three per word, four per
utterance, plus separators. Registering them leaves `len(tokenizer)` at 262144, so there is
no embedding resize and no id drift. `<unused8>` (id 14) is skipped — it is a known
llama.cpp degenerate-loop trigger for Gemma 4.

The RAIL/RAFT path scores *in the text channel*: a register is a trigger, and the token the
model emits after it carries the score. Every value a score type can take is therefore its
own atomic token. Those tokens are derived from `SCORE_GRID` rather than listed, so the
grid spacing is the only thing to edit.

**Every token this module adds is registered with `special=False`, and that is deliberate.**

- `special` does not gate emission. `generate()` suppresses tokens only from
  `generation_config.suppress_tokens`, and nothing reads `all_special_ids` there. A special
  token is emittable anyway, so the flag buys nothing on the output side.
- The flag only decides whether `decode(skip_special_tokens=True)` deletes the token. With a
  mixed split, that default deletes the anchors and keeps the scores: a string that looks
  clean but cannot be parsed. Uniform `special=False` makes the default decode and the full
  decode the same string, so no call site can silently get the unparseable half.
- The flag is not sticky: the last `add_tokens` call for a slot wins. A mixed policy would
  therefore depend on call order, and one wrong order would quietly change what a run parses.
- This touches only tokens we add. The real specials (`<bos>`, `<pad>`, `<eos>`, the audio
  and image markers) are untouched and still strip by default.

The cost is that a decoded dump shows `<unusedN>` rather than a digit. Display code strips
anchors with `strip_anchors`; parsing never does, because the anchors are what make the
stream parseable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: One unused Gemma 4 vocabulary entry, e.g. `<unused2968>`. Every token this module adds is
#: one of these, so the pattern is also the test for "is this an anchor we emitted?".
_UNUSED = re.compile(r"<unused(\d+)>")

#: Any anchor in a decoded string. Structural markers, triggers and score tokens all match,
#: which is what lets the parser treat them uniformly without a registry of live slots.
ANCHOR = re.compile(r"<unused\d+>")


def strip_anchors(text: str) -> str:
    """Remove every anchor. For **display only** — parsing needs the anchors present.

    A decoded scored turn reads as `<unused0>\\nWE<unused10>W<unused32>...` otherwise. Strip
    it for a human, never before the parser: the anchors are how the parser knows which
    field a score belongs to.
    """
    return ANCHOR.sub("", text)

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

    Registers are added with `special=False`, like the grid tokens. Under RAIL/RAFT the model
    *emits* them as triggers, so a decode must keep them for the stream to be parseable, and
    a mixed policy would make the default decode unparseable. See the module docstring.
    """
    from transformers import AddedToken

    size_before = len(tokenizer)
    tokenizer.add_tokens(
        [AddedToken(reg.token, special=False, normalized=False) for reg in ALL_REGISTERS],
        special_tokens=False,
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


# ---------------------------------------------------------------------------
# score grids: the values the model emits as text under RAIL/RAFT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    """The discrete support one score type renders onto, derived from its spacing.

    `spacing` alone decides the rendering: a value is written with exactly as many decimals
    as the step needs in order to be exact. A step of 1 renders `2`; a step of 0.2 renders
    `2.0`. So `1` and `1.0` are *different* tokens, and which one a score type uses is a
    property of the score type and never of the float it happens to hold — the corpus
    writes `2` for a phone score of 2.0, and that must still render as the decimal token.

    Nothing here is read off the data. `low` and `high` are the ends of the target space,
    not the observed support: the training split omits word accuracy 4 and utterance
    accuracy 0 and 2, yet those values exist, and a regression-aware loss needs a grid that
    spans the whole space so the expectation can land anywhere in it.
    """

    spacing: float
    low: float
    high: float

    @property
    def decimals(self) -> int:
        """Decimal places the step needs to be exact: 1 -> 0, 0.2 -> 1, 0.05 -> 2."""
        for places in range(0, 7):
            if abs(round(self.spacing, places) - self.spacing) < 1e-12:
                return places
        raise ValueError(f"spacing {self.spacing} needs more than 6 decimal places")

    @property
    def values(self) -> tuple[float, ...]:
        """Every value on the grid, low to high, inclusive at both ends."""
        steps = int(round((self.high - self.low) / self.spacing))
        return tuple(round(self.low + i * self.spacing, self.decimals) for i in range(steps + 1))

    @property
    def cls(self) -> str:
        """`integer` or `decimal` — the rendering class the spacing implies."""
        return "integer" if self.decimals == 0 else "decimal"

    def render(self, value: float) -> str:
        """The grid string for `value`, snapped to the nearest grid point.

        Keyed by the grid, never by the float: this is what stops a phone score of 2.0 from
        being formatted as `2` and landing on the integer token instead of the decimal one.
        Snapping is lossless on gold, since every released label sits on its own grid.
        """
        values = self.values
        index = int(round((value - self.low) / self.spacing))
        if not 0 <= index < len(values):
            raise ValueError(
                f"{value} is outside the grid {self.low}..{self.high} at spacing {self.spacing}"
            )
        return f"{values[index]:.{self.decimals}f}"


#: Grid per score type, keyed by (level, field) to match `FIELD_RANGE` and `FIELD_SUPPORT`.
#:
#: `word.stress` is the **vote count** grid, not the released 5/10 median: the released label
#: is the median of five annotators and is 99% one value, so the head is trained on how many
#: of the five agreed. Reporting still happens on the 5/10 scale.
#:
#: `utterance.completeness` is deliberately absent. It is untrained and has no register, and
#: its released values (1.7, 6.7, 7.1, 7.5, 8.6, 8.9) sit on no step that is 1 or 0.2, so it
#: would force a 0.1 grid and about 100 more tokens. Train it only with that decided.
SCORE_GRID: dict[tuple[str, str], ScoreGrid] = {
    ("phone", "accuracy"): ScoreGrid(spacing=0.2, low=0.0, high=2.0),
    ("word", "accuracy"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
    ("word", "stress"): ScoreGrid(spacing=1.0, low=0.0, high=5.0),
    ("word", "total"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
    ("utterance", "accuracy"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
    ("utterance", "prosodic"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
    ("utterance", "fluency"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
    ("utterance", "total"): ScoreGrid(spacing=1.0, low=0.0, high=10.0),
}

#: Every grid string across every score type, integers first then decimals, each ascending.
#: Shared by class: all integer score types draw from one set, all decimal ones from the
#: other, so the same token means the same number wherever the model emits it.
GRID_STRINGS: tuple[str, ...] = tuple(
    sorted(
        {grid.render(v) for grid in SCORE_GRID.values() for v in grid.values},
        key=lambda s: ("." in s, float(s)),
    )
)


def grid_for(level: str, field: str) -> ScoreGrid:
    """The grid a score type renders onto. Raises for a field with no grid."""
    try:
        return SCORE_GRID[(level, field)]
    except KeyError:
        raise KeyError(
            f"no score grid for ({level!r}, {field!r}); scored types: "
            f"{sorted(SCORE_GRID)}"
        ) from None


def grid_token_map(tokenizer, grid_ids: dict[str, int]) -> dict[str, str]:
    """{grid string: the literal token that carries it}, for writing a score into text.

    A score never reaches the text as digits. `1.0` is one registered `<unusedN>` token, and
    writing the literal `1.0` would tokenize to three tokens and put the score outside the
    support the expectation is taken over. So the renderer needs this map, not the ids.
    """
    return {text: tokenizer.convert_ids_to_tokens(tid) for text, tid in grid_ids.items()}


def slots_for(tokenizer, count: int, taken: set[str]) -> list[str]:
    """The `count` lowest unused Gemma slots, as `<unusedN>` strings.

    Deterministic and free of both hazards: `<unused8>` (id 14, the llama.cpp loop trigger)
    and every slot a register already holds. The low block is preferred because
    `RegisterDelta` builds a lookup table sized by the largest id it sees, so slots at
    256001+ would grow that table from tens to a quarter of a million entries.
    """
    free = [
        (token_id, token)
        for token, token_id in tokenizer.get_vocab().items()
        if token not in taken and token != "<unused8>" and _UNUSED.match(token)
    ]
    free.sort()
    if len(free) < count:
        raise RuntimeError(f"only {len(free)} unused slots for {count} grid tokens")
    if free[count - 1][0] > 1024:
        raise RuntimeError(
            f"the low block cannot hold {count} tokens: the {count}th free slot is id "
            f"{free[count - 1][0]}. RegisterDelta's lookup table would jump to that width."
        )
    return [token for _, token in free[:count]]


def register_grid_tokens(tokenizer, taken: set[str] | None = None) -> dict[str, int]:
    """Make every grid string atomic and return {grid string: token id}.

    Idempotent, and it re-checks the same two invariants `register_tokens` does: the
    vocabulary does not grow, and each string is one token.

    Uses `special=False`, the same policy as `register_tokens`, so every token this package
    adds is non-special. That keeps the default decode parseable: a score token must survive
    `decode(skip_special_tokens=True)` or the score channel is erased from the text it lives
    in. See the module docstring for why the split policy was dropped.
    """
    from transformers import AddedToken

    held = set(taken or ()) | {reg.token for reg in ALL_REGISTERS}
    slots = slots_for(tokenizer, len(GRID_STRINGS), taken=held)
    mapping = dict(zip(slots, GRID_STRINGS))

    size_before = len(tokenizer)
    tokenizer.add_tokens(
        [AddedToken(slot, special=False, normalized=False) for slot in slots],
        special_tokens=False,
    )
    if len(tokenizer) != size_before:
        raise RuntimeError(
            f"registering the grid grew the vocabulary {size_before} -> {len(tokenizer)}. "
            "Every slot must already exist in the base vocab; a grown vocab would require "
            "resize_token_embeddings and silently invalidate the checkpoint."
        )

    ids: dict[str, int] = {}
    for slot, text in mapping.items():
        encoded = tokenizer.encode(slot, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"grid token {slot} (value {text!r}) tokenized to {len(encoded)} tokens "
                f"{encoded}; it must be atomic"
            )
        ids[text] = encoded[0]

    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"grid token ids collide: {ids}")
    overlap = set(ids.values()) & set(
        tokenizer.convert_tokens_to_ids(reg.token) for reg in ALL_REGISTERS
    )
    if overlap:
        raise RuntimeError(f"grid token ids collide with register ids: {sorted(overlap)}")
    return ids
