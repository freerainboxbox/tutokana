"""Shared fixtures. Nothing here downloads a model or touches the network."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tutokana.data import SAMPLE_RATE, Utterance, Word

#: The Gemma 4 tokenizer, if a snapshot happens to be in the local Hub cache. Tests that
#: need real tokenization skip cleanly when it is not — the rest of the suite still runs.
_TOKENIZER_GLOB = (
    Path.home() / ".cache/huggingface/hub/models--*gemma-4-12B-it*/snapshots/*"
)


def _find_tokenizer_dir() -> Path | None:
    import glob

    for path in sorted(glob.glob(str(_TOKENIZER_GLOB))):
        if (Path(path) / "tokenizer.json").exists():
            return Path(path)
    return None


@pytest.fixture(scope="session")
def tokenizer():
    path = _find_tokenizer_dir()
    if path is None:
        pytest.skip("no local Gemma 4 tokenizer snapshot")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


#: A handful of real ARPAbet entries so the fixtures exercise the phone vocabulary, and so
#: `mix.sample_negatives` can actually find word-disjoint pairs (every utterance sharing one
#: vocabulary would make a disjoint pair impossible by construction).
LEXICON: dict[str, tuple[str, ...]] = {
    "WE": ("W", "IY0"),
    "CALL": ("K", "AO0", "L"),
    "IT": ("IH0", "T"),
    "BEAR": ("B", "EH0", "R"),
    "HELLO": ("HH", "AH0", "L", "OW1"),
    "BIRD": ("B", "ER1", "D"),
    "FAST": ("F", "AE0", "S", "T"),
    "TREE": ("T", "R", "IY1"),
}
WORD_GROUPS: tuple[tuple[str, ...], ...] = (
    ("WE", "CALL"),
    ("IT", "BEAR"),
    ("HELLO", "BIRD"),
    ("FAST", "TREE"),
)

#: Phone scores cycle through the low end of the range so the fixtures are not saturated at
#: 2.0 the way the real corpus is — a constant target would make several tests vacuous.
_PHONE_SCORES = (2.0, 1.8, 0.4, 1.0, 1.6, 0.0, 2.0, 1.2)


def make_utterance(
    index: int = 0, speaker: str = "0001", words: tuple[str, ...] = ("WE", "CALL")
) -> Utterance:
    """A short utterance with hand-picked scores spread across each field's range."""
    built = []
    cursor = index
    for position, text in enumerate(words):
        phones = LEXICON[text]
        scores = tuple(
            _PHONE_SCORES[(cursor + position + i) % len(_PHONE_SCORES)]
            for i in range(len(phones))
        )
        built.append(
            Word(
                text=text,
                accuracy=float(10 - 2 * ((cursor + position) % 4)),
                stress=10.0 if (cursor + position) % 7 else 5.0,
                total=float(10 - 2 * ((cursor + position) % 4)),
                phones=phones,
                phone_accuracy=scores,
            )
        )
    return Utterance(
        index=index,
        speaker=speaker,
        text=" ".join(words),
        audio=np.zeros(SAMPLE_RATE + 640 * (index % 5), dtype=np.float32),
        accuracy=float(6 + index % 5),
        prosodic=float(5 + index % 6),
        fluency=float(7 + index % 4),
        total=float(6 + index % 5),
        completeness=10.0,
        words=tuple(built),
    )


@pytest.fixture
def utterance():
    return make_utterance()


@pytest.fixture
def utterances():
    return [
        make_utterance(i, speaker=f"{i % 5:04d}", words=WORD_GROUPS[i % len(WORD_GROUPS)])
        for i in range(20)
    ]
