"""speechocean762 loading, target extraction, splits and normalisation statistics.

The corpus is 2500 train / 2500 test utterances of L1-Mandarin English learners, scored by
five experts at three granularities. Facts that shape everything downstream, all measured
against the live dataset rather than quoted (reproduce with experiments/audit_data.py):

  * Phone accuracy is NOT the {0,1,2} rubric handed to each annotator. The released label
    is their MEAN, quantised to 0.2 — eleven values in [0.0, 2.0], with 2.0 covering 80.1%
    of train phones. Rounding it to three classes throws away roughly a fifth of the
    label's real signal, and caps achievable correlation.
  * Word stress is 5 or 10 with only 1.0% at 5 (160 of 15849 train words). It is a 99:1
    binary problem wearing a regression costume, so `stress_binary()` maps it to {0,1} for
    a weighted BCE head. Published SOTA is 0.366; a fine-tuned Qwen2-Audio managed -0.01.
    Report it; do not tune against it.
  * Utterance completeness is 10.0 for 99.6% of train (sigma 0.11). It is measured and
    reported but never trained — see tokens.UNTRAINED_FIELDS.
  * Word accuracy is 88% exactly 10, and bimodal: the value 4 appears zero times in train.
  * Audio is short — mean 4.1 s, p95 8.1 s, max 20.4 s. At Gemma 4's 40 ms per audio token
    that is ~100 tokens median and ~510 worst case, comfortably inside both the 750-token
    audio budget and the 1024 sliding-attention window.

The official train/test split is speaker-disjoint (125 speakers each). `speaker_split()`
preserves that property when carving a validation slice out of train: a random subset would
share speakers with what remains and read optimistically.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

DATASET_ID = "mispeech/speechocean762"
SAMPLE_RATE = 16000

#: Field ranges on their native scales, used for soft-class supports and sanity checks.
FIELD_RANGE: dict[tuple[str, str], tuple[float, float]] = {
    ("utterance", "accuracy"): (0.0, 10.0),
    ("utterance", "prosodic"): (0.0, 10.0),
    ("utterance", "fluency"): (0.0, 10.0),
    ("utterance", "total"): (0.0, 10.0),
    ("utterance", "completeness"): (0.0, 10.0),
    ("word", "accuracy"): (0.0, 10.0),
    ("word", "stress"): (5.0, 10.0),
    ("word", "total"): (0.0, 10.0),
    ("phone", "accuracy"): (0.0, 2.0),
}

#: The scale each field is actually *trained* on. Identical to FIELD_RANGE except word
#: stress, which is carried as binary {0,1} (see `stress_binary`).
TRAINING_RANGE: dict[tuple[str, str], tuple[float, float]] = {
    **FIELD_RANGE,
    ("word", "stress"): (0.0, 1.0),
}

#: Smallest standard deviation used for z-scoring, as a fraction of the field's span. A
#: near-degenerate field would otherwise divide by ~0 and turn an ordinary label into a
#: target of magnitude thousands, which log-cosh faithfully reports as an enormous loss. At
#: 2% of span the worst-case z-score is bounded near 50.
MIN_STD_FRACTION = 0.02

#: Discrete support per field, for `soft_class` heads. Phone accuracy really is 0.2-spaced.
FIELD_SUPPORT: dict[tuple[str, str], tuple[float, ...]] = {
    ("phone", "accuracy"): tuple(round(0.2 * i, 1) for i in range(11)),
    **{
        key: tuple(float(v) for v in range(11))
        for key in FIELD_RANGE
        if key[0] in ("utterance", "word") and key[1] != "stress"
    },
    ("word", "stress"): (5.0, 10.0),
}


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    accuracy: float
    stress: float
    total: float
    phones: tuple[str, ...]
    phone_accuracy: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Utterance:
    """One scored recording, flattened out of the raw dataset row.

    `audio` is float32 mono at 16 kHz. `index` is the row's position in its split, kept so
    every downstream record can be traced back to the source.
    """

    index: int
    speaker: str
    text: str
    audio: np.ndarray
    accuracy: float
    prosodic: float
    fluency: float
    total: float
    completeness: float
    words: tuple[Word, ...]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    def utterance_targets(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "prosodic": self.prosodic,
            "fluency": self.fluency,
            "total": self.total,
            "completeness": self.completeness,
        }


def stress_binary(stress: float) -> float:
    """5/10 -> 0/1, where 1 means the stress position is correct (or monosyllabic)."""
    return 1.0 if stress >= 7.5 else 0.0


def from_binary_stress(p: float) -> float:
    """Inverse of stress_binary on the expectation, back onto the reported 5-10 scale."""
    return 5.0 + 5.0 * p


def to_utterance(row: dict, index: int) -> Utterance:
    """Flatten one raw dataset row. Audio is copied to float32 once, here and nowhere else."""
    words = tuple(
        Word(
            text=w["text"],
            accuracy=float(w["accuracy"]),
            stress=float(w["stress"]),
            total=float(w["total"]),
            phones=tuple(w["phones"]),
            phone_accuracy=tuple(float(a) for a in w["phones-accuracy"]),
        )
        for w in row["words"]
    )
    for w in words:
        if len(w.phones) != len(w.phone_accuracy):
            raise ValueError(
                f"row {index} word {w.text!r}: {len(w.phones)} phones but "
                f"{len(w.phone_accuracy)} phone scores"
            )
    return Utterance(
        index=index,
        speaker=str(row["speaker"]),
        text=row["text"],
        audio=np.asarray(row["audio"]["array"], dtype=np.float32),
        accuracy=float(row["accuracy"]),
        prosodic=float(row["prosodic"]),
        fluency=float(row["fluency"]),
        total=float(row["total"]),
        completeness=float(row["completeness"]),
        words=words,
    )


def load_split(split: str, limit: int | None = None) -> list[Utterance]:
    """Load and flatten a speechocean762 split.

    Materialised as a plain list: the whole corpus is under 3 hours of audio, and holding
    it in memory removes a per-epoch decode cost that dominated the predecessor's runtime.
    """
    from datasets import load_dataset

    raw = load_dataset(DATASET_ID, split=split)
    n = len(raw) if limit is None else min(limit, len(raw))
    return [to_utterance(raw[i], i) for i in range(n)]


def speaker_split(
    utterances: list[Utterance], val_speakers: int, seed: int
) -> tuple[list[Utterance], list[Utterance]]:
    """Carve a speaker-disjoint validation slice out of a split.

    The official train/test division is speaker-disjoint by construction; a validation slice
    that shares speakers with training would leak speaker identity and read optimistically,
    which is exactly the mistake that makes a mid-training metric useless.
    """
    speakers = sorted({u.speaker for u in utterances})
    if not 0 < val_speakers < len(speakers):
        raise ValueError(
            f"val_speakers must be in (0, {len(speakers)}), got {val_speakers}"
        )
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(np.asarray(speakers))[:val_speakers].tolist())
    train = [u for u in utterances if u.speaker not in held]
    val = [u for u in utterances if u.speaker in held]
    return train, val


# --- Target normalisation -----------------------------------------------------------
# Targets are z-scored per field with train-split statistics. Correlation is invariant to
# affine transforms so this costs nothing at evaluation time, and it equalises gradient
# scale across fields whose native sigma differ by ~4x (utterance 1.5 on 0-10 vs phone 0.36
# on 0-2). Without it the utterance heads would quietly dominate the shared trunk.


@dataclass(frozen=True, slots=True)
class TargetStats:
    """Per-field mean, standard deviation and label support, keyed "<level>.<field>".

    The support is the set of values the training labels actually take — 11 for phone
    accuracy (0.0 to 2.0 in steps of 0.2), integers for the 0-10 fields. It is persisted
    because snapping predictions back onto it at eval must use the *training* grid, and
    because deriving it from the split under evaluation would be reading its labels.
    """

    mean: dict[str, float]
    std: dict[str, float]
    support: dict[str, list[float]] = dc_field(default_factory=dict)

    @staticmethod
    def key(level: str, field: str) -> str:
        return f"{level}.{field}"

    def normalize(self, level: str, field: str, value):
        k = self.key(level, field)
        return (value - self.mean[k]) / self.std[k]

    def denormalize(self, level: str, field: str, value):
        k = self.key(level, field)
        return value * self.std[k] + self.mean[k]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"mean": self.mean, "std": self.std, "support": self.support}, indent=2
        ))

    @classmethod
    def load(cls, path: Path) -> "TargetStats":
        # `support` is absent from runs trained before it was recorded; an empty mapping
        # means "no grid known", which callers treat as "do not snap" rather than guessing.
        blob = json.loads(Path(path).read_text())
        return cls(mean=blob["mean"], std=blob["std"], support=blob.get("support", {}))


def compute_target_stats(utterances: list[Utterance]) -> TargetStats:
    """Field statistics over a split. `word.stress` is stated on its binary {0,1} scale."""
    pools: dict[str, list[float]] = {}

    def add(level: str, field: str, value: float) -> None:
        pools.setdefault(TargetStats.key(level, field), []).append(value)

    for u in utterances:
        for field, value in u.utterance_targets().items():
            add("utterance", field, value)
        for w in u.words:
            add("word", "accuracy", w.accuracy)
            add("word", "stress", stress_binary(w.stress))
            add("word", "total", w.total)
            for a in w.phone_accuracy:
                add("phone", "accuracy", a)

    mean, std, support, degenerate = {}, {}, {}, []
    for key, values in pools.items():
        level, field = key.split(".", 1)
        arr = np.asarray(values, dtype=np.float64)
        low, high = TRAINING_RANGE[(level, field)]
        floor = MIN_STD_FRACTION * (high - low)
        mean[key] = float(arr.mean())
        std[key] = float(max(arr.std(), floor))
        support[key] = sorted(float(v) for v in np.unique(arr))
        if arr.std() < floor:
            degenerate.append(f"{key} (sd {arr.std():.4g} < {floor:.4g})")
    if degenerate:
        # Worth saying out loud rather than silently flooring: on the full corpus only
        # completeness is this flat, so hitting it elsewhere means the split is too small
        # or the labels are not what they are assumed to be.
        warnings.warn(
            "near-constant target fields, standard deviation floored: "
            + ", ".join(degenerate),
            stacklevel=2,
        )
    return TargetStats(mean=mean, std=std, support=support)


def phone_vocabulary(utterances: list[Utterance]) -> tuple[str, ...]:
    """Sorted ARPAbet-with-stress symbols observed in a split (66 in train, 65 in test).

    Used to size the FiLM conditioning table in heads.py. The table always carries one extra
    row for symbols absent from training, so an unseen phone degrades to a shared fallback
    rather than crashing.
    """
    return tuple(sorted({p for u in utterances for w in u.words for p in w.phones}))


def stratified_indices(utterances: list[Utterance], n: int) -> list[int]:
    """A deterministic, score-spread subset of positions into `utterances`.

    Half the picks are evenly spaced over the split sorted by utterance total; half are
    evenly spaced over the mispronunciation-carrying subset, sorted the same way. A
    contiguous prefix would be score-range-restricted and would understate correlation,
    which is the trap the predecessor documented after evaluating on offset/limit slices.
    """
    n = min(n, len(utterances))
    if n <= 0:
        return []
    order = sorted(range(len(utterances)), key=lambda i: (utterances[i].total, i))
    hard = [
        i
        for i in order
        if any(a <= 1.0 for w in utterances[i].words for a in w.phone_accuracy)
    ]

    def spread(pool: list[int], k: int) -> list[int]:
        if k <= 0 or not pool:
            return []
        step = len(pool) / k
        return [pool[min(len(pool) - 1, int(math.floor(j * step)))] for j in range(k)]

    picked = list(dict.fromkeys(spread(order, n - n // 2) + spread(hard, n // 2)))
    for i in order:  # top up if the two pools overlapped
        if len(picked) >= n:
            break
        if i not in picked:
            picked.append(i)
    return sorted(picked[:n])
