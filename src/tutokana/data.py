"""speechocean762 loading and per-field targets.

Targets are z-scored per field with training-split statistics, persisted to the run directory
and restored at eval. Correlation is affine-invariant so this is free, and it equalises
gradient scale across fields whose standard deviations differ several-fold.

Standard deviations are floored at `MIN_STD_FRACTION` of the field's range. Without it a
near-constant field (completeness is 10.0 for 99.6% of the corpus) produces z-scores in the
thousands.

`TargetStats.support` records the values each label actually takes. Predictions can be
rounded back onto that grid at eval (`evaluate.py --snap`); it must be the *training* grid,
since deriving it from the split under evaluation would be reading that split's labels.

Word stress additionally carries `Word.stress_votes`, the five experts' individual verdicts
recovered from the corpus's `scores-detail.json`. See the README's "Word stress" note.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

DATASET_ID = "mispeech/speechocean762"
SAMPLE_RATE = 16000

#: Every utterance is scored by five experts independently. HuggingFace ships only the
#: aggregate; the individual verdicts live in the corpus's own repository.
STRESS_RATERS = 5
_CORPUS_RAW = "https://raw.githubusercontent.com/jimbozhang/speechocean762/master"
STRESS_DETAIL_URL = f"{_CORPUS_RAW}/resource/scores-detail.json"
SPLIT_INDEX_URL = _CORPUS_RAW + "/{split}/text"

#: Beta-binomial concentration for the five stress verdicts, by moment matching on the
#: training split: Var(k) = n*p*(1-p) + n*(n-1)*Var(p) gives Var(p) = 0.0052 about a mean
#: p of 0.947, hence nu = p*(1-p)/Var(p) - 1 = 8.63. Held fixed; see README.
STRESS_CONCENTRATION = 8.6

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
    """One scored word.

    `stress` is the released label, which is the *median* of the five experts' {5, 10}
    verdicts and therefore takes only those two values. `stress_votes` is how many of the
    five judged the stress correct, 0-5, which is the label the head is trained on.
    """

    text: str
    accuracy: float
    stress: float
    stress_votes: int
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


def stress_median(votes: int) -> float:
    """The released 5/10 label: the median of the five experts' {5, 10} verdicts.

    A word is released as 10 ("stress correct, or monosyllabic") as soon as three of the
    five say so, which is why the published label has no intermediate values.
    """
    return 10.0 if 2 * votes > STRESS_RATERS else 5.0


def _cache_dir() -> Path:
    """Where the corpus-repository files land — the Hugging Face cache, never the repo."""
    root = os.environ.get("HF_HOME")
    return (Path(root) if root else Path.home() / ".cache" / "huggingface") / "tutokana"


def _fetch(url: str, name: str) -> Path:
    """Download `url` once, atomically, and return the cached path."""
    path = _cache_dir() / name
    if path.exists():
        return path
    from urllib.request import urlopen

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urlopen(url) as response:
        partial.write_bytes(response.read())
    partial.rename(path)  # a half-written file must never be mistaken for a cached one
    return path


def load_stress_votes(split: str) -> list[tuple[int, ...]]:
    """Per-word counts of experts who judged the stress correct, in split order.

    HuggingFace ships the aggregated `scores.json` only, so the five individual verdicts
    come from the corpus repository. Neither file carries an utterance id into the
    HuggingFace rows, so alignment is positional against the official split index and is
    verified against the aggregate in `load_split` rather than trusted.
    """
    detail = json.loads(_fetch(STRESS_DETAIL_URL, "scores-detail.json").read_text())
    index = _fetch(SPLIT_INDEX_URL.format(split=split), f"{split}-text").read_text()
    ids = [line.split("\t", 1)[0] for line in index.splitlines() if line.strip()]
    return [
        tuple(
            sum(1 for v in word["stress"] if v == 10.0) for word in detail[key]["words"]
        )
        for key in ids
    ]


def to_utterance(row: dict, index: int, stress_votes: tuple[int, ...]) -> Utterance:
    """Flatten one raw dataset row. Audio is copied to float32 once, here and nowhere else."""
    if len(stress_votes) != len(row["words"]):
        raise ValueError(
            f"row {index}: {len(row['words'])} words but {len(stress_votes)} vote counts "
            f"— the split index and the HuggingFace rows are out of order"
        )
    words = tuple(
        Word(
            text=w["text"],
            accuracy=float(w["accuracy"]),
            stress=float(w["stress"]),
            stress_votes=int(votes),
            total=float(w["total"]),
            phones=tuple(w["phones"]),
            phone_accuracy=tuple(float(a) for a in w["phones-accuracy"]),
        )
        for w, votes in zip(row["words"], stress_votes)
    )
    for w in words:
        if len(w.phones) != len(w.phone_accuracy):
            raise ValueError(
                f"row {index} word {w.text!r}: {len(w.phones)} phones but "
                f"{len(w.phone_accuracy)} phone scores"
            )
        if stress_median(w.stress_votes) != w.stress:
            raise ValueError(
                f"row {index} word {w.text!r}: {w.stress_votes}/5 experts scored the "
                f"stress correct, whose median is {stress_median(w.stress_votes)}, but the "
                f"released label is {w.stress} — the vote file is misaligned with the split"
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
    it in memory removes a per-epoch decode cost.
    """
    from datasets import load_dataset

    raw = load_dataset(DATASET_ID, split=split)
    votes = load_stress_votes(split)
    if len(votes) != len(raw):
        raise ValueError(
            f"split {split!r} has {len(raw)} rows but the official index lists "
            f"{len(votes)} utterances"
        )
    n = len(raw) if limit is None else min(limit, len(raw))
    return [to_utterance(raw[i], i, votes[i]) for i in range(n)]


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
    """Field statistics over a split, each field on its own reported scale.

    `word.stress` is stated as the released 5/10 median, not as the vote count the head is
    trained on: these statistics normalise the *reported* target, which is what the
    correlation term and every metric speak.
    """
    pools: dict[str, list[float]] = {}

    def add(level: str, field: str, value: float) -> None:
        pools.setdefault(TargetStats.key(level, field), []).append(value)

    for u in utterances:
        for field, value in u.utterance_targets().items():
            add("utterance", field, value)
        for w in u.words:
            add("word", "accuracy", w.accuracy)
            add("word", "stress", w.stress)
            add("word", "total", w.total)
            for a in w.phone_accuracy:
                add("phone", "accuracy", a)

    mean, std, support, degenerate = {}, {}, {}, []
    for key, values in pools.items():
        level, field = key.split(".", 1)
        arr = np.asarray(values, dtype=np.float64)
        low, high = FIELD_RANGE[(level, field)]
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
    which is what a contiguous offset/limit slice would give.
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
