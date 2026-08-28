"""Training mix: entropy-greedy oversampling plus synthetic negatives.

80.1% of phone accuracies are exactly 2.0 and 88.0% of word accuracies are exactly 10, so the
raw split is dominated by correct speech. `compute_multiplicities` greedily repeats
utterances to maximise phone-class entropy, and `sample_negatives` pairs transcripts with
mismatched audio using disjoint word sets to manufacture unambiguous mispronunciations.
"""

from __future__ import annotations

import random

import numpy as np

from .data import Utterance, Word


def phone_class_counts(utterances: list[Utterance]) -> np.ndarray:
    """(N, 3) per-utterance counts of phone scores rounded into classes {0, 1, 2}."""
    counts = np.zeros((len(utterances), 3), dtype=np.int64)
    for i, u in enumerate(utterances):
        for w in u.words:
            for score in w.phone_accuracy:
                counts[i, int(round(score))] += 1
    return counts


def _entropy(class_totals: np.ndarray) -> float:
    p = class_totals / class_totals.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def compute_multiplicities(utterances: list[Utterance], k: int) -> np.ndarray:
    """Integer multiplicities in [1, k] greedily maximising phone-class entropy.

    Deterministic: no RNG, ties break by lowest index through `argmax`.
    """
    counts = phone_class_counts(utterances)
    mult = np.ones(len(utterances), dtype=np.int64)
    totals = counts.sum(axis=0).astype(np.float64)
    best = _entropy(totals)
    # Only utterances holding at least one non-majority phone can raise the entropy.
    candidates = np.where(counts[:, :2].sum(axis=1) > 0)[0]
    while True:
        available = candidates[mult[candidates] < k]
        if len(available) == 0:
            break
        proposed = totals[None, :] + counts[available]
        p = proposed / proposed.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropies = -(np.where(p > 0, p * np.log(p), 0.0)).sum(axis=1)
        j = int(np.argmax(entropies))
        if entropies[j] <= best + 1e-12:
            break
        best = entropies[j]
        totals += counts[available[j]]
        mult[available[j]] += 1
    return mult


def make_negative(a: Utterance, b: Utterance, rng: random.Random) -> Utterance:
    """Transcript and canonical phones from A, audio and delivery scores from B."""
    severity = rng.choice([0, 1, 2])
    words = tuple(
        Word(
            text=w.text,
            accuracy=(0.0 if severity == 0 else float(rng.choice([0, 1]))),
            stress=10.0,  # the dataset's default where stress is unratable
            total=(0.0 if severity == 0 else float(rng.choice([0, 1]))),
            phones=w.phones,  # canonical phones are transcript-derived, so A's stand
            phone_accuracy=(0.0,) * len(w.phones),
        )
        for w in a.words
    )
    return Utterance(
        index=-1 - a.index,  # negative index marks a synthetic row in any trace
        speaker=b.speaker,
        text=a.text,
        audio=b.audio,
        accuracy=float(severity),
        prosodic=b.prosodic,
        fluency=b.fluency,
        total=float(severity),
        completeness=0.0,
        words=words,
    )


def sample_negatives(
    utterances: list[Utterance], n_pairs: int, seed: int
) -> list[Utterance]:
    """`n_pairs` synthetic transcript/audio mismatches with disjoint word sets.

    One RNG stream drives both pair selection and label construction, so the whole set is
    reproducible from (seed, n_pairs) alone.
    """
    if n_pairs <= 0:
        return []
    rng = random.Random(seed)
    word_sets = [frozenset(w.text for w in u.words) for u in utterances]
    n = len(utterances)
    negatives: list[Utterance] = []
    used: set[tuple[int, int]] = set()
    attempts = 0
    while len(negatives) < n_pairs:
        attempts += 1
        if attempts > n_pairs * 200:
            raise RuntimeError(
                f"could not find {n_pairs} word-disjoint pairs "
                f"(found {len(negatives)} in {attempts} attempts)"
            )
        a, b = rng.randrange(n), rng.randrange(n)
        if a == b or (a, b) in used or (word_sets[a] & word_sets[b]):
            continue
        used.add((a, b))
        negatives.append(make_negative(utterances[a], utterances[b], rng))
    return negatives


def build_mix(
    utterances: list[Utterance], k: int, n_negatives: int, seed: int
) -> tuple[list[Utterance], dict]:
    """The epoch's sample list plus an auditable summary of how it was built.

    Oversampled duplicates are shared references, not copies, so the mix costs no extra
    memory beyond the list itself; the per-epoch shuffle spreads them apart.
    """
    mult = compute_multiplicities(utterances, k) if k > 1 else np.ones(len(utterances), np.int64)
    negatives = sample_negatives(utterances, n_negatives, seed)

    mixed: list[Utterance] = []
    for u, m in zip(utterances, mult):
        mixed.extend([u] * int(m))
    mixed.extend(negatives)

    counts = phone_class_counts(utterances)
    base_totals = counts.sum(axis=0).astype(np.float64)
    mixed_totals = (counts * mult[:, None]).sum(axis=0).astype(np.float64)
    with_mispronunciation = sum(
        1 for u in utterances if any(a <= 1.0 for w in u.words for a in w.phone_accuracy)
    )
    stats = {
        "base_samples": len(utterances),
        "oversampled_samples": int(mult.sum()),
        "n_negatives": len(negatives),
        "epoch_samples": len(mixed),
        "multiplicity_max": int(mult.max()),
        "phone_entropy_base": _entropy(base_totals) / float(np.log(3)),
        "phone_entropy_mixed": _entropy(mixed_totals) / float(np.log(3)),
        "mispronunciation_share_base": with_mispronunciation / max(len(utterances), 1),
    }
    return mixed, stats
