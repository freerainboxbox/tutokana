"""The localisation probe's ablation and bucketing, checked without loading a model.

The probe decides an architecture question — whether a forced aligner would add anything —
so its arithmetic needs to be right before a result is read off it. These tests confirm the
zeroing touches only the intended span, and that the distance bucketing actually separates
the two hypotheses it is meant to distinguish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from probe_phone_localization import zero_segment


class _Utterance:
    """Enough of an Utterance for `dataclasses.replace` to be unnecessary."""

    def __init__(self, audio):
        self.audio = audio


def _bucket_means(counts, segments, kernel):
    position = np.concatenate([(np.arange(n) + 0.5) / n for n in counts])
    buckets: dict[int, list[float]] = {}
    for index in range(segments):
        centre = (index + 0.5) / segments
        distance = np.abs(position - centre) * segments
        moved = kernel(distance / segments * 2)
        for b in range(segments):
            mask = (distance >= b) & (distance < b + 1)
            if mask.any():
                buckets.setdefault(b, []).extend(moved[mask].tolist())
    return [float(np.mean(buckets[b])) for b in sorted(buckets)]


def test_zeroing_silences_only_the_named_span():
    from dataclasses import dataclass

    @dataclass
    class U:
        audio: np.ndarray

    original = np.ones(1000, dtype=np.float32)
    altered = zero_segment(U(audio=original), 2, 5)
    assert np.all(altered.audio[400:600] == 0.0)
    assert np.all(altered.audio[:400] == 1.0)
    assert np.all(altered.audio[600:] == 1.0)


def test_zeroing_does_not_mutate_the_original():
    """The reference pass is scored from the same objects; mutating them would silently
    compare altered audio against altered audio and report no effect at all."""
    from dataclasses import dataclass

    @dataclass
    class U:
        audio: np.ndarray

    original = np.ones(500, dtype=np.float32)
    utterance = U(audio=original)
    zero_segment(utterance, 0, 5)
    assert np.all(utterance.audio == 1.0)
    assert np.all(original == 1.0)


def test_segments_tile_the_audio_exactly():
    from dataclasses import dataclass

    @dataclass
    class U:
        audio: np.ndarray

    audio = np.ones(997, dtype=np.float32)  # deliberately not divisible
    silenced = np.zeros(997, dtype=bool)
    for index in range(5):
        silenced |= zero_segment(U(audio=audio), index, 5).audio == 0.0
    assert silenced.all(), "every sample must fall in exactly one segment"


def test_bucketing_shows_a_peak_for_a_localised_model():
    means = _bucket_means([20, 15, 25], 5, lambda d: np.exp(-8 * d**2))
    assert means[0] > 5 * means[1]
    assert means == sorted(means, reverse=True)


def test_bucketing_is_flat_for_a_model_that_reads_everything():
    means = _bucket_means([20, 15, 25], 5, np.ones_like)
    assert max(means) - min(means) < 1e-9
