"""Metrics. The standard eight correlations, plus the columns that catch a lie.

speechocean762 is scored by Pearson correlation per aspect per level, flattened over the
whole test split (all 2500 utterances, all ~16k words, all ~47k phones) — not macro-averaged
per utterance. That is the protocol GOPT established and every paper since has followed, so
`pearson` is what goes in the comparison column.

Two additions the predecessor's table lacked, both of which exist because a bare correlation
can hide the failure this project is about:

* **Spearman.** Both recent Microsoft papers argue Pearson is inflated on speechocean762's
  skewed marginals — on their private set Pearson runs 0.87-0.95 while Spearman sits at
  0.57-0.62. Reporting both makes that gap visible instead of flattering the result.
* **sigma_pred / sigma_gold.** This is the column that exposed the original problem. A model
  that has learned the marginal and nothing else produces a ratio near 0 while its
  correlation may still look respectable; two of the predecessor's fields were literally
  constant. Anything much below ~0.8 means the predictions are shrunk toward the mean.

`pearson` returns NaN when either side is constant, which is the honest answer and is
exactly how the collapse showed up before. Do not paper over it with a zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FieldMetrics:
    """Everything reported for one (level, field) pair."""

    n: int
    pearson: float
    pearson_lo: float
    pearson_hi: float
    spearman: float
    mse: float
    mae: float
    pred_mean: float
    pred_std: float
    gold_mean: float
    gold_std: float

    @property
    def sigma_ratio(self) -> float:
        return float("nan") if self.gold_std == 0 else self.pred_std / self.gold_std


def pearson(prediction, gold) -> float:
    """Pearson correlation; NaN if either side is constant (correlation is undefined)."""
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(gold, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(prediction, gold) -> float:
    """Rank correlation, with average ranks for ties (labels here are heavily tied)."""
    from scipy.stats import rankdata

    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(gold, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def bootstrap_pearson(
    prediction, gold, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for the correlation.

    Worth the cost: differences below roughly 0.02 on this test set are inside the interval,
    and the literature routinely reports single-seed gaps smaller than that as improvements.
    """
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(gold, dtype=np.float64)
    if a.size < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, a.size, a.size)
        draws[i] = pearson(a[idx], b[idx])
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def field_metrics(prediction, gold, bootstrap: bool = True, seed: int = 0) -> FieldMetrics:
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(gold, dtype=np.float64)
    lo, hi = bootstrap_pearson(a, b, seed=seed) if bootstrap else (float("nan"),) * 2
    return FieldMetrics(
        n=int(a.size),
        pearson=pearson(a, b),
        pearson_lo=lo,
        pearson_hi=hi,
        spearman=spearman(a, b),
        mse=float(np.mean((a - b) ** 2)) if a.size else float("nan"),
        mae=float(np.mean(np.abs(a - b))) if a.size else float("nan"),
        pred_mean=float(a.mean()) if a.size else float("nan"),
        pred_std=float(a.std()) if a.size else float("nan"),
        gold_mean=float(b.mean()) if b.size else float("nan"),
        gold_std=float(b.std()) if b.size else float("nan"),
    )


# --- Transcription metrics (generative mode only) ------------------------------------


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def strip_stress(phones: list[str]) -> list[str]:
    """Drop the trailing stress digit, so PER can be reported with and without it."""
    return [p[:-1] if p and p[-1].isdigit() else p for p in phones]


@dataclass(frozen=True, slots=True)
class TranscriptionMetrics:
    phone_error_rate: float
    phone_error_rate_no_stress: float
    word_exact_match: float
    n_words: int
    n_phones: int


def transcription_metrics(
    predicted: list[list[tuple[str, list[str]]]],
    gold: list[list[tuple[str, list[str]]]],
) -> TranscriptionMetrics:
    """Phone error rate and word exact-match over paired (word, phones) transcripts.

    Words are aligned positionally; a length mismatch counts every unmatched gold word as
    fully wrong rather than being skipped. Silently dropping misaligned words is how a
    transcription number gets flattered, and the predecessor's phone correlation was
    computed over only the 98% of words whose lengths happened to agree.
    """
    edits = edits_no_stress = phones_total = 0
    exact = words_total = 0
    for pred_words, gold_words in zip(predicted, gold):
        for i, (gold_word, gold_phones) in enumerate(gold_words):
            words_total += 1
            phones_total += len(gold_phones)
            if i < len(pred_words):
                pred_word, pred_phones = pred_words[i]
            else:
                pred_word, pred_phones = "", []
            exact += int(pred_word == gold_word and pred_phones == gold_phones)
            edits += levenshtein(pred_phones, gold_phones)
            edits_no_stress += levenshtein(
                strip_stress(pred_phones), strip_stress(gold_phones)
            )
    return TranscriptionMetrics(
        phone_error_rate=edits / max(phones_total, 1),
        phone_error_rate_no_stress=edits_no_stress / max(phones_total, 1),
        word_exact_match=exact / max(words_total, 1),
        n_words=words_total,
        n_phones=phones_total,
    )
