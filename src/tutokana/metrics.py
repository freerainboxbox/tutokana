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


# --- detection ---------------------------------------------------------------------------
# Spearman on this corpus is a detection metric in disguise. Decomposing the achievable SCC
# on the test split shows where it actually lives:
#
#   phone.accuracy       perfect predictor (ceiling)                    0.680
#                        perfect ceiling-vs-not, random ordering below  0.671   <- 99%
#                        perfect ordering below, ceiling not detected   0.077   <-  1%
#
# So "improve the rank correlation" means "improve mispronunciation detection", and a
# differentiable ranking loss would spend its gradient on orderings the metric cannot
# reward. Reporting detection directly says so out loud instead of leaving it to be inferred
# from a rank correlation.


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Separating imperfect labels from the ceiling, treating the score as a detector."""

    positives: int          #: labels below the maximum — the mispronunciations
    share: float            #: their share of the field, i.e. the base rate
    auc: float              #: threshold-free ranking quality
    f1: float               #: best F1 over all thresholds
    precision: float
    recall: float
    threshold: float        #: predictions at or below this are called imperfect


def detection_auc(prediction, gold) -> float:
    """Probability a randomly chosen imperfect label scores below a randomly chosen perfect
    one. Computed from ranks (Mann-Whitney U), so ties are handled and no sweep is needed."""
    from scipy.stats import rankdata

    pred = np.asarray(prediction, dtype=np.float64)
    positive = np.asarray(gold, dtype=np.float64) < np.max(gold) if len(gold) else np.array([])
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Lower prediction should mean "more likely imperfect", so rank the negated score.
    ranks = rankdata(-pred)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detection_metrics(prediction, gold) -> DetectionMetrics:
    """Detection of "label below the maximum", scored from the continuous prediction.

    The threshold is chosen to maximise F1 rather than fixed at a midpoint: the base rate is
    0.8-19% depending on the field, so any fixed threshold would be arbitrary and would
    understate a model that ranks well but is poorly calibrated.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(gold, dtype=np.float64)
    positive = truth < truth.max() if truth.size else np.array([], dtype=bool)
    n_pos = int(positive.sum())
    if truth.size == 0 or n_pos == 0 or n_pos == truth.size:
        nan = float("nan")
        return DetectionMetrics(n_pos, n_pos / max(truth.size, 1), nan, nan, nan, nan, nan)

    # Sweep every distinct threshold in one pass: sort ascending, and calling everything up
    # to position i "imperfect" makes true positives the cumulative count of positives.
    order = np.argsort(pred, kind="mergesort")
    hits = np.cumsum(positive[order])
    called = np.arange(1, pred.size + 1)
    precision = hits / called
    recall = hits / n_pos
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(precision), where=denominator > 0)
    best = int(np.argmax(f1))
    return DetectionMetrics(
        positives=n_pos,
        share=n_pos / truth.size,
        auc=detection_auc(pred, truth),
        f1=float(f1[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        threshold=float(pred[order][best]),
    )


def snap_to_support(prediction, support) -> np.ndarray:
    """Round predictions onto the values the labels actually take.

    Gold is quantised — phone accuracy to 0.2, the 0-10 fields to integers — and a continuous
    readout is not. Snapping restores the ties that quantisation created, which is what a
    rank correlation is comparing against, and removes error smaller than half a grid step.
    It is a change of scale, not a change of model: no retraining, and it cannot invent
    ordering the predictions did not already have.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    grid = np.asarray(sorted(set(float(v) for v in support)), dtype=np.float64)
    if grid.size == 0:
        return pred
    return grid[np.abs(pred[:, None] - grid[None, :]).argmin(axis=1)]


def spearman_ceiling(gold, seed: int = 0, repeats: int = 5) -> float:
    """The highest Spearman any *continuous* predictor can reach on these labels.

    A continuous readout never produces ties, while gold is heavily tied — 81% of phone
    accuracies at 2.0, 90% of word accuracies at 10. Reproducing gold exactly and breaking
    its ties arbitrarily is therefore the best such a predictor can do, and that is what this
    measures. On the test split it comes to 0.680 for phone accuracy and 0.528 for word
    accuracy, against 0.96 for the utterance fields, which is most of why the levels look so
    different. Without this column a reader compares 0.43 against 0.72 and draws the wrong
    conclusion.
    """
    truth = np.asarray(gold, dtype=np.float64)
    if truth.size < 2 or truth.std() == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    scale = 1e-9 * max(float(np.abs(truth).max()), 1.0)
    return float(
        np.mean([spearman(truth + rng.normal(0, scale, truth.size), truth)
                 for _ in range(repeats)])
    )
