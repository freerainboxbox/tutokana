"""The composite objective.

    L = sum_level w_level * [ mean(pointwise) + lambda_ccc * (1 - CCC) ]
        + lambda_detect * BCE(below ceiling)  +  lambda_lm * CE(text)

The labels are saturated — 80.1% of phone accuracies are exactly 2.0, 88.0% of word
accuracies exactly 10 — so "predict the mode" is a strong local optimum for any purely
pointwise loss, and the failure mode is variance collapse. Four things push back:

* **log-cosh** rather than MSE: smooth, gradient bounded by 1, so the ~20% of low scores are
  not drowned out by an outlier-squared term.
* **Concordance correlation, not Pearson.** CCC penalises shrunken variance and shifted mean;
  Pearson is invariant to both and would accept a collapsed prediction rescaled after the fact.
* **A buffer**, because one batch is not a population. `CorrelationBuffer` holds a detached
  FIFO of recent (prediction, target) pairs and estimates the statistic over live batch plus
  buffer. Gradient flows only through the live batch.
* **Label-frequency reweighting**, capped in dynamic range.

Detection (`lambda_detect`) is separate from all of that: it optimises "is this label below
the ceiling", which is ~99% of the achievable rank correlation.

Level losses are normalised by element count before weighting, or phones (~19 per utterance)
drown out the four utterance registers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def log_cosh(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise log(cosh(x)), evaluated stably for large residuals.

    The naive form overflows once |x| passes ~89 in float32; this identity is exact and
    costs one extra softplus.
    """
    diff = prediction - target
    return diff.abs() + F.softplus(-2.0 * diff.abs()) - math.log(2.0)


def concordance(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Lin's concordance correlation coefficient, in [-1, 1]."""
    if prediction.numel() < 2:
        return prediction.new_tensor(0.0)
    pred_mean, target_mean = prediction.mean(), target.mean()
    pred_centered, target_centered = prediction - pred_mean, target - target_mean
    covariance = (pred_centered * target_centered).mean()
    pred_var, target_var = pred_centered.pow(2).mean(), target_centered.pow(2).mean()
    denominator = pred_var + target_var + (pred_mean - target_mean).pow(2) + eps
    return 2.0 * covariance / denominator


class CorrelationBuffer:
    """Detached FIFO of recent (prediction, target) pairs, one ring per head.

    Only the live batch carries gradient; the buffer contributes moments alone. That makes
    the correlation term usable at any batch size, which matters because the batch sizes
    that fit a 12B model are far below what a stable utterance-level correlation needs.

    Entries stay on whatever device they arrive on. A few thousand floats is nothing next to
    the model, and copying to host every step would add an accelerator synchronisation per
    head per step for no benefit.
    """

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._predictions: dict[str, torch.Tensor] = {}
        self._targets: dict[str, torch.Tensor] = {}

    def extend(self, key: str, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        pred = prediction.detach().flatten().float()
        tgt = target.detach().flatten().float()
        # History follows the live batch's device: a resumed run restores this buffer from
        # the host, and concatenating across devices segfaults on MPS rather than raising.
        self._predictions[key] = torch.cat(
            [self._history(self._predictions, key, pred), pred]
        )[-self.capacity :]
        self._targets[key] = torch.cat(
            [self._history(self._targets, key, tgt), tgt]
        )[-self.capacity :]

    @staticmethod
    def _history(store: dict[str, torch.Tensor], key: str, live: torch.Tensor) -> torch.Tensor:
        held = store.get(key)
        return live[:0] if held is None else held.to(live.device)

    def augmented(
        self, key: str, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_pred = self._predictions.get(key)
        if history_pred is None or history_pred.numel() == 0:
            return prediction, target
        device = prediction.device
        return (
            torch.cat([prediction, history_pred.to(device)]),
            torch.cat([target, self._targets[key].to(device)]),
        )

    def reset(self) -> None:
        self._predictions.clear()
        self._targets.clear()

    def state_dict(self) -> dict:
        """Saved across a resume: a cold buffer would leave the correlation term estimating
        its moments from a handful of points for the first few hundred steps."""
        return {
            "capacity": self.capacity,
            "predictions": {k: v.cpu() for k, v in self._predictions.items()},
            "targets": {k: v.cpu() for k, v in self._targets.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = state["capacity"]
        self._predictions = dict(state["predictions"])
        self._targets = dict(state["targets"])


@dataclass
class LossConfig:
    """Knobs for the composite objective.

    `level_weights` multiply each level's element-count-normalised loss. `lambda_ccc`
    weights the correlation term; 0 disables it, which is the ablation arm that isolates
    whether the pointwise loss alone is enough.

    `reweight_levels` names the levels whose pointwise loss is reweighted by inverse label
    frequency. It defaults to phone alone, because reweighting answers a classification
    question — "this decision boundary sees too few examples of that class" — and only the
    phone head is a classifier. On a `regression` head, a label weight multiplies a log-cosh
    *residual*, which does not rebalance anything; it just declares that being wrong about a
    rare label costs more, and the rarest labels are also the noisiest. `--reweight-levels
    phone,word` restores the previous behaviour as an ablation.

    `word.stress` is reweighted regardless of this setting: even over the five annotators'
    vote count, 80% of words are a clean sweep, and there is no sane configuration in which
    an unbalanced stress head is the intended experiment.
    """

    level_weights: dict[str, float] = field(
        default_factory=lambda: {"phone": 1.0, "word": 1.0, "utterance": 1.0}
    )
    lambda_ccc: float = 0.5
    lambda_lm: float = 0.5
    buffer_capacity: int = 512
    label_smoothing: float = 0.05
    reweight_levels: tuple[str, ...] = ("phone",)
    reweight_strength: float = 1.0
    #: Largest ratio between any two label weights, applied before the mean-1 pass. Uncapped
    #: inverse frequency spans a 4176x range here (word accuracy 9 occurs 3 times in 15849
    #: words), which makes the loss a sawtooth driven by whichever rare label is in the batch.
    reweight_max: float = 10.0
    #: Weight on the auxiliary "is this label below the ceiling?" term; 0 is the ablation.
    #: Detection is ~99% of the achievable Spearman, and the multi-class cross-entropy alone
    #: spends most of its mass separating 1.8 from 2.0 instead.
    lambda_detect: float = 0.5
    #: Positive-class weight in that term; 4.3 balances phone accuracy's 19% base rate, and
    #: happens to suit word stress too — 20.0% of words draw at least one dissenting expert.
    detect_pos_weight: float = 4.3


#: Reweighted whether or not its level is listed — see LossConfig.
ALWAYS_REWEIGHTED = ("word.stress",)


class LabelReweighter:
    """Inverse-frequency weights over a field's discrete support.

    Weights are normalised to mean 1 over the training marginal, so turning reweighting on
    does not implicitly change the learning rate — only the relative pull of rare labels.
    `strength` interpolates: 0 is uniform, 1 is full inverse frequency.

    Mean 1 is necessary but not sufficient: the tail is what destabilises training, and these
    label distributions have a very long one. Word accuracy 9 occurs 3 times in 15849 words,
    so uncapped inverse frequency hands it a 476x multiplier — a single word outweighing
    several hundred ordinary ones, which is how the loss became a 10-148 sawtooth instead of
    a curve.

    `maximum` bounds the *dynamic range*: no label may pull more than `maximum` times the
    least-weighted one. Clamping the ratio before normalising keeps both guarantees exact in
    one pass — clamping afterwards does not, because the renormalisation that restores mean 1
    scales the clamped values back above the cap.
    """

    def __init__(
        self,
        histograms: dict[str, dict[float, int]],
        strength: float = 1.0,
        maximum: float = 10.0,
    ):
        self.tables: dict[str, tuple[tuple[float, ...], torch.Tensor]] = {}
        for key, histogram in histograms.items():
            values = tuple(sorted(histogram))
            counts = torch.tensor([histogram[v] for v in values], dtype=torch.float32)
            probability = counts / counts.sum()
            weights = probability.clamp_min(1e-8).pow(-strength)
            if maximum and maximum > 0:
                weights = weights.clamp(max=float(weights.min()) * maximum)
            weights = weights / (weights * probability).sum()  # mean 1 under the marginal
            self.tables[key] = (values, weights)

    def weights_for(self, key: str, raw: torch.Tensor) -> torch.Tensor | None:
        table = self.tables.get(key)
        if table is None:
            return None
        values, weights = table
        support = raw.new_tensor(values)
        index = (raw.unsqueeze(-1) - support).abs().argmin(dim=-1)
        return weights.to(raw.device)[index]


def detection_logit(head, logits: torch.Tensor) -> torch.Tensor | None:
    """Log-odds that this label is *below* the head's ceiling, from the existing logits.

    A `soft_class` head already carries the answer: `log(1 - p_top) - log(p_top)` is exactly
    the logit of "not at the top class", and computing it as a logsumexp difference keeps it
    stable without ever materialising a probability. Zero new parameters — the readout was
    throwing this away by collapsing the distribution to its expectation, which conflates
    "confidently 1.6" with "unsure between 1.0 and 2.0" though only the second is suspicious.

    A `binomial` head carries the same thing over its vote distribution: "below the ceiling"
    is "not a clean sweep of the panel".

    Returns None for heads with no distribution to read (`regression`, `binary`).
    """
    if head.mode == "binomial":
        return head.flagged_logit(logits)
    if head.mode != "soft_class" or head.support.numel() == 0:
        return None
    top = int(torch.argmax(head.support))
    others = [i for i in range(logits.shape[-1]) if i != top]
    if not others:
        return None
    return torch.logsumexp(logits[..., others], dim=-1) - logits[..., top]


def detection_loss(head, logits: torch.Tensor, batch, config: LossConfig) -> torch.Tensor | None:
    """Binary cross-entropy on "below the ceiling", or None if the head cannot express it."""
    logit = detection_logit(head, logits)
    if logit is None:
        return None
    # `raw` is the vote count for a binomial head and the native score otherwise, so the
    # ceiling is whatever a top rating looks like on the scale the loss is reading.
    ceiling = float(head.n_raters) if head.mode == "binomial" else float(head.support.max())
    target = (batch.raw < ceiling).to(logit.dtype)
    weight = logit.new_tensor(config.detect_pos_weight)
    return F.binary_cross_entropy_with_logits(logit, target, pos_weight=weight)


def head_loss(
    head,
    logits: torch.Tensor,
    batch,
    config: LossConfig,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pointwise loss for one head, plus its normalized prediction for the CCC term."""
    if head.mode == "regression":
        elementwise = log_cosh(logits.squeeze(-1), batch.target)
    elif head.mode == "binomial":
        # `raw` is the annotator vote count, not a score: the likelihood of the panel.
        elementwise = -head.vote_log_likelihood(logits, batch.raw)
    elif head.mode == "binary":
        # Class balance comes from the mean-1 inverse-frequency weights below rather than
        # `pos_weight`, which would upweight whichever class happens to be the majority.
        elementwise = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), batch.raw, reduction="none"
        )
    else:
        elementwise = F.cross_entropy(
            logits,
            head.class_targets(batch.raw),
            label_smoothing=config.label_smoothing,
            reduction="none",
        )

    if sample_weight is not None:
        elementwise = elementwise * sample_weight
    return elementwise.mean(), head.predict_normalized(logits)


def composite_loss(
    head_bank,
    head_outputs: dict[str, torch.Tensor],
    heads: dict,
    config: LossConfig,
    buffer: CorrelationBuffer | None = None,
    reweighter: LabelReweighter | None = None,
    lm_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The full objective, and a flat dict of its parts for logging.

    Returns `(total, parts)`. `parts` is plain floats — safe to hand straight to wandb or a
    log line without dragging the graph along.
    """
    device = next(iter(head_outputs.values())).device if head_outputs else None
    total = torch.zeros((), device=device) if device is not None else torch.zeros(())
    parts: dict[str, float] = {}

    for key, logits in head_outputs.items():
        level = key.split(".", 1)[0]
        batch = heads[key]
        weight = config.level_weights.get(level, 1.0)
        if weight == 0.0:
            continue
        head = head_bank.head(key)

        sample_weight = None
        if reweighter is not None and (
            level in config.reweight_levels or key in ALWAYS_REWEIGHTED
        ):
            sample_weight = reweighter.weights_for(key, batch.raw)

        pointwise, normalized = head_loss(head, logits, batch, config, sample_weight)
        term = pointwise
        parts[f"loss/{key}"] = float(pointwise.detach())

        if config.lambda_detect > 0.0:
            detection = detection_loss(head, logits, batch, config)
            if detection is not None:
                term = term + config.lambda_detect * detection
                parts[f"detect/{key}"] = float(detection.detach())

        if config.lambda_ccc > 0.0:
            if buffer is not None:
                augmented_pred, augmented_target = buffer.augmented(
                    key, normalized, batch.target
                )
            else:
                augmented_pred, augmented_target = normalized, batch.target
            ccc = concordance(augmented_pred, augmented_target)
            term = term + config.lambda_ccc * (1.0 - ccc)
            parts[f"ccc/{key}"] = float(ccc.detach())

        if buffer is not None:
            buffer.extend(key, normalized, batch.target)

        total = total + weight * term

    if lm_loss is not None and config.lambda_lm > 0.0:
        total = total + config.lambda_lm * lm_loss
        parts["loss/lm"] = float(lm_loss.detach())

    parts["loss/total"] = float(total.detach())
    return total, parts


def build_reweighter(utterances, config: LossConfig) -> LabelReweighter:
    """Label histograms over the training split, for the inverse-frequency weights."""
    from .data import TargetStats

    histograms: dict[str, dict[float, int]] = {}

    def add(level: str, field_name: str, value: float) -> None:
        key = TargetStats.key(level, field_name)
        histograms.setdefault(key, {})
        histograms[key][value] = histograms[key].get(value, 0) + 1

    for u in utterances:
        for name, value in u.utterance_targets().items():
            add("utterance", name, float(value))
        for w in u.words:
            add("word", "accuracy", float(w.accuracy))
            add("word", "stress", float(w.stress_votes))
            add("word", "total", float(w.total))
            for score in w.phone_accuracy:
                add("phone", "accuracy", float(score))

    wanted = {
        key
        for key in histograms
        if key.split(".", 1)[0] in config.reweight_levels or key in ALWAYS_REWEIGHTED
    }
    return LabelReweighter(
        {k: v for k, v in histograms.items() if k in wanted},
        config.reweight_strength,
        config.reweight_max,
    )
