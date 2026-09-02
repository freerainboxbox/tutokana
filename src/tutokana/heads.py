"""Score heads reading register hidden states.

Each head takes a per-level learned softmax mixture over the last `n_layers` hidden states,
initialised biased low for phone and high for utterance, then
`RMSNorm -> Linear(512) -> GELU -> Dropout -> Linear(n_out)` in fp32. A register is read at
its own position; the -1 shift belongs to language-model cross-entropy only.

Four output modes: `regression` (scalar), `soft_class` (logits over the label's discrete
support, read out as an expectation), `binary` (one logit) and `binomial` (one logit read as
the probability that a single annotator flags the item, with the panel's verdict recovered
through a beta-binomial).

A head may also read a sibling head's readout for the same word (`--stress-siblings`),
appended to its input as extra scalars and detached, so the coupling is one-way.

Phone conditioning supplies a per-phone difficulty prior: `film` scales and shifts the output
from a 2-parameter embedding, `concat` appends a 32-dimensional symbol embedding to the input.
One head per symbol is deliberately not offered — rare phones would get too few examples and
share no statistics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

#: Per-symbol conditioning strategies for the phone head. `none` shares one function across
#: every symbol; `film` scales and shifts its output per symbol; `concat` appends a learned
#: per-symbol embedding to its input. A head per symbol is deliberately absent — see the
#: module docstring.
PHONE_CONDITIONING = ("none", "film", "concat")

#: What `--stress-siblings` feeds the stress head: the same word's other two readouts, in
#: this order. Both are dense-target heads, which is the whole point — see build_head_specs.
STRESS_SIBLING_SOURCES = ("word.accuracy", "word.total")


class LayerMixture(nn.Module):
    """A learned convex combination over the last `n_layers` hidden states.

    `bias_toward` in [0, 1] tilts the initial weights: 0 favours the earliest layer in the
    window, 1 the last. It is only an initialisation — the mixture is free to move.
    """

    def __init__(self, n_layers: int, bias_toward: float = 1.0, temperature: float = 2.0):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        position = torch.linspace(0.0, 1.0, n_layers) if n_layers > 1 else torch.zeros(1)
        self.weight = nn.Parameter(-temperature * (position - bias_toward).abs())

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        window = hidden_states[-self.n_layers :]
        if len(window) != self.n_layers:
            raise ValueError(
                f"layer mixture wants {self.n_layers} hidden states but the model exposed "
                f"{len(hidden_states)}; build_model clamps head_layers to the model depth, "
                f"so reaching this means the mixture and the model disagree"
            )
        stacked = torch.stack([h.float() for h in window], dim=0)
        weights = torch.softmax(self.weight, dim=0).view(-1, *([1] * (stacked.dim() - 1)))
        return (stacked * weights).sum(dim=0)

    def distribution(self) -> list[float]:
        """The current mixture weights, oldest layer in the window first."""
        return torch.softmax(self.weight.detach(), dim=0).tolist()


class ScoreHead(nn.Module):
    """One (level, field) head.

    `mode`:
      * `regression`  — a scalar, supervised with log-cosh on the z-scored target.
      * `soft_class`  — logits over the field's discrete support, supervised with
                        cross-entropy against a smoothed one-hot and read out as the
                        expectation, so the prediction stays continuous.
      * `binary`      — one logit, supervised with weighted BCE.
      * `binomial`    — one logit read as p, the probability that a single annotator scores
                        this item at the top of its scale. The panel's `n_raters` verdicts
                        are modelled as Beta-Binomial(n, nu*p, nu*(1-p)), so the loss is the
                        likelihood of the observed vote count and the readout is the
                        probability that the panel's *median* lands at the top — which is
                        the label the corpus actually publishes. Word stress only.

    `n_conditions > 0` enables per-symbol conditioning, with index 0 reserved as the shared
    fallback for unseen symbols. `conditioning="film"` applies a learned scale and shift to
    the head's output; `"concat"` appends a learned embedding to the input instead.

    Every head exposes both `predict_native` (the field's own scale, what metrics report)
    and `predict_normalized` (z-scored, what the correlation term consumes), so the four
    modes stay interchangeable behind one interface and callers never have to remember which
    scale a particular head happens to speak.
    """

    def __init__(
        self,
        hidden_size: int,
        mode: str = "regression",
        support: tuple[float, ...] | None = None,
        n_raters: int = 0,
        concentration: float = 0.0,
        n_conditions: int = 0,
        conditioning: str = "film",
        condition_dim: int = 32,
        context_dim: int = 0,
        proj_size: int = 512,
        dropout: float = 0.1,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__()
        if mode not in ("regression", "soft_class", "binary", "binomial"):
            raise ValueError(f"unknown head mode {mode!r}")
        if conditioning not in PHONE_CONDITIONING:
            raise ValueError(
                f"unknown conditioning {conditioning!r}; known: {sorted(PHONE_CONDITIONING)}"
            )
        if mode == "soft_class" and not support:
            raise ValueError("soft_class heads require a discrete support")
        if mode == "binomial":
            if not support or len(support) != 2:
                raise ValueError(
                    "binomial heads report onto a two-valued median scale, so they need a "
                    "support of exactly (low, high)"
                )
            if n_raters < 2 or concentration <= 0.0:
                raise ValueError(
                    f"binomial heads need n_raters >= 2 and concentration > 0, got "
                    f"{n_raters} and {concentration}"
                )
        self.mode = mode
        self.n_raters = n_raters
        self.context_dim = context_dim
        self.n_outputs = len(support) if mode == "soft_class" else 1
        self.register_buffer(
            "support",
            torch.tensor(support if support else (), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "concentration", torch.tensor(float(concentration)), persistent=False
        )
        self.register_buffer("mean", torch.tensor(float(mean)), persistent=False)
        self.register_buffer("std", torch.tensor(float(std)), persistent=False)
        self.conditioning = conditioning if n_conditions > 0 else "none"
        self.film = self.embedding = None
        if self.conditioning == "film":
            # Zero-initialised, so conditioning starts as an exact identity and only earns
            # its keep if the per-symbol prior is real.
            self.film = nn.Embedding(n_conditions, 2)
            nn.init.zeros_(self.film.weight)
        elif self.conditioning == "concat":
            self.embedding = nn.Embedding(n_conditions, condition_dim)
            nn.init.zeros_(self.embedding.weight)
            hidden_size = hidden_size + condition_dim
        # Sibling readouts are appended last, so the conditioning block above keeps its
        # existing slice of the input and old checkpoints load unchanged at context_dim 0.
        hidden_size = hidden_size + context_dim

        self.body = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_size, self.n_outputs),
        )

    def forward(
        self,
        states: torch.Tensor,
        condition: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.conditioning != "none" and condition is None:
            raise ValueError(
                f"this head is {self.conditioning}-conditioned but got no condition index"
            )
        if bool(self.context_dim) != (context is not None):
            raise ValueError(
                f"this head expects context_dim={self.context_dim} but was given "
                f"{'no context' if context is None else 'a context tensor'}"
            )
        states = states.float()
        if self.embedding is not None:
            states = torch.cat([states, self.embedding(condition)], dim=-1)
        if context is not None:
            states = torch.cat([states, context.float()], dim=-1)
        out = self.body(states)
        if self.film is not None:
            gamma, beta = self.film(condition).unbind(dim=-1)
            out = out * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return out

    def vote_log_pmf(self, logits: torch.Tensor) -> torch.Tensor:
        """(..., n_raters + 1) log-probabilities over the panel's vote count.

        The head predicts p, the chance one annotator scores this item at the top. If the
        annotators were exchangeable and independent given p the count would be Binomial,
        but they are not: the model's p is an estimate, and the residual item-to-item
        spread it cannot resolve shows up as overdispersion. Beta-Binomial absorbs that
        with a single fixed concentration, and it is what makes the likelihood safe — a
        Binomial assigns ~1e-10 to a unanimous dissent from a confident prediction, and
        the resulting gradient spike is the whole reason a plain count loss misbehaves here.
        """
        # Both shape parameters come from a sigmoid of the logit rather than from p and
        # 1 - p: the second form rounds to exactly zero once the logit passes ~16 in fp32,
        # and lgamma(0) is infinite. Written this way the pair stays positive out to |z| ~ 67
        # and, more importantly, the gradient survives saturation — a confidently wrong head
        # is exactly where the loss still has to pull. Only that far tail is clamped.
        z = logits.squeeze(-1).unsqueeze(-1)
        alpha = (self.concentration * torch.sigmoid(z)).clamp_min(1e-30)
        beta = (self.concentration * torch.sigmoid(-z)).clamp_min(1e-30)
        n = float(self.n_raters)
        k = torch.arange(self.n_raters + 1, device=logits.device, dtype=alpha.dtype)
        log_choose = (
            math.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        )
        return (
            log_choose
            + torch.lgamma(k + alpha)
            + torch.lgamma(n - k + beta)
            - torch.lgamma(n + alpha + beta)
            + torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )

    def vote_log_likelihood(self, logits: torch.Tensor, votes: torch.Tensor) -> torch.Tensor:
        """Log-probability of the observed vote count — the binomial head's pointwise loss."""
        index = votes.long().clamp(0, self.n_raters).unsqueeze(-1)
        return self.vote_log_pmf(logits).gather(-1, index).squeeze(-1)

    def flagged_logit(self, logits: torch.Tensor) -> torch.Tensor:
        """Log-odds that at least one annotator flagged this item (vote count below n).

        Free from the same distribution the loss already uses, and a far better posed
        question than the published label asks: 16% of words draw at least one dissent
        against the 1% the median calls wrong.
        """
        log_pmf = self.vote_log_pmf(logits)
        return log_pmf[..., :-1].logsumexp(dim=-1) - log_pmf[..., -1]

    def predict_native(self, logits: torch.Tensor) -> torch.Tensor:
        """Head output on the field's own scale (0-10, 0.0-2.0, or P(correct stress))."""
        if self.mode == "regression":
            return logits.squeeze(-1) * self.std + self.mean
        if self.mode == "binary":
            return torch.sigmoid(logits.squeeze(-1))
        if self.mode == "binomial":
            # The published label is the panel's median, so the quantity to report is the
            # probability that the median lands at the top — P(votes > n/2) — mapped onto
            # the field's reported endpoints. Pearson is affine-invariant, so this is the
            # comparable number and not a rescaling choice.
            majority = self.n_raters // 2 + 1
            probability = self.vote_log_pmf(logits)[..., majority:].logsumexp(dim=-1).exp()
            low, high = self.support[0], self.support[-1]
            return low + (high - low) * probability
        return (torch.softmax(logits, dim=-1) * self.support).sum(dim=-1)

    def predict_normalized(self, logits: torch.Tensor) -> torch.Tensor:
        """Head output z-scored with the train-split statistics for this field."""
        if self.mode == "regression":
            return logits.squeeze(-1)
        return (self.predict_native(logits) - self.mean) / self.std

    def class_targets(self, raw: torch.Tensor) -> torch.Tensor:
        """Nearest support index for each native-scale label (soft_class heads only)."""
        return (raw.unsqueeze(-1) - self.support).abs().argmin(dim=-1)


class HeadBank(nn.Module):
    """All heads plus their per-level layer mixtures, held together.

    `read` gathers the hidden state at each register position and runs the matching head, so
    the caller never has to reason about indexing into a padded batch.
    """

    def __init__(
        self,
        hidden_size: int,
        specs: dict[str, dict],
        levels: tuple[str, ...],
        n_layers: int = 8,
        proj_size: int = 512,
        dropout: float = 0.1,
        layer_mixture: bool = True,
    ):
        super().__init__()
        self.levels = levels
        #: {consumer key: source keys} — heads fed a sibling head's readout for the same word.
        self.context_sources = {
            key: tuple(spec["context_from"])
            for key, spec in specs.items()
            if spec.get("context_from")
        }
        self.mixtures = nn.ModuleDict(
            {
                level: LayerMixture(
                    n_layers=n_layers if layer_mixture else 1,
                    bias_toward=bias,
                )
                for level, bias in (
                    ("phone", 0.0),
                    ("word", 0.5),
                    ("utterance", 1.0),
                )
                if level in levels
            }
        )
        self.heads = nn.ModuleDict(
            {
                key.replace(".", "__"): ScoreHead(
                    hidden_size=hidden_size,
                    mode=spec["mode"],
                    support=spec.get("support"),
                    n_raters=spec.get("n_raters", 0),
                    concentration=spec.get("concentration", 0.0),
                    n_conditions=spec.get("n_conditions", 0),
                    conditioning=spec.get("conditioning", "none"),
                    context_dim=len(spec.get("context_from", ())),
                    proj_size=proj_size,
                    dropout=dropout,
                    mean=spec.get("mean", 0.0),
                    std=spec.get("std", 1.0),
                )
                for key, spec in specs.items()
            }
        )

    def head(self, key: str) -> ScoreHead:
        return self.heads[key.replace(".", "__")]

    def sibling_context(
        self, key: str, batch, heads: dict, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """(M, len(sources)) of sibling readouts for the same words, detached.

        Normalized rather than native, so every source arrives on one z-scored scale
        whatever mode it is. Detached on purpose: this head's loss is carried by ~1% of
        words and is reweighted up to 10x, and letting that gradient into the accuracy and
        total heads would risk the two columns that currently work to rescue the one that
        does not. The pairing is asserted, never assumed — see `HeadBatch.word_index`.
        """
        columns = []
        for source in self.context_sources[key]:
            if source not in outputs:
                raise RuntimeError(
                    f"{key} reads {source}, which produced no output this batch; a context "
                    f"source must be a trained head at the same level"
                )
            other = heads[source]
            if batch.word_index is None or other.word_index is None:
                raise RuntimeError(
                    f"{key} reads {source} but one of them carries no word_index; sibling "
                    f"context is only defined for word-level heads"
                )
            if not torch.equal(batch.word_index, other.word_index):
                raise RuntimeError(
                    f"{key} and {source} cover different words ({len(batch)} vs "
                    f"{len(other)} rows); their register slots are no longer emitted in "
                    f"lockstep, so pairing them by row would mislabel every score"
                )
            columns.append(self.head(source).predict_normalized(outputs[source]).detach())
        return torch.stack(columns, dim=-1)

    def read(
        self, hidden_states: tuple[torch.Tensor, ...], heads: dict
    ) -> dict[str, torch.Tensor]:
        """{head key: raw head output} for every populated head in the batch.

        Two passes: heads that read a sibling's output run after the heads they read.
        """
        pooled: dict[str, torch.Tensor] = {}
        outputs: dict[str, torch.Tensor] = {}
        deferred: list[tuple[str, object, torch.Tensor]] = []
        for key, batch in heads.items():
            if len(batch) == 0:
                continue
            level = key.split(".", 1)[0]
            if level not in pooled:
                pooled[level] = self.mixtures[level](hidden_states)
            states = pooled[level][batch.positions[:, 0], batch.positions[:, 1]]
            if key in self.context_sources:
                deferred.append((key, batch, states))
                continue
            outputs[key] = self.head(key)(states, batch.phone_id)
        for key, batch, states in deferred:
            context = self.sibling_context(key, batch, heads, outputs)
            outputs[key] = self.head(key)(states, batch.phone_id, context)
        return outputs

    def mixture_report(self) -> dict[str, list[float]]:
        return {level: mix.distribution() for level, mix in self.mixtures.items()}


def build_head_specs(
    levels: tuple[str, ...],
    head_modes: dict[str, str],
    n_phone_conditions: int,
    phone_conditioning: str,
    stats,
    stress_concentration: float | None = None,
    stress_siblings: bool = False,
) -> dict[str, dict]:
    """Assemble the per-(level, field) head configuration.

    `head_modes` is keyed by level; word stress always overrides to `binomial`. Its released
    label is the median of five annotators, so on its own it is 99:1 binary and a regressor
    on it converges to a constant. Modelling the five verdicts instead restores a six-valued
    target without changing what is reported.

    `stress_siblings` hands the stress head the word accuracy and word total readouts for the
    same word, as two extra input scalars. It is the control for the sibling-coupling
    question: those two heads already predict the stress label better than the stress head
    does, so if two numbers recover most of that, no richer coupling is warranted.
    """
    if phone_conditioning not in PHONE_CONDITIONING:
        raise ValueError(
            f"unknown phone_conditioning {phone_conditioning!r}; "
            f"known: {sorted(PHONE_CONDITIONING)}"
        )
    from .data import (
        FIELD_SUPPORT,
        STRESS_CONCENTRATION,
        STRESS_RATERS,
        TargetStats,
    )
    from .tokens import REGISTERS_BY_LEVEL, UNTRAINED_FIELDS

    if stress_concentration is None:
        stress_concentration = STRESS_CONCENTRATION

    specs: dict[str, dict] = {}
    for level in levels:
        for reg in REGISTERS_BY_LEVEL[level]:
            if (level, reg.field) in UNTRAINED_FIELDS:
                continue
            key = TargetStats.key(level, reg.field)
            stress = (level, reg.field) == ("word", "stress")
            mode = "binomial" if stress else head_modes[level]
            spec: dict = {"mode": mode, "mean": stats.mean[key], "std": stats.std[key]}
            if mode in ("soft_class", "binomial"):
                spec["support"] = FIELD_SUPPORT[(level, reg.field)]
            if mode == "binomial":
                spec["n_raters"] = STRESS_RATERS
                spec["concentration"] = stress_concentration
            if level == "phone" and phone_conditioning != "none":
                spec["n_conditions"] = n_phone_conditions
                spec["conditioning"] = phone_conditioning
            specs[f"{level}.{reg.field}"] = spec

    if stress_siblings:
        if "word.stress" not in specs:
            raise ValueError(
                "stress_siblings needs the word.stress head, which this level selection "
                "does not build"
            )
        sources = tuple(k for k in STRESS_SIBLING_SOURCES if k in specs)
        if not sources:
            raise ValueError(
                f"stress_siblings needs at least one of {list(STRESS_SIBLING_SOURCES)}, "
                f"none of which is being built"
            )
        specs["word.stress"]["context_from"] = sources
    return specs
