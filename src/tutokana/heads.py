"""Score heads: what reads the register hidden states, and how.

Three design points, each of which is a deliberate departure from the obvious version.

**Where to read.** Not the last layer. HMamba's architecture ablation is explicit that
"phone-level and word-level scores should be predicted in lower layers" — its flat
last-layer variant scores 0.694 phone correlation against 0.739 for the hierarchical one.
`LayerMixture` therefore gives every level its own learned softmax over the hidden stack,
initialised biased toward a lower band for phones and a higher one for utterances. The
learned weights are also a free diagnostic: print them and you can see which depth the model
actually found the signal at.

**No off-by-one.** The hidden state at index i already includes token i, so a register is
read at its own position. The -1 shift belongs to language-model cross-entropy, which
predicts the *next* token; applying it here would read the state of whatever precedes the
register and quietly discard the register's own attention.

**How to condition on the phone.** The obvious reading of "a head per phone" is 66
independent heads. That is the wrong shape of the right idea: rare symbols get too few
examples, no statistics are shared, and routing by the *gold* phone during training but the
*emitted* phone at inference introduces a train/test skew. `film` buys the thing routing was
actually reaching for — a per-phone difficulty prior — with two parameters per symbol, on
top of a trunk trained on all the data at once.

Output modes are per level. `soft_class` exists because 80.1% of phone accuracy labels are
exactly 2.0: a plain regressor under any pointwise loss drifts to the mode, whereas a
distribution over the eleven-value support read out as an expectation stays continuous and
takes class weighting naturally.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
      * `binary`      — one logit, supervised with weighted BCE (word stress only).

    `n_conditions > 0` enables FiLM conditioning: a per-symbol scale and shift applied to
    the head's output, with index 0 reserved as the shared fallback for unseen symbols.

    Every head exposes both `predict_native` (the field's own scale, what metrics report)
    and `predict_normalized` (z-scored, what the correlation term consumes), so the three
    modes stay interchangeable behind one interface and callers never have to remember which
    scale a particular head happens to speak.
    """

    def __init__(
        self,
        hidden_size: int,
        mode: str = "regression",
        support: tuple[float, ...] | None = None,
        n_conditions: int = 0,
        proj_size: int = 512,
        dropout: float = 0.1,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__()
        if mode not in ("regression", "soft_class", "binary"):
            raise ValueError(f"unknown head mode {mode!r}")
        if mode == "soft_class" and not support:
            raise ValueError("soft_class heads require a discrete support")
        self.mode = mode
        self.n_outputs = len(support) if mode == "soft_class" else 1
        self.register_buffer(
            "support",
            torch.tensor(support if support else (), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("mean", torch.tensor(float(mean)), persistent=False)
        self.register_buffer("std", torch.tensor(float(std)), persistent=False)
        self.body = nn.Sequential(
            nn.RMSNorm(hidden_size),
            nn.Linear(hidden_size, proj_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_size, self.n_outputs),
        )
        self.film = None
        if n_conditions > 0:
            # Initialised to identity so conditioning starts as a no-op and only earns its
            # keep if the per-symbol prior is real.
            self.film = nn.Embedding(n_conditions, 2)
            nn.init.zeros_(self.film.weight)

    def forward(self, states: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        out = self.body(states.float())
        if self.film is not None:
            if condition is None:
                raise ValueError("this head is FiLM-conditioned but got no condition index")
            gamma, beta = self.film(condition).unbind(dim=-1)
            out = out * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return out

    def predict_native(self, logits: torch.Tensor) -> torch.Tensor:
        """Head output on the field's own scale (0-10, 0.0-2.0, or P(correct stress))."""
        if self.mode == "regression":
            return logits.squeeze(-1) * self.std + self.mean
        if self.mode == "binary":
            return torch.sigmoid(logits.squeeze(-1))
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
                    n_conditions=spec.get("n_conditions", 0),
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

    def read(
        self, hidden_states: tuple[torch.Tensor, ...], heads: dict
    ) -> dict[str, torch.Tensor]:
        """{head key: raw head output} for every populated head in the batch."""
        pooled: dict[str, torch.Tensor] = {}
        outputs: dict[str, torch.Tensor] = {}
        for key, batch in heads.items():
            if len(batch) == 0:
                continue
            level = key.split(".", 1)[0]
            if level not in pooled:
                pooled[level] = self.mixtures[level](hidden_states)
            states = pooled[level][batch.positions[:, 0], batch.positions[:, 1]]
            outputs[key] = self.head(key)(states, batch.phone_id)
        return outputs

    def mixture_report(self) -> dict[str, list[float]]:
        return {level: mix.distribution() for level, mix in self.mixtures.items()}


def build_head_specs(
    levels: tuple[str, ...],
    head_modes: dict[str, str],
    n_phone_conditions: int,
    phone_conditioning: str,
    stats,
) -> dict[str, dict]:
    """Assemble the per-(level, field) head configuration.

    `head_modes` is keyed by level; word stress always overrides to `binary` because a 99:1
    label split is a detection problem, not a regression one, and a regressor on it converges
    to the constant 10 that made the predecessor's stress correlation undefined.
    """
    from .data import FIELD_SUPPORT, TargetStats
    from .tokens import REGISTERS_BY_LEVEL, UNTRAINED_FIELDS

    specs: dict[str, dict] = {}
    for level in levels:
        for reg in REGISTERS_BY_LEVEL[level]:
            if (level, reg.field) in UNTRAINED_FIELDS:
                continue
            key = TargetStats.key(level, reg.field)
            mode = "binary" if (level, reg.field) == ("word", "stress") else head_modes[level]
            spec: dict = {"mode": mode, "mean": stats.mean[key], "std": stats.std[key]}
            if mode == "soft_class":
                spec["support"] = FIELD_SUPPORT[(level, reg.field)]
            if level == "phone" and phone_conditioning == "film":
                spec["n_conditions"] = n_phone_conditions
            specs[f"{level}.{reg.field}"] = spec
    return specs
