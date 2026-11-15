"""RAIL: read a score off the grid, then feed it back into the stream.

A score is not the token the model samples. It is the **expectation** of the next-token
distribution, restricted to that score type's own grid and renormalised over it. That is the
whole of RAIL, and it matters because one sampled token keeps about half the achievable
correlation while the expectation keeps nearly all of it: the grid is dense near the mode and
the tail carries the information a sample throws away.

The expectation is continuous and the stream is not, so the value is **snapped** to the
nearest grid token before it is written back. That reinjection is the reason this is a unified
model rather than a feature extractor with a regressor bolted on: the score the model believes
becomes context that every later token attends to.

Two readouts, because the grid scale and the reported scale differ for exactly one field:

- `expectation` is the number the loss supervises and the number the reinjection snaps.
- `report` is the number the metrics compare against. For `word.stress` the grid holds the
  five annotators' vote count, while the published label is their 5/10 median, so the
  readout is `P(votes >= 3)` mapped onto that range. The `binomial` head does the same
  mapping, which is what keeps the two paths comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import FIELD_RANGE, STRESS_RATERS
from .tokens import SCORE_GRID


@dataclass(frozen=True)
class GridSupport:
    """One score type's grid as vocabulary ids and the numbers they name.

    `token_ids[i]` and `values[i]` stay paired, so a restriction, an expectation and a snap
    are all one gather apart and cannot fall out of step with each other.
    """

    level: str
    field: str
    token_ids: torch.Tensor
    values: torch.Tensor

    @property
    def key(self) -> str:
        return f"{self.level}.{self.field}"

    def _values(self, like: torch.Tensor) -> torch.Tensor:
        return self.values.to(like.device, dtype=like.dtype)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """The next-token distribution, renormalised onto the grid.

        Restricting first is what "the grid distribution" means: mass sitting on the rest of
        the 262144-token vocabulary is not part of the score, and leaving it in the
        denominator would shrink every prediction toward zero.
        """
        return torch.softmax(logits[..., self.token_ids.to(logits.device)], dim=-1)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """The score, on the grid's own scale."""
        return (self.probabilities(logits) * self._values(logits)).sum(dim=-1)

    def report(self, logits: torch.Tensor) -> torch.Tensor:
        """The score on the scale the field is published on, which is what metrics read."""
        probabilities = self.probabilities(logits)
        if (self.level, self.field) == ("word", "stress"):
            low, high = FIELD_RANGE["word", "stress"]
            majority = STRESS_RATERS // 2 + 1
            p_correct = probabilities[..., self._values(logits) >= majority].sum(dim=-1)
            return low + (high - low) * p_correct
        return (probabilities * self._values(logits)).sum(dim=-1)

    def snap(self, values: torch.Tensor) -> torch.Tensor:
        """The grid token nearest each value. Lossless on a value already on the grid."""
        values = values.reshape(-1)
        grid = self._values(values)
        distance = (grid[None, :] - values[:, None]).abs()
        return self.token_ids.to(values.device)[distance.argmin(dim=-1)]


def grid_supports(grid_ids: dict[str, int]) -> dict[str, GridSupport]:
    """{`level.field`: GridSupport} for every scored type, from the registered tokens.

    Derived from `SCORE_GRID`, never listed: widening a grid or adding a score type changes
    one line in `tokens.py` and this follows it.
    """
    supports: dict[str, GridSupport] = {}
    for (level, field), grid in SCORE_GRID.items():
        strings = [grid.render(v) for v in grid.values]
        missing = [s for s in strings if s not in grid_ids]
        if missing:
            raise RuntimeError(
                f"({level}, {field}) grid needs {len(strings)} registered tokens and is "
                f"missing {len(missing)}: {missing[:4]}. Register the grid before scoring."
            )
        supports[f"{level}.{field}"] = GridSupport(
            level=level,
            field=field,
            token_ids=torch.tensor([grid_ids[s] for s in strings], dtype=torch.long),
            values=torch.tensor(grid.values, dtype=torch.float32),
        )
    return supports
