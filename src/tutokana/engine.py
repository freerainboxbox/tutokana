"""Training and scoring loops.

Two things here are worth explaining rather than reading off the code.

**Preflight.** A full run is measured in tens of hours, so every invariant that can be
checked in one micro-batch is checked before the first optimizer step: audio features
present, at least one register located per level, no NaN targets, and — the one that
actually caught a bug in the predecessor — a backward pass that reaches every head. A head
that receives no gradient is a head whose registers were never found, and finding that out
at hour thirty is expensive.

**Validation measures correlation, not loss.** The predecessor ran a validation *loss* pass
that its own documentation admitted was incomparable across configurations. Cross-entropy on
the transcript is not the objective any more, and now that the scores live in heads the real
metric costs one teacher-forced forward pass with no generation — a few hundred samples in
a couple of minutes. So validation reports per-level correlation and the
sigma_pred/sigma_gold ratio, which is what catches variance collapse at step 200 instead of
after the run. The slice is speaker-disjoint from what is trained on; the test split is
never touched here.
"""

from __future__ import annotations

import json
import math
import random
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import get_logger
from .data import Utterance, from_binary_stress
from .losses import LossConfig, composite_loss
from .metrics import FieldMetrics, field_metrics

logger = get_logger()

#: A z-scored target beyond this cannot come from a healthy field: the floor in
#: `data.compute_target_stats` bounds the worst case near 50.
MAX_ABS_TARGET = 60.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def graceful_interrupt():
    """Turn Ctrl-C into a flag checked at accumulation boundaries.

    Raising mid-window would leave a half-accumulated gradient and an unsaved run; setting a
    flag lets the current step finish and checkpoint cleanly.
    """
    state = {"requested": False}
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        if state["requested"]:  # a second Ctrl-C means "now"
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        state["requested"] = True
        logger.warning("interrupt requested — finishing this step, then saving")

    signal.signal(signal.SIGINT, handler)
    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, previous)


def move_batch(batch: dict, device) -> dict:
    moved = {}
    for key, value in batch.items():
        if key == "heads":
            moved[key] = {k: v.to(device) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --- preflight ------------------------------------------------------------------------


def preflight(model, collator, samples: list[Utterance], device, loss_config: LossConfig,
              reweighter=None) -> None:
    """One forward+backward over a tiny batch, asserting everything that can be asserted."""
    batch = move_batch(collator(samples), device)

    if batch.get("input_features") is None:
        raise SystemExit("[preflight] collator produced no audio features")
    if not batch["heads"]:
        raise SystemExit("[preflight] no register positions were located in the batch")
    for key, head_batch in batch["heads"].items():
        if len(head_batch) == 0:
            raise SystemExit(f"[preflight] head {key} has no supervised positions")
        if torch.isnan(head_batch.target).any():
            raise SystemExit(f"[preflight] head {key} has NaN targets")
        worst = float(head_batch.target.abs().max())
        if worst > MAX_ABS_TARGET:
            raise SystemExit(
                f"[preflight] head {key} has a z-scored target of {worst:.1f}. That means "
                f"the field is near-constant in the training split, so its standard "
                f"deviation was floored — check the split size and the label distribution "
                f"(experiments/audit_data.py) before spending a run on it."
            )
    if (batch["labels"] != -100).sum() == 0:
        raise SystemExit("[preflight] language-model loss mask is empty")

    model.train()
    head_outputs, lm_loss = model(batch)
    total, parts = composite_loss(
        model.heads, head_outputs, batch["heads"], loss_config, None, reweighter, lm_loss
    )
    total.backward()

    starved = [
        key
        for key in head_outputs
        if all(p.grad is None or not p.grad.any() for p in model.heads.head(key).parameters())
    ]
    if starved:
        raise SystemExit(f"[preflight] heads received no gradient: {starved}")
    model.zero_grad(set_to_none=True)

    counts = {k: len(v) for k, v in batch["heads"].items()}
    logger.info(
        "[preflight] ok — %d tokens, %d supervised text positions, registers %s, loss %.4f",
        batch["input_ids"].shape[1],
        int((batch["labels"] != -100).sum()),
        counts,
        parts["loss/total"],
    )


# --- scoring --------------------------------------------------------------------------


@dataclass
class ScoreResult:
    metrics: dict[str, FieldMetrics]
    predictions: dict[str, np.ndarray]
    gold: dict[str, np.ndarray]


@torch.no_grad()
def score(
    model,
    collator,
    utterances: list[Utterance],
    device,
    batch_size: int = 8,
    bootstrap: bool = True,
    progress_every: int = 0,
) -> ScoreResult:
    """Teacher-forced scoring: one forward pass per batch, every head read at once.

    The canonical phones come from the dataset, so every phone register aligns one-to-one
    with its gold score. Letting the model generate the phone sequence here would leave a
    few percent of positions misaligned and make the phone correlation incomparable to the
    published baselines — the generative capability is measured separately.
    """
    model.eval()
    collected: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    started = time.time()

    for done, chunk in enumerate(batches(utterances, batch_size), 1):
        batch = move_batch(collator(chunk), device)
        head_outputs, _ = model(batch)
        for key, logits in head_outputs.items():
            head = model.heads.head(key)
            prediction = head.predict_native(logits).float().cpu().numpy()
            gold = batch["heads"][key].raw.float().cpu().numpy()
            if key == "word.stress":
                # Reported on the corpus's 5-10 scale so the number lines up with published
                # tables; correlation is unchanged by the affine map.
                prediction = np.vectorize(from_binary_stress)(prediction)
                gold = np.vectorize(from_binary_stress)(gold)
            collected.setdefault(key, []).append((prediction, gold))
        if progress_every and done % progress_every == 0:
            seen = done * batch_size
            rate = seen / max(time.time() - started, 1e-6)
            logger.info("[score] %d/%d utterances (%.1f/s)", seen, len(utterances), rate)

    predictions = {k: np.concatenate([p for p, _ in v]) for k, v in collected.items()}
    gold = {k: np.concatenate([g for _, g in v]) for k, v in collected.items()}
    metrics = {
        key: field_metrics(predictions[key], gold[key], bootstrap=bootstrap)
        for key in predictions
    }
    return ScoreResult(metrics=metrics, predictions=predictions, gold=gold)


def summarize(metrics: dict[str, FieldMetrics]) -> str:
    """A one-line digest for the training log: correlation and dispersion per field."""
    return "  ".join(
        f"{key.split('.')[0][0]}.{key.split('.')[1][:4]} "
        f"{m.pearson:.3f}/{m.sigma_ratio:.2f}"
        for key, m in sorted(metrics.items())
    )


# --- training -------------------------------------------------------------------------


def build_optimizer(model, config):
    """Two parameter groups: adapters at the base rate, freshly initialised heads faster.

    The heads and the register delta start from scratch and have to travel much further than
    a LoRA adapter perturbing a pretrained function, so they get their own learning rate.
    """
    head_parameters = list(model.heads.parameters()) + list(model.register_delta.parameters())
    head_ids = {id(p) for p in head_parameters}
    base_parameters = [
        p for p in model.base.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": config.train.learning_rate},
            {"params": head_parameters, "lr": config.train.head_learning_rate},
        ],
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.95),
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --- interruption and resume ----------------------------------------------------------
# Interrupting is only half of it. A resume is *clean* when continuing produces the same
# trajectory the uninterrupted run would have had, which needs more than the weights:
# optimizer moments, the schedule position, the random streams that drive shuffling and
# dropout, the correlation buffer's population, and where in the epoch the run stopped.
# Everything else — the training mix, the speaker split, the epoch orders, the target
# statistics — is a deterministic function of the saved config, so it is recomputed rather
# than stored.

RESUME_META = "resume.json"
RESUME_STATE = "resume_state.pt"


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_resume(
    run_dir,
    *,
    meta: dict,
    optimizer,
    scheduler,
    buffer,
) -> None:
    """Write everything needed to continue, alongside the adapter and heads.

    Called both on interrupt and at every periodic checkpoint, so an ordinary crash or a
    killed job resumes from the last checkpoint rather than from nothing.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(),
            "buffer": buffer.state_dict() if buffer is not None else None,
        },
        run_dir / RESUME_STATE,
    )
    (run_dir / RESUME_META).write_text(json.dumps(meta, indent=2, default=str))


def load_resume(run_dir) -> tuple[dict, dict]:
    """(metadata, tensors) for a run that was interrupted. Raises if it was not."""
    run_dir = Path(run_dir)
    meta_path = run_dir / RESUME_META
    if not meta_path.exists():
        raise SystemExit(
            f"{run_dir} has no {RESUME_META} — it either finished cleanly or never "
            f"reached its first checkpoint, so there is nothing to resume."
        )
    meta = json.loads(meta_path.read_text())
    state = torch.load(run_dir / RESUME_STATE, map_location="cpu", weights_only=False)
    return meta, state


def clear_resume(run_dir) -> None:
    """Drop the resume bundle once a run finishes, so it cannot leak into a later one."""
    for name in (RESUME_META, RESUME_STATE):
        (Path(run_dir) / name).unlink(missing_ok=True)
