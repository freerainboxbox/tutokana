"""Training and scoring loops.

**Preflight** checks every invariant that fits in one micro-batch before a multi-hour run:
audio features present, registers located per level, no NaN or unbounded targets, and a
backward pass reaching every head. A head with no gradient is a head whose registers were
never found.

**Validation measures correlation, not loss.** Scoring is one teacher-forced forward pass, so
a few hundred samples cost a couple of minutes and report per-level correlation and
sigma_pred/sigma_gold — which is what catches variance collapse early. The slice is
speaker-disjoint from training; the test split is never touched here.

**Resume** stores what continuing needs: optimizer moments, schedule position, RNG streams,
the correlation buffer and the epoch/sample offset. Everything else — mix, splits, epoch
orders, target statistics — is recomputed from the saved config.
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
from .data import Utterance
from .losses import LossConfig, composite_loss
from .metrics import FieldMetrics, field_metrics
from .progress import Progress, memory_note
from .rail import READOUTS, grid_supports

logger = get_logger()

#: A z-scored target beyond this cannot come from a healthy field: the floor in
#: `data.compute_target_stats` bounds the worst case near 50.
MAX_ABS_TARGET = 60.0


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _has_mps():
        torch.mps.manual_seed(seed)
    if getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def graceful_interrupt():
    """Turn Ctrl-C into a flag checked at accumulation boundaries.

    Raising mid-window would leave a half-accumulated gradient and an unsaved run; setting a
    flag lets the current step finish and checkpoint cleanly. A second press raises at once,
    which is safe only because every checkpoint write runs under `deferred_interrupt`: the
    escape hatch can abandon a step, never a half-written checkpoint.
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


@contextmanager
def deferred_interrupt(state: dict | None = None, what: str = "checkpoint"):
    """Hold every Ctrl-C until the block finishes; record it in `state` instead of raising.

    A checkpoint is several files (adapter, heads, optimizer bundle, resume.json) that only
    mean something together. A KeyboardInterrupt between two of those writes leaves weights
    from one step beside the optimizer and data position of an earlier one, and `--resume`
    would continue from that silently. Inside this block no number of presses can do that:
    each one only sets `state["requested"]`, and the previous handler — usually
    `graceful_interrupt`'s, with its second-press escape hatch — returns on exit.
    """
    state = state if state is not None else {"requested": False}
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        state["requested"] = True
        logger.warning("interrupt received during the %s — finishing it first", what)

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
              reweighter=None, buffer=None) -> None:
    """One forward+backward over a tiny batch, asserting everything that can be asserted.

    `buffer` is passed through rather than stubbed out: on a resumed run it arrives holding
    host-side tensors from the checkpoint, and that is exactly the state whose first
    interaction with an accelerator batch needs to be proven here rather than at step one.
    """
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

    # Snapshot the correlation buffer so preflight still exercises host->device concatenation
    # on a resumed buffer without polluting its FIFO history or evicting real training entries.
    saved_buffer = (
        (
            {k: v.clone() for k, v in buffer._predictions.items()},
            {k: v.clone() for k, v in buffer._targets.items()},
        )
        if buffer is not None
        else None
    )

    try:
        model.train()
        head_outputs, lm_loss, score_logits = model(batch)
        total, parts = composite_loss(
            model.heads, head_outputs, batch["heads"], loss_config, buffer, reweighter, lm_loss,
            score_logits=score_logits, supports=model.supports,
        )
        total.backward()

        starved = [
            key
            for key in head_outputs
            if all(p.grad is None or not p.grad.any() for p in model.heads.head(key).parameters())
        ]
        if starved:
            raise SystemExit(f"[preflight] heads received no gradient: {starved}")
    finally:
        model.zero_grad(set_to_none=True)
        if buffer is not None and saved_buffer is not None:
            buffer._predictions, buffer._targets = saved_buffer

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
    #: Utterances actually scored, which is fewer than asked for after an interrupt.
    scored: int = 0
    partial: bool = False


@torch.no_grad()
def score(
    model,
    collator,
    utterances: list[Utterance],
    device,
    batch_size: int = 8,
    bootstrap: bool = True,
    progress: "Progress | None" = None,
    interrupt: dict | None = None,
    readout: str = "expectation",
) -> ScoreResult:
    """Score `utterances`, by whichever channel the run's collator describes.

    One entry point on purpose. The two channels are different measurements, and a caller that
    had to remember which one to call is a caller that can quietly score a RAIL run with the
    head reader and get an empty table back.
    """
    if collator.spec.emit_scores:
        return score_emitted(
            model, collator, utterances, device,
            bootstrap=bootstrap, progress=progress, interrupt=interrupt,
            readout=readout,
        )
    return _score_teacher_forced(
        model, collator, utterances, device,
        batch_size=batch_size, bootstrap=bootstrap, progress=progress, interrupt=interrupt,
    )


@torch.no_grad()
def _score_teacher_forced(
    model,
    collator,
    utterances: list[Utterance],
    device,
    batch_size: int = 8,
    bootstrap: bool = True,
    progress: "Progress | None" = None,
    interrupt: dict | None = None,
) -> ScoreResult:
    """Teacher-forced scoring: one forward pass per batch, every head read at once.

    The canonical phones come from the dataset, so every phone register aligns one-to-one
    with its gold score. Letting the model generate the phone sequence here would leave a
    few percent of positions misaligned and make the phone correlation incomparable to the
    published baselines — the generative capability is measured separately.

    `with_lm_loss=False` is the memory fix: scoring discards the text loss, but asking for it
    materialises logits over the full vocabulary, which was the largest allocation in the
    pass and grew with sequence length.

    `interrupt` is a `graceful_interrupt` flag. When it trips, scoring stops at the current
    batch and returns what it has, marked `partial` — a correlation over 1800 of 2500
    utterances is still worth reading, and losing it to a KeyboardInterrupt is not.
    """
    model.eval()
    collected: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    scored = 0
    stopped = False

    for chunk in batches(utterances, batch_size):
        batch = move_batch(collator(chunk), device)
        head_outputs, _, _ = model(batch, with_lm_loss=False)
        for key, logits in head_outputs.items():
            head = model.heads.head(key)
            prediction = head.predict_native(logits).float().cpu().numpy()
            # `native`, not `raw`: word stress trains on the annotator vote count but is
            # reported on the corpus's published 5-10 median, as the baselines are.
            gold = batch["heads"][key].native.float().cpu().numpy()
            collected.setdefault(key, []).append((prediction, gold))
        scored += len(chunk)
        if progress is not None:
            progress.update(len(chunk), detail=memory_note(device))
        if interrupt is not None and interrupt.get("requested"):
            stopped = True
            break

    predictions = {k: np.concatenate([p for p, _ in v]) for k, v in collected.items()}
    gold = {k: np.concatenate([g for _, g in v]) for k, v in collected.items()}
    metrics = {
        key: field_metrics(predictions[key], gold[key], bootstrap=bootstrap)
        for key in predictions
    }
    return ScoreResult(
        metrics=metrics, predictions=predictions, gold=gold,
        scored=scored, partial=stopped,
    )


# --- emitted-score scoring (RAIL) ------------------------------------------------------

#: What the multimodal front end needs on the first forward, and nothing needs after: the
#: audio is scattered into the embeddings once, and the KV cache carries the result.
_AUDIO_KEYS = ("input_features", "input_features_mask", "mm_token_type_ids", "token_type_ids")


@torch.no_grad()
def emitted_row(
    model, batch: dict, supports: dict, readout: str = "expectation"
) -> dict[str, torch.Tensor]:
    """One utterance, scored on the grid with every emitted score fed back.

    The transcript stays teacher-forced and only the score tokens are the model's own. That is
    deliberate: the published protocol fixes the phone sequence, so letting the model write its
    own phones would misalign a few percent of registers and the phone correlation would stop
    meaning what the baselines mean. The reinjection is still real, because each emitted score
    changes what every later token attends to.

    `readout` selects how a score is read off the restricted grid logits and written back:
    - `"expectation"` (RAIL, `--ce-rail` / `--raft-rail`): report the grid expectation and
      reinject the nearest grid token to that expectation.
    - `"mode"` (`--ce-std`): naive mode-seeking autoregression restricted to the field's grid
      tokens — pick the argmax grid token, report its value, and reinject that token.

    The turn is walked left to right, feeding one segment per trigger — everything up to and
    including that trigger — and the KV cache carries the rest. That is one forward pass per
    score rather than per token, and each pass adds only the one to three tokens between two
    triggers. Scores are read strictly left to right, so each one sits in the context of the
    scores the model actually emitted, never the gold ones.
    """
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r}; expected one of {READOUTS}")

    device = batch["input_ids"].device
    ids = batch["input_ids"][0].tolist()
    read = {key: torch.zeros(len(heads)) for key, heads in batch["heads"].items()}
    # One event per supervised score slot, in the order the model will meet them.
    events = sorted(
        (int(position), key, row)
        for key, head_batch in batch["heads"].items()
        for row, (_, position) in enumerate(head_batch.positions.tolist())
    )
    if not events:
        return read

    audio = {k: batch[k] for k in _AUDIO_KEYS if batch.get(k) is not None}
    cache = None
    consumed = 0
    for position, key, row in events:
        support = supports[key]
        length = position + 1
        # The first pass carries the audio; after it the cache holds everything needed, and
        # re-sending features would scatter them into positions that are already past.
        first = (
            {
                k: (v[:, :length] if k in ("mm_token_type_ids", "token_type_ids") else v)
                for k, v in audio.items()
            }
            if cache is None
            else {}
        )
        outputs = model.base(
            input_ids=torch.tensor([ids[consumed:length]], device=device),
            past_key_values=cache,
            attention_mask=torch.ones(1, length, dtype=torch.long, device=device),
            cache_position=torch.arange(consumed, length, device=device),
            use_cache=True,
            **first,
        )
        # Restricted to the grid once, here, so every readout downstream sees one shape.
        grid_logits = support.restrict(outputs.logits[:, -1, :])
        if readout == "mode":
            read[key][row] = support.report_mode(grid_logits)[0].float().cpu()
            # Mode-seeking autoregression: write the argmax grid token back into the stream.
            ids[position + 1] = int(support.mode_token(grid_logits)[0])
        else:
            read[key][row] = support.report(grid_logits)[0].float().cpu()
            # Snap and write back, so the next segment is conditioned on this score.
            ids[position + 1] = int(support.snap(support.expectation(grid_logits))[0])
        cache = outputs.past_key_values
        consumed = position + 1
    return read


@torch.no_grad()
def score_emitted(
    model,
    collator,
    utterances: list[Utterance],
    device,
    bootstrap: bool = True,
    progress: "Progress | None" = None,
    interrupt: dict | None = None,
    readout: str = "expectation",
) -> ScoreResult:
    """Score an emitted-score run (`--ce-std`, `--ce-rail`, or `--raft-rail`).

    One utterance at a time, unlike the teacher-forced path. Reinjection makes the sequence
    sequential, so a batch would have to advance in lockstep across utterances whose triggers
    sit in different places, and padding to the longest would write pad tokens into the cache
    and change what every later token attends to. Correctness first, and the cache keeps the
    cost near one forward pass per utterance anyway.
    """
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r}; expected one of {READOUTS}")
    model.eval()
    supports = grid_supports(collator.grid_ids)
    collected: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    scored = 0
    stopped = False

    for utterance in utterances:
        batch = move_batch(collator([utterance]), device)
        read = emitted_row(model, batch, supports, readout=readout)
        for key, prediction in read.items():
            # `native`, not `raw`: stress trains and scores on the vote count but reports on
            # the corpus's published 5-10 median, exactly as the head path does.
            gold = batch["heads"][key].native.float().cpu()
            collected.setdefault(key, []).append((prediction.numpy(), gold.numpy()))
        scored += 1
        if progress is not None:
            progress.update(1, detail=memory_note(device))
        if interrupt is not None and interrupt.get("requested"):
            stopped = True
            break

    if not collected:
        raise SystemExit(
            "[score] the emitted path found no supervised score slots. Check that the "
            "collator renders with emit_scores=True and that the split carries labels."
        )
    predictions = {k: np.concatenate([p for p, _ in v]) for k, v in collected.items()}
    gold = {k: np.concatenate([g for _, g in v]) for k, v in collected.items()}
    metrics = {
        key: field_metrics(predictions[key], gold[key], bootstrap=bootstrap)
        for key in predictions
    }
    return ScoreResult(
        metrics=metrics, predictions=predictions, gold=gold,
        scored=scored, partial=stopped,
    )


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

RESUME_META = "resume.json"
RESUME_STATE = "resume_state.pt"


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            [s.cpu() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "mps": torch.mps.get_rng_state().cpu() if _has_mps() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    if torch.cuda.is_available():
        if state.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in state["cuda"]])
        else:
            logger.warning(
                "[resume] checkpoint has no saved CUDA RNG state; "
                "dropout stream will not match an uninterrupted run"
            )
    if _has_mps():
        if state.get("mps") is not None:
            torch.mps.set_rng_state(state["mps"].to(torch.uint8).cpu())
        else:
            logger.warning(
                "[resume] checkpoint has no saved MPS RNG state; "
                "dropout stream will not match an uninterrupted run"
            )


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
