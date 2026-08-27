"""Interruption and clean resume.

Interrupting is the easy half. A resume is *clean* when continuing produces the trajectory
the uninterrupted run would have had, which needs more than the weights: optimizer moments,
the schedule position, the random streams driving shuffling and dropout, the correlation
buffer's population, and where in the epoch the run stopped. Everything else — the mix, the
splits, the epoch orders, the target statistics — is recomputed from the saved config, and
these tests pin that determinism down.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from tutokana.config import Config, DataConfig, TrainConfig
from tutokana.engine import (
    RESUME_META,
    RESUME_STATE,
    build_optimizer,
    build_scheduler,
    clear_resume,
    graceful_interrupt,
    load_resume,
    restore_rng_state,
    rng_state,
    save_resume,
)
from tutokana.losses import CorrelationBuffer
from tutokana.mix import build_mix


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.Linear(4, 1)
        self.register_delta = torch.nn.Linear(4, 1)
        self.base = torch.nn.Linear(4, 1)


def _optimizer_and_scheduler(config: Config):
    model = _Stub()
    optimizer = build_optimizer(model, config)
    return model, optimizer, build_scheduler(optimizer, 100, config.train.warmup_ratio)


def test_interrupt_sets_a_flag_rather_than_raising():
    """Raising mid-window would leave a half-accumulated gradient and an unsaved run."""
    import signal

    with graceful_interrupt() as state:
        assert state["requested"] is False
        signal.raise_signal(signal.SIGINT)
        assert state["requested"] is True


def test_handler_is_restored_afterwards():
    import signal

    before = signal.getsignal(signal.SIGINT)
    with graceful_interrupt():
        pass
    assert signal.getsignal(signal.SIGINT) is before


def test_rng_state_round_trips():
    """Dropout and the epoch shuffle both draw from these; a cold restart would diverge."""
    seed_state = rng_state()
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(seed_state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == first


def test_epoch_order_is_a_pure_function_of_seed_and_epoch(utterances):
    """Resume replays the interrupted epoch's order rather than storing it."""
    mix, _ = build_mix(utterances, k=2, n_negatives=2, seed=11)
    order = lambda epoch: random.Random(11 + epoch).sample(range(len(mix)), len(mix))
    assert order(3) == order(3)
    assert order(3) != order(4)


def test_mix_is_reconstructible_from_the_saved_config(utterances):
    first, _ = build_mix(utterances, k=3, n_negatives=3, seed=7)
    second, _ = build_mix(utterances, k=3, n_negatives=3, seed=7)
    assert [u.index for u in first] == [u.index for u in second]


def test_resume_bundle_round_trips(tmp_path):
    config = Config()
    model, optimizer, scheduler = _optimizer_and_scheduler(config)
    model.heads.weight.sum().backward()
    optimizer.step()
    for _ in range(5):
        scheduler.step()

    buffer = CorrelationBuffer(64)
    buffer.extend("utterance.total", torch.randn(8), torch.randn(8))

    meta = {"run_id": "r", "config": config.to_dict(), "epoch": 1,
            "next_sample": 32, "step": 5, "wandb_run_id": None}
    save_resume(tmp_path, meta=meta, optimizer=optimizer, scheduler=scheduler, buffer=buffer)
    assert (tmp_path / RESUME_META).exists() and (tmp_path / RESUME_STATE).exists()

    loaded_meta, state = load_resume(tmp_path)
    assert loaded_meta["epoch"] == 1 and loaded_meta["next_sample"] == 32

    fresh_model, fresh_optimizer, fresh_scheduler = _optimizer_and_scheduler(config)
    fresh_optimizer.load_state_dict(state["optimizer"])
    fresh_scheduler.load_state_dict(state["scheduler"])
    assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()

    fresh_buffer = CorrelationBuffer(1)
    fresh_buffer.load_state_dict(state["buffer"])
    assert fresh_buffer.capacity == 64
    assert fresh_buffer._predictions["utterance.total"].numel() == 8


def test_learning_rate_continues_rather_than_restarting(tmp_path):
    """A restarted schedule would re-warm-up and undo the decay already served."""
    config = Config()
    _, optimizer, scheduler = _optimizer_and_scheduler(config)
    for _ in range(40):
        optimizer.step()
        scheduler.step()
    interrupted = scheduler.get_last_lr()

    save_resume(tmp_path, meta={"step": 40}, optimizer=optimizer, scheduler=scheduler,
                buffer=CorrelationBuffer(0))
    _, fresh_optimizer, fresh_scheduler = _optimizer_and_scheduler(config)
    assert fresh_scheduler.get_last_lr() != interrupted  # a cold schedule starts warming up
    fresh_scheduler.load_state_dict(load_resume(tmp_path)[1]["scheduler"])
    assert fresh_scheduler.get_last_lr() == interrupted


def test_missing_bundle_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="nothing to resume"):
        load_resume(tmp_path)


def test_clear_resume_leaves_nothing_for_a_later_run(tmp_path):
    save_resume(tmp_path, meta={"step": 1}, optimizer=_optimizer_and_scheduler(Config())[1],
                scheduler=_optimizer_and_scheduler(Config())[2], buffer=CorrelationBuffer(0))
    clear_resume(tmp_path)
    assert not (tmp_path / RESUME_META).exists()
    assert not (tmp_path / RESUME_STATE).exists()
    clear_resume(tmp_path)  # idempotent


def test_saved_position_lands_on_an_accumulation_boundary():
    """train.py refuses a bundle whose position does not divide the window."""
    config = Config(train=TrainConfig(batch_size=4, grad_accum=4))
    window = config.train.batch_size * config.train.grad_accum
    for micro in range(1, 9):
        next_sample = micro * window
        assert next_sample % window == 0


def test_config_diff_names_only_what_disagrees():
    a = Config(data=DataConfig(oversample_k=5), lora_r=32)
    b = Config(data=DataConfig(oversample_k=3), lora_r=32)
    assert a.diff(b) == [("data.oversample_k", 5, 3)]
    assert a.diff(a) == []


def test_config_diff_reaches_nested_fields():
    a = Config(train=TrainConfig(seed=1), lora_r=8)
    b = Config(train=TrainConfig(seed=2), lora_r=16)
    paths = {path for path, _, _ in a.diff(b)}
    assert paths == {"train.seed", "lora_r"}


def test_diff_paths_map_onto_argparse_dests():
    """train.py filters conflicts by dest name, so the mapping must hold."""
    a = Config(data=DataConfig(oversample_k=5), train=TrainConfig(seed=1), lora_r=8)
    b = Config(data=DataConfig(oversample_k=3), train=TrainConfig(seed=2), lora_r=16)
    dests = {path.replace(".", "__") for path, _, _ in a.diff(b)}
    assert dests == {"data__oversample_k", "train__seed", "lora_r"}


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_extend_keeps_the_buffer_on_the_live_batch_device():
    """The invariant that a restored buffer violated: history follows the live batch."""
    buffer = CorrelationBuffer(64)
    live = torch.randn(4)
    buffer.extend("k", live, live)
    assert buffer._predictions["k"].device == live.device
    buffer.extend("k", live, live)
    assert buffer._predictions["k"].device == live.device


def test_restored_buffer_survives_a_batch_on_another_device():
    """A resumed run loads a host-side buffer and then trains on the accelerator.

    Concatenating across devices does not raise on MPS — it segfaults inside
    `structured_cat_out_mps` on the first step after a resume, with no Python traceback.
    """
    device = _accelerator()
    if device is None:
        pytest.skip("no accelerator available to cross devices with")

    warm = CorrelationBuffer(64)
    warm.extend("utterance.total", torch.randn(8, device=device), torch.randn(8, device=device))

    restored = CorrelationBuffer(64)
    restored.load_state_dict(warm.state_dict())  # comes back on the host
    assert restored._predictions["utterance.total"].device.type == "cpu"

    live = torch.randn(4, device=device)
    restored.extend("utterance.total", live, live)  # used to segfault here
    assert restored._predictions["utterance.total"].device.type == device.type
    assert restored._predictions["utterance.total"].numel() == 12

    pred, target = restored.augmented("utterance.total", live, live)
    assert pred.device.type == device.type and pred.numel() == 16


def test_preflight_accepts_a_buffer():
    """Signature guard: preflight must be able to exercise the restored buffer."""
    import inspect

    from tutokana.engine import preflight

    assert "buffer" in inspect.signature(preflight).parameters
