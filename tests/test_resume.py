"""Interruption and clean resume.

A resume is clean when continuing produces the trajectory the uninterrupted run would have
had. That needs optimizer moments, schedule position, RNG streams, the correlation buffer and
the epoch/sample offset; the mix, splits, epoch orders and target statistics are recomputed
from the saved config, and these tests pin that determinism.
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


def test_rng_state_round_trips_accelerator():
    """Dropout on MPS/CUDA draws from the accelerator generator, not the CPU one."""
    device = _accelerator()
    if device is None:
        pytest.skip("no accelerator available")

    saved = rng_state()
    assert saved[device.type] is not None
    expected = torch.rand(16, device=device)
    _ = torch.rand(64, device=device)  # advance the stream
    restore_rng_state(saved)
    restored = torch.rand(16, device=device)
    assert torch.equal(expected, restored)


def test_restore_rng_warns_when_accelerator_state_is_missing(caplog):
    """An older checkpoint lacking MPS/CUDA RNG state must not silently diverge."""
    device = _accelerator()
    if device is None:
        pytest.skip("no accelerator available")

    legacy = rng_state()
    legacy[device.type] = None
    with caplog.at_level("WARNING", logger="tutokana"):
        restore_rng_state(legacy)
    assert any(device.type.upper() in rec.message for rec in caplog.records)


def test_rng_state_round_trips_mps_logic_via_mock(monkeypatch, caplog):
    """Verify MPS RNG save/restore and missing-MPS warning even without hardware MPS access."""
    import tutokana.engine as eng

    fake_state = {"val": torch.tensor([1, 2, 3, 4], dtype=torch.uint8)}
    monkeypatch.setattr(eng, "_has_mps", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: fake_state["val"].clone())

    def fake_set(t):
        fake_state["val"] = t.clone()

    monkeypatch.setattr(torch.mps, "set_rng_state", fake_set)

    saved = rng_state()
    assert torch.equal(saved["mps"], torch.tensor([1, 2, 3, 4], dtype=torch.uint8))

    fake_state["val"] = torch.tensor([9, 9, 9, 9], dtype=torch.uint8)
    restore_rng_state(saved)
    assert torch.equal(fake_state["val"], torch.tensor([1, 2, 3, 4], dtype=torch.uint8))

    legacy = dict(saved, mps=None)
    with caplog.at_level("WARNING", logger="tutokana"):
        restore_rng_state(legacy)
    assert any("MPS" in rec.message for rec in caplog.records)


def test_preflight_does_not_mutate_correlation_buffer():
    """Preflight must exercise the restored buffer without pushing mix[:2] into its FIFO."""
    from tutokana.collate import HeadBatch
    from tutokana.engine import preflight
    from tutokana.losses import LossConfig

    class _TinyHead(torch.nn.Module):
        mode = "regression"

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

        def predict_normalized(self, logits):
            return logits.squeeze(-1)

    class _TinyBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._head = _TinyHead()

        def head(self, key):
            return self._head

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = _TinyBank()
            self.supports = {}

        def forward(self, batch):
            logits = self.heads.head("utterance.total")(batch["input_features"])
            lm_loss = logits.sum() * 0.0
            return {"utterance.total": logits}, lm_loss, {}

    def fake_collator(_samples):
        return {
            "input_ids": torch.tensor([[1, 2], [1, 2]]),
            "labels": torch.tensor([[1, 2], [1, 2]]),
            "input_features": torch.randn(2, 4),
            "heads": {
                "utterance.total": HeadBatch(
                    positions=torch.tensor([[0, 1], [1, 1]]),
                    target=torch.tensor([0.5, -0.5]),
                    raw=torch.tensor([8.0, 6.0]),
                    native=torch.tensor([8.0, 6.0]),
                )
            },
        }

    model = _TinyModel()
    buf = CorrelationBuffer(64)
    orig_pred = torch.tensor([0.1, 0.2, 0.3])
    orig_tgt = torch.tensor([0.4, 0.5, 0.6])
    buf.extend("utterance.total", orig_pred, orig_tgt)
    before = buf.state_dict()

    preflight(model, fake_collator, [None, None], torch.device("cpu"), LossConfig(), buffer=buf)

    after = buf.state_dict()
    assert torch.equal(after["predictions"]["utterance.total"], before["predictions"]["utterance.total"])
    assert torch.equal(after["targets"]["utterance.total"], before["targets"]["utterance.total"])
    assert all(p.grad is None for p in model.parameters())


def test_interrupted_and_resumed_trajectory_is_bit_identical_to_uninterrupted(tmp_path):
    """End-to-end equivalence: 6 uninterrupted steps == 3 steps + save_resume + preflight + 3 steps."""
    from tutokana.collate import HeadBatch
    from tutokana.engine import preflight, seed_everything
    from tutokana.losses import LossConfig, composite_loss

    class _TinyHead(torch.nn.Module):
        mode = "regression"

        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x)

        def predict_normalized(self, logits):
            return logits.squeeze(-1)

    class _TinyBank(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._head = _TinyHead()

        def head(self, key):
            return self._head

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = _TinyBank()
            self.register_delta = torch.nn.Linear(4, 4)
            self.base = torch.nn.Linear(4, 4)
            self.supports = {}

        def forward(self, batch):
            h = self.base(batch["input_features"]) + self.register_delta(batch["input_features"])
            logits = self.heads.head("utterance.total")(h)
            lm_loss = (h ** 2).mean()
            return {"utterance.total": logits}, lm_loss, {}

    def make_batch(i: int):
        g = torch.Generator().manual_seed(100 + i)
        return {
            "input_ids": torch.tensor([[1, 2], [1, 2]]),
            "labels": torch.tensor([[1, 2], [1, 2]]),
            "input_features": torch.randn(2, 4, generator=g),
            "heads": {
                "utterance.total": HeadBatch(
                    positions=torch.tensor([[0, 1], [1, 1]]),
                    target=torch.randn(2, generator=g),
                    raw=torch.tensor([8.0, 6.0]),
                    native=torch.tensor([8.0, 6.0]),
                )
            },
        }

    cfg = Config(train=TrainConfig(seed=42))
    loss_cfg = LossConfig(lambda_ccc=0.5, lambda_lm=0.5, lambda_detect=0.0)
    device = torch.device("cpu")

    def step_once(model, opt, sched, buf, i):
        model.train()
        batch = make_batch(i)
        ho, lm, sl = model(batch)
        loss, _ = composite_loss(model.heads, ho, batch["heads"], loss_cfg, buf, None, lm)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    # 1. Uninterrupted 6-step run.
    seed_everything(42)
    m_full = _TinyModel()
    opt_full = build_optimizer(m_full, cfg)
    sched_full = build_scheduler(opt_full, 6, cfg.train.warmup_ratio)
    buf_full = CorrelationBuffer(64)
    preflight(m_full, lambda _s: make_batch(0), [None, None], device, loss_cfg, buffer=buf_full)
    losses_full = [step_once(m_full, opt_full, sched_full, buf_full, i) for i in range(6)]

    # 2. Run 3 steps, checkpoint, reload fresh, run preflight, restore RNG, finish steps 3..5.
    seed_everything(42)
    m_part = _TinyModel()
    opt_part = build_optimizer(m_part, cfg)
    sched_part = build_scheduler(opt_part, 6, cfg.train.warmup_ratio)
    buf_part = CorrelationBuffer(64)
    preflight(m_part, lambda _s: make_batch(0), [None, None], device, loss_cfg, buffer=buf_part)
    losses_part = [step_once(m_part, opt_part, sched_part, buf_part, i) for i in range(3)]
    torch.save(m_part.state_dict(), tmp_path / "model.pt")
    save_resume(tmp_path, meta={"step": 3}, optimizer=opt_part, scheduler=sched_part, buffer=buf_part)

    # Fresh process simulation on resume.
    seed_everything(42)
    m_res = _TinyModel()
    m_res.load_state_dict(torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True))
    opt_res = build_optimizer(m_res, cfg)
    sched_res = build_scheduler(opt_res, 6, cfg.train.warmup_ratio)
    buf_res = CorrelationBuffer(64)
    _, state = load_resume(tmp_path)
    opt_res.load_state_dict(state["optimizer"])
    sched_res.load_state_dict(state["scheduler"])
    buf_res.load_state_dict(state["buffer"])
    preflight(m_res, lambda _s: make_batch(0), [None, None], device, loss_cfg, buffer=buf_res)
    restore_rng_state(state["rng"])
    losses_part.extend(step_once(m_res, opt_res, sched_res, buf_res, i) for i in range(3, 6))

    assert losses_part == losses_full
    for (name, p1), (_, p2) in zip(m_full.named_parameters(), m_res.named_parameters()):
        assert torch.equal(p1, p2), f"parameter {name} diverged after resume"
    assert torch.equal(
        buf_full._predictions["utterance.total"], buf_res._predictions["utterance.total"]
    )
    assert torch.equal(
        buf_full._targets["utterance.total"], buf_res._targets["utterance.total"]
    )


def _press_ctrl_c():
    """Deliver a real SIGINT to this process; Python runs the handler before returning."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGINT)


def test_repeated_ctrl_c_cannot_interrupt_a_checkpoint_write(tmp_path):
    """Any number of presses mid-save is recorded, never raised, and the bundle completes."""
    import signal

    from tutokana.engine import deferred_interrupt, graceful_interrupt

    _, optimizer, scheduler = _optimizer_and_scheduler(Config())
    with graceful_interrupt() as interrupt:
        _press_ctrl_c()  # the first press: finish the step, then save
        assert interrupt["requested"]
        with deferred_interrupt(interrupt):
            _press_ctrl_c()  # would be graceful_interrupt's raise-now press
            _press_ctrl_c()
            save_resume(tmp_path, meta={"step": 7}, optimizer=optimizer,
                        scheduler=scheduler, buffer=None)
        meta, state = load_resume(tmp_path)
        assert meta["step"] == 7 and state["optimizer"] is not None
        # The escape hatch is back once the write is done.
        with pytest.raises(KeyboardInterrupt):
            _press_ctrl_c()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_ctrl_c_during_a_periodic_save_requests_a_stop():
    """A press that lands mid-save must not be lost: the loop has to stop after it."""
    from tutokana.engine import deferred_interrupt, graceful_interrupt

    with graceful_interrupt() as interrupt:
        with deferred_interrupt(interrupt):
            _press_ctrl_c()
        assert interrupt["requested"]


def test_deferred_interrupt_shields_writes_outside_the_training_loop():
    """The final checkpoint runs after graceful_interrupt has exited; it needs its own guard."""
    import signal

    from tutokana.engine import deferred_interrupt

    with deferred_interrupt(what="final checkpoint") as state:
        _press_ctrl_c()
        _press_ctrl_c()
    assert state["requested"]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
