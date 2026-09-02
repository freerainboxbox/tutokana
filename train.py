"""Train tutokana on speechocean762.

Fine-tunes Gemma 4 12B with LoRA plus score-register heads. The assistant turn carries a word
and phoneme transcript with a register token at every score position; the scores are read out
of the hidden state at those registers rather than generated as text.

    python train.py                          # current best configuration
    python train.py --train-probe 400        # short run for iteration
    python train.py --resume <run-id>        # continue an interrupted run
    python train.py --lambda-detect 0        # ablate the detection term

Ctrl-C finishes the current step, checkpoints, and prints the resume command. Every run writes
to `runs/<run_id>/` and logs to `logs/train-<run_id>-<completion>.log`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Importable without installing: the package lives in src/, so put it on the path when the
# script is run straight out of a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tutokana import config as cfg
from tutokana.collate import Collator
from tutokana.data import (
    compute_target_stats,
    load_split,
    phone_vocabulary,
    speaker_split,
    stratified_indices,
)
from tutokana.engine import (
    build_optimizer,
    build_scheduler,
    batches,
    clear_resume,
    graceful_interrupt,
    load_resume,
    move_batch,
    preflight,
    restore_rng_state,
    save_resume,
    score,
    seed_everything,
    summarize,
)
from tutokana.heads import build_head_specs
from tutokana.losses import (
    CorrelationBuffer,
    LossConfig,
    build_reweighter,
    composite_loss,
)
from tutokana.mix import build_mix
from tutokana.model import (
    ModelConfig,
    build_model,
    load_processor,
    resolve_device,
    save_run_metadata,
)
from tutokana.prompting import TargetSpec
from tutokana.progress import Progress, memory_note
from tutokana.reporting import wandb_metrics
from tutokana.tokens import register_tokens


def parse_args() -> tuple[argparse.Namespace, set[str]]:
    """Parsed arguments, plus the set of flags the caller actually typed.

    The second half matters only for --resume: comparing the assembled config against the
    saved one would flag every default that differs from the interrupted run's settings,
    burying the handful of flags the user really did pass. Re-parsing an empty argument list
    gives the defaults to compare against.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default="", help="run directory name (default: timestamped)")
    parser.add_argument("--notes", default="", help="free-text note stored with the run")
    parser.add_argument(
        "--resume", default="", metavar="RUN_ID",
        help="continue an interrupted run. Every other setting comes from its resume.json; "
             "conflicting flags are named and ignored rather than silently applied",
    )
    cfg.add_config_arguments(parser)
    args = parser.parse_args()
    defaults = parser.parse_args([])
    explicit = {
        name for name, value in vars(args).items() if value != getattr(defaults, name)
    }
    return args, explicit


def main() -> None:
    args, explicit = parse_args()
    config = cfg.config_from_args(args)

    # The mix, splits, shuffles and schedule are deterministic functions of the saved
    # config, so a conflicting flag would silently continue a different run.
    resume_meta = resume_state = None
    if args.resume:
        if args.run_id:
            raise SystemExit("--run-id and --resume are mutually exclusive")
        run_id = args.resume
        run_dir = cfg.RUN_DIR / run_id
        resume_meta, resume_state = load_resume(run_dir)
        saved = cfg.Config.from_dict(resume_meta["config"])
        conflicts = [
            entry
            for entry in config.diff(saved)
            if entry[0].replace(".", "__") in explicit
        ]
        config = saved
    else:
        run_id = args.run_id or cfg.new_run_id("probe" if config.train.probe else "run")
        run_dir = cfg.RUN_DIR / run_id
        conflicts = []

    with cfg.run_logging("train", run_id) as log:
        started = time.time()
        log.info("run %s", run_id)
        if resume_meta is not None:
            log.info(
                "[resume] from epoch %d, sample %d, step %d (saved %s)",
                resume_meta["epoch"], resume_meta["next_sample"], resume_meta["step"],
                resume_meta.get("saved_at", "?"),
            )
            for path, given, kept in conflicts:
                log.warning("[resume] ignoring --%s=%r; using the saved %r",
                            path.replace("_", "-").replace(".", "-"), given, kept)
        log.info("config %s", json.dumps(config.to_dict(), default=str))

        seed_everything(config.train.seed)
        device = resolve_device(config.device)
        log.info("device %s", device)

        # -- data ---------------------------------------------------------------------
        limit = config.data.limit or None
        full_train = load_split("train", limit=limit)
        train_utterances, val_utterances = speaker_split(
            full_train, config.data.val_speakers, config.train.seed
        )
        val_subset = [
            val_utterances[i]
            for i in stratified_indices(val_utterances, config.data.val_samples)
        ]
        log.info(
            "[data] %d train / %d val utterances (%d val speakers held out), scoring on %d",
            len(train_utterances), len(val_utterances), config.data.val_speakers,
            len(val_subset),
        )

        stats = compute_target_stats(train_utterances)
        phones = phone_vocabulary(train_utterances)
        log.info("[data] %d phone symbols in train", len(phones))

        mix, mix_stats = build_mix(
            train_utterances,
            k=config.data.oversample_k,
            n_negatives=config.data.n_negatives,
            seed=config.train.seed,
        )
        log.info("[mix] %s", json.dumps(mix_stats))
        if config.train.probe:
            mix = random.Random(config.train.seed).sample(
                mix, min(config.train.probe, len(mix))
            )
            log.info("[mix] probe mode: %d samples", len(mix))

        # -- model --------------------------------------------------------------------
        spec = TargetSpec(levels=tuple(config.data.levels), emit_phones=config.data.emit_phones)
        processor = load_processor(config.model_id)
        register_ids = register_tokens(getattr(processor, "tokenizer", processor))
        head_specs = build_head_specs(
            levels=spec.levels,
            head_modes=config.head_modes.as_dict(),
            n_phone_conditions=len(phones) + 1,
            phone_conditioning=config.phone_conditioning,
            stats=stats,
            stress_concentration=config.stress_concentration,
            stress_siblings=config.stress_siblings,
        )
        log.info("[model] heads %s", sorted(head_specs))

        model = build_model(
            ModelConfig(
                model_id=config.model_id,
                lora_r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                train_audio_projection=config.train_audio_projection,
                head_layers=config.head_layers,
                layer_mixture=config.layer_mixture,
            ),
            head_specs=head_specs,
            levels=spec.levels,
            register_ids=register_ids,
            device=device,
            adapter_dir=(run_dir / "adapter") if resume_meta is not None else None,
            trainable_adapter=resume_meta is not None,
        )
        if resume_meta is not None:
            model.load_trained(run_dir)
        trainable, total = model.trainable_parameters()
        log.info("[model] %.2fM trainable / %.2fM total (%.3f%%)",
                 trainable / 1e6, total / 1e6, 100 * trainable / total)

        collator = Collator(
            processor, spec, stats, phones, max_length=config.data.max_length
        )

        loss_config = LossConfig(
            level_weights=config.level_weights(),
            lambda_ccc=config.lambda_ccc,
            lambda_lm=config.lambda_lm,
            buffer_capacity=config.buffer_capacity,
            reweight_levels=tuple(config.reweight_levels),
            reweight_strength=config.reweight_strength,
            reweight_max=config.reweight_max,
            lambda_detect=config.lambda_detect,
            detect_pos_weight=config.detect_pos_weight,
        )
        reweighter = build_reweighter(train_utterances, loss_config)
        buffer = CorrelationBuffer(loss_config.buffer_capacity)

        # -- optimisation -------------------------------------------------------------
        steps_per_epoch = max(
            1, len(mix) // (config.train.batch_size * config.train.grad_accum)
        )
        total_steps = steps_per_epoch * config.train.epochs
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, total_steps, config.train.warmup_ratio)
        log.info("[train] %d steps (%d/epoch x %d epochs)",
                 total_steps, steps_per_epoch, config.train.epochs)

        # Restored before preflight so preflight exercises the state training will use; the
        # RNG comes after, since preflight's dropout would consume the stream being restored.
        start_epoch, start_sample, step = 0, 0, 0
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            if resume_state.get("buffer") is not None:
                buffer.load_state_dict(resume_state["buffer"])
            start_epoch = resume_meta["epoch"]
            start_sample = resume_meta["next_sample"]
            step = resume_meta["step"]
            window = config.train.batch_size * config.train.grad_accum
            if start_sample % window:
                raise SystemExit(
                    f"[resume] saved position {start_sample} is not on an accumulation "
                    f"boundary ({window}); the bundle does not match this configuration"
                )

        preflight(model, collator, mix[:2], device, loss_config, reweighter, buffer)

        if resume_state is not None:
            restore_rng_state(resume_state["rng"])

        run = None
        if config.train.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=config.train.wandb_project,
                name=run_id,
                mode=config.train.wandb_mode,
                id=resume_meta.get("wandb_run_id") if resume_meta else None,
                resume="must" if resume_meta and resume_meta.get("wandb_run_id") else None,
                config=config.to_dict() | {"mix": mix_stats, "notes": args.notes},
            )

        # Written before the first step as well as after the last: the statistics are
        # needed to interpret any checkpoint, including one left behind by a crash.
        run_dir.mkdir(parents=True, exist_ok=True)
        stats.save(run_dir / "target_stats.json")

        def resume_snapshot(epoch: int, next_sample: int) -> dict:
            return {
                "run_id": run_id,
                "wandb_run_id": run.id if run is not None else None,
                "config": config.to_dict(),
                "phones": list(phones),
                "epoch": epoch,
                "next_sample": next_sample,
                "step": step,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }

        stopped = False
        bar = Progress(total_steps, log, "train", every=config.train.log_every,
                       unit="step", start_at=step)
        with graceful_interrupt() as interrupt:
            for epoch in range(start_epoch, config.train.epochs):
                # Reconstructed, not stored: the order is a pure function of (seed, epoch),
                # so a resumed epoch replays exactly the sequence it was part way through.
                order = random.Random(config.train.seed + epoch).sample(
                    range(len(mix)), len(mix)
                )
                shuffled = [mix[i] for i in order]
                consumed = start_sample if epoch == start_epoch else 0
                if consumed:
                    log.info("[resume] skipping %d samples already seen this epoch", consumed)
                model.train()
                window_loss, window_count = 0.0, 0

                for micro, chunk in enumerate(
                    batches(shuffled[consumed:], config.train.batch_size)
                ):
                    batch = move_batch(collator(chunk), device)
                    head_outputs, lm_loss = model(batch)
                    loss, parts = composite_loss(
                        model.heads, head_outputs, batch["heads"], loss_config,
                        buffer, reweighter, lm_loss,
                    )
                    (loss / config.train.grad_accum).backward()
                    window_loss += parts["loss/total"]
                    window_count += 1

                    if (micro + 1) % config.train.grad_accum:
                        continue

                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        config.train.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    mean_loss = window_loss / max(window_count, 1)
                    bar.update(
                        1,
                        suffix=f"  loss {mean_loss:.3f}",
                        detail=f"epoch {epoch} loss {mean_loss:.4f} "
                               f"lr {scheduler.get_last_lr()[0]:.2e} {memory_note(device)}",
                    )

                    if step % config.train.log_every == 0:
                        if run is not None:
                            run.log({**parts, "train/step": step,
                                     "train/lr": scheduler.get_last_lr()[0]}, step=step)
                        window_loss, window_count = 0.0, 0

                    if config.train.val_every and step % config.train.val_every == 0:
                        result = score(model, collator, val_subset, device,
                                       batch_size=config.train.batch_size, bootstrap=False)
                        bar.write(f"[val] step {step}  {summarize(result.metrics)}")
                        if run is not None:
                            run.log(wandb_metrics(result.metrics, "val"), step=step)
                        model.train()

                    next_sample = consumed + (micro + 1) * config.train.batch_size

                    if config.train.save_every and step % config.train.save_every == 0:
                        model.save(run_dir / f"checkpoint-{step}")
                        model.save(run_dir)
                        save_resume(
                            run_dir, meta=resume_snapshot(epoch, next_sample),
                            optimizer=optimizer, scheduler=scheduler, buffer=buffer,
                        )
                        bar.write(f"[save] checkpoint-{step} (resumable)")

                    if interrupt["requested"]:
                        model.save(run_dir)
                        save_resume(
                            run_dir, meta=resume_snapshot(epoch, next_sample),
                            optimizer=optimizer, scheduler=scheduler, buffer=buffer,
                        )
                        bar.write(
                            f"[save] interrupted at epoch {epoch} sample {next_sample} "
                            f"step {step} — continue with: python train.py --resume {run_id}"
                        )
                        stopped = True
                        break
                if stopped:
                    break

        bar.close()
        model.save(run_dir)
        stats.save(run_dir / "target_stats.json")
        if not stopped:
            # A finished run must leave nothing behind that a later --resume could pick up.
            clear_resume(run_dir)
        result = score(model, collator, val_subset, device,
                       batch_size=config.train.batch_size, bootstrap=False)
        log.info("[val] final  %s", summarize(result.metrics))
        log.info("[val] layer mixtures %s", json.dumps(model.heads.mixture_report()))

        save_run_metadata(
            run_dir,
            {
                "run_id": run_id,
                "config": config.to_dict(),
                "phones": list(phones),
                "mix": mix_stats,
                "steps": step,
                "interrupted": stopped,
                "notes": args.notes,
                "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - started, 1),
            },
        )
        log.info(
            "[done] %s in %.1f min%s", run_dir, (time.time() - started) / 60,
            f" (interrupted — resume with --resume {run_id})" if stopped else "",
        )
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
