"""Train tutokana on speechocean762.

Fine-tunes `google/gemma-4-12B-it` with LoRA plus score-register heads. The assistant turn
carries a word and phoneme transcript with a dedicated register token at every score
position; the scores themselves never appear as text, they are read out of the hidden state
at those registers by small regression heads. See `src/tutokana/prompting.py` for the exact
layout and `src/tutokana/losses.py` for why the objective looks the way it does.

    python train.py                          # full run, all three levels
    python train.py --train-probe 400        # short run for hyperparameter iteration
    python train.py --data-levels utterance  # the joint-prediction ablation
    python train.py --lambda-ccc 0           # pointwise loss only, no correlation term

Every run writes to `runs/<run_id>/` (adapter, heads, target statistics, config) and logs to
`logs/train-<run_id>-<completion timestamp>.log` alongside the console.

The base weights are downloaded to the standard Hugging Face cache, never into the repo; set
HF_HOME to move it.
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
    graceful_interrupt,
    move_batch,
    preflight,
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
from tutokana.reporting import wandb_metrics
from tutokana.tokens import register_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default="", help="run directory name (default: timestamped)")
    parser.add_argument("--notes", default="", help="free-text note stored with the run")
    cfg.add_config_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = cfg.config_from_args(args)
    run_id = args.run_id or cfg.new_run_id("probe" if config.train.probe else "run")
    run_dir = cfg.RUN_DIR / run_id

    with cfg.run_logging("train", run_id) as log:
        started = time.time()
        log.info("run %s", run_id)
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
        )
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
        )
        reweighter = build_reweighter(train_utterances, loss_config)
        buffer = CorrelationBuffer(loss_config.buffer_capacity)

        preflight(model, collator, mix[:2], device, loss_config, reweighter)

        # -- optimisation -------------------------------------------------------------
        steps_per_epoch = max(
            1, len(mix) // (config.train.batch_size * config.train.grad_accum)
        )
        total_steps = steps_per_epoch * config.train.epochs
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, total_steps, config.train.warmup_ratio)
        log.info("[train] %d steps (%d/epoch x %d epochs)",
                 total_steps, steps_per_epoch, config.train.epochs)

        run = None
        if config.train.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=config.train.wandb_project,
                name=run_id,
                mode=config.train.wandb_mode,
                config=config.to_dict() | {"mix": mix_stats, "notes": args.notes},
            )

        # Written before the first step as well as after the last: the statistics are
        # needed to interpret any checkpoint, including one left behind by a crash.
        run_dir.mkdir(parents=True, exist_ok=True)
        stats.save(run_dir / "target_stats.json")

        step = 0
        stopped = False
        with graceful_interrupt() as interrupt:
            for epoch in range(config.train.epochs):
                order = random.Random(config.train.seed + epoch).sample(
                    range(len(mix)), len(mix)
                )
                shuffled = [mix[i] for i in order]
                model.train()
                window_loss, window_start = 0.0, time.time()

                for micro, chunk in enumerate(batches(shuffled, config.train.batch_size)):
                    batch = move_batch(collator(chunk), device)
                    head_outputs, lm_loss = model(batch)
                    loss, parts = composite_loss(
                        model.heads, head_outputs, batch["heads"], loss_config,
                        buffer, reweighter, lm_loss,
                    )
                    (loss / config.train.grad_accum).backward()
                    window_loss += parts["loss/total"]

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

                    if step % config.train.log_every == 0:
                        window = config.train.log_every * config.train.grad_accum
                        rate = window / max(time.time() - window_start, 1e-6)
                        log.info(
                            "[train] epoch %d step %d/%d loss %.4f lr %.2e (%.2f batch/s)",
                            epoch, step, total_steps, window_loss / window,
                            scheduler.get_last_lr()[0], rate,
                        )
                        if run is not None:
                            run.log({**parts, "train/step": step,
                                     "train/lr": scheduler.get_last_lr()[0]}, step=step)
                        window_loss, window_start = 0.0, time.time()

                    if config.train.val_every and step % config.train.val_every == 0:
                        result = score(model, collator, val_subset, device,
                                       batch_size=config.train.batch_size, bootstrap=False)
                        log.info("[val] step %d  %s", step, summarize(result.metrics))
                        if run is not None:
                            run.log(wandb_metrics(result.metrics, "val"), step=step)
                        model.train()

                    if config.train.save_every and step % config.train.save_every == 0:
                        model.save(run_dir / f"checkpoint-{step}")
                        log.info("[save] checkpoint-%d", step)

                    if interrupt["requested"]:
                        stopped = True
                        break
                if stopped:
                    break

        model.save(run_dir)
        stats.save(run_dir / "target_stats.json")
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
        log.info("[done] %s in %.1f min%s", run_dir, (time.time() - started) / 60,
                 " (interrupted)" if stopped else "")
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
