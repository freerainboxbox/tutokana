"""Evaluate a trained run on the speechocean762 test split.

Scoring is teacher-forced: the canonical phone sequence comes from the dataset, so every
register aligns one-to-one with its gold score and one forward pass reads all eight fields.
A model generating its own phone sequence would misalign a few percent of positions and the
phone correlation would no longer mean what the published tables mean.

    python evaluate.py                     # full diagnostic table
    python evaluate.py --lite              # only what the literature reports
    python evaluate.py --run-id run-...    # a specific run ("base" = untuned floor)
    python evaluate.py --generative        # also decode transcripts, report PER

`--generative` measures the generative capability separately. The table is printed and
written verbatim to `logs/eval-<run_id>-<timestamp>.log`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tutokana import config as cfg
from tutokana.collate import Collator
from tutokana.data import TargetStats, load_split, stratified_indices
from tutokana.engine import graceful_interrupt, score, seed_everything
from tutokana.heads import build_head_specs
from tutokana.metrics import (
    detection_metrics,
    field_metrics,
    snap_to_support,
    spearman_ceiling,
    transcription_metrics,
)
from tutokana.model import ModelConfig, build_model, load_processor, resolve_device
from tutokana.prompting import TargetSpec, parse_generated
from tutokana.progress import Progress
from tutokana.reporting import (
    render_baselines,
    render_detection,
    render_snapping,
    render_table,
    render_transcription,
)
from tutokana.tokens import register_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id", default="",
        help='run directory name (default: latest completed). "base" reuses the latest '
             "run's configuration but skips its adapter and heads, giving an untuned floor",
    )
    parser.add_argument("--run-dir", default="", help="explicit path to a run directory")
    parser.add_argument("--split", default="test", help="dataset split (default: test)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N utterances")
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="utterances per forward pass. Memory scales with batch x sequence length, so "
             "raise this only after watching the memory column in the log (default: 2)",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="skip the correlation intervals")
    parser.add_argument(
        "--lite", action="store_true",
        help="report only PCC and MSE against the published baselines, omitting the "
             "rank correlation, dispersion, detection and snapping sections",
    )
    parser.add_argument("--generative", action="store_true",
                        help="also decode transcripts and report phone error rate")
    parser.add_argument("--generative-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default="", help="write per-field predictions to this JSON file")
    parser.add_argument(
        "--no-detection", action="store_true",
        help="skip the detection table. Spearman here is mostly a detection metric: perfect "
             "ceiling-vs-not detection reaches 99%% of the achievable rank correlation, "
             "perfect ordering below it reaches 1%%",
    )
    parser.add_argument(
        "--snap", action="store_true",
        help="also score predictions rounded onto the training label grid. Gold is "
             "quantised and a continuous readout is not; snapping restores the ties a rank "
             "correlation is comparing against. Applied after the fact, no retraining",
    )
    return parser.parse_args()


BASE_SENTINEL = "base"


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Which run's configuration to evaluate. `--run-id base` still needs one to read."""
    if args.run_dir:
        return Path(args.run_dir)
    if args.run_id and args.run_id != BASE_SENTINEL:
        path = cfg.RUN_DIR / args.run_id
        if not (path / "heads.safetensors").exists():
            raise SystemExit(f"{path} has no trained heads — is that the right run id?")
        return path
    return cfg.latest_run()


@torch.no_grad()
def generative_pass(model, processor, collator, utterances, device, max_new_tokens: int):
    """Decode the transcript from the prompt alone and pair it with the gold transcript."""
    tokenizer = getattr(processor, "tokenizer", processor)
    model.eval()
    predicted, gold = [], []
    for utterance in utterances:
        batch = collator([utterance])
        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
            if k != "utterances"
        }
        output = model.base.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        predicted.append(parse_generated(text))
        gold.append([(w.text, list(w.phones)) for w in utterance.words])
    return predicted, gold


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    meta = json.loads((run_dir / "config.json").read_text())
    config = cfg.Config.from_dict(meta["config"])
    run_id = meta.get("run_id", run_dir.name)

    with cfg.run_logging("eval", run_id) as log:
        log.info("run %s (trained %s)", run_dir, meta.get("completed_at", "?"))
        seed_everything(config.train.seed)
        device = resolve_device(config.device)

        utterances = load_split(args.split, limit=args.limit or None)
        log.info("[data] %d utterances from split %s", len(utterances), args.split)

        stats = TargetStats.load(run_dir / "target_stats.json")
        phones = tuple(meta["phones"])
        spec = TargetSpec(
            levels=tuple(config.data.levels), emit_phones=config.data.emit_phones
        )

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
        # The untuned floor: the same wiring and the same target statistics, but random
        # heads over the stock model. Whatever it scores is what the architecture gets for
        # free, and every trained number should be read against it.
        untuned = args.run_id == BASE_SENTINEL
        model = build_model(
            ModelConfig(
                model_id=config.model_id,
                head_layers=config.head_layers,
                layer_mixture=config.layer_mixture,
                gradient_checkpointing=False,
            ),
            head_specs=head_specs,
            levels=spec.levels,
            register_ids=register_ids,
            device=device,
            adapter_dir=None if untuned else run_dir / "adapter",
        )
        if untuned:
            run_id = f"{run_id}-UNTUNED"
            log.info("[model] untuned baseline: stock weights, randomly initialised heads")
        else:
            model.load_trained(run_dir)
            log.info("[model] loaded adapter and heads from %s", run_dir)

        collator = Collator(
            processor, spec, stats, phones, max_length=config.data.max_length
        )
        with graceful_interrupt() as interrupt:
            with Progress(len(utterances), log, "score", every=10, unit="utt") as bar:
                result = score(
                    model, collator, utterances, device,
                    batch_size=args.batch_size,
                    bootstrap=not args.no_bootstrap,
                    progress=bar,
                    interrupt=interrupt,
                )
        if result.partial:
            log.warning(
                "[score] interrupted after %d/%d utterances — the table below is computed "
                "on what was scored, not the full split",
                result.scored, len(utterances),
            )

        title = f"{run_id} / {args.split}"
        if result.partial:
            title += f" (PARTIAL: {result.scored}/{len(utterances)})"
        table = render_table(result.metrics, title=title, lite=args.lite)
        comparison = render_baselines(result.metrics)
        sections = [table, comparison]

        detection = {}
        if not args.no_detection and not args.lite:
            detection = {
                key: detection_metrics(result.predictions[key], result.gold[key])
                for key in result.predictions
            }
            ceilings = {
                key: (result.metrics[key].spearman, spearman_ceiling(result.gold[key]))
                for key in result.predictions
            }
            sections.append(render_detection(detection, ceilings))

        snapped_metrics = {}
        if args.snap and not args.lite:
            missing = [k for k in result.predictions if k not in (stats.support or {})]
            if missing:
                log.warning(
                    "[snap] no label grid recorded for %s — this run predates support being "
                    "written to target_stats.json, so those fields are left unsnapped",
                    ", ".join(sorted(missing)),
                )
            # A two-point grid is not a grid. Snapping a continuous score onto {5, 10} is
            # thresholding, which destroys the ranking outright: every stress prediction
            # lands on 10, the field becomes constant, and its correlation is undefined.
            binary_grid = [
                k for k in result.predictions
                if len((stats.support or {}).get(k, ())) <= 2
            ]
            if binary_grid:
                log.info(
                    "[snap] skipping %s — a two-value support makes snapping a threshold, "
                    "not a rounding", ", ".join(sorted(binary_grid)),
                )
            snapped = {
                key: snap_to_support(values, stats.support[key])
                for key, values in result.predictions.items()
                if key in (stats.support or {}) and key not in binary_grid
            }
            snapped_metrics = {
                key: field_metrics(values, result.gold[key], bootstrap=False)
                for key, values in snapped.items()
            }
            if snapped_metrics:
                sections.append(render_snapping(result.metrics, snapped_metrics))

        for line in "\n\n".join(sections).splitlines():
            log.info("%s", line)

        if args.generative:
            subset = [
                utterances[i]
                for i in stratified_indices(utterances, args.generative_samples)
            ]
            decode_collator = Collator(
                processor, spec, stats, phones,
                max_length=config.data.max_length, with_target=False,
            )
            predicted, gold = generative_pass(
                model, processor, decode_collator, subset, device, args.max_new_tokens
            )
            transcription = transcription_metrics(predicted, gold)
            for line in render_transcription(transcription).splitlines():
                log.info("%s", line)

        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "split": args.split,
                        "metrics": {
                            k: asdict(v) | {"sigma_ratio": v.sigma_ratio}
                            for k, v in result.metrics.items()
                        },
                        "detection": {k: asdict(v) for k, v in detection.items()},
                        "spearman_ceiling": {
                            k: spearman_ceiling(v) for k, v in result.gold.items()
                        },
                        "snapped_metrics": {
                            k: asdict(v) | {"sigma_ratio": v.sigma_ratio}
                            for k, v in snapped_metrics.items()
                        },
                        "predictions": {k: v.tolist() for k, v in result.predictions.items()},
                        "gold": {k: v.tolist() for k, v in result.gold.items()},
                    },
                    indent=2,
                )
            )
            log.info("[out] %s", args.out)


if __name__ == "__main__":
    main()
