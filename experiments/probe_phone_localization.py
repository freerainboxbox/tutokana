"""Has the model already solved alignment, or would a forced aligner add something?

`probe_audio_ablation.py` asks whether the model listens at all. This asks something
narrower: when it scores phone *i*, is it reading the part of the audio where phone *i*
actually is?

The question matters because it decides an architecture choice. GOPT/3MH/HMamba are fed
GOP features computed on force-aligned frames; this model is handed the canonical phone
sequence and the raw audio and has to work out the correspondence itself. If it has, an
aligner buys nothing. If it has not, boundary markers or an alignment embedding on the phone
register have real headroom.

**Method.** Split each utterance's audio into `--segments` equal spans, zero one span at a
time, and re-score. Every phone's prediction is compared against the untouched run, and the
shift is bucketed by how far that phone sits from the zeroed span — using a uniform-time
assumption, since the corpus ships no alignment. A model that localises shows shift
concentrated at distance 0 and decaying; a model that reads the whole utterance as one blob
shows a flat profile.

**What the uniform-time assumption costs.** Phones are not equal length, so a phone's true
span may be a bucket or so from its assumed one. That blurs the profile — it cannot
manufacture a peak that is not there. A flat result is therefore weak evidence (blur could
hide a real peak), while a peaked result is strong evidence (nothing but real localisation
puts it there). Read it in that direction.

    uv run python experiments/probe_phone_localization.py [--run-id RUN] [--n 120]

The `identity` row re-scores untouched audio: its shift is pure numerical noise and every
other number should be read against it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tutokana import config as cfg
from tutokana.collate import Collator
from tutokana.data import TargetStats, load_split, phone_vocabulary, stratified_indices
from tutokana.engine import batches, move_batch
from tutokana.heads import build_head_specs
from tutokana.model import ModelConfig, build_model, load_processor, resolve_device
from tutokana.progress import Progress
from tutokana.prompting import TargetSpec
from tutokana.tokens import register_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default="", help="run directory (default: latest completed)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=120, help="utterances to probe")
    parser.add_argument("--segments", type=int, default=5,
                        help="equal spans the audio is divided into; each is zeroed in turn")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def zero_segment(utterance, index: int, segments: int):
    """A copy with one of `segments` equal spans of the waveform silenced."""
    audio = np.array(utterance.audio, copy=True)
    edges = np.linspace(0, len(audio), segments + 1).astype(int)
    audio[edges[index] : edges[index + 1]] = 0.0
    return replace(utterance, audio=audio)


@torch.no_grad()
def phone_predictions(model, collator, utterances, device, batch_size: int) -> np.ndarray:
    """Flat phone-accuracy predictions, in dataset order."""
    model.eval()
    out = []
    for chunk in batches(utterances, batch_size):
        batch = move_batch(collator(chunk), device)
        head_outputs, _ = model(batch, with_lm_loss=False)
        logits = head_outputs["phone.accuracy"]
        head = model.heads.head("phone.accuracy")
        out.append(head.predict_native(logits).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    run_dir = cfg.RUN_DIR / args.run_id if args.run_id else cfg.latest_run()

    with cfg.run_logging("localization", run_dir.name) as log:
        import json

        meta = json.loads((run_dir / "config.json").read_text())
        config = cfg.Config.from_dict(meta["config"])
        device = resolve_device(config.device)
        log.info("run %s on %s", run_dir, device)

        utterances = load_split(args.split)
        subset = [utterances[i] for i in stratified_indices(utterances, args.n)]
        counts = [sum(len(w.phones) for w in u.words) for u in subset]
        log.info("[probe] %d utterances, %d phones, %d segments",
                 len(subset), sum(counts), args.segments)

        stats = TargetStats.load(run_dir / "target_stats.json")
        phones = tuple(meta["phones"])
        spec = TargetSpec(levels=tuple(config.data.levels),
                          emit_phones=config.data.emit_phones)
        processor = load_processor(config.model_id)
        model = build_model(
            ModelConfig(
                model_id=config.model_id, lora_r=config.lora_r,
                lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                head_layers=config.head_layers, layer_mixture=config.layer_mixture,
            ),
            head_specs=build_head_specs(
                levels=spec.levels, head_modes=config.head_modes.as_dict(),
                n_phone_conditions=len(phones) + 1,
                phone_conditioning=config.phone_conditioning, stats=stats,
            ),
            levels=spec.levels,
            register_ids=register_tokens(getattr(processor, "tokenizer", processor)),
            device=device,
            adapter_dir=run_dir / "adapter",
        )
        model.load_trained(run_dir)
        collator = Collator(processor, spec, stats, phones,
                            max_length=config.data.max_length)

        with Progress(args.segments + 2, log, "localize", every=1, unit="pass") as bar:
            reference = phone_predictions(model, collator, subset, device, args.batch_size)
            bar.update(detail="reference")
            control = phone_predictions(model, collator, subset, device, args.batch_size)
            bar.update(detail="identity control")

            # distance is measured in segment widths, so it is comparable across utterances
            # of different lengths; the fractional position of each phone comes from the
            # uniform-time assumption the docstring qualifies.
            position = np.concatenate([(np.arange(n) + 0.5) / n for n in counts])
            shifts: dict[int, list[float]] = {}
            for index in range(args.segments):
                altered = [zero_segment(u, index, args.segments) for u in subset]
                moved = np.abs(
                    phone_predictions(model, collator, altered, device, args.batch_size)
                    - reference
                )
                centre = (index + 0.5) / args.segments
                distance = np.abs(position - centre) * args.segments
                for bucket in range(args.segments):
                    mask = (distance >= bucket) & (distance < bucket + 1)
                    if mask.any():
                        shifts.setdefault(bucket, []).extend(moved[mask].tolist())
                bar.update(detail=f"segment {index}")

        noise = float(np.abs(control - reference).mean())
        log.info("")
        log.info("=== how far the audio was zeroed from the phone being scored ===")
        log.info("%-26s%10s%14s%14s", "distance (segments)", "n phones", "mean |shift|",
                 "vs identity")
        log.info("-" * 64)
        for bucket in sorted(shifts):
            values = np.asarray(shifts[bucket])
            ratio = values.mean() / noise if noise > 1e-12 else float("inf")
            log.info("%-26s%10d%14.4f%13.1fx",
                     f"{bucket} - {bucket + 1}", values.size, values.mean(), ratio)
        log.info("%-26s%10d%14.4f%13s", "identity (control)", control.size, noise, "1.0x")

        near = np.asarray(shifts.get(0, [0.0])).mean()
        far = np.asarray(shifts.get(max(shifts), [0.0])).mean()
        log.info("")
        log.info(
            "nearest/furthest ratio %.2fx — a model that localises concentrates the shift "
            "at distance 0; a flat profile means it reads the utterance as one blob. "
            "Uniform-time bucketing blurs a real peak but cannot invent one, so a peak is "
            "strong evidence and flatness is weak evidence.",
            near / far if far > 1e-12 else float("inf"),
        )


if __name__ == "__main__":
    main()
