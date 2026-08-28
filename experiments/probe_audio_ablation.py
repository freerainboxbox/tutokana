"""Is the model actually listening? Score the same transcripts against altered audio.

A model can be functionally deaf without any loss curve showing it — supervising the constant
`<|audio|>` id collapses the representations at exactly the positions carrying speech. This
probe catches that.

Utterance-level scores are re-read under each condition and compared against the real-audio
run. A listening model shifts substantially under `zeros`, `noise` and `swapped`, and by ~0
under `identity`; a deaf model shifts by ~0 everywhere.

    uv run python experiments/probe_audio_ablation.py [--run-id RUN] [--n 32]

`identity` re-scores untouched audio, so its shift is numerical noise and every other row
should be read against it.
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
from tutokana.data import TargetStats, load_split, stratified_indices
from tutokana.engine import score, seed_everything
from tutokana.heads import build_head_specs
from tutokana.model import ModelConfig, build_model, load_processor, resolve_device
from tutokana.prompting import TargetSpec

UTTERANCE_FIELDS = (
    "utterance.accuracy",
    "utterance.prosodic",
    "utterance.fluency",
    "utterance.total",
)


def conditions(utterances, seed: int):
    """(name, utterances) pairs — the same transcripts against different waveforms."""
    rng = np.random.default_rng(seed)
    swapped = utterances[len(utterances) // 2 :] + utterances[: len(utterances) // 2]
    return [
        ("identity", list(utterances)),
        ("zeros", [replace(u, audio=np.zeros_like(u.audio)) for u in utterances]),
        (
            "noise",
            [
                replace(
                    u,
                    audio=(0.05 * rng.standard_normal(u.audio.shape)).astype(np.float32),
                )
                for u in utterances
            ],
        ),
        (
            "swapped",
            [replace(u, audio=other.audio) for u, other in zip(utterances, swapped)],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import json

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else (cfg.RUN_DIR / args.run_id if args.run_id else cfg.latest_run())
    )
    meta = json.loads((run_dir / "config.json").read_text())
    config = cfg.Config.from_dict(meta["config"])
    seed_everything(config.train.seed)
    device = resolve_device(config.device)

    utterances = load_split(args.split)
    subset = [utterances[i] for i in stratified_indices(utterances, args.n)]

    stats = TargetStats.load(run_dir / "target_stats.json")
    phones = tuple(meta["phones"])
    spec = TargetSpec(levels=tuple(config.data.levels), emit_phones=config.data.emit_phones)
    processor = load_processor(config.model_id)
    from tutokana.tokens import register_tokens

    model = build_model(
        ModelConfig(
            model_id=config.model_id,
            head_layers=config.head_layers,
            layer_mixture=config.layer_mixture,
            gradient_checkpointing=False,
        ),
        head_specs=build_head_specs(
            spec.levels,
            config.head_modes.as_dict(),
            len(phones) + 1,
            config.phone_conditioning,
            stats,
        ),
        levels=spec.levels,
        register_ids=register_tokens(getattr(processor, "tokenizer", processor)),
        device=device,
        adapter_dir=run_dir / "adapter",
    )
    model.load_trained(run_dir)
    collator = Collator(processor, spec, stats, phones, max_length=config.data.max_length)

    baseline = None
    print(f"=== audio ablation: {run_dir.name}, {len(subset)} utterances ===")
    header = f"{'condition':<12}" + "".join(f"{f.split('.')[1][:5]:>10}" for f in UTTERANCE_FIELDS)
    print(header + f"{'mean |d|':>11}")
    print("-" * len(header + f"{'mean |d|':>11}"))

    for name, altered in conditions(subset, config.train.seed):
        result = score(model, collator, altered, device, batch_size=args.batch_size,
                       bootstrap=False)
        values = {f: result.predictions[f] for f in UTTERANCE_FIELDS if f in result.predictions}
        if baseline is None:
            baseline = values
        shifts = {f: float(np.abs(v - baseline[f]).mean()) for f, v in values.items()}
        row = f"{name:<12}" + "".join(f"{values[f].mean():>10.2f}" for f in values)
        print(row + f"{np.mean(list(shifts.values())):>11.3f}")

    print(
        "\nA listening model shifts substantially under zeros/noise/swapped and by ~0 under\n"
        "identity. Near-zero shifts everywhere mean the scores do not depend on the audio."
    )


if __name__ == "__main__":
    main()
