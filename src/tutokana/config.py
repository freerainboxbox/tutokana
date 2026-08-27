"""Configuration and logging.

Config is a frozen dataclass tree bound to argparse, which buys one specific thing: the
resume file is `asdict(config)` rather than a hand-maintained list of keys. The predecessor
kept that list by hand and it drifted — a resumed run silently reconstructed a different
training mix than the one it was continuing.

Log files are named for when the run *finished*, which means the name cannot be known when
the handler is created. `run_logging` writes to `<name>.partial.log` and renames on the way
out through a `finally`, so a crashed or interrupted run still leaves a completely written,
correctly named file rather than nothing at all.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path

from .tokens import LEVELS

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PACKAGE_ROOT / "logs"
RUN_DIR = PACKAGE_ROOT / "runs"

LOG_FORMAT = "%(asctime)s %(levelname).1s %(message)s"
LOG_DATEFMT = "%H:%M:%S"


@dataclass(frozen=True)
class DataConfig:
    levels: tuple[str, ...] = LEVELS
    emit_phones: bool = True
    oversample_k: int = 5
    n_negatives: int = 300
    val_speakers: int = 12
    val_samples: int = 250
    max_length: int = 2048
    limit: int = 0  # 0 = the whole split


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 3407
    epochs: int = 2
    # Effective batch size = batch_size * grad_accum
    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 1e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    val_every: int = 200
    save_every: int = 500
    log_every: int = 10
    probe: int = 0  # 0 = full run; otherwise train on N mix samples
    wandb_project: str = "tutokana"
    wandb_mode: str = "online"


@dataclass(frozen=True)
class HeadModeConfig:
    phone: str = "soft_class"
    word: str = "regression"
    utterance: str = "regression"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    head_modes: HeadModeConfig = field(default_factory=HeadModeConfig)
    model_id: str = "google/gemma-4-12B-it"
    device: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    train_audio_projection: bool = False
    head_layers: int = 8
    layer_mixture: bool = True
    phone_conditioning: str = "film"
    lambda_ccc: float = 0.5
    lambda_lm: float = 0.5
    buffer_capacity: int = 512
    reweight_strength: float = 1.0
    reweight_max: float = 10.0
    reweight_levels: tuple[str, ...] = ("phone",)
    level_weight_phone: float = 1.0
    level_weight_word: float = 1.0
    level_weight_utterance: float = 1.0

    def level_weights(self) -> dict[str, float]:
        return {
            "phone": self.level_weight_phone,
            "word": self.level_weight_word,
            "utterance": self.level_weight_utterance,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def diff(self, other: "Config") -> list[tuple[str, object, object]]:
        """(field path, mine, theirs) for every field that disagrees, nested included.

        A resumed run reconstructs its mix, splits and schedule from the saved config, so a
        conflicting flag cannot be honoured — but it must be named rather than dropped
        silently, which is how the predecessor's resumed runs quietly drifted.
        """

        def walk(a, b, prefix=""):
            found = []
            for f in fields(a):
                mine, theirs = getattr(a, f.name), getattr(b, f.name)
                path = f"{prefix}{f.name}"
                if is_dataclass(mine):
                    found.extend(walk(mine, theirs, f"{path}."))
                elif mine != theirs:
                    found.append((path, mine, theirs))
            return found

        return walk(self, other)

    @classmethod
    def from_dict(cls, blob: dict) -> "Config":
        def build(kind, values):
            kwargs = {}
            for f in fields(kind):
                if f.name not in values:
                    continue
                value = values[f.name]
                if is_dataclass(f.type) if isinstance(f.type, type) else False:
                    kwargs[f.name] = build(f.type, value)
                elif isinstance(value, list):
                    kwargs[f.name] = tuple(value)
                else:
                    kwargs[f.name] = value
            return kind(**kwargs)

        nested = {
            "data": DataConfig,
            "train": TrainConfig,
            "head_modes": HeadModeConfig,
        }
        kwargs = {}
        for f in fields(cls):
            if f.name not in blob:
                continue
            if f.name in nested:
                kwargs[f.name] = build(nested[f.name], blob[f.name])
            elif isinstance(blob[f.name], list):
                kwargs[f.name] = tuple(blob[f.name])
            else:
                kwargs[f.name] = blob[f.name]
        return cls(**kwargs)


# --- argparse binding ----------------------------------------------------------------


NESTED_GROUPS: dict[str, type] = {
    "data": DataConfig,
    "train": TrainConfig,
    "head_modes": HeadModeConfig,
}


def _comma_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _add_field(group, field_def, default, flag: str, dest: str) -> None:
    # The dest carries the group prefix so `config_from_args` can reassemble the tree, but
    # the metavar shows the plain field name — `--data-levels LEVELS`, not `DATA__LEVELS`.
    metavar = field_def.name.upper()
    if isinstance(default, bool):
        group.add_argument(
            flag, dest=dest, action=argparse.BooleanOptionalAction,
            default=default, help=f"(default: {default})",
        )
    elif isinstance(default, tuple):
        group.add_argument(
            flag, dest=dest, type=_comma_tuple, default=default, metavar=metavar,
            help=f"comma-separated (default: {','.join(map(str, default))})",
        )
    else:
        group.add_argument(
            flag, dest=dest, type=type(default), default=default, metavar=metavar,
            help=f"(default: {default})",
        )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose every config field as a flag: --data-levels, --train-seed, --lora-r."""
    for prefix, kind in NESTED_GROUPS.items():
        group = parser.add_argument_group(prefix)
        defaults = kind()
        for f in fields(kind):
            flag = f"--{prefix.replace('_', '-')}-{f.name.replace('_', '-')}"
            _add_field(group, f, getattr(defaults, f.name), flag, f"{prefix}__{f.name}")

    group = parser.add_argument_group("model")
    defaults = Config()
    for f in fields(Config):
        if f.name in NESTED_GROUPS:
            continue
        flag = f"--{f.name.replace('_', '-')}"
        _add_field(group, f, getattr(defaults, f.name), flag, f.name)


def config_from_args(args: argparse.Namespace) -> Config:
    grouped: dict[str, dict] = {name: {} for name in NESTED_GROUPS}
    top: dict = {}
    top_names = {f.name for f in fields(Config)} - set(NESTED_GROUPS)
    for key, value in vars(args).items():
        if "__" in key:
            prefix, name = key.split("__", 1)
            if prefix in grouped:
                grouped[prefix][name] = value
        elif key in top_names:
            top[key] = value
    return Config(
        **{name: kind(**grouped[name]) for name, kind in NESTED_GROUPS.items()},
        **top,
    )


# --- logging --------------------------------------------------------------------------


def get_logger(name: str = "tutokana") -> logging.Logger:
    return logging.getLogger(name)


@contextlib.contextmanager
def run_logging(kind: str, run_id: str, log_dir: Path | None = None):
    """Console + file logging, with the file named for the completion time.

    `kind` is "train" or "eval". The final name is
    `logs/<kind>-<run_id>-<YYYYmmdd-HHMMSS>.log`.
    """
    log_dir = Path(log_dir or LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    partial = log_dir / f"{kind}-{run_id}.partial.log"

    logger = logging.getLogger("tutokana")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(partial, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)

    try:
        yield logger
    finally:
        logger.removeHandler(console)
        logger.removeHandler(file_handler)
        file_handler.close()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        final = log_dir / f"{kind}-{run_id}-{stamp}.log"
        partial.rename(final)
        print(f"[log] {final}")


# --- run directories ------------------------------------------------------------------


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def latest_run(run_dir: Path | None = None) -> Path:
    """The most recently *completed* run directory.

    Ordered by the `completed_at` stamp written into `config.json` rather than by mtime,
    because touching a directory (copying it, listing it, a stray editor save) must not
    change which run is considered current.
    """
    root = Path(run_dir or RUN_DIR)
    candidates = []
    for path in sorted(root.glob("*/")):
        meta = path / "config.json"
        if not (path / "heads.safetensors").exists() or not meta.exists():
            continue
        try:
            stamp = json.loads(meta.read_text()).get("completed_at", "")
        except json.JSONDecodeError:
            stamp = ""
        candidates.append((stamp, path))
    if not candidates:
        raise SystemExit(
            f"no completed run found under {root} — train one first, or pass --run-id"
        )
    return max(candidates)[1]
