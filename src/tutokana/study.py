"""Batch study orchestration helpers, state management, Fisher Z aggregation, and summaries.

A batch study evaluates a set of modes (subset of `heads`, `ce-std`, `ce-rail`, `raft-rail`,
`veil-rail`) across a sequence of fixed seeds in round-robin order, serially, storing every
artifact except the standard train/eval logs inside `batch-studies/<prefix>/`.

Modes are what gets evaluated and reported; training runs are what gets trained. `ce-std` and
`ce-rail` share one training objective and differ only in their inference readout, so both are
evaluated from a single `ce` training run.

At the end of a study, per-run Pearson correlations are combined across seeds using Fisher's
Z-transformation: average in `arctanh` space, compute the Student-t 95% confidence interval in
`arctanh` space, and transform both the mean and CI bounds back to PCC space with `tanh`.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .engine import RESUME_META
from .reporting import BASELINES, REPORT_FIELDS, Baseline, read_metrics_csv
from .tokens import UNTRAINED_FIELDS

#: Canonical order of evaluated (reported) modes.
MODES: tuple[str, ...] = ("heads", "ce-std", "ce-rail", "raft-rail", "veil-rail")

#: Canonical round-robin order of training runs, and the modes each one is evaluated as.
TRAIN_MODES: dict[str, tuple[str, ...]] = {
    "heads": ("heads",),
    "ce": ("ce-std", "ce-rail"),
    "raft-rail": ("raft-rail",),
    "veil-rail": ("veil-rail",),
}

#: CLI flag passed to `train.py` for each training mode. `ce` trains under `--ce-std`; the
#: objective is identical to `--ce-rail`, only the readout used for validation logging differs.
MODE_TO_FLAG: dict[str, tuple[str, ...]] = {
    "heads": (),
    "ce": ("--ce-std",),
    "raft-rail": ("--raft-rail",),
    "veil-rail": ("--veil-rail",),
}

#: `evaluate.py --eval-variant` for each evaluated mode that needs one.
EVAL_VARIANT: dict[str, str] = {"ce-std": "ce_std", "ce-rail": "ce_rail"}

STUDY_STATE_FILE = "study_state.json"

#: Default applicable score types (the 8 trained/reported fields; completeness is untrained).
APPLICABLE_FIELDS: tuple[str, ...] = tuple(
    f"{level}.{field}"
    for level, field in REPORT_FIELDS
    if (level, field) not in UNTRAINED_FIELDS
)

_PREFIX_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_prefix(prefix: str) -> str:
    """Ensure the batch prefix is a safe, single path component."""
    cleaned = prefix.strip()
    if not cleaned:
        raise SystemExit("--prefix is required and must be non-empty")
    if cleaned in (".", "..") or not _PREFIX_PATTERN.match(cleaned):
        raise SystemExit(
            f"invalid --prefix {prefix!r}: use only letters, digits, '.', '_' or '-' "
            "(no spaces or path separators)"
        )
    return cleaned


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated list of integer seeds, preserving order."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--seeds must contain at least one integer seed (e.g. --seeds 42,43,44)")
    seeds: list[int] = []
    for part in parts:
        try:
            seeds.append(int(part))
        except ValueError:
            raise SystemExit(f"invalid seed {part!r} in --seeds {raw!r}; expected integers") from None
    return seeds


def resolve_modes(excluded: set[str]) -> list[str]:
    """Return active modes in canonical round-robin order after removing exclusions."""
    unknown = excluded - set(MODES)
    if unknown:
        raise SystemExit(f"unknown excluded mode(s): {', '.join(sorted(unknown))}")
    active = [mode for mode in MODES if mode not in excluded]
    if not active:
        raise SystemExit("all modes were excluded; at least one mode must remain enabled")
    return active


def make_run_id(prefix: str, mode: str, round_num: int, seed: int) -> str:
    """Build `<prefix>_<mode>_<round>_<seed>` (1-based round)."""
    return f"{prefix}_{mode}_{round_num}_{seed}"


def new_eval_entry(mode: str) -> dict:
    return {
        "mode": mode,
        "status": "pending",
        "csv_path": None,
        "json_path": None,
        "last_error": None,
    }


def build_schedule(prefix: str, modes: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    """Round-robin schedule: for round i (1..N) with seed[i], train each needed run in order.

    A training run is scheduled when at least one of its evaluated modes is active, and carries
    one evaluation entry per active mode it serves.
    """
    active = set(modes)
    runs: list[dict] = []
    for round_num, seed in enumerate(seeds, start=1):
        for train_mode, eval_modes in TRAIN_MODES.items():
            evals = [m for m in eval_modes if m in active]
            if not evals:
                continue
            runs.append(
                {
                    "run_id": make_run_id(prefix, train_mode, round_num, seed),
                    "mode": train_mode,
                    "round": round_num,
                    "seed": seed,
                    "status": "pending",
                    "resumable": False,
                    "last_error": None,
                    "evals": [new_eval_entry(m) for m in evals],
                }
            )
    return runs


def new_study_state(
    *,
    prefix: str,
    seeds: Sequence[int],
    modes: Sequence[str],
    excluded_modes: Sequence[str],
    split: str,
    eval_batch_size: int,
    eval_limit: int,
    no_bootstrap: bool,
    snap: bool,
    extra_train_args: Sequence[str],
) -> dict:
    """Initial study resume metadata payload."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "prefix": prefix,
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "paused_at": None,
        "abandoned_at": None,
        "completed_at": None,
        "seeds": list(seeds),
        "modes": list(modes),
        "excluded_modes": list(excluded_modes),
        "split": split,
        "eval_batch_size": eval_batch_size,
        "eval_limit": eval_limit,
        "no_bootstrap": no_bootstrap,
        "snap": snap,
        "extra_train_args": list(extra_train_args),
        "current_run_id": None,
        "runs": build_schedule(prefix, modes, seeds),
        "summary_csv": None,
        "summary_txt": None,
    }


def save_study_state(study_dir: Path, state: dict) -> Path:
    """Atomically persist `study_state.json` so a crash mid-write cannot corrupt progress."""
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    target = study_dir / STUDY_STATE_FILE
    tmp = study_dir / f".{STUDY_STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_study_state(study_dir: Path) -> dict:
    """Load `study_state.json` from a batch study directory."""
    path = Path(study_dir) / STUDY_STATE_FILE
    if not path.exists():
        raise SystemExit(
            f"no batch study state found at {path} — check the --prefix and try again"
        )
    state = json.loads(path.read_text(encoding="utf-8"))
    if any("evals" not in entry for entry in state.get("runs", [])):
        raise SystemExit(
            f"{path} uses the old one-run-per-mode schema (before ce-std and ce-rail shared a "
            "training run); migrate it before resuming"
        )
    return state


def find_paused_studies(batch_root: Path) -> list[dict]:
    """Find every batch study under `batch_root` that is in progress, paused, or has a paused run."""
    root = Path(batch_root)
    if not root.exists():
        return []
    found: list[dict] = []
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        state_path = study_dir / STUDY_STATE_FILE
        paused_run_ids = sorted(
            p.parent.name for p in (study_dir / "runs").glob(f"*/{RESUME_META}")
        ) if (study_dir / "runs").exists() else []

        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {"prefix": study_dir.name, "status": "corrupted"}
            status = state.get("status", "unknown")
            if status == "abandoned":
                continue
            if status != "completed" or paused_run_ids:
                found.append(
                    {
                        "prefix": state.get("prefix", study_dir.name),
                        "study_dir": study_dir,
                        "status": status,
                        "current_run_id": state.get("current_run_id"),
                        "paused_runs": paused_run_ids,
                    }
                )
        elif paused_run_ids:
            found.append(
                {
                    "prefix": study_dir.name,
                    "study_dir": study_dir,
                    "status": "paused",
                    "current_run_id": paused_run_ids[0],
                    "paused_runs": paused_run_ids,
                }
            )
    return found


# --- Fisher Z-transformation and summary aggregation ----------------------------------


@dataclass(frozen=True, slots=True)
class FisherSummary:
    """Combined PCC statistics across multiple runs using Fisher's Z-transformation."""

    n_total: int
    n_valid: int
    mean_pcc: float
    stddev_pcc: float
    ci95_lo: float
    ci95_hi: float


def fisher_z_summary(values: Sequence[float | None]) -> FisherSummary:
    """Combine Pearson correlations via Fisher's Z-transformation.

    - Mean PCC: `tanh(mean(arctanh(r_i)))`.
    - 95% CI: Student-t interval in Fisher-z space (`mean_z +- t_{0.975, k-1} * sd_z / sqrt(k)`),
      transformed back into PCC space with `tanh`.
    - Stddev: sample standard deviation (`ddof=1`) of the finite run PCC values.

    Non-finite entries (NaN from a collapsed field) are excluded from the valid count rather
    than replaced with zero. When fewer than 2 valid runs exist, stddev and CI 95 are NaN.
    """
    from scipy.stats import t as student_t

    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n_total = len(values)
    n_valid = len(finite)
    nan = float("nan")
    if n_valid == 0:
        return FisherSummary(n_total, 0, nan, nan, nan, nan)
    if n_valid == 1:
        return FisherSummary(n_total, 1, finite[0], nan, nan, nan)

    arr = np.asarray(finite, dtype=np.float64)
    # Clamp strictly inside (-1, 1) only if a degenerate correlation hits +-1.0 exactly.
    eps = 1e-15
    clamped = np.clip(arr, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(clamped)
    z_mean = float(np.mean(z))
    z_std = float(np.std(z, ddof=1))
    mean_pcc = float(np.tanh(z_mean))
    stddev_pcc = float(np.std(arr, ddof=1))
    se_z = z_std / math.sqrt(n_valid)
    t_crit = float(student_t.ppf(0.975, df=n_valid - 1))
    ci95_lo = float(np.tanh(z_mean - t_crit * se_z))
    ci95_hi = float(np.tanh(z_mean + t_crit * se_z))
    return FisherSummary(
        n_total=n_total,
        n_valid=n_valid,
        mean_pcc=mean_pcc,
        stddev_pcc=stddev_pcc,
        ci95_lo=ci95_lo,
        ci95_hi=ci95_hi,
    )


def collect_study_pccs(
    study_dir: Path, state: dict
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """Collect per-run PCCs from each run's exported CSV, aligned by (mode, score_type, round)."""
    study_dir = Path(study_dir)
    n_rounds = len(state["seeds"])
    modes: list[str] = list(state["modes"])
    pccs: dict[str, dict[str, list[float]]] = {mode: {} for mode in modes}
    observed_keys: set[str] = set()

    for entry, ev in ((e, ev) for e in state["runs"] for ev in e.get("evals", [])):
        csv_rel = ev.get("csv_path")
        if not csv_rel:
            continue
        csv_path = Path(csv_rel) if Path(csv_rel).is_absolute() else study_dir / csv_rel
        if not csv_path.exists():
            continue
        mode = ev["mode"]
        round_idx = int(entry["round"]) - 1
        for row in read_metrics_csv(csv_path):
            key = row["metric_key"]
            observed_keys.add(key)
            series = pccs.setdefault(mode, {}).setdefault(key, [float("nan")] * n_rounds)
            raw_val = row.get("pearson", "")
            try:
                series[round_idx] = float(raw_val) if raw_val != "" else float("nan")
            except ValueError:
                series[round_idx] = float("nan")

    if observed_keys:
        ordered = [f"{lvl}.{fld}" for lvl, fld in REPORT_FIELDS if f"{lvl}.{fld}" in observed_keys]
        for extra in sorted(observed_keys - set(ordered)):
            ordered.append(extra)
    else:
        ordered = list(APPLICABLE_FIELDS)

    for mode in modes:
        for key in ordered:
            pccs[mode].setdefault(key, [float("nan")] * n_rounds)
    return pccs, ordered


def _csv_stat(value: float | None) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return repr(float(value))


def _csv_run_pcc(value: float | None) -> str:
    if value is None:
        return "N/A"
    val = float(value)
    if math.isnan(val):
        return "nan"
    return repr(val)


def write_study_summary_csv(
    path: Path | str,
    state: dict,
    pccs_by_mode: dict[str, dict[str, list[float]]],
    score_types: Sequence[str],
    baselines: Sequence[Baseline] = BASELINES,
) -> Path:
    """Write the final summary CSV with individual run PCCs, Fisher-Z means, stddevs, and 95% CIs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seeds: list[int] = list(state["seeds"])
    round_cols = [f"round_{i}_seed_{seed}" for i, seed in enumerate(seeds, start=1)]
    fieldnames = [
        "model",
        "source",
        "score_type",
        "n_runs",
        *round_cols,
        "mean_pcc",
        "stddev_pcc",
        "ci95_lo",
        "ci95_hi",
    ]
    source_label = f"study:{state['prefix']}"

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for mode in state["modes"]:
            for score_type in score_types:
                values = pccs_by_mode.get(mode, {}).get(score_type, [float("nan")] * len(seeds))
                summary = fisher_z_summary(values)
                row = {
                    "model": mode,
                    "source": source_label,
                    "score_type": score_type,
                    "n_runs": str(summary.n_valid),
                    "mean_pcc": _csv_stat(summary.mean_pcc),
                    "stddev_pcc": _csv_stat(summary.stddev_pcc),
                    "ci95_lo": _csv_stat(summary.ci95_lo),
                    "ci95_hi": _csv_stat(summary.ci95_hi),
                }
                for col, val in zip(round_cols, values):
                    row[col] = _csv_run_pcc(val)
                writer.writerow(row)

        for baseline in baselines:
            for score_type in score_types:
                val = baseline.pearson.get(score_type)
                row = {
                    "model": baseline.name,
                    "source": baseline.source,
                    "score_type": score_type,
                    "n_runs": "N/A",
                    "mean_pcc": _csv_stat(val),
                    "stddev_pcc": "N/A",
                    "ci95_lo": "N/A",
                    "ci95_hi": "N/A",
                }
                for col in round_cols:
                    row[col] = "N/A"
                writer.writerow(row)
    return target


def _fmt_cell(value: float | None, width: int = 8, na: str = "N/A") -> str:
    if value is None or math.isnan(float(value)):
        return f"{na:>{width}}"
    return f"{float(value):>{width}.3f}"


def _short_col(score_type: str) -> str:
    level, _, field = score_type.partition(".")
    if not field:
        return score_type[:8]
    return f"{level[0]}.{field[:5]}"


def render_study_summary(
    state: dict,
    pccs_by_mode: dict[str, dict[str, list[float]]],
    score_types: Sequence[str],
    baselines: Sequence[Baseline] = BASELINES,
) -> str:
    """Render a human-readable summary report matching tutokana's table conventions."""
    seeds: list[int] = list(state["seeds"])
    modes: list[str] = list(state["modes"])
    excluded: list[str] = list(state.get("excluded_modes", []))
    lines: list[str] = [
        f"=== Batch Study Summary: {state['prefix']} ===",
        f"modes:     {', '.join(modes)}"
        + (f" (excluded: {', '.join(excluded)})" if excluded else ""),
        f"seeds:     {', '.join(str(s) for s in seeds)} ({len(seeds)} round(s))",
        f"split:     {state.get('split', 'test')}",
        "combining: Fisher Z-transformation — mean = tanh(mean(arctanh(r))), "
        "95% CI = tanh(mean_z +- t_{0.975, n-1} * sd_z / sqrt(n)), "
        "stddev = sample SD (ddof=1) of run PCCs",
        "",
    ]

    # Section 1: compact overview table (models x score_types), matching README / render_baselines
    col_labels = [_short_col(st) for st in score_types]
    overview_header = f"{'model':<22}" + "".join(f"{c:>10}" for c in col_labels)
    lines.append("=== Mean PCC Overview (Fisher-Z combined vs published) ===")
    lines.append(overview_header)
    lines.append("-" * len(overview_header))
    for mode in modes:
        row = f"{mode:<22}"
        for st in score_types:
            summary = fisher_z_summary(pccs_by_mode.get(mode, {}).get(st, []))
            row += _fmt_cell(summary.mean_pcc, width=10)
        lines.append(row)
    for baseline in baselines:
        row = f"{baseline.name:<22}"
        for st in score_types:
            row += _fmt_cell(baseline.pearson.get(st), width=10)
        lines.append(row)
    lines.append("")

    # Section 2: full detailed table by model architecture with individual PCCs, mean, sd, 95% CI
    run_headers = [f"r{i}({seed})" for i, seed in enumerate(seeds, start=1)]
    run_col_w = max(9, max((len(h) + 1 for h in run_headers), default=9))
    detail_header = (
        f"{'model':<20}{'score_type':<22}{'n':>4}"
        + "".join(f"{h:>{run_col_w}}" for h in run_headers)
        + f"{'mean':>9}{'stddev':>9}{'CI95 lo':>9}{'CI95 hi':>9}"
    )
    lines.append("=== Detailed PCC Summary by Model Architecture ===")
    lines.append(detail_header)
    lines.append("-" * len(detail_header))

    for mode in modes:
        for st in score_types:
            values = pccs_by_mode.get(mode, {}).get(st, [float("nan")] * len(seeds))
            summary = fisher_z_summary(values)
            run_cells = "".join(
                f"{'nan':>{run_col_w}}" if math.isnan(v) else f"{v:>{run_col_w}.3f}"
                for v in values
            )
            lines.append(
                f"{mode:<20}{st:<22}{summary.n_valid:>4}"
                f"{run_cells}"
                f"{_fmt_cell(summary.mean_pcc, 9)}"
                f"{_fmt_cell(summary.stddev_pcc, 9)}"
                f"{_fmt_cell(summary.ci95_lo, 9)}"
                f"{_fmt_cell(summary.ci95_hi, 9)}"
            )
        lines.append("-" * len(detail_header))

    for baseline in baselines:
        for st in score_types:
            val = baseline.pearson.get(st)
            run_cells = "".join(f"{'N/A':>{run_col_w}}" for _ in seeds)
            lines.append(
                f"{baseline.name:<20}{st:<22}{'N/A':>4}"
                f"{run_cells}"
                f"{_fmt_cell(val, 9)}"
                f"{'N/A':>9}{'N/A':>9}{'N/A':>9}"
            )
        lines.append("-" * len(detail_header))

    lines.append("")
    lines.append("sources: " + "; ".join(f"{b.name} = {b.source}" for b in baselines))
    return "\n".join(lines) + "\n"


def write_study_summaries(
    study_dir: Path,
    state: dict,
    baselines: Sequence[Baseline] = BASELINES,
) -> tuple[Path, Path]:
    """Compute Fisher-Z summaries from all completed run CSVs and write CSV + text reports."""
    study_dir = Path(study_dir)
    pccs_by_mode, score_types = collect_study_pccs(study_dir, state)
    csv_path = write_study_summary_csv(
        study_dir / "summary.csv", state, pccs_by_mode, score_types, baselines=baselines
    )
    txt_path = study_dir / "summary.txt"
    txt_path.write_text(
        render_study_summary(state, pccs_by_mode, score_types, baselines=baselines),
        encoding="utf-8",
    )
    return csv_path, txt_path
