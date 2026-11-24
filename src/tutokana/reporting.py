"""Result tables. One renderer for console, log file and wandb.

`render_table` is the full diagnostic view; `render_table(lite=True)` reports only the
columns the literature reports (PCC and MSE). Published baselines in `BASELINES` are
transcribed from their sources and are never computed here.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .metrics import FieldMetrics

#: Column order for the report: coarsest last, matching the published tables.
REPORT_FIELDS: tuple[tuple[str, str], ...] = (
    ("phone", "accuracy"),
    ("word", "accuracy"),
    ("word", "stress"),
    ("word", "total"),
    ("utterance", "accuracy"),
    ("utterance", "completeness"),
    ("utterance", "fluency"),
    ("utterance", "prosodic"),
    ("utterance", "total"),
)


@dataclass(frozen=True, slots=True)
class Baseline:
    name: str
    source: str
    pearson: dict[str, float]


#: Pearson correlations as published. HMamba's Table 1 is the field's standard comparison.
BASELINES: tuple[Baseline, ...] = (
    Baseline(
        "GOPT (2022)",
        "arXiv:2205.03432",
        {
            "phone.accuracy": 0.612,
            "word.accuracy": 0.533,
            "word.stress": 0.291,
            "word.total": 0.549,
            "utterance.accuracy": 0.714,
            "utterance.completeness": 0.155,
            "utterance.fluency": 0.753,
            "utterance.prosodic": 0.760,
            "utterance.total": 0.742,
        },
    ),
    Baseline(
        "3MH (2023)",
        "HMamba Table 1",
        {
            "phone.accuracy": 0.693,
            "word.accuracy": 0.682,
            "word.stress": 0.361,
            "word.total": 0.694,
            "utterance.accuracy": 0.782,
            "utterance.completeness": 0.374,
            "utterance.fluency": 0.843,
            "utterance.prosodic": 0.836,
            "utterance.total": 0.811,
        },
    ),
    Baseline(
        "HMamba (2025)",
        "aclanthology 2025.naacl-long.98",
        {
            "phone.accuracy": 0.739,
            "word.accuracy": 0.708,
            "word.stress": 0.366,
            "word.total": 0.718,
            "utterance.accuracy": 0.807,
            "utterance.completeness": 0.278,
            "utterance.fluency": 0.848,
            "utterance.prosodic": 0.843,
            "utterance.total": 0.829,
        },
    ),
    Baseline(
        "Qwen2-Audio LoRA",
        "arXiv:2509.15701",
        {
            "phone.accuracy": 0.38,
            "word.accuracy": 0.51,
            "word.stress": 0.11,
            "word.total": 0.52,
            "utterance.accuracy": 0.69,
            "utterance.fluency": 0.74,
            "utterance.prosodic": 0.73,
            "utterance.total": 0.72,
        },
    ),
)


def _fmt(value: float, spec: str = "6.3f") -> str:
    if value != value:  # NaN
        # ">+9" is not a valid string format, so drop the sign flag before reusing the width.
        return f"{'nan':>{spec.split('.')[0].lstrip('+- ')}}"
    return f"{value:{spec}}"


def render_table(
    results: dict[str, FieldMetrics], title: str = "results", lite: bool = False
) -> str:
    """Per-field results. `lite` drops everything the published tables do not carry."""
    if lite:
        header = f"{'field':<24}{'n':>8}{'PCC':>8}{'MSE':>9}"
    else:
        header = (
            f"{'field':<24}{'n':>8}{'PCC':>8}{'95% CI':>17}{'SCC':>8}"
            f"{'MSE':>8}{'MAE':>7}{'pred mu+-sd':>16}{'gold mu+-sd':>16}{'sd ratio':>10}"
        )
    lines = [f"=== {title} ===", header, "-" * len(header)]
    for level, field in REPORT_FIELDS:
        key = f"{level}.{field}"
        m = results.get(key)
        if m is None:
            continue
        if lite:
            lines.append(
                f"{key:<24}{m.n:>8}{_fmt(m.pearson, '8.3f')}{_fmt(m.mse, '9.3f')}"
            )
            continue
        interval = (
            "        --       "
            if m.pearson_lo != m.pearson_lo
            else f"[{m.pearson_lo:6.3f},{m.pearson_hi:6.3f}]"
        )
        lines.append(
            f"{key:<24}{m.n:>8}{_fmt(m.pearson, '8.3f')}{interval:>17}"
            f"{_fmt(m.spearman, '8.3f')}{_fmt(m.mse, '8.3f')}{_fmt(m.mae, '7.2f')}"
            f"{m.pred_mean:>9.2f}+-{m.pred_std:<5.2f}{m.gold_mean:>9.2f}+-{m.gold_std:<5.2f}"
            f"{_fmt(m.sigma_ratio, '10.2f')}"
        )
    return "\n".join(lines)


def render_baselines(results: dict[str, FieldMetrics]) -> str:
    """This run's correlations next to the published ones, same column order."""
    columns = [f"{level[0]}.{field[:5]}" for level, field in REPORT_FIELDS]
    header = f"{'model':<26}" + "".join(f"{c:>10}" for c in columns)
    lines = ["=== Pearson vs published ===", header, "-" * len(header)]

    row = f"{'THIS RUN':<26}"
    for level, field in REPORT_FIELDS:
        m = results.get(f"{level}.{field}")
        row += f"{'--':>10}" if m is None else _fmt(m.pearson, "10.3f")
    lines.append(row)

    for baseline in BASELINES:
        row = f"{baseline.name:<26}"
        for level, field in REPORT_FIELDS:
            value = baseline.pearson.get(f"{level}.{field}")
            row += f"{'--':>10}" if value is None else f"{value:>10.3f}"
        lines.append(row)

    lines.append("")
    lines.append("sources: " + "; ".join(f"{b.name} = {b.source}" for b in BASELINES))
    return "\n".join(lines)


def render_transcription(metrics) -> str:
    return "\n".join(
        [
            "=== transcription (generative mode) ===",
            f"phone error rate            {metrics.phone_error_rate:7.3%}",
            f"phone error rate, no stress {metrics.phone_error_rate_no_stress:7.3%}",
            f"word exact match            {metrics.word_exact_match:7.3%}",
            f"words / phones              {metrics.n_words} / {metrics.n_phones}",
        ]
    )


def wandb_metrics(results: dict[str, FieldMetrics], prefix: str = "eval") -> dict[str, float]:
    """Flatten the table for wandb. sigma_ratio is logged as the variance-collapse warning."""
    flat: dict[str, float] = {}
    for key, m in results.items():
        flat[f"{prefix}/pcc/{key}"] = m.pearson
        flat[f"{prefix}/scc/{key}"] = m.spearman
        flat[f"{prefix}/mse/{key}"] = m.mse
        flat[f"{prefix}/sigma_ratio/{key}"] = m.sigma_ratio
    return flat


#: Fields where more than this share of labels sit below the maximum have no dominant
#: ceiling, so "detect the imperfect ones" is not a detection problem and F1 is maximised by
#: calling everything positive.
CEILING_MAJORITY = 0.5


def render_detection(results: dict, ceilings: dict[str, float] | None = None) -> str:
    """Detection of labels below the field's maximum, with the achievable SCC alongside.

    On this corpus Spearman is largely a detection metric: perfect ceiling-vs-not detection
    with random ordering below reaches 0.671 of the 0.680 achievable on phone accuracy, while
    perfect ordering without detection reaches 0.077. The ceiling column states how much rank
    correlation was ever available, which is set by how tied the labels are.
    """
    header = (
        f"{'field':<24}{'imperfect':>11}{'base rate':>11}{'AUC':>8}{'F1':>8}"
        f"{'prec':>8}{'recall':>8}{'thresh':>9}"
    )
    if ceilings:
        header += f"{'SCC':>8}{'ceiling':>9}{'of ceiling':>12}"
    lines = ["=== detection: is this label below the maximum? ===", header, "-" * len(header)]
    skipped = []
    for level, field in REPORT_FIELDS:
        key = f"{level}.{field}"
        d = results.get(key)
        if d is None:
            continue
        # The framing only holds where the ceiling is the majority class. Utterance scores
        # run 2-10 with no dominant value, so "below the maximum" is 95-99% of them and F1
        # reaches 0.999 by calling everything positive — a number that means nothing. Those
        # fields have no ceiling problem to begin with: their SCC already tracks their PCC.
        if d.share == d.share and d.share > CEILING_MAJORITY:
            skipped.append(f"{key} ({d.share:.0%})")
            continue
        row = (
            f"{key:<24}{d.positives:>11}{d.share:>10.2%}{_fmt(d.auc, '8.3f')}"
            f"{_fmt(d.f1, '8.3f')}{_fmt(d.precision, '8.3f')}{_fmt(d.recall, '8.3f')}"
            f"{_fmt(d.threshold, '9.3f')}"
        )
        if ceilings and key in ceilings:
            scc, ceiling = ceilings[key]
            share = 100 * scc / ceiling if ceiling and ceiling == ceiling else float("nan")
            row += f"{_fmt(scc, '8.3f')}{_fmt(ceiling, '9.3f')}{_fmt(share, '11.0f')}%"
        lines.append(row)
    if skipped:
        lines.append(
            "  (no dominant ceiling, detection not meaningful: " + ", ".join(skipped) + ")"
        )
    return "\n".join(lines)


def render_snapping(raw: dict[str, FieldMetrics], snapped: dict[str, FieldMetrics]) -> str:
    """What rounding predictions onto the training label grid is worth.

    Gold is quantised and a continuous readout is not, so a rank correlation compares a
    strict ordering against one that is mostly a single tied block. Snapping restores those
    ties and removes error below half a grid step. It is a change of scale applied after the
    fact, not a change of model, and it cannot invent ordering the predictions lacked.
    """
    header = (
        f"{'field':<24}{'PCC raw':>10}{'PCC snap':>10}{'delta':>9}"
        f"{'SCC raw':>10}{'SCC snap':>10}{'delta':>9}"
    )
    lines = ["=== snapped to the training label grid ===", header, "-" * len(header)]
    for level, field in REPORT_FIELDS:
        key = f"{level}.{field}"
        a, b = raw.get(key), snapped.get(key)
        if a is None or b is None:
            continue
        lines.append(
            f"{key:<24}{_fmt(a.pearson, '10.3f')}{_fmt(b.pearson, '10.3f')}"
            f"{_fmt(b.pearson - a.pearson, '+9.3f')}"
            f"{_fmt(a.spearman, '10.3f')}{_fmt(b.spearman, '10.3f')}"
            f"{_fmt(b.spearman - a.spearman, '+9.3f')}"
        )
    return "\n".join(lines)


METRICS_CSV_COLUMNS: tuple[str, ...] = (
    "run_id",
    "mode",
    "seed",
    "split",
    "readout",
    "level",
    "field",
    "metric_key",
    "n",
    "pearson",
    "pearson_lo",
    "pearson_hi",
    "spearman",
    "mse",
    "mae",
    "pred_mean",
    "pred_std",
    "gold_mean",
    "gold_std",
    "sigma_ratio",
    "spearman_ceiling",
    "detect_positives",
    "detect_share",
    "detect_auc",
    "detect_f1",
    "detect_precision",
    "detect_recall",
    "detect_threshold",
    "snapped_pearson",
    "snapped_spearman",
    "snapped_mse",
    "snapped_mae",
)


def _csv_num(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    val = float(value)
    if math.isnan(val):
        return "nan"
    return repr(val)


def write_metrics_csv(
    path: Path | str,
    *,
    run_id: str,
    mode: str,
    seed: int,
    split: str,
    readout: str,
    metrics: dict[str, FieldMetrics],
    detection: dict | None = None,
    spearman_ceilings: dict[str, float] | None = None,
    snapped_metrics: dict[str, FieldMetrics] | None = None,
) -> Path:
    """Export one run's evaluation metrics as a flat CSV for independent statistical analysis."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    detection = detection or {}
    spearman_ceilings = spearman_ceilings or {}
    snapped_metrics = snapped_metrics or {}

    ordered_keys: list[tuple[str, str]] = [
        (level, field)
        for level, field in REPORT_FIELDS
        if f"{level}.{field}" in metrics
    ]
    seen = {f"{level}.{field}" for level, field in ordered_keys}
    for key in sorted(metrics):
        if key not in seen:
            level, _, field = key.partition(".")
            ordered_keys.append((level, field))

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_COLUMNS)
        writer.writeheader()
        for level, field in ordered_keys:
            key = f"{level}.{field}"
            m = metrics[key]
            d = detection.get(key)
            s = snapped_metrics.get(key)
            writer.writerow(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "seed": str(seed),
                    "split": split,
                    "readout": readout,
                    "level": level,
                    "field": field,
                    "metric_key": key,
                    "n": _csv_num(m.n),
                    "pearson": _csv_num(m.pearson),
                    "pearson_lo": _csv_num(m.pearson_lo),
                    "pearson_hi": _csv_num(m.pearson_hi),
                    "spearman": _csv_num(m.spearman),
                    "mse": _csv_num(m.mse),
                    "mae": _csv_num(m.mae),
                    "pred_mean": _csv_num(m.pred_mean),
                    "pred_std": _csv_num(m.pred_std),
                    "gold_mean": _csv_num(m.gold_mean),
                    "gold_std": _csv_num(m.gold_std),
                    "sigma_ratio": _csv_num(m.sigma_ratio),
                    "spearman_ceiling": _csv_num(spearman_ceilings.get(key)),
                    "detect_positives": _csv_num(d.positives if d is not None else None),
                    "detect_share": _csv_num(d.share if d is not None else None),
                    "detect_auc": _csv_num(d.auc if d is not None else None),
                    "detect_f1": _csv_num(d.f1 if d is not None else None),
                    "detect_precision": _csv_num(d.precision if d is not None else None),
                    "detect_recall": _csv_num(d.recall if d is not None else None),
                    "detect_threshold": _csv_num(d.threshold if d is not None else None),
                    "snapped_pearson": _csv_num(s.pearson if s is not None else None),
                    "snapped_spearman": _csv_num(s.spearman if s is not None else None),
                    "snapped_mse": _csv_num(s.mse if s is not None else None),
                    "snapped_mae": _csv_num(s.mae if s is not None else None),
                }
            )
    return target


def read_metrics_csv(path: Path | str) -> list[dict[str, str]]:
    """Read rows from a per-run metrics CSV written by `write_metrics_csv`."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
