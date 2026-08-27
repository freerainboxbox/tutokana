"""One renderer for the results table — console, log file and wandb all read from here.

Published reference rows are carried alongside the measured ones so a run is always read
against the field rather than against itself. The numbers below are transcribed from the
sources named in `BASELINES`; they are context, never anything this code computes.

The bar is HMamba, not the Phi-4-multimodal paper this project started from. That paper
reports no word-level or phoneme-level scores at all — only utterance accuracy/fluency/
prosodic/total plus transcription error rates — and its utterance numbers (0.645 / 0.733 /
0.714 / 0.668 for LoRA at four epochs) are within noise of what the predecessor already
achieved. Its headline claim of state-of-the-art "phoneme-level accuracy" is mislabelled
utterance-level accuracy, and at 0.743 it sits well below HMamba's 0.807 from earlier the
same year.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    Baseline(
        "gemma-4-12b-so762",
        "predecessor, glowing-moon-28",
        {
            "phone.accuracy": 0.358,
            "word.accuracy": 0.371,
            "word.total": 0.381,
            "utterance.accuracy": 0.639,
            "utterance.fluency": 0.724,
            "utterance.prosodic": 0.708,
            "utterance.total": 0.667,
        },
    ),
)


def _fmt(value: float, spec: str = "6.3f") -> str:
    if value != value:  # NaN
        return f"{'nan':>{spec.split('.')[0]}}"
    return f"{value:{spec}}"


def render_table(results: dict[str, FieldMetrics], title: str = "results") -> str:
    """The measured table: correlation with interval, rank correlation, error, dispersion."""
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
    """Flatten the table for wandb. Dispersion is logged too — it is the early warning."""
    flat: dict[str, float] = {}
    for key, m in results.items():
        flat[f"{prefix}/pcc/{key}"] = m.pearson
        flat[f"{prefix}/scc/{key}"] = m.spearman
        flat[f"{prefix}/mse/{key}"] = m.mse
        flat[f"{prefix}/sigma_ratio/{key}"] = m.sigma_ratio
    return flat


#: Above this share of labels below the maximum, "detect the imperfect ones" stops being a
#: detection problem — everything is imperfect, and F1 is maximised by saying so.
CEILING_MAJORITY = 0.5


def render_detection(results: dict, ceilings: dict[str, float] | None = None) -> str:
    """Detection of imperfect labels, plus how much of the achievable SCC is being reached.

    Spearman on this corpus is a detection metric wearing a rank correlation's clothes: on
    the test split, perfect ceiling-vs-not detection with random ordering below reaches 0.671
    of the 0.680 achievable on phone accuracy, and perfect ordering without detection reaches
    0.077. Reporting detection directly saves inferring it from SCC, and the ceiling column
    says how much of the rank correlation was ever available — word accuracy caps at 0.528
    for any continuous predictor simply because 90% of its labels are tied at 10.
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
