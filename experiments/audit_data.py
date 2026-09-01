"""Reproduce the label-distribution facts the design rests on, from the live dataset.

Run this before anything trains. Every number the architecture was chosen against is
printed here from the dataset as it exists today, so a silent upstream revision shows up in
one minute rather than as an unexplained result three days later.

    uv run python experiments/audit_data.py [--split train] [--json out.json]

Expected on the train split, as measured 2026-08-26:
  phone accuracy    11 distinct values in 0.2 steps, 80.1% at 2.0   (NOT the {0,1,2} rubric)
  word accuracy     88.0% at 10, and the value 4 never occurs
  word stress       99.0% at 10 released, but only 80.0% a clean sweep of the 5 experts
                    panel concentration nu = 8.6, the value the stress head is built with
  completeness      99.6% at 10.0, sigma 0.11                       (measured, never trained)
  audio             mean 4.1 s, p95 8.1 s, max 20.4 s -> ~100 audio tokens median
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tutokana.data import STRESS_RATERS, SAMPLE_RATE, load_split, phone_vocabulary


def stress_panel(votes: Counter, n_raters: int) -> dict:
    """Moment-match the per-word agreement rate p behind the panel's vote counts.

    p is never observed — only k, how many of the `n_raters` experts called the stress
    correct. Its spread is still recoverable, because the variance of k splits into two
    parts that do not overlap:

        Var(k) = n*p_bar*(1 - p_bar)  +  n*(n - 1)*Var(p)
                 \________________/      \_____________/
                  the panel disagreeing    words genuinely
                  even at a fixed rate     differing in p

    The first term is what five people would produce on their own, and it is a closed form.
    Whatever variance is left over is the second term. Solving for Var(p) and matching it to
    a Beta, whose variance is m*(1 - m)/(nu + 1), gives the concentration in one step.
    `config.stress_concentration` is this number, held fixed rather than learned.
    """
    k = np.repeat(
        np.array(sorted(votes), dtype=float), [votes[v] for v in sorted(votes)]
    )
    p_bar = float(k.mean()) / n_raters
    independent = n_raters * p_bar * (1.0 - p_bar)
    var_p = (float(k.var()) - independent) / (n_raters * (n_raters - 1))
    concentration = p_bar * (1.0 - p_bar) / var_p - 1.0
    return {
        "p_bar": p_bar,
        "var_k": float(k.var()),
        "var_k_if_p_were_constant": independent,
        "var_p": var_p,
        "concentration": concentration,
        # How much wider the counts spread than independent experts would manage, and the
        # panel size that spread is really worth.
        "design_effect": float(k.var()) / independent,
        "effective_raters": n_raters * independent / float(k.var()),
    }


def audit(split: str, limit: int | None) -> dict:
    utterances = load_split(split, limit=limit)

    phone_scores = Counter()
    word_accuracy = Counter()
    word_stress = Counter()
    stress_votes = Counter()
    for u in utterances:
        for w in u.words:
            word_accuracy[w.accuracy] += 1
            word_stress[w.stress] += 1
            stress_votes[w.stress_votes] += 1
            phone_scores.update(w.phone_accuracy)

    utterance_fields = {}
    for name in ("accuracy", "completeness", "fluency", "prosodic", "total"):
        values = np.array([u.utterance_targets()[name] for u in utterances], dtype=float)
        utterance_fields[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "share_at_max": float((values == values.max()).mean()),
        }

    durations = np.array([u.duration_s for u in utterances])
    audio_tokens = np.ceil(durations * SAMPLE_RATE / 640)
    n_phones = sum(phone_scores.values())

    return {
        "split": split,
        "utterances": len(utterances),
        "speakers": len({u.speaker for u in utterances}),
        "words": sum(word_accuracy.values()),
        "phones": n_phones,
        "phone_symbols": len(phone_vocabulary(utterances)),
        "utterance_fields": utterance_fields,
        "phone_accuracy": {
            "distinct_values": len(phone_scores),
            "share_at_2": phone_scores[2.0] / n_phones,
            "histogram": {str(k): v for k, v in sorted(phone_scores.items())},
        },
        "word_accuracy": {
            "share_at_10": word_accuracy[10.0] / max(sum(word_accuracy.values()), 1),
            "histogram": {str(k): v for k, v in sorted(word_accuracy.items())},
        },
        "word_stress": {
            "share_correct": word_stress[10.0] / max(sum(word_stress.values()), 1),
            "histogram": {str(k): v for k, v in sorted(word_stress.items())},
            # The released label is the median of five experts, so it hides every split
            # verdict. The vote histogram is the target the stress head is trained on.
            "share_unanimous": stress_votes[STRESS_RATERS]
            / max(sum(stress_votes.values()), 1),
            "vote_histogram": {str(k): v for k, v in sorted(stress_votes.items())},
            "panel": stress_panel(stress_votes, STRESS_RATERS),
        },
        "audio": {
            "mean_s": float(durations.mean()),
            "p50_s": float(np.percentile(durations, 50)),
            "p95_s": float(np.percentile(durations, 95)),
            "max_s": float(durations.max()),
            "median_audio_tokens": float(np.median(audio_tokens)),
            "max_audio_tokens": float(audio_tokens.max()),
        },
    }


def report(result: dict) -> str:
    utterance = result["utterance_fields"]
    audio = result["audio"]
    panel = result["word_stress"]["panel"]
    lines = [
        f"=== speechocean762 / {result['split']} ===",
        f"{result['utterances']} utterances, {result['speakers']} speakers, "
        f"{result['words']} words, {result['phones']} phones, "
        f"{result['phone_symbols']} phone symbols",
        "",
        f"{'utterance field':<16}{'mean':>8}{'sd':>8}{'min':>7}{'max':>7}{'@max':>8}",
    ]
    for name, stats in utterance.items():
        lines.append(
            f"{name:<16}{stats['mean']:>8.2f}{stats['std']:>8.2f}"
            f"{stats['min']:>7.1f}{stats['max']:>7.1f}{stats['share_at_max']:>8.1%}"
        )
    phone = result["phone_accuracy"]
    lines += [
        "",
        f"phone accuracy: {phone['distinct_values']} distinct values, "
        f"{phone['share_at_2']:.1%} at 2.0"
        + ("  <-- continuous, NOT {0,1,2}" if phone["distinct_values"] > 3 else ""),
        f"  {phone['histogram']}",
        f"word accuracy:  {result['word_accuracy']['share_at_10']:.1%} at 10",
        f"word stress:    {result['word_stress']['share_correct']:.1%} released correct, "
        f"{result['word_stress']['share_unanimous']:.1%} unanimous across the 5 experts",
        f"  votes {result['word_stress']['vote_histogram']}",
        f"  panel: p {panel['p_bar']:.4f}, Var(k) {panel['var_k']:.4f} against "
        f"{panel['var_k_if_p_were_constant']:.4f} at a fixed rate",
        f"         concentration nu {panel['concentration']:.2f}, "
        f"{panel['effective_raters']:.2f} of {STRESS_RATERS} experts' worth of "
        f"independent information",
        "",
        f"audio: mean {audio['mean_s']:.2f}s  p50 {audio['p50_s']:.2f}s  "
        f"p95 {audio['p95_s']:.2f}s  max {audio['max_s']:.2f}s",
        f"       {audio['median_audio_tokens']:.0f} audio tokens median, "
        f"{audio['max_audio_tokens']:.0f} max (budget 750)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", default="", help="also write the raw numbers here")
    args = parser.parse_args()

    result = audit(args.split, args.limit or None)
    print(report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\n[out] {args.json}")


if __name__ == "__main__":
    main()
