"""Tests for batch_study.py, Fisher Z aggregation, per-run CSV export, and pause/resume."""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

import batch_study
from tutokana.metrics import DetectionMetrics, FieldMetrics
from tutokana.reporting import BASELINES, METRICS_CSV_COLUMNS, read_metrics_csv, write_metrics_csv
from tutokana.study import (
    APPLICABLE_FIELDS,
    MODES,
    STUDY_STATE_FILE,
    build_schedule,
    collect_study_pccs,
    find_paused_studies,
    fisher_z_summary,
    load_study_state,
    make_run_id,
    new_study_state,
    parse_seeds,
    render_study_summary,
    resolve_modes,
    save_study_state,
    validate_prefix,
    write_study_summaries,
)


# --- helpers --------------------------------------------------------------------------


def _sample_field_metrics(pearson: float, n: int = 100) -> FieldMetrics:
    return FieldMetrics(
        n=n,
        pearson=pearson,
        pearson_lo=pearson - 0.02 if math.isfinite(pearson) else float("nan"),
        pearson_hi=pearson + 0.02 if math.isfinite(pearson) else float("nan"),
        spearman=pearson - 0.05 if math.isfinite(pearson) else float("nan"),
        mse=0.12,
        mae=0.25,
        pred_mean=8.0,
        pred_std=1.2,
        gold_mean=8.1,
        gold_std=1.5,
    )


def _write_fake_scripts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create lightweight stub train.py and evaluate.py scripts that mirror the real protocols.

    The stub train script:
    - Records every invocation to a shared trace file.
    - Handles `--resume` and `--run-dir` identically to `train.py`.
    - When `FAKE_PAUSE_RUN_ID` matches the current run_id (and it has not resumed yet), signals
      readiness via a marker file and waits for SIGINT via `graceful_interrupt`, saving
      `resume.json`, `resume_state.pt`, and `config.json` with `"interrupted": true`, then
      exiting 130.
    - Otherwise writes `config.json` (`"interrupted": false`), `heads.safetensors`, removes
      `resume.json` if present, and exits 0.

    The stub evaluate script:
    - Records every invocation to the shared trace file.
    - When `FAKE_PAUSE_EVAL_RUN_ID` matches the current run_id (and `FAKE_PAUSE_EVAL_MODE`, if
      set, matches the evaluated mode) on its first pause, signals readiness and waits for
      SIGINT, writing a partial JSON and exiting 130.
    - Honours `--eval-variant`, reporting the run as that mode.
    - Otherwise writes deterministic per-run CSV and JSON files via `write_metrics_csv`.
    """
    trace_file = tmp_path / "trace.jsonl"
    fake_train = tmp_path / "fake_train.py"
    fake_eval = tmp_path / "fake_eval.py"
    repo_src = str(Path(__file__).resolve().parents[1] / "src")

    fake_train.write_text(
        f"""
import argparse, json, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, {repo_src!r})
from tutokana.engine import RESUME_META, RESUME_STATE, graceful_interrupt

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", default="")
parser.add_argument("--run-dir", default="")
parser.add_argument("--resume", default="")
parser.add_argument("--train-seed", type=int, default=3407)
parser.add_argument("--ce-std", action="store_true")
parser.add_argument("--ce-rail", action="store_true")
parser.add_argument("--raft-rail", action="store_true")
parser.add_argument("--min-battery-soc", type=int, default=0)
args, extra = parser.parse_known_args()

run_id = args.resume or args.run_id
run_dir = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
mode = "heads"
if args.ce_std:
    mode = "ce-std"
elif args.ce_rail:
    mode = "ce-rail"
elif args.raft_rail:
    mode = "raft-rail"
elif args.resume and (run_dir / RESUME_META).exists():
    saved = json.loads((run_dir / RESUME_META).read_text())
    mode = saved["mode"]
    args.train_seed = saved["seed"]

with open({str(trace_file)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{
        "cmd": "train", "run_id": run_id, "run_dir": str(run_dir),
        "resumed": bool(args.resume), "mode": mode, "seed": args.train_seed,
        "min_battery_soc": args.min_battery_soc, "extra": extra,
    }}) + "\\n")

battery_pause_target = os.environ.get("FAKE_BATTERY_PAUSE_RUN_ID", "")
if battery_pause_target == run_id and not args.resume:
    from tutokana.power import BATTERY_EXIT_CODE
    (run_dir / RESUME_META).write_text(json.dumps({{
        "run_id": run_id, "mode": mode, "seed": args.train_seed, "step": 10
    }}))
    (run_dir / RESUME_STATE).write_bytes(b"state")
    (run_dir / "config.json").write_text(json.dumps({{
        "run_id": run_id, "interrupted": True, "mode": mode, "seed": args.train_seed
    }}))
    sys.exit(BATTERY_EXIT_CODE)

pause_target = os.environ.get("FAKE_PAUSE_RUN_ID", "")
ready_file = os.environ.get("FAKE_READY_FILE", "")
if pause_target == run_id and not args.resume:
    with graceful_interrupt() as interrupt:
        if ready_file:
            Path(ready_file).write_text("ready")
        deadline = time.time() + 10.0
        while not interrupt["requested"] and time.time() < deadline:
            time.sleep(0.02)
        if interrupt["requested"]:
            (run_dir / RESUME_META).write_text(json.dumps({{
                "run_id": run_id, "mode": mode, "seed": args.train_seed, "step": 10
            }}))
            (run_dir / RESUME_STATE).write_bytes(b"state")
            (run_dir / "config.json").write_text(json.dumps({{
                "run_id": run_id, "interrupted": True, "mode": mode, "seed": args.train_seed
            }}))
            sys.exit(130)

(run_dir / RESUME_META).unlink(missing_ok=True)
(run_dir / RESUME_STATE).unlink(missing_ok=True)
(run_dir / "heads.safetensors").write_bytes(b"weights")
(run_dir / "config.json").write_text(json.dumps({{
    "run_id": run_id, "interrupted": False, "mode": mode, "seed": args.train_seed
}}))
""",
        encoding="utf-8",
    )

    fake_eval.write_text(
        f"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, {repo_src!r})
from tutokana.engine import graceful_interrupt
from tutokana.metrics import FieldMetrics
from tutokana.reporting import write_metrics_csv
from tutokana.study import APPLICABLE_FIELDS

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--split", default="test")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--no-bootstrap", action="store_true")
parser.add_argument("--snap", action="store_true")
parser.add_argument("--out", default="")
parser.add_argument("--out-csv", default="")
parser.add_argument("--eval-variant", default="")
args = parser.parse_args()

run_dir = Path(args.run_dir)
meta = json.loads((run_dir / "config.json").read_text())
run_id = meta["run_id"]
mode = args.eval_variant.replace("_", "-") if args.eval_variant else meta["mode"]
seed = int(meta["seed"])

with open({str(trace_file)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{
        "cmd": "eval", "run_id": run_id, "run_dir": str(run_dir),
        "split": args.split, "out_csv": args.out_csv, "mode": mode,
    }}) + "\\n")

pause_eval = os.environ.get("FAKE_PAUSE_EVAL_RUN_ID", "")
pause_eval_mode = os.environ.get("FAKE_PAUSE_EVAL_MODE", "")
ready_file = os.environ.get("FAKE_EVAL_READY_FILE", "")
paused_marker = run_dir / ".eval_paused_once"
if (
    pause_eval == run_id
    and pause_eval_mode in ("", mode)
    and not paused_marker.exists()
):
    with graceful_interrupt() as interrupt:
        if ready_file:
            Path(ready_file).write_text("ready")
        deadline = time.time() + 10.0
        while not interrupt["requested"] and time.time() < deadline:
            time.sleep(0.02)
        if interrupt["requested"]:
            paused_marker.write_text("1")
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps({{"run_id": run_id, "partial": True}}))
            sys.exit(130)

mode_offset = {{"heads": 0.72, "ce-std": 0.45, "ce-rail": 0.58, "raft-rail": 0.70}}[mode]
seed_delta = ((seed % 10) - 2) * 0.005
metrics = {{}}
for i, key in enumerate(APPLICABLE_FIELDS):
    pcc = min(0.95, max(0.05, mode_offset + seed_delta - i * 0.01))
    metrics[key] = FieldMetrics(
        n=100, pearson=pcc, pearson_lo=pcc - 0.02, pearson_hi=pcc + 0.02,
        spearman=pcc - 0.05, mse=0.2, mae=0.3,
        pred_mean=8.0, pred_std=1.0, gold_mean=8.0, gold_std=1.2,
    )

if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({{"run_id": run_id, "partial": False}}))
if args.out_csv:
    write_metrics_csv(
        args.out_csv,
        run_id=run_id,
        mode=mode,
        seed=seed,
        split=args.split,
        readout="heads" if mode == "heads" else ("mode" if mode == "ce-std" else "expectation"),
        metrics=metrics,
    )
""",
        encoding="utf-8",
    )
    return fake_train, fake_eval, trace_file


# --- Fisher Z-transformation tests ----------------------------------------------------


def test_fisher_z_summary_matches_exact_formula():
    values = [0.730, 0.735, 0.740, 0.745, 0.750]
    summary = fisher_z_summary(values)

    z = np.arctanh(values)
    z_mean = float(np.mean(z))
    z_sd = float(np.std(z, ddof=1))
    t_crit = float(student_t.ppf(0.975, df=len(values) - 1))
    expected_mean = math.tanh(z_mean)
    expected_lo = math.tanh(z_mean - t_crit * z_sd / math.sqrt(len(values)))
    expected_hi = math.tanh(z_mean + t_crit * z_sd / math.sqrt(len(values)))
    expected_sd = float(np.std(values, ddof=1))

    assert summary.n_total == 5
    assert summary.n_valid == 5
    assert summary.mean_pcc == pytest.approx(expected_mean, abs=1e-12)
    assert summary.stddev_pcc == pytest.approx(expected_sd, abs=1e-12)
    assert summary.ci95_lo == pytest.approx(expected_lo, abs=1e-12)
    assert summary.ci95_hi == pytest.approx(expected_hi, abs=1e-12)
    # Fisher Z weights larger correlations more strongly than arithmetic mean.
    assert summary.mean_pcc > float(np.mean(values))
    assert summary.ci95_lo < summary.mean_pcc < summary.ci95_hi


def test_fisher_z_handles_nan_single_and_empty():
    single = fisher_z_summary([0.65, float("nan")])
    assert single.n_total == 2 and single.n_valid == 1
    assert single.mean_pcc == pytest.approx(0.65)
    assert math.isnan(single.stddev_pcc)
    assert math.isnan(single.ci95_lo)
    assert math.isnan(single.ci95_hi)

    empty = fisher_z_summary([float("nan"), float("nan")])
    assert empty.n_valid == 0
    assert math.isnan(empty.mean_pcc)


def test_fisher_z_handles_identical_values():
    same = fisher_z_summary([0.74, 0.74, 0.74])
    assert same.mean_pcc == pytest.approx(0.74, abs=1e-12)
    assert same.stddev_pcc == pytest.approx(0.0, abs=1e-12)
    assert same.ci95_lo == pytest.approx(0.74, abs=1e-12)
    assert same.ci95_hi == pytest.approx(0.74, abs=1e-12)


# --- per-run CSV export tests ---------------------------------------------------------


def test_metrics_csv_round_trips_all_columns(tmp_path):
    metrics = {
        "phone.accuracy": _sample_field_metrics(0.741, n=47369),
        "word.stress": _sample_field_metrics(float("nan"), n=15967),
    }
    detection = {
        "phone.accuracy": DetectionMetrics(
            positives=8850, share=0.187, auc=0.91, f1=0.72,
            precision=0.75, recall=0.69, threshold=1.85,
        )
    }
    ceilings = {"phone.accuracy": 0.680, "word.stress": 0.095}
    snapped = {"phone.accuracy": _sample_field_metrics(0.739, n=47369)}

    csv_path = write_metrics_csv(
        tmp_path / "run.csv",
        run_id="alpha_heads_1_42",
        mode="heads",
        seed=42,
        split="test",
        readout="heads",
        metrics=metrics,
        detection=detection,
        spearman_ceilings=ceilings,
        snapped_metrics=snapped,
    )
    rows = read_metrics_csv(csv_path)
    assert tuple(rows[0].keys()) == METRICS_CSV_COLUMNS
    assert len(rows) == 2
    assert rows[0]["run_id"] == "alpha_heads_1_42"
    assert rows[0]["mode"] == "heads"
    assert rows[0]["seed"] == "42"
    assert rows[0]["metric_key"] == "phone.accuracy"
    assert float(rows[0]["pearson"]) == pytest.approx(0.741)
    assert float(rows[0]["detect_auc"]) == pytest.approx(0.91)
    assert float(rows[0]["snapped_pearson"]) == pytest.approx(0.739)
    assert rows[1]["metric_key"] == "word.stress"
    assert math.isnan(float(rows[1]["pearson"]))
    assert rows[1]["detect_auc"] == ""
    assert rows[1]["snapped_pearson"] == ""


# --- scheduling, naming, and CLI validation tests -------------------------------------


def test_schedule_is_round_robin_with_seed_depending_only_on_round():
    modes = resolve_modes(set())
    assert modes == ["heads", "ce-std", "ce-rail", "raft-rail"]
    schedule = build_schedule("alpha", modes, [42, 43])
    assert [r["run_id"] for r in schedule] == [
        "alpha_heads_1_42",
        "alpha_ce_1_42",
        "alpha_raft-rail_1_42",
        "alpha_heads_2_43",
        "alpha_ce_2_43",
        "alpha_raft-rail_2_43",
    ]
    # ce-std and ce-rail share one training run and are evaluated from it separately.
    assert [[ev["mode"] for ev in r["evals"]] for r in schedule[:3]] == [
        ["heads"], ["ce-std", "ce-rail"], ["raft-rail"],
    ]
    assert make_run_id("alpha", "raft-rail", 1, 42) == "alpha_raft-rail_1_42"


def test_schedule_drops_excluded_evals_and_runs_with_none_left():
    schedule = build_schedule("alpha", resolve_modes({"ce-std"}), [42])
    assert [(r["run_id"], [ev["mode"] for ev in r["evals"]]) for r in schedule] == [
        ("alpha_heads_1_42", ["heads"]),
        ("alpha_ce_1_42", ["ce-rail"]),
        ("alpha_raft-rail_1_42", ["raft-rail"]),
    ]
    schedule = build_schedule("alpha", resolve_modes({"ce-std", "ce-rail"}), [42])
    assert [r["run_id"] for r in schedule] == ["alpha_heads_1_42", "alpha_raft-rail_1_42"]


def test_old_schema_study_state_is_refused(tmp_path):
    (tmp_path / STUDY_STATE_FILE).write_text(
        json.dumps({"prefix": "old", "runs": [{"run_id": "old_ce-std_1_1", "mode": "ce-std"}]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="old one-run-per-mode schema"):
        load_study_state(tmp_path)


def test_mode_exclusions_preserve_round_robin_order():
    modes = resolve_modes({"ce-std", "raft-rail"})
    assert modes == ["heads", "ce-rail"]
    with pytest.raises(SystemExit, match="all modes were excluded"):
        resolve_modes(set(MODES))


def test_prefix_and_seed_validation():
    assert validate_prefix("alpha-1_v2") == "alpha-1_v2"
    for bad in ("", "   ", "a/b", "..", "has space"):
        with pytest.raises(SystemExit):
            validate_prefix(bad)

    assert parse_seeds("42, 43,44") == [42, 43, 44]
    for bad in ("", "42,abc"):
        with pytest.raises(SystemExit):
            parse_seeds(bad)


def test_resume_without_prefix_explicitly_requires_prefix(tmp_path):
    with pytest.raises(SystemExit, match="must specify --prefix"):
        batch_study.run_batch_study(["--resume"], batch_root=tmp_path)


def test_starting_new_study_while_any_study_is_paused_warns_and_exits(tmp_path):
    study_dir = tmp_path / "alpha"
    state = new_study_state(
        prefix="alpha", seeds=[42, 43], modes=list(MODES), excluded_modes=[],
        split="test", eval_batch_size=2, eval_limit=0, no_bootstrap=False, snap=False,
        extra_train_args=[],
    )
    state["status"] = "paused"
    state["current_run_id"] = "alpha_heads_1_42"
    save_study_state(study_dir, state)

    assert len(find_paused_studies(tmp_path)) == 1
    with pytest.raises(SystemExit) as excinfo:
        batch_study.run_batch_study(
            ["--prefix", "beta", "--seeds", "10,11"], batch_root=tmp_path
        )
    msg = str(excinfo.value)
    assert "paused" in msg.lower()
    assert "--resume --prefix alpha" in msg


def test_batch_controlled_train_flags_are_rejected():
    for bad in (["--train-seed", "99"], ["--ce-rail"], ["--run-id", "x"], ["--run-dir", "y"]):
        with pytest.raises(SystemExit, match="managed by batch_study.py"):
            batch_study.parse_args(["--prefix", "a", "--seeds", "1", *bad])


def test_unknown_extra_flags_are_rejected():
    with pytest.raises(SystemExit):
        batch_study.parse_args(["--prefix", "a", "--seeds", "1", "--not-a-real-flag"])


def test_valid_extra_train_flags_pass_through():
    args, extra = batch_study.parse_args(
        ["--prefix", "a", "--seeds", "1,2", "--no-ce-std", "--train-probe", "20", "--train-wandb-mode", "disabled"]
    )
    assert args.prefix == "a"
    assert args.no_ce_std is True
    assert extra == ["--train-probe", "20", "--train-wandb-mode", "disabled"]


# --- end-to-end orchestration, summary files, and Ctrl-C pause/resume -----------------


def test_end_to_end_study_runs_serial_round_robin_and_writes_summaries(tmp_path):
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"

    study_dir = batch_study.run_batch_study(
        ["--prefix", "alpha", "--seeds", "42,43,44"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: None,
    )
    assert study_dir == batch_root / "alpha"
    assert find_paused_studies(batch_root) == []

    # Verify exact serial round-robin execution order: train then its eval(s) for each run.
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    seeds = [(1, 42), (2, 43), (3, 44)]
    expected_train = [
        f"alpha_{m}_{r}_{s}" for r, s in seeds for m in ("heads", "ce", "raft-rail")
    ]
    expected_sequence = []
    for r, s in seeds:
        expected_sequence += [
            ("train", f"alpha_heads_{r}_{s}"), ("eval", f"alpha_heads_{r}_{s}", "heads"),
            ("train", f"alpha_ce_{r}_{s}"),
            ("eval", f"alpha_ce_{r}_{s}", "ce-std"), ("eval", f"alpha_ce_{r}_{s}", "ce-rail"),
            ("train", f"alpha_raft-rail_{r}_{s}"),
            ("eval", f"alpha_raft-rail_{r}_{s}", "raft-rail"),
        ]
    assert [
        (e["cmd"], e["run_id"]) if e["cmd"] == "train" else (e["cmd"], e["run_id"], e["mode"])
        for e in events
    ] == expected_sequence
    # The shared ce run trains once, under --ce-std.
    assert {e["mode"] for e in events if e["cmd"] == "train" and "_ce_" in e["run_id"]} == {"ce-std"}

    # Every training run has weights under runs/<run_id>; every evaluated mode has a CSV.
    for rid in expected_train:
        assert (study_dir / "runs" / rid / "heads.safetensors").exists()
    for r, s in seeds:
        for mode in MODES:
            assert (study_dir / "results" / f"alpha_{mode}_{r}_{s}.csv").exists()

    state = load_study_state(study_dir)
    assert state["status"] == "completed"
    assert (study_dir / "summary.csv").exists()
    assert (study_dir / "summary.txt").exists()

    # Validate summary.csv structure and values.
    with (study_dir / "summary.csv").open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == [
        "model", "source", "score_type", "n_runs",
        "round_1_seed_42", "round_2_seed_43", "round_3_seed_44",
        "mean_pcc", "stddev_pcc", "ci95_lo", "ci95_hi",
    ]

    expected_models = list(MODES) + [b.name for b in BASELINES]
    assert len(rows) == len(expected_models) * len(APPLICABLE_FIELDS)
    assert "utterance.completeness" not in {r["score_type"] for r in rows}

    # Check study row Fisher-Z consistency against individual round columns.
    heads_phone = next(
        r for r in rows if r["model"] == "heads" and r["score_type"] == "phone.accuracy"
    )
    individual = [
        float(heads_phone["round_1_seed_42"]),
        float(heads_phone["round_2_seed_43"]),
        float(heads_phone["round_3_seed_44"]),
    ]
    ref = fisher_z_summary(individual)
    # The two ce readouts come from one training run but are reported as separate models.
    ce_std = next(r for r in rows if r["model"] == "ce-std" and r["score_type"] == "phone.accuracy")
    ce_rail = next(r for r in rows if r["model"] == "ce-rail" and r["score_type"] == "phone.accuracy")
    assert ce_std["round_1_seed_42"] != ce_rail["round_1_seed_42"]
    assert float(heads_phone["mean_pcc"]) == pytest.approx(ref.mean_pcc, abs=1e-12)
    assert float(heads_phone["stddev_pcc"]) == pytest.approx(ref.stddev_pcc, abs=1e-12)
    assert float(heads_phone["ci95_lo"]) == pytest.approx(ref.ci95_lo, abs=1e-12)
    assert float(heads_phone["ci95_hi"]) == pytest.approx(ref.ci95_hi, abs=1e-12)

    # Check published baseline rows have published PCC and N/A for unavailable fields.
    hmamba_phone = next(
        r for r in rows if r["model"] == "HMamba (2025)" and r["score_type"] == "phone.accuracy"
    )
    assert float(hmamba_phone["mean_pcc"]) == pytest.approx(0.739)
    assert hmamba_phone["n_runs"] == "N/A"
    assert hmamba_phone["round_1_seed_42"] == "N/A"
    assert hmamba_phone["stddev_pcc"] == "N/A"
    assert hmamba_phone["ci95_lo"] == "N/A"
    assert hmamba_phone["ci95_hi"] == "N/A"

    # Check human-readable summary text contains overview, details, baselines, and N/A.
    txt = (study_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Batch Study Summary: alpha" in txt
    assert "Fisher Z-transformation" in txt
    assert "HMamba (2025)" in txt
    assert "GOPT (2022)" in txt
    assert "N/A" in txt

    # Re-running with the same completed prefix without --resume is refused.
    with pytest.raises(SystemExit, match="already exists"):
        batch_study.run_batch_study(
            ["--prefix", "alpha", "--seeds", "42"],
            batch_root=batch_root,
            train_script=fake_train,
            eval_script=fake_eval,
        )


def test_sigint_pauses_training_and_resume_completes_study(tmp_path):
    """Real SIGINT during run #2 pauses the study, blocks new studies, and resumes cleanly."""
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"
    ready_file = tmp_path / "ready.txt"
    repo_root = Path(__file__).resolve().parents[1]

    runner = tmp_path / "run_wrapper.py"
    runner.write_text(
        f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(repo_root)!r})
import batch_study
batch_study.run_batch_study(
    sys.argv[1:],
    batch_root=Path({str(batch_root)!r}),
    train_script=Path({str(fake_train)!r}),
    eval_script=Path({str(fake_eval)!r}),
    read_soc=lambda: None,
)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["FAKE_PAUSE_RUN_ID"] = "alpha_ce_1_42"
    env["FAKE_READY_FILE"] = str(ready_file)

    proc = subprocess.Popen(
        [
            sys.executable, str(runner),
            "--prefix", "alpha", "--seeds", "42,43",
            "--no-ce-rail", "--no-raft-rail",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10.0
    while not ready_file.exists() and time.time() < deadline:
        assert proc.poll() is None, f"process exited early: {proc.stderr.read()}"
        time.sleep(0.02)
    assert ready_file.exists(), "stub train never reached ready state"

    # Send Ctrl-C (SIGINT) to batch_study; it must forward once to stub train and save state.
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 130, f"expected 130, got {proc.returncode}\n{stdout}\n{stderr}"
    assert "--resume --prefix alpha" in stdout

    # Verify study state and paused run checkpoint exist.
    study_dir = batch_root / "alpha"
    state = load_study_state(study_dir)
    assert state["status"] == "paused"
    assert state["current_run_id"] == "alpha_ce_1_42"
    assert (study_dir / "runs" / "alpha_ce_1_42" / "resume.json").exists()
    statuses = {r["run_id"]: r["status"] for r in state["runs"]}
    assert statuses["alpha_heads_1_42"] == "completed"
    assert statuses["alpha_ce_1_42"] == "training"
    assert statuses["alpha_heads_2_43"] == "pending"

    # Starting any new study while paused must warn and exit.
    with pytest.raises(SystemExit, match="paused"):
        batch_study.run_batch_study(
            ["--prefix", "other", "--seeds", "99"],
            batch_root=batch_root,
            train_script=fake_train,
            eval_script=fake_eval,
        )

    # Resuming without --prefix must fail and tell the user to specify it.
    with pytest.raises(SystemExit, match="must specify --prefix"):
        batch_study.run_batch_study(
            ["--resume"],
            batch_root=batch_root,
            train_script=fake_train,
            eval_script=fake_eval,
        )

    # Resume with the right prefix: finishes the paused run via --resume and completes the rest.
    batch_study.run_batch_study(
        ["--resume", "--prefix", "alpha"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: None,
    )
    final_state = load_study_state(study_dir)
    assert final_state["status"] == "completed"
    assert not (study_dir / "runs" / "alpha_ce_1_42" / "resume.json").exists()
    assert (study_dir / "summary.csv").exists()
    assert (study_dir / "summary.txt").exists()

    # Verify alpha_heads_1_42 was NOT retrained, and alpha_ce_1_42 resumed with --resume.
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    heads_1_trains = [e for e in events if e["cmd"] == "train" and e["run_id"] == "alpha_heads_1_42"]
    assert len(heads_1_trains) == 1
    cestd_1_trains = [e for e in events if e["cmd"] == "train" and e["run_id"] == "alpha_ce_1_42"]
    assert [e["resumed"] for e in cestd_1_trains] == [False, True]


def test_sigint_during_evaluation_resumes_directly_to_eval(tmp_path):
    """If paused during evaluate.py, resuming skips re-training and re-runs evaluation."""
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"
    ready_file = tmp_path / "eval_ready.txt"
    repo_root = Path(__file__).resolve().parents[1]

    runner = tmp_path / "run_wrapper.py"
    runner.write_text(
        f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(repo_root)!r})
import batch_study
batch_study.run_batch_study(
    sys.argv[1:],
    batch_root=Path({str(batch_root)!r}),
    train_script=Path({str(fake_train)!r}),
    eval_script=Path({str(fake_eval)!r}),
    read_soc=lambda: None,
)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["FAKE_PAUSE_EVAL_RUN_ID"] = "alpha_heads_1_42"
    env["FAKE_EVAL_READY_FILE"] = str(ready_file)

    proc = subprocess.Popen(
        [
            sys.executable, str(runner),
            "--prefix", "alpha", "--seeds", "42",
            "--no-ce-std", "--no-ce-rail", "--no-raft-rail",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10.0
    while not ready_file.exists() and time.time() < deadline:
        assert proc.poll() is None, f"process exited early: {proc.stderr.read()}"
        time.sleep(0.02)
    assert ready_file.exists()

    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=10)
    assert proc.returncode == 130

    study_dir = batch_root / "alpha"
    state = load_study_state(study_dir)
    assert state["status"] == "paused"
    assert state["runs"][0]["status"] == "trained"
    # Partial eval files must not remain on disk.
    assert not (study_dir / "results" / "alpha_heads_1_42.csv").exists()
    assert not (study_dir / "results" / "alpha_heads_1_42.json").exists()

    # Resume: must run evaluate.py without re-running train.py.
    batch_study.run_batch_study(
        ["--resume", "--prefix", "alpha"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: None,
    )
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert [e["cmd"] for e in events] == ["train", "eval", "eval"]
    assert load_study_state(study_dir)["status"] == "completed"


def test_sigint_during_second_ce_eval_resumes_only_that_eval(tmp_path):
    """Pausing the ce-rail eval of a shared ce run keeps ce-std's result and redoes only ce-rail."""
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"
    ready_file = tmp_path / "eval_ready.txt"
    repo_root = Path(__file__).resolve().parents[1]

    runner = tmp_path / "run_wrapper.py"
    runner.write_text(
        f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(repo_root)!r})
import batch_study
batch_study.run_batch_study(
    sys.argv[1:],
    batch_root=Path({str(batch_root)!r}),
    train_script=Path({str(fake_train)!r}),
    eval_script=Path({str(fake_eval)!r}),
    read_soc=lambda: None,
)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["FAKE_PAUSE_EVAL_RUN_ID"] = "alpha_ce_1_42"
    env["FAKE_PAUSE_EVAL_MODE"] = "ce-rail"
    env["FAKE_EVAL_READY_FILE"] = str(ready_file)

    proc = subprocess.Popen(
        [
            sys.executable, str(runner),
            "--prefix", "alpha", "--seeds", "42",
            "--no-heads", "--no-raft-rail",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10.0
    while not ready_file.exists() and time.time() < deadline:
        assert proc.poll() is None, f"process exited early: {proc.stderr.read()}"
        time.sleep(0.02)
    assert ready_file.exists()

    proc.send_signal(signal.SIGINT)
    proc.communicate(timeout=10)
    assert proc.returncode == 130

    study_dir = batch_root / "alpha"
    state = load_study_state(study_dir)
    entry = state["runs"][0]
    assert entry["run_id"] == "alpha_ce_1_42"
    assert entry["status"] == "trained"
    assert {ev["mode"]: ev["status"] for ev in entry["evals"]} == {
        "ce-std": "completed", "ce-rail": "pending",
    }
    assert (study_dir / "results" / "alpha_ce-std_1_42.csv").exists()
    assert not (study_dir / "results" / "alpha_ce-rail_1_42.csv").exists()

    batch_study.run_batch_study(
        ["--resume", "--prefix", "alpha"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: None,
    )
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert [(e["cmd"], e.get("mode")) for e in events if e["cmd"] == "eval"] == [
        ("eval", "ce-std"), ("eval", "ce-rail"), ("eval", "ce-rail"),
    ]
    assert len([e for e in events if e["cmd"] == "train"]) == 1
    assert load_study_state(study_dir)["status"] == "completed"
    assert (study_dir / "results" / "alpha_ce-rail_1_42.csv").exists()


def test_train_and_evaluate_cli_accept_new_flags(monkeypatch):
    import evaluate
    import train

    monkeypatch.setattr("sys.argv", ["train.py", "--run-id", "alpha_heads_1_42", "--run-dir", "batch-studies/alpha/runs/alpha_heads_1_42"])
    t_args, _ = train.parse_args()
    assert t_args.run_dir == "batch-studies/alpha/runs/alpha_heads_1_42"

    monkeypatch.setattr("sys.argv", ["evaluate.py", "--run-dir", "batch-studies/alpha/runs/alpha_heads_1_42", "--out-csv", "batch-studies/alpha/results/alpha_heads_1_42.csv"])
    e_args = evaluate.parse_args()
    assert e_args.out_csv == "batch-studies/alpha/results/alpha_heads_1_42.csv"
    assert e_args.eval_variant == ""

    monkeypatch.setattr("sys.argv", ["evaluate.py", "--eval-variant", "ce_rail"])
    assert evaluate.parse_args().eval_variant == "ce_rail"
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--eval-variant", "raft_rail"])
    with pytest.raises(SystemExit):
        evaluate.parse_args()


# --- --purge tests --------------------------------------------------------------------


def test_purge_requires_prefix_and_rejects_resume_combination(tmp_path):
    with pytest.raises(SystemExit, match="must specify --prefix"):
        batch_study.run_batch_study(["--purge"], batch_root=tmp_path)
    with pytest.raises(SystemExit, match="mutually exclusive"):
        batch_study.run_batch_study(["--purge", "--resume", "--prefix", "alpha"], batch_root=tmp_path)
    with pytest.raises(SystemExit, match="No batch study found"):
        batch_study.run_batch_study(["--purge", "--prefix", "missing"], batch_root=tmp_path)


def test_purge_reports_in_progress_status_and_respects_confirmation(tmp_path, monkeypatch, capsys):
    study_dir = tmp_path / "alpha"
    state = new_study_state(
        prefix="alpha", seeds=[42, 43], modes=["heads", "ce-std"], excluded_modes=["ce-rail", "raft-rail"],
        split="test", eval_batch_size=2, eval_limit=0, no_bootstrap=False, snap=False,
        extra_train_args=[],
    )
    state["status"] = "paused"
    state["current_run_id"] = "alpha_ce_1_42"
    state["runs"][0]["status"] = "completed"
    save_study_state(study_dir, state)
    run_dir = study_dir / "runs" / "alpha_ce_1_42"
    run_dir.mkdir(parents=True)
    (run_dir / "resume.json").write_text('{"step": 5}', encoding="utf-8")
    (run_dir / "heads.safetensors").write_bytes(b"weights")

    # Default (empty Enter) or 'n' must cancel and preserve everything.
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    batch_study.run_batch_study(["--purge", "--prefix", "alpha"], batch_root=tmp_path)
    out = capsys.readouterr().out
    assert "still in progress" in out
    assert "1/4 runs completed" in out
    assert "purge cancelled" in out
    assert study_dir.exists()
    assert (run_dir / "resume.json").exists()

    # Confirming with 'y' deletes the study directory and unblocks new studies.
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    batch_study.run_batch_study(["--purge", "--prefix", "alpha"], batch_root=tmp_path)
    out = capsys.readouterr().out
    assert "still in progress" in out
    assert "purged batch study 'alpha'" in out
    assert not study_dir.exists()
    assert find_paused_studies(tmp_path) == []


def test_purge_reports_finished_status_and_deletes_on_yes(tmp_path, monkeypatch, capsys):
    study_dir = tmp_path / "done_study"
    state = new_study_state(
        prefix="done_study", seeds=[42], modes=["heads"], excluded_modes=["ce-std", "ce-rail", "raft-rail"],
        split="test", eval_batch_size=2, eval_limit=0, no_bootstrap=False, snap=False,
        extra_train_args=[],
    )
    state["status"] = "completed"
    state["runs"][0]["status"] = "completed"
    save_study_state(study_dir, state)
    (study_dir / "summary.csv").write_text("col\n1\n", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    batch_study.run_batch_study(["--purge", "--prefix", "done_study"], batch_root=tmp_path)
    out = capsys.readouterr().out
    assert "finished (1/1 runs completed)" in out
    assert not study_dir.exists()



# --- W&B logging of evaluation results ------------------------------------------------


def test_wandb_prefix_separates_the_ce_readouts():
    import evaluate

    assert evaluate.wandb_prefix("test", "heads") == "test"
    assert evaluate.wandb_prefix("test", "raft_rail") == "test"
    assert evaluate.wandb_prefix("test", "ce_std") == "test-ce-std"
    assert evaluate.wandb_prefix("test", "ce_rail") == "test-ce-rail"


class _FakeRun:
    def __init__(self, rid, starting_step=0):
        self.id = rid
        self.starting_step = starting_step
        self.logged: list[tuple[dict, int | None, bool | None]] = []
        self.finished = False

    def log(self, data, step=None, commit=None):
        self.logged.append((dict(data), step, commit))
        next_step = (step if step is not None else self.starting_step) + 1
        self.starting_step = max(self.starting_step, next_step)

    def finish(self):
        self.finished = True


class _FakeApi:
    default_entity = "me"
    runs_by_id: dict = {}
    names: dict = {}

    def runs(self, path, filters):
        assert path == "me/proj"
        return [self.runs_by_id[r] for r in self.names.get(filters["display_name"], [])]


def test_push_wandb_metrics_logs_history_with_monotonic_step(monkeypatch):
    import types

    from tutokana.reporting import push_wandb_metrics

    _FakeApi.runs_by_id = {
        "a1": _FakeRun("a1", starting_step=1961),
        "b1": _FakeRun("b1", starting_step=1961),
        "b2": _FakeRun("b2", starting_step=10),
    }
    _FakeApi.names = {"alpha_ce_1_12": ["a1"], "dup": ["b1", "b2"]}
    init_calls: list[dict] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        assert kwargs["resume"] == "must"
        assert Path(kwargs["dir"]).exists()
        run = _FakeApi.runs_by_id[kwargs["id"]]
        run.finished = False
        return run

    fake_wandb = types.SimpleNamespace(
        Api=_FakeApi,
        init=fake_init,
        Settings=lambda **kw: kw,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    # Recorded id logs at requested step when >= starting_step, and finishes cleanly.
    assert push_wandb_metrics("proj", "ignored", "b1", {"x": 1.0}, step=1964) == "b1"
    assert _FakeApi.runs_by_id["b1"].logged == [({"x": 1.0}, 1964, True)]
    assert _FakeApi.runs_by_id["b1"].finished is True
    assert init_calls[-1]["settings"]["x_disable_stats"] is True
    assert init_calls[-1]["settings"]["console"] == "off"

    # Second eval on the same run advances past starting_step instead of dropping step=1964.
    assert push_wandb_metrics("proj", "ignored", "b1", {"x2": 2.0}, step=1964) == "b1"
    assert _FakeApi.runs_by_id["b1"].logged[-1] == ({"x2": 2.0}, 1965, True)

    # Missing id falls back to unique display_name lookup and handles step=None.
    assert push_wandb_metrics("proj", "alpha_ce_1_12", None, {"y": 2.0}) == "a1"
    assert _FakeApi.runs_by_id["a1"].logged == [({"y": 2.0}, None, None)]
    assert _FakeApi.runs_by_id["a1"].finished is True

    with pytest.raises(LookupError, match="2 runs"):
        push_wandb_metrics("proj", "dup", None, {"z": 3.0})
    with pytest.raises(LookupError, match="0 runs"):
        push_wandb_metrics("proj", "missing", None, {"z": 3.0})


# --- macOS battery SoC guard tests ----------------------------------------------------


def test_parse_pmset_soc_and_non_mac_noop(monkeypatch):
    from tutokana import power

    assert power.parse_pmset_soc(
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=22610019)\t68%; charging; 16:47 remaining present: true\n"
    ) == 68
    assert power.parse_pmset_soc(
        "Now drawing from 'Battery Power'\n"
        " -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true\n"
    ) == 100
    assert power.parse_pmset_soc("Now drawing from 'AC Power'\n") is None
    assert power.parse_pmset_soc("") is None

    monkeypatch.setattr(power.sys, "platform", "linux")
    assert power.is_mac_host() is False
    assert power.read_battery_soc() is None


def test_wait_for_battery_polls_every_5s_until_80():
    from tutokana.power import wait_for_battery

    readings = iter([68, 68, 74, 79, 80])
    sleeps: list[float] = []
    messages: list[str] = []

    final = wait_for_battery(
        min_soc=80,
        poll_seconds=5.0,
        read_soc=lambda: next(readings),
        sleep=sleeps.append,
        reason="after run 1",
        emit=messages.append,
    )
    assert final == 80
    assert sleeps == [5.0, 5.0, 5.0, 5.0]
    assert any("68%" in m and "80%" in m for m in messages)
    assert any("reached 80%" in m for m in messages)


def test_batch_study_waits_for_80_between_runs_and_resumes_after_mid_run_battery_pause(
    tmp_path, monkeypatch
):
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"

    monkeypatch.setattr(batch_study, "is_mac_host", lambda: True)
    monkeypatch.setenv("FAKE_BATTERY_PAUSE_RUN_ID", "beta_heads_1_12")

    # Sequence of SoC queries:
    # 1. Mid-run pause during beta_heads_1_12 -> 48%, 65%, 80% (then auto-resumes run 1)
    # 2. After beta_heads_1_12 finishes (before beta_heads_2_17) -> 72%, 81%
    # 3. After beta_heads_2_17 (last run): no query needed
    readings = iter([48, 65, 80, 72, 81])
    sleeps: list[float] = []

    study_dir = batch_study.run_batch_study(
        [
            "--prefix", "beta", "--seeds", "12,17",
            "--no-ce-std", "--no-ce-rail", "--no-raft-rail",
        ],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: next(readings),
        sleep=sleeps.append,
        battery_poll_seconds=5.0,
    )
    assert load_study_state(study_dir)["status"] == "completed"
    # 2 sleeps during mid-run recharge (48->65->80) + 1 sleep between runs (72->81).
    assert sleeps == [5.0, 5.0, 5.0]
    assert list(readings) == []

    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    trains = [e for e in events if e["cmd"] == "train"]
    assert [(e["run_id"], e["resumed"], e["min_battery_soc"]) for e in trains] == [
        ("beta_heads_1_12", False, 50),
        ("beta_heads_1_12", True, 50),
        ("beta_heads_2_17", False, 50),
    ]


def test_batch_study_omits_min_battery_soc_on_non_mac_hosts(tmp_path, monkeypatch):
    fake_train, fake_eval, trace_file = _write_fake_scripts(tmp_path)
    batch_root = tmp_path / "batch-studies"

    monkeypatch.setattr(batch_study, "is_mac_host", lambda: False)
    batch_study.run_batch_study(
        ["--prefix", "linuxrun", "--seeds", "12", "--no-ce-std", "--no-ce-rail", "--no-raft-rail"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
        read_soc=lambda: None,
    )
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    trains = [e for e in events if e["cmd"] == "train"]
    assert trains[0]["min_battery_soc"] == 0


def test_min_battery_soc_is_accepted_by_train_and_owned_by_batch_study(monkeypatch):
    import train

    monkeypatch.setattr("sys.argv", ["train.py", "--min-battery-soc", "50"])
    args, _ = train.parse_args()
    assert args.min_battery_soc == 50

    with pytest.raises(SystemExit, match="managed by batch_study.py"):
        batch_study.parse_args(["--prefix", "a", "--seeds", "1", "--min-battery-soc", "50"])
