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
    - When `FAKE_PAUSE_EVAL_RUN_ID` matches the current run_id (on its first eval attempt),
      signals readiness and waits for SIGINT, writing a partial JSON and exiting 130.
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
        "extra": extra,
    }}) + "\\n")

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
args = parser.parse_args()

run_dir = Path(args.run_dir)
meta = json.loads((run_dir / "config.json").read_text())
run_id = meta["run_id"]
mode = meta["mode"]
seed = int(meta["seed"])

with open({str(trace_file)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{
        "cmd": "eval", "run_id": run_id, "run_dir": str(run_dir),
        "split": args.split, "out_csv": args.out_csv,
    }}) + "\\n")

pause_eval = os.environ.get("FAKE_PAUSE_EVAL_RUN_ID", "")
ready_file = os.environ.get("FAKE_EVAL_READY_FILE", "")
paused_marker = run_dir / ".eval_paused_once"
if pause_eval == run_id and not paused_marker.exists():
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
        "alpha_ce-std_1_42",
        "alpha_ce-rail_1_42",
        "alpha_raft-rail_1_42",
        "alpha_heads_2_43",
        "alpha_ce-std_2_43",
        "alpha_ce-rail_2_43",
        "alpha_raft-rail_2_43",
    ]
    assert make_run_id("alpha", "raft-rail", 1, 42) == "alpha_raft-rail_1_42"


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
        ["--prefix", "alpha", "--seeds", "42,43,44", "--no-ce-std"],
        batch_root=batch_root,
        train_script=fake_train,
        eval_script=fake_eval,
    )
    assert study_dir == batch_root / "alpha"
    assert find_paused_studies(batch_root) == []

    # Verify exact serial round-robin execution order: train then eval for each run.
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    expected_ids = [
        "alpha_heads_1_42", "alpha_ce-rail_1_42", "alpha_raft-rail_1_42",
        "alpha_heads_2_43", "alpha_ce-rail_2_43", "alpha_raft-rail_2_43",
        "alpha_heads_3_44", "alpha_ce-rail_3_44", "alpha_raft-rail_3_44",
    ]
    assert [e["run_id"] for e in events if e["cmd"] == "train"] == expected_ids
    assert [e["run_id"] for e in events if e["cmd"] == "eval"] == expected_ids
    # Every train event is immediately followed by its eval event.
    for i in range(0, len(events), 2):
        assert events[i]["cmd"] == "train"
        assert events[i + 1]["cmd"] == "eval"
        assert events[i]["run_id"] == events[i + 1]["run_id"]

    # Every run has weights under batch-studies/alpha/runs/<run_id> and a CSV under results/.
    for rid in expected_ids:
        assert (study_dir / "runs" / rid / "heads.safetensors").exists()
        assert (study_dir / "results" / f"{rid}.csv").exists()

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

    expected_models = ["heads", "ce-rail", "raft-rail"] + [b.name for b in BASELINES]
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
)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["FAKE_PAUSE_RUN_ID"] = "alpha_ce-std_1_42"
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
    assert state["current_run_id"] == "alpha_ce-std_1_42"
    assert (study_dir / "runs" / "alpha_ce-std_1_42" / "resume.json").exists()
    statuses = {r["run_id"]: r["status"] for r in state["runs"]}
    assert statuses["alpha_heads_1_42"] == "completed"
    assert statuses["alpha_ce-std_1_42"] == "training"
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
    )
    final_state = load_study_state(study_dir)
    assert final_state["status"] == "completed"
    assert not (study_dir / "runs" / "alpha_ce-std_1_42" / "resume.json").exists()
    assert (study_dir / "summary.csv").exists()
    assert (study_dir / "summary.txt").exists()

    # Verify alpha_heads_1_42 was NOT retrained, and alpha_ce-std_1_42 resumed with --resume.
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    heads_1_trains = [e for e in events if e["cmd"] == "train" and e["run_id"] == "alpha_heads_1_42"]
    assert len(heads_1_trains) == 1
    cestd_1_trains = [e for e in events if e["cmd"] == "train" and e["run_id"] == "alpha_ce-std_1_42"]
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
    )
    events = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert [e["cmd"] for e in events] == ["train", "eval", "eval"]
    assert load_study_state(study_dir)["status"] == "completed"


def test_train_and_evaluate_cli_accept_new_flags(monkeypatch):
    import evaluate
    import train

    monkeypatch.setattr("sys.argv", ["train.py", "--run-id", "alpha_heads_1_42", "--run-dir", "batch-studies/alpha/runs/alpha_heads_1_42"])
    t_args, _ = train.parse_args()
    assert t_args.run_dir == "batch-studies/alpha/runs/alpha_heads_1_42"

    monkeypatch.setattr("sys.argv", ["evaluate.py", "--run-dir", "batch-studies/alpha/runs/alpha_heads_1_42", "--out-csv", "batch-studies/alpha/results/alpha_heads_1_42.csv"])
    e_args = evaluate.parse_args()
    assert e_args.out_csv == "batch-studies/alpha/results/alpha_heads_1_42.csv"


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
    state["current_run_id"] = "alpha_ce-std_1_42"
    state["runs"][0]["status"] = "completed"
    save_study_state(study_dir, state)
    run_dir = study_dir / "runs" / "alpha_ce-std_1_42"
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

