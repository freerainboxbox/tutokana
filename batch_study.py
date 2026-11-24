"""Run a multi-seed, multi-mode training and evaluation batch study in serial.

Trains and evaluates each active mode (`heads`, `ce-std`, `ce-rail`, `raft-rail`) across every
given seed in round-robin order:

    python batch_study.py --prefix alpha --seeds 42,43,44,45,46
    python batch_study.py --prefix alpha --seeds 42,43 --no-ce-std
    python batch_study.py --resume --prefix alpha
    python batch_study.py --purge --prefix alpha

Each run is named `<prefix>_<mode>_<round>_<seed>` (1-based round, e.g. `alpha_raft-rail_1_42`).
Standard train/eval logs still write to `logs/` unchanged; every other study artifact — model
weights, checkpoints, per-run CSVs/JSONs, study resume metadata, and final Fisher-Z summaries —
is collected under `batch-studies/<prefix>/`.

Press Ctrl-C to pause cleanly: the active run saves its own checkpoint via the existing
graceful-interrupt mechanics and the batch study saves its progress in
`batch-studies/<prefix>/study_state.json`. Starting another batch study while any study is
paused warns and exits; resume with `python batch_study.py --resume --prefix <prefix>`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tutokana import config as cfg
from tutokana.engine import RESUME_META
from tutokana.study import (
    MODE_TO_FLAG,
    MODES,
    STUDY_STATE_FILE,
    find_paused_studies,
    load_study_state,
    new_study_state,
    parse_seeds,
    resolve_modes,
    save_study_state,
    validate_prefix,
    write_study_summaries,
)

#: Train flags owned by `batch_study.py`; passing them through as extra train args is refused.
_BATCH_CONTROLLED_TRAIN_FLAGS = frozenset(
    {
        "--run-id",
        "--run-dir",
        "--resume",
        "--train-seed",
        "--variant",
        "--ce-std",
        "--ce-rail",
        "--raft-rail",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="required batch study prefix; run IDs are <prefix>_<mode>_<round>_<seed>",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated integer seeds, one per round (e.g. 42,43,44,45,46)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a paused batch study (requires --prefix <prefix>)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete a batch study's artifacts and resume metadata after confirmation "
        "(requires --prefix <prefix>)",
    )
    for mode in MODES:
        parser.add_argument(
            f"--no-{mode}",
            dest=f"no_{mode.replace('-', '_')}",
            action="store_true",
            help=f"exclude the {mode} mode from the study",
        )

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument(
        "--split", default="test", help="dataset split to evaluate on (default: test)"
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="batch size forwarded to evaluate.py (default: 2)",
    )
    eval_group.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="if > 0, evaluate only the first N utterances (default: 0 = full split)",
    )
    eval_group.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="forward --no-bootstrap to evaluate.py",
    )
    eval_group.add_argument(
        "--snap",
        action="store_true",
        help="forward --snap to evaluate.py",
    )
    return parser


def validate_extra_train_args(extra: list[str]) -> list[str]:
    """Ensure any forwarded flags are real `train.py` flags and don't collide with batch control."""
    if not extra:
        return []
    for token in extra:
        flag = token.split("=", 1)[0]
        if flag in _BATCH_CONTROLLED_TRAIN_FLAGS:
            raise SystemExit(
                f"{flag} is managed by batch_study.py and cannot be passed as an extra train flag"
            )
    train_parser = argparse.ArgumentParser(prog="batch_study.py (extra train flags)", add_help=False)
    train_parser.add_argument("--notes", default="")
    cfg.add_config_arguments(train_parser)
    # Raises SystemExit with a clear argparse error if any flag is unknown or malformed.
    train_parser.parse_args(extra)
    return list(extra)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    extra = validate_extra_train_args(extra)
    return args, extra


def _excluded_from_args(args: argparse.Namespace) -> set[str]:
    return {
        mode for mode in MODES if getattr(args, f"no_{mode.replace('-', '_')}", False)
    }


def _format_paused_list(paused: list[dict]) -> str:
    lines = []
    for item in paused:
        detail = f"status={item['status']}"
        if item.get("current_run_id"):
            detail += f", current_run={item['current_run_id']}"
        elif item.get("paused_runs"):
            detail += f", paused_runs={','.join(item['paused_runs'])}"
        lines.append(f"  - {item['prefix']} ({detail}) -> python batch_study.py --resume --prefix {item['prefix']}")
    return "\n".join(lines)


def _run_subprocess_interruptible(cmd: list[str]) -> tuple[int, bool]:
    """Run a child command in its own process group and forward Ctrl-C cleanly once per press.

    Isolating the child in `process_group=0` prevents a terminal Ctrl-C from being delivered
    twice (once by the kernel to the foreground group and again by this script), which would
    trigger the child's immediate-abort second-Ctrl-C branch before it can save its checkpoint.
    """
    state = {"count": 0}
    proc = subprocess.Popen(cmd, process_group=0)
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        state["count"] += 1
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        if state["count"] == 1:
            os.write(
                2,
                b"\n[batch] Ctrl-C received - waiting for current run to pause and save cleanly...\n",
            )
        else:
            os.write(
                2,
                b"\n[batch] second Ctrl-C received - forwarding immediate interrupt...\n",
            )

    signal.signal(signal.SIGINT, handler)
    try:
        rc = proc.wait()
    finally:
        signal.signal(signal.SIGINT, previous)
    return rc, state["count"] > 0


def _training_completed_cleanly(run_dir: Path, rc: int) -> bool:
    if rc != 0:
        return False
    if (run_dir / RESUME_META).exists():
        return False
    meta_path = run_dir / "config.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return not bool(meta.get("interrupted", False))


def _pause_and_exit(study_dir: Path, state: dict, run_id: str, reason: str, code: int = 130) -> None:
    state["status"] = "paused"
    state["paused_at"] = datetime.now().isoformat(timespec="seconds")
    state["current_run_id"] = run_id
    save_study_state(study_dir, state)
    print(
        f"\n[batch] study {state['prefix']!r} paused ({reason}).\n"
        f"[batch] progress saved to {study_dir / STUDY_STATE_FILE}\n"
        f"[batch] resume with: python batch_study.py --resume --prefix {state['prefix']}",
        flush=True,
    )
    sys.exit(code)


def _train_command(
    train_script: Path,
    run_id: str,
    run_dir: Path,
    mode: str,
    seed: int,
    extra_train_args: list[str],
) -> list[str]:
    if (run_dir / RESUME_META).exists():
        return [
            sys.executable,
            str(train_script),
            "--resume",
            run_id,
            "--run-dir",
            str(run_dir),
        ]
    return [
        sys.executable,
        str(train_script),
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--train-seed",
        str(seed),
        *MODE_TO_FLAG[mode],
        *extra_train_args,
    ]


def _eval_command(
    eval_script: Path,
    run_dir: Path,
    csv_path: Path,
    json_path: Path,
    state: dict,
) -> list[str]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--run-dir",
        str(run_dir),
        "--split",
        str(state["split"]),
        "--batch-size",
        str(state["eval_batch_size"]),
        "--out-csv",
        str(csv_path),
        "--out",
        str(json_path),
    ]
    if state.get("eval_limit"):
        cmd.extend(["--limit", str(state["eval_limit"])])
    if state.get("no_bootstrap"):
        cmd.append("--no-bootstrap")
    if state.get("snap"):
        cmd.append("--snap")
    return cmd


def _describe_study_status(study_dir: Path) -> str:
    """Human-readable status summary ('finished' vs 'still in progress') for --purge."""
    state_path = study_dir / STUDY_STATE_FILE
    paused_runs = (
        sorted(p.parent.name for p in (study_dir / "runs").glob(f"*/{RESUME_META}"))
        if (study_dir / "runs").exists()
        else []
    )
    if not state_path.exists():
        if paused_runs:
            return f"still in progress (no {STUDY_STATE_FILE}; paused runs: {', '.join(paused_runs)})"
        return f"unknown (no {STUDY_STATE_FILE} found)"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"still in progress (corrupted {STUDY_STATE_FILE})"

    raw_status = state.get("status", "unknown")
    runs = state.get("runs", [])
    done = sum(1 for r in runs if r.get("status") == "completed")
    total = len(runs)
    if raw_status == "completed" and not paused_runs:
        return f"finished ({done}/{total} runs completed)"

    details = [raw_status, f"{done}/{total} runs completed"]
    if state.get("current_run_id"):
        details.append(f"current run: {state['current_run_id']}")
    if paused_runs:
        details.append(f"paused checkpoint(s): {', '.join(paused_runs)}")
    return f"still in progress ({'; '.join(details)})"


def _purge_study(prefix_arg: str, batch_root: Path) -> Path:
    if not prefix_arg:
        existing = (
            sorted(p.name for p in batch_root.iterdir() if p.is_dir())
            if batch_root.exists()
            else []
        )
        hint = f"\nExisting studies: {', '.join(existing)}" if existing else ""
        raise SystemExit(
            f"You must specify --prefix <prefix> when using --purge.{hint}"
        )
    prefix = validate_prefix(prefix_arg)
    study_dir = batch_root / prefix
    if not study_dir.exists():
        raise SystemExit(
            f"No batch study found for prefix {prefix!r} under {batch_root}."
        )

    status_text = _describe_study_status(study_dir)
    print(f"[batch] study {prefix!r}: {study_dir}", flush=True)
    print(f"[batch] status: {status_text}", flush=True)
    print(
        "[batch] this will permanently remove all study artifacts (model weights, checkpoints, "
        "per-run CSVs/JSONs, summaries, and resume files). Log files in logs/ are not deleted.",
        flush=True,
    )
    try:
        answer = input(f"Delete batch study {prefix!r}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[batch] purge cancelled.", flush=True)
        return study_dir

    if answer not in ("y", "yes"):
        print("[batch] purge cancelled.", flush=True)
        return study_dir

    shutil.rmtree(study_dir)
    print(f"[batch] purged batch study {prefix!r} ({study_dir}).", flush=True)
    return study_dir


def run_batch_study(
    argv: list[str] | None = None,
    *,
    batch_root: Path | None = None,
    train_script: Path | None = None,
    eval_script: Path | None = None,
) -> Path:
    args, extra_train_args = parse_args(argv)
    batch_root = Path(batch_root) if batch_root is not None else cfg.BATCH_DIR
    train_script = Path(train_script) if train_script is not None else ROOT / "train.py"
    eval_script = Path(eval_script) if eval_script is not None else ROOT / "evaluate.py"

    if args.purge and args.resume:
        raise SystemExit("--purge and --resume are mutually exclusive")
    if args.purge:
        return _purge_study(args.prefix, batch_root)

    paused = find_paused_studies(batch_root)

    if args.resume:
        if not args.prefix:
            hint = f"\nPaused studies found:\n{_format_paused_list(paused)}" if paused else ""
            raise SystemExit(
                "You must specify --prefix <prefix> when using --resume so the intended "
                f"stopped study is resumed.{hint}"
            )
        prefix = validate_prefix(args.prefix)
        study_dir = batch_root / prefix
        if not (study_dir / STUDY_STATE_FILE).exists():
            hint = f"\nPaused studies found:\n{_format_paused_list(paused)}" if paused else ""
            raise SystemExit(
                f"No batch study found for prefix {prefix!r} under {batch_root}.{hint}"
            )
        state = load_study_state(study_dir)
        has_paused_child = bool(
            (study_dir / "runs").exists()
            and list((study_dir / "runs").glob(f"*/{RESUME_META}"))
        )
        if state.get("status") == "completed" and not has_paused_child:
            raise SystemExit(
                f"Batch study {prefix!r} is already completed ({study_dir / STUDY_STATE_FILE}); "
                "there is nothing to resume."
            )

        # Warn and ignore conflicting CLI flags on resume, mirroring `train.py --resume`.
        if args.seeds:
            given_seeds = parse_seeds(args.seeds)
            if given_seeds != state["seeds"]:
                print(
                    f"[batch] warning: ignoring --seeds={given_seeds}; "
                    f"using saved seeds {state['seeds']}",
                    file=sys.stderr,
                )
        given_excluded = _excluded_from_args(args)
        if given_excluded and sorted(given_excluded) != sorted(state.get("excluded_modes", [])):
            print(
                f"[batch] warning: ignoring mode exclusions {sorted(given_excluded)}; "
                f"using saved modes {state['modes']}",
                file=sys.stderr,
            )
        if extra_train_args and extra_train_args != state.get("extra_train_args", []):
            print(
                f"[batch] warning: ignoring extra train flags {extra_train_args}; "
                f"using saved flags {state.get('extra_train_args', [])}",
                file=sys.stderr,
            )
        print(
            f"[batch] resuming study {prefix!r} "
            f"(modes={state['modes']}, seeds={state['seeds']})",
            flush=True,
        )
    else:
        if paused:
            raise SystemExit(
                "WARNING: A previous batch study is still in progress or paused. "
                "Cannot start a new batch study until it is resumed and finished.\n"
                f"{_format_paused_list(paused)}\n"
                "Specify `python batch_study.py --resume --prefix <prefix>` to continue."
            )
        prefix = validate_prefix(args.prefix)
        if not args.seeds:
            raise SystemExit("--seeds is required when starting a new batch study (e.g. --seeds 42,43,44,45,46)")
        seeds = parse_seeds(args.seeds)
        excluded = _excluded_from_args(args)
        modes = resolve_modes(excluded)
        study_dir = batch_root / prefix
        if study_dir.exists() and (study_dir / STUDY_STATE_FILE).exists():
            raise SystemExit(
                f"Batch study directory {study_dir} already exists. "
                "Choose a different --prefix so existing results are not overwritten."
            )
        state = new_study_state(
            prefix=prefix,
            seeds=seeds,
            modes=modes,
            excluded_modes=sorted(excluded),
            split=args.split,
            eval_batch_size=args.eval_batch_size,
            eval_limit=args.eval_limit,
            no_bootstrap=args.no_bootstrap,
            snap=args.snap,
            extra_train_args=extra_train_args,
        )
        save_study_state(study_dir, state)
        print(
            f"[batch] starting study {prefix!r}: {len(state['runs'])} runs "
            f"({len(modes)} modes x {len(seeds)} seeds) -> {study_dir}",
            flush=True,
        )

    runs_root = study_dir / "runs"
    results_root = study_dir / "results"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    state["status"] = "in_progress"
    state["paused_at"] = None
    save_study_state(study_dir, state)

    total_runs = len(state["runs"])
    for idx, entry in enumerate(state["runs"], start=1):
        run_id = entry["run_id"]
        mode = entry["mode"]
        seed = int(entry["seed"])
        round_num = int(entry["round"])
        run_dir = runs_root / run_id
        csv_path = results_root / f"{run_id}.csv"
        json_path = results_root / f"{run_id}.json"

        if entry["status"] == "completed" and csv_path.exists():
            print(f"[batch] [{idx}/{total_runs}] skipping completed {run_id}", flush=True)
            continue
        if entry["status"] == "completed" and not csv_path.exists():
            entry["status"] = "trained"

        # --- training phase -----------------------------------------------------------
        if entry["status"] in ("pending", "training"):
            resuming_run = (run_dir / RESUME_META).exists()
            action = "resuming training" if resuming_run else "training"
            print(
                f"\n[batch] [{idx}/{total_runs}] {action} {run_id} "
                f"(round {round_num}, mode {mode}, seed {seed})",
                flush=True,
            )
            entry["status"] = "training"
            entry["last_error"] = None
            state["current_run_id"] = run_id
            save_study_state(study_dir, state)

            cmd = _train_command(
                train_script, run_id, run_dir, mode, seed, state.get("extra_train_args", [])
            )
            rc, interrupted = _run_subprocess_interruptible(cmd)

            if _training_completed_cleanly(run_dir, rc):
                entry["status"] = "trained"
                entry["resumable"] = False
                save_study_state(study_dir, state)
                if interrupted:
                    _pause_and_exit(
                        study_dir,
                        state,
                        run_id,
                        f"interrupted after training finished for {run_id}",
                        code=130,
                    )
            elif interrupted or rc == 130 or (run_dir / RESUME_META).exists():
                entry["status"] = "training"
                entry["resumable"] = (run_dir / RESUME_META).exists()
                _pause_and_exit(
                    study_dir,
                    state,
                    run_id,
                    f"interrupted during training of {run_id}",
                    code=130,
                )
            else:
                entry["status"] = "training"
                entry["resumable"] = (run_dir / RESUME_META).exists()
                entry["last_error"] = f"train.py exited with code {rc}"
                _pause_and_exit(
                    study_dir,
                    state,
                    run_id,
                    f"train.py failed for {run_id} with exit code {rc}",
                    code=rc or 1,
                )

        # --- evaluation phase ---------------------------------------------------------
        if entry["status"] in ("trained", "evaluating"):
            print(
                f"\n[batch] [{idx}/{total_runs}] evaluating {run_id} "
                f"(split={state['split']})",
                flush=True,
            )
            entry["status"] = "evaluating"
            entry["last_error"] = None
            state["current_run_id"] = run_id
            save_study_state(study_dir, state)

            cmd = _eval_command(eval_script, run_dir, csv_path, json_path, state)
            rc, interrupted = _run_subprocess_interruptible(cmd)

            eval_partial = False
            if json_path.exists():
                try:
                    eval_partial = bool(
                        json.loads(json_path.read_text(encoding="utf-8")).get("partial", False)
                    )
                except json.JSONDecodeError:
                    eval_partial = True

            if rc == 0 and not interrupted and not eval_partial and csv_path.exists():
                entry["status"] = "completed"
                entry["csv_path"] = str(csv_path.relative_to(study_dir))
                entry["json_path"] = (
                    str(json_path.relative_to(study_dir)) if json_path.exists() else None
                )
                entry["last_error"] = None
                save_study_state(study_dir, state)
            else:
                # Never keep partial evaluation outputs as study results.
                csv_path.unlink(missing_ok=True)
                json_path.unlink(missing_ok=True)
                entry["status"] = "trained"
                if interrupted or rc == 130 or eval_partial:
                    _pause_and_exit(
                        study_dir,
                        state,
                        run_id,
                        f"interrupted during evaluation of {run_id}",
                        code=130,
                    )
                entry["last_error"] = f"evaluate.py exited with code {rc}"
                _pause_and_exit(
                    study_dir,
                    state,
                    run_id,
                    f"evaluate.py failed for {run_id} with exit code {rc}",
                    code=rc or 1,
                )

    # --- final aggregation ------------------------------------------------------------
    csv_summary, txt_summary = write_study_summaries(study_dir, state)
    state["status"] = "completed"
    state["completed_at"] = datetime.now().isoformat(timespec="seconds")
    state["current_run_id"] = None
    state["summary_csv"] = str(csv_summary.relative_to(study_dir))
    state["summary_txt"] = str(txt_summary.relative_to(study_dir))
    save_study_state(study_dir, state)

    print("\n" + txt_summary.read_text(encoding="utf-8"), flush=True)
    print(f"[batch] summary CSV:  {csv_summary}", flush=True)
    print(f"[batch] summary text: {txt_summary}", flush=True)
    return study_dir


def main() -> None:
    run_batch_study()


if __name__ == "__main__":
    main()
