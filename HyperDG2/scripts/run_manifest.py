#!/usr/bin/env python3
"""Run one reviewer-round manifest on one GPU, with safe whole-run resume.

Use one live process per GPU.  A configuration is skipped only when its last
logged step is 4999 and ``model_last.pt`` exists.  Incomplete runs are rerun
from the beginning because the parent trainer does not persist optimizer state
at intermediate checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
EXPECTED_LAST_STEP = 4999
EXPECTED_STEPS = tuple(range(200, 5000, 200)) + (EXPECTED_LAST_STEP,)
REQUIRED_EVAL_KEYS = tuple(
    f"env{env}_{split}_acc"
    for env in range(4)
    for split in ("in", "out")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next configuration after a failed run.",
    )
    return parser.parse_args()


@contextmanager
def output_lock(output_dir: Path):
    """Prevent two queue workers from training the same output cell."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".reviewer_round2.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"output is locked by another worker: {output_dir}") from error
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            + "\n"
        )
        lock_handle.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def read_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def read_last_step(metrics_path: Path) -> int | None:
    rows = read_metrics(metrics_path)
    try:
        return int(rows[-1]["step"]) if rows else None
    except (KeyError, TypeError, ValueError):
        return None


def metrics_complete(metrics_path: Path) -> bool:
    rows = read_metrics(metrics_path)
    try:
        steps = tuple(int(row["step"]) for row in rows)
    except (KeyError, TypeError, ValueError):
        return False
    if steps != EXPECTED_STEPS:
        return False
    for row in rows:
        for key in REQUIRED_EVAL_KEYS:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                return False
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return False
    return True


def config_output_dir(config_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    path = Path(config["output_dir"])
    return path if path.is_absolute() else REPO_ROOT / path


def is_complete(output_dir: Path) -> bool:
    return (
        metrics_complete(output_dir / "metrics.jsonl")
        and (output_dir / "model_last.pt").is_file()
    )


def tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def main() -> None:
    args = parse_args()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    config_paths = [
        REPO_ROOT / line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    worker_name = manifest.stem
    log_dir = (
        STUDY_ROOT
        / "logs"
        / "reviewer_round2"
        / f"gpu{args.gpu}"
        / worker_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "queue_summary.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    source_paths = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "third_party" / "domainbed"),
    ]
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{os.pathsep.join(source_paths)}{os.pathsep}{old_pythonpath}"
        if old_pythonpath
        else os.pathsep.join(source_paths)
    )

    completed: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    queue_started = time.time()

    print(
        f"[queue] gpu={args.gpu} manifest={manifest.relative_to(REPO_ROOT)} "
        f"configs={len(config_paths)}",
        flush=True,
    )
    for index, config_path in enumerate(config_paths, start=1):
        output_dir = config_output_dir(config_path)
        label = str(config_path.relative_to(REPO_ROOT))
        if is_complete(output_dir):
            skipped.append(label)
            print(f"[{index}/{len(config_paths)}] skip complete {label}", flush=True)
            continue

        config_root = STUDY_ROOT / "configs" / "reviewer_round2"
        try:
            relative_config = config_path.relative_to(config_root)
        except ValueError:
            relative_config = Path(config_path.name)
        attempt_dir = log_dir / relative_config.with_suffix("")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = attempt_dir / f"attempt_{attempt_stamp}.log"
        started = time.time()
        print(f"[{index}/{len(config_paths)}] start {label}", flush=True)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--device",
            "cuda",
        ]
        try:
            with output_lock(output_dir):
                if is_complete(output_dir):
                    skipped.append(label)
                    print(
                        f"[{index}/{len(config_paths)}] skip complete after lock "
                        f"{label}",
                        flush=True,
                    )
                    continue
                with log_path.open("w", encoding="utf-8") as log_handle:
                    result = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
        except RuntimeError as error:
            failure = {
                "config": label,
                "returncode": None,
                "last_step": read_last_step(output_dir / "metrics.jsonl"),
                "log": str(log_path.relative_to(REPO_ROOT)),
                "error": str(error),
            }
            failed.append(failure)
            print(f"[{index}/{len(config_paths)}] LOCKED {json.dumps(failure)}")
            if not args.keep_going:
                break
            continue
        elapsed_minutes = (time.time() - started) / 60.0

        if result.returncode == 0 and is_complete(output_dir):
            completed.append(label)
            print(
                f"[{index}/{len(config_paths)}] done {label} "
                f"minutes={elapsed_minutes:.1f}",
                flush=True,
            )
        else:
            failure = {
                "config": label,
                "returncode": result.returncode,
                "last_step": read_last_step(output_dir / "metrics.jsonl"),
                "log": str(log_path.relative_to(REPO_ROOT)),
            }
            failed.append(failure)
            print(
                f"[{index}/{len(config_paths)}] FAILED {json.dumps(failure)}",
                flush=True,
            )
            failure_tail = tail(log_path)
            if failure_tail:
                print(failure_tail, flush=True)
            if not args.keep_going:
                break

        summary_path.write_text(
            json.dumps(
                {
                    "gpu": args.gpu,
                    "manifest": str(manifest.relative_to(REPO_ROOT)),
                    "completed": completed,
                    "skipped": skipped,
                    "failed": failed,
                    "elapsed_hours": (time.time() - queue_started) / 3600.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = {
        "gpu": args.gpu,
        "manifest": str(manifest.relative_to(REPO_ROOT)),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "elapsed_hours": (time.time() - queue_started) / 3600.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[queue] finished gpu={args.gpu} completed={len(completed)} "
        f"skipped={len(skipped)} failed={len(failed)} "
        f"hours={summary['elapsed_hours']:.2f}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
