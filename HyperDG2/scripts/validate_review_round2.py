#!/usr/bin/env python3
"""Preflight validation for the reviewer-round experiment grid."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
INVENTORY = STUDY_ROOT / "manifests" / "inventory.csv"
EXPECTED_LAST_STEP = 4999
EXPECTED_STEPS = tuple(range(200, 5000, 200)) + (EXPECTED_LAST_STEP,)
REQUIRED_EVAL_KEYS = tuple(
    f"env{env}_{split}_acc"
    for env in range(4)
    for split in ("in", "out")
)

EXPECTED_COMMON = {
    "lr": 5.0e-5,
    "weight_decay": 0.0,
    "nonlinear_classifier": False,
    "data_augmentation": True,
    "resnet18": False,
    "resnet50_augmix": True,
    "class_balanced": False,
    "freeze_bn": False,
    "resnet_dropout": 0.0,
    "checkpoint_freq": 200,
    "n_steps": 5000,
    "batch_size": 32,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def last_step(metrics_path: Path) -> int | None:
    if not metrics_path.is_file():
        return None
    last: dict[str, Any] | None = None
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return int(last["step"]) if last is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        steps = tuple(int(row["step"]) for row in rows)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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


def main() -> None:
    errors: list[str] = []
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 128:
        errors.append(f"Expected 128 inventory rows, found {len(rows)}")

    identities: set[tuple[str, str, str, str]] = set()
    outputs: set[str] = set()
    counts: dict[str, int] = {}

    for row in rows:
        identity = (
            row["condition"],
            row["dataset"],
            row["seed"],
            row["target"],
        )
        if identity in identities:
            errors.append(f"Duplicate identity: {identity}")
        identities.add(identity)
        if row["planned_output"] in outputs:
            errors.append(f"Duplicate output: {row['planned_output']}")
        outputs.add(row["planned_output"])
        counts[row["status"]] = counts.get(row["status"], 0) + 1

        config_path = REPO_ROOT / row["config"]
        if not config_path.is_file():
            errors.append(f"Missing config: {row['config']}")
            continue
        config = load_yaml(config_path)
        condition = row["condition"]
        expected_algorithm = "hyperdg" if condition == "shared_learnable" else "ERM"
        if config.get("algorithm") != expected_algorithm:
            errors.append(
                f"{row['config']}: algorithm={config.get('algorithm')}, "
                f"expected {expected_algorithm}"
            )
        if int(config.get("seed", -1)) != int(row["seed"]):
            errors.append(f"{row['config']}: seed mismatch")
        if config.get("dataset", {}).get("test_envs") != [int(row["target"])]:
            errors.append(f"{row['config']}: target mismatch")
        if config.get("output_dir") != row["planned_output"]:
            errors.append(f"{row['config']}: output_dir mismatch")
        if config.get("eval_on_start") is not False:
            errors.append(f"{row['config']}: eval_on_start must be false")
        for key, value in EXPECTED_COMMON.items():
            actual = config.get("hparams", {}).get(key)
            if actual != value:
                errors.append(
                    f"{row['config']}: hparams.{key}={actual!r}, expected {value!r}"
                )
        if condition == "shared_learnable":
            hparams = config["hparams"]
            if hparams.get("single_curvature") is not True:
                errors.append(f"{row['config']}: shared flag is not true")
            if float(hparams.get("curvature_lr", 0.0)) <= 0:
                errors.append(f"{row['config']}: curvature is not learnable")
            expected_shared = {
                "curvature_lr": 1.0e-4,
                "grad_clip_norm": 5.0,
                "alpha_curv": 0.2,
                "beta_align": 0.5,
                "gamma_sharp": 0.05,
                "delta_local_flat": 0.1,
                "delta_global_flat": 0.1,
                "curvature_epsilon": 1.0e-4,
                "single_curvature": True,
            }
            for key, value in expected_shared.items():
                if hparams.get(key) != value:
                    errors.append(
                        f"{row['config']}: hparams.{key}="
                        f"{hparams.get(key)!r}, expected {value!r}"
                    )
        elif "single_curvature" in config["hparams"]:
            errors.append(f"{row['config']}: ERM should not receive curvature flags")

        dataset_config = REPO_ROOT / config["dataset"]["config"]
        if not dataset_config.is_file():
            errors.append(f"{row['config']}: missing nested dataset config")
        expected_dataset = {
            "pacs": ("configs/datasets/pacs.yaml", "data"),
            "vlcs": ("configs/datasets/vlcs.yaml", "data"),
            "officehome": (
                "configs/datasets/officehome.yaml",
                "data/officehome_clean",
            ),
            "terraincognita": (
                "configs/datasets/terraincognita.yaml",
                "data",
            ),
        }[row["dataset"]]
        if config["dataset"].get("config") != expected_dataset[0]:
            errors.append(f"{row['config']}: nested dataset config drift")
        if config["dataset"].get("data_dir") != expected_dataset[1]:
            errors.append(f"{row['config']}: data_dir drift")

        if row["status"] in {"complete_local", "reuse_legacy"}:
            evidence = REPO_ROOT / row["evidence_output"]
            if not metrics_complete(evidence / "metrics.jsonl"):
                errors.append(
                    f"{row['evidence_output']}: marked complete but metrics "
                    "schedule/keys are invalid"
                )
            if not (evidence / "model_last.pt").is_file():
                errors.append(
                    f"{row['evidence_output']}: marked complete but checkpoint missing"
                )

    manifest_configs: list[str] = []
    for manifest in sorted((STUDY_ROOT / "manifests").glob("gpu*.txt")):
        manifest_configs.extend(
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    pending_configs = sorted(
        row["config"] for row in rows if row["status"] == "pending"
    )
    if sorted(manifest_configs) != pending_configs:
        errors.append(
            "GPU manifests do not contain exactly the pending configurations"
        )

    report = {
        "inventory_rows": len(rows),
        "unique_identities": len(identities),
        "unique_outputs": len(outputs),
        "status_counts": counts,
        "queued_configs": len(manifest_configs),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
