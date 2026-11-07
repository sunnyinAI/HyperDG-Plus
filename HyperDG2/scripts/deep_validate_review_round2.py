#!/usr/bin/env python3
"""Deep, fail-closed validation of all reviewer-round run artifacts.

The lightweight queue validator checks schedules while experiments are
running.  This final validator additionally loads every checkpoint, verifies
condition-specific state structure and tensor finiteness, checks the
configuration-to-output identity, and records SHA-256 digests for the config,
metrics, and checkpoint.  It exits successfully only for a complete 384-cell
study unless ``--allow-incomplete`` is explicitly supplied for diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

import analyze_review_round2 as analysis


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
DEFAULT_OUTPUT = (
    STUDY_ROOT / "outputs" / "reviewer_round2" / "deep_validation.json"
)
EXPECTED_CELLS = 384
NUM_CLASSES = {
    "pacs": 7,
    "vlcs": 5,
    "officehome": 65,
    "terraincognita": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing cells for an explicitly diagnostic partial run.",
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Speed up a diagnostic pass; never use for final validation.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path(
    condition: str,
    dataset: str,
    seed: int,
    target: int,
    inventory: dict[tuple[str, str, int, int], dict[str, str]],
) -> Path:
    if condition in {"standard_erm", "shared_learnable"}:
        return REPO_ROOT / inventory[
            (condition, dataset, seed, target)
        ]["config"]

    suffix = f"{dataset}_split{target}.yaml"
    if seed:
        token = {
            "matched_control": "erm",
            "full_per_domain": "baseline",
            "frozen": "fixed",
            "no_variance": "novar",
        }[condition]
        return (
            REPO_ROOT
            / "configs"
            / "batch4"
            / f"seed{seed}"
            / token
            / suffix
        )
    prefix = {
        "matched_control": ("batch2", "erm"),
        "full_per_domain": ("batch3", "baseline"),
        "frozen": ("batch1", "fixed"),
        "no_variance": ("batch1", "novar"),
    }[condition]
    return REPO_ROOT / "configs" / prefix[0] / prefix[1] / suffix


def validate_config(
    path: Path,
    run_dir: Path,
    condition: str,
    dataset: str,
    seed: int,
    target: int,
) -> list[str]:
    errors: list[str] = []
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return [f"cannot read config {path}: {error}"]
    if not isinstance(config, dict):
        return [f"config is not a mapping: {path}"]

    expected_algorithm = (
        "ERM" if condition == "standard_erm" else "hyperdg"
    )
    if config.get("algorithm") != expected_algorithm:
        errors.append(
            f"algorithm={config.get('algorithm')!r}, expected "
            f"{expected_algorithm!r}"
        )
    if int(config.get("seed", -1)) != seed:
        errors.append(f"seed={config.get('seed')!r}, expected {seed}")
    test_envs = config.get("dataset", {}).get("test_envs")
    if test_envs != [target]:
        errors.append(f"test_envs={test_envs!r}, expected [{target}]")
    configured_output = Path(str(config.get("output_dir", "")))
    configured_output = (
        configured_output
        if configured_output.is_absolute()
        else REPO_ROOT / configured_output
    )
    if configured_output.resolve() != run_dir.resolve():
        errors.append(
            f"output_dir resolves to {configured_output}, expected {run_dir}"
        )
    expected_dataset_config = f"configs/datasets/{dataset}.yaml"
    if config.get("dataset", {}).get("config") != expected_dataset_config:
        errors.append(
            "nested dataset config mismatch: "
            f"{config.get('dataset', {}).get('config')!r}"
        )
    hparams = config.get("hparams", {})
    if condition == "standard_erm":
        curvature_keys = {
            key for key in hparams if "curv" in key or key == "single_curvature"
        }
        if curvature_keys:
            errors.append(
                "standard ERM config contains curvature keys: "
                + ",".join(sorted(curvature_keys))
            )
    elif condition == "shared_learnable":
        if hparams.get("single_curvature") is not True:
            errors.append("shared_learnable requires single_curvature=true")
        if float(hparams.get("curvature_lr", 0.0)) <= 0.0:
            errors.append("shared_learnable curvature_lr must be positive")
    else:
        if hparams.get("single_curvature") is not False:
            errors.append(f"{condition} requires single_curvature=false")
        if condition in {"matched_control", "frozen"} and float(
            hparams.get("curvature_lr", math.nan)
        ) != 0.0:
            errors.append(f"{condition} requires curvature_lr=0")
        if condition == "no_variance" and float(
            hparams.get("alpha_curv", math.nan)
        ) != 0.0:
            errors.append("no_variance requires alpha_curv=0")
    return errors


def checkpoint_errors(
    checkpoint_path: Path, condition: str, dataset: str
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    metadata: dict[str, Any] = {}
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as error:  # checkpoint corruption may raise many types
        return [f"checkpoint load failed: {type(error).__name__}: {error}"], {}
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"state_dict"}:
        return [
            "checkpoint must be a mapping containing exactly state_dict"
        ], {}
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not state:
        return ["state_dict must be a non-empty mapping"], {}
    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        errors.append(
            "non-tensor state entries: " + ",".join(non_tensors[:5])
        )
    nonfinite: list[str] = []
    for key, value in state.items():
        if torch.is_tensor(value) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all()):
                nonfinite.append(key)
    if nonfinite:
        errors.append(
            "non-finite checkpoint tensors: " + ",".join(nonfinite[:5])
        )

    curvature = {
        key: list(value.shape)
        for key, value in state.items()
        if "curvature" in key and torch.is_tensor(value)
    }
    metadata["state_entries"] = len(state)
    metadata["curvature_state_shapes"] = curvature
    expected_classes = NUM_CLASSES[dataset]
    classifier_weight = state.get("classifier.weight")
    if not torch.is_tensor(classifier_weight):
        errors.append("missing classifier.weight")
    elif tuple(classifier_weight.shape) != (expected_classes, 2048):
        errors.append(
            "classifier.weight shape "
            f"{tuple(classifier_weight.shape)}, expected "
            f"({expected_classes}, 2048)"
        )
    conv = state.get("featurizer.network.conv1.weight")
    if not torch.is_tensor(conv) or tuple(conv.shape) != (64, 3, 7, 7):
        errors.append("missing or invalid featurizer.network.conv1.weight")

    if condition == "standard_erm":
        if curvature:
            errors.append("standard ERM checkpoint contains curvature state")
        network_classifier = state.get("network.1.weight")
        if not torch.is_tensor(network_classifier):
            errors.append("standard ERM is missing network.1.weight")
        elif torch.is_tensor(classifier_weight) and not torch.equal(
            classifier_weight, network_classifier
        ):
            errors.append("standard ERM classifier aliases disagree")
        network_conv = state.get("network.0.network.conv1.weight")
        if not torch.is_tensor(network_conv):
            errors.append(
                "standard ERM is missing network.0.network.conv1.weight"
            )
        elif torch.is_tensor(conv) and not torch.equal(conv, network_conv):
            errors.append("standard ERM featurizer aliases disagree")
    else:
        expected_local_shape = (
            [1] if condition == "shared_learnable" else [3]
        )
        expected_curvature = {
            "log_curvatures": expected_local_shape,
            "log_global_curvature": [],
        }
        if curvature != expected_curvature:
            errors.append(
                f"curvature state shapes {curvature!r}, expected "
                f"{expected_curvature!r}"
            )
        local = state.get("log_curvatures")
        global_value = state.get("log_global_curvature")
        if torch.is_tensor(local) and torch.is_tensor(global_value):
            epsilon = 1.0e-4
            metadata["final_local_curvatures"] = (
                torch.nn.functional.softplus(local).add(epsilon).tolist()
            )
            metadata["final_global_curvature"] = float(
                torch.nn.functional.softplus(global_value).add(epsilon)
            )
    return errors, metadata


def main() -> None:
    args = parse_args()
    inventory = analysis.load_inventory()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    missing: list[dict[str, Any]] = []

    for condition in analysis.CONDITIONS:
        for dataset in analysis.DATASETS:
            for seed in range(4):
                for target in range(4):
                    identity = {
                        "condition": condition,
                        "dataset": dataset,
                        "seed": seed,
                        "target": target,
                    }
                    try:
                        directory = analysis.run_dir(
                            condition, dataset, seed, target, inventory
                        )
                        config = config_path(
                            condition, dataset, seed, target, inventory
                        )
                    except (KeyError, ValueError) as error:
                        missing.append(
                            {**identity, "reason": f"routing_error:{error}"}
                        )
                        continue
                    metrics_path = directory / "metrics.jsonl"
                    checkpoint_path = directory / "model_last.pt"
                    metric_rows = analysis.read_metrics(metrics_path)
                    metric_error = analysis.metrics_validation_error(metric_rows)
                    if metric_error is not None or not checkpoint_path.is_file():
                        missing.append(
                            {
                                **identity,
                                "run_dir": str(
                                    directory.relative_to(REPO_ROOT)
                                ),
                                "reason": (
                                    metric_error
                                    if metric_error is not None
                                    else "missing_checkpoint"
                                ),
                            }
                        )
                        continue

                    cell_errors = validate_config(
                        config,
                        directory,
                        condition,
                        dataset,
                        seed,
                        target,
                    )
                    checkpoint_cell_errors, metadata = checkpoint_errors(
                        checkpoint_path, condition, dataset
                    )
                    cell_errors.extend(checkpoint_cell_errors)
                    record = {
                        **identity,
                        "run_dir": str(directory.relative_to(REPO_ROOT)),
                        "config": str(config.relative_to(REPO_ROOT)),
                        "config_sha256": sha256(config),
                        "metrics_sha256": sha256(metrics_path),
                        "checkpoint_sha256": (
                            None
                            if args.skip_checkpoint_hash
                            else sha256(checkpoint_path)
                        ),
                        "checkpoint_bytes": checkpoint_path.stat().st_size,
                        "n_metric_records": len(metric_rows),
                        "last_step": int(metric_rows[-1]["step"]),
                        **metadata,
                        "errors": cell_errors,
                    }
                    records.append(record)
                    errors.extend(
                        [
                            f"{condition}/{dataset}/seed{seed}/target{target}: "
                            f"{error}"
                            for error in cell_errors
                        ]
                    )

    complete = len(records)
    if not args.allow_incomplete and complete != EXPECTED_CELLS:
        errors.append(
            f"complete validated cells={complete}, expected {EXPECTED_CELLS}"
        )
    if not args.allow_incomplete and args.skip_checkpoint_hash:
        errors.append("final validation may not skip checkpoint SHA-256")
    report = {
        "contract": {
            "expected_cells": EXPECTED_CELLS,
            "checkpoint_hash_required_for_final": True,
            "metrics_schedule": list(analysis.EXPECTED_STEPS),
            "required_eval_keys": list(analysis.REQUIRED_EVAL_KEYS),
        },
        "complete_cells": complete,
        "missing_cells": len(missing),
        "error_count": len(errors),
        "all_checkpoints_loaded": complete == EXPECTED_CELLS,
        "all_artifact_hashes_recorded": bool(
            complete == EXPECTED_CELLS
            and not args.skip_checkpoint_hash
            and all(record["checkpoint_sha256"] for record in records)
        ),
        "missing": missing,
        "errors": errors,
        "records": records,
    }
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "complete_cells",
                    "missing_cells",
                    "error_count",
                    "all_checkpoints_loaded",
                    "all_artifact_hashes_recorded",
                )
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
