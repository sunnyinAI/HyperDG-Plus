#!/usr/bin/env python3
"""Generate the reviewer-requested shared-curvature and standard-ERM grid.

The generated study has two new conditions:

* ``shared_learnable``: the full HyperDG objective with one learned curvature
  scalar shared by all source domains.
* ``standard_erm``: the unmodified DomainBed ERM implementation, using the
  same data splits, ResNet-50, augmentation, learning rate, batch size, number
  of updates, checkpoint cadence, and model-selection inputs.

There are 4 datasets x 4 targets x 4 seeds x 2 conditions = 128 configs.
Legacy seed-0 shared-curvature runs are not reused by default because twelve
were produced under an older Torch/Torchvision environment.  An explicit
``--allow-legacy-reuse`` escape hatch exists only for exploratory reconstruction
and must not be used for the confirmatory reviewer response.  No standard-ERM
accuracy run is reusable because the prior "ERM" arm went through HyperDG's
zero-penalty code path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
CONFIG_ROOT = STUDY_ROOT / "configs" / "reviewer_round2"
MANIFEST_ROOT = STUDY_ROOT / "manifests"
EXPECTED_LAST_STEP = 4999
EXPECTED_STEPS = tuple(range(200, 5000, 200)) + (EXPECTED_LAST_STEP,)
REQUIRED_EVAL_KEYS = tuple(
    f"env{env}_{split}_acc"
    for env in range(4)
    for split in ("in", "out")
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    config: str
    data_dir: str
    runtime_weight: float


DATASETS = (
    DatasetSpec("pacs", "configs/datasets/pacs.yaml", "data", 1.00),
    DatasetSpec("vlcs", "configs/datasets/vlcs.yaml", "data", 1.10),
    DatasetSpec(
        "officehome",
        "configs/datasets/officehome.yaml",
        "data/officehome_clean",
        1.20,
    ),
    DatasetSpec(
        "terraincognita",
        "configs/datasets/terraincognita.yaml",
        "data",
        1.15,
    ),
)
SEEDS = range(4)
TARGETS = range(4)
CONDITIONS = ("shared_learnable", "standard_erm")


COMMON_HPARAMS: dict[str, Any] = {
    "lr": 5.0e-5,
    "weight_decay": 0.0,
    "nonlinear_classifier": False,
    # Freeze DomainBed defaults explicitly.  In this checkout,
    # resnet50_augmix=True routes to timm's resnet50.ram_in1k.
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

SHARED_HPARAMS: dict[str, Any] = {
    **COMMON_HPARAMS,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Number of balanced queue manifests to create.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help=(
            "Physical GPU ids to use, for example --gpu-ids 2 3. "
            "Overrides the default range implied by --gpus."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "Number of disjoint queue workers per physical GPU. "
            "Use 2 only when memory and thermals permit."
        ),
    )
    parser.add_argument(
        "--allow-legacy-reuse",
        action="store_true",
        help=(
            "Exploratory only: reuse complete legacy seed-0 shared runs even "
            "though their software environment differs."
        ),
    )
    return parser.parse_args()


def output_dir(condition: str, seed: int, dataset: str, target: int) -> Path:
    return (
        STUDY_ROOT
        / "experiments"
        / "reviewer_round2"
        / condition
        / f"seed{seed}"
        / f"{dataset}_split{target}"
    )


def config_path(condition: str, seed: int, dataset: str, target: int) -> Path:
    return (
        CONFIG_ROOT
        / condition
        / f"seed{seed}"
        / f"{dataset}_split{target}.yaml"
    )


def legacy_shared_dir(seed: int, dataset: str, target: int) -> Path | None:
    if seed != 0:
        return None
    return (
        REPO_ROOT
        / "experiments"
        / "reviewer_hQnG"
        / "single_curv"
        / f"{dataset}_split{target}"
    )


def metrics_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    rows: list[dict[str, Any]] = []
    try:
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return False
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


def is_complete(run_dir: Path) -> bool:
    return (
        metrics_complete(run_dir / "metrics.jsonl")
        and (run_dir / "model_last.pt").is_file()
    )


def make_config(
    condition: str,
    seed: int,
    dataset: DatasetSpec,
    target: int,
) -> dict[str, Any]:
    if condition == "shared_learnable":
        algorithm = "hyperdg"
        hparams = dict(SHARED_HPARAMS)
    elif condition == "standard_erm":
        algorithm = "ERM"
        hparams = dict(COMMON_HPARAMS)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    run_output = output_dir(condition, seed, dataset.key, target)
    return {
        "experiment_name": (
            f"reviewer_round2_{condition}_{dataset.key}_seed{seed}_split{target}"
        ),
        "seed": seed,
        "output_dir": str(run_output.relative_to(REPO_ROOT)),
        "device": "cuda",
        "eval_on_start": False,
        "dataset": {
            "config": dataset.config,
            "data_dir": dataset.data_dir,
            "test_envs": [target],
        },
        "algorithm": algorithm,
        "hparams": hparams,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_smoke_configs() -> None:
    pacs = DATASETS[0]
    for condition in CONDITIONS:
        payload = make_config(condition, 97, pacs, 0)
        payload["experiment_name"] = f"reviewer_round2_smoke_{condition}"
        payload["output_dir"] = (
            f"HyperDG2/experiments/smoke/{condition}"
        )
        payload["hparams"]["n_steps"] = 2
        payload["hparams"]["checkpoint_freq"] = 1
        payload["hparams"]["batch_size"] = 4
        write_yaml(STUDY_ROOT / "configs" / "smoke" / f"{condition}.yaml", payload)


def main() -> None:
    args = parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be positive")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be positive")
    gpu_ids = args.gpu_ids if args.gpu_ids is not None else list(range(args.gpus))
    if not gpu_ids or any(gpu < 0 for gpu in gpu_ids):
        raise SystemExit("--gpu-ids must contain non-negative ids")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("--gpu-ids must be unique")
    workers = [
        (gpu, worker)
        for gpu in gpu_ids
        for worker in range(args.workers_per_gpu)
    ]

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, str]] = []
    pending: list[tuple[float, Path]] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            for dataset in DATASETS:
                for target in TARGETS:
                    cfg_path = config_path(condition, seed, dataset.key, target)
                    write_yaml(
                        cfg_path,
                        make_config(condition, seed, dataset, target),
                    )
                    planned_dir = output_dir(condition, seed, dataset.key, target)
                    evidence_dir = planned_dir
                    status = "pending"

                    if is_complete(planned_dir):
                        status = "complete_local"
                    elif condition == "shared_learnable" and args.allow_legacy_reuse:
                        legacy_dir = legacy_shared_dir(seed, dataset.key, target)
                        if legacy_dir is not None and is_complete(legacy_dir):
                            evidence_dir = legacy_dir
                            status = "reuse_legacy"

                    if status == "pending":
                        pending.append((dataset.runtime_weight, cfg_path))

                    inventory.append(
                        {
                            "condition": condition,
                            "dataset": dataset.key,
                            "seed": str(seed),
                            "target": str(target),
                            "config": str(cfg_path.relative_to(REPO_ROOT)),
                            "planned_output": str(planned_dir.relative_to(REPO_ROOT)),
                            "evidence_output": str(evidence_dir.relative_to(REPO_ROOT)),
                            "status": status,
                        }
                    )

    inventory_path = MANIFEST_ROOT / "inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    # Greedy longest-processing-time assignment gives balanced, deterministic
    # queues while keeping each individual training process single-GPU.
    queues: list[list[Path]] = [[] for _ in workers]
    loads = [0.0 for _ in workers]
    for weight, cfg_path in sorted(
        pending, key=lambda item: (-item[0], str(item[1]))
    ):
        worker_index = min(
            range(len(workers)), key=lambda idx: (loads[idx], idx)
        )
        queues[worker_index].append(cfg_path)
        loads[worker_index] += weight

    expected_manifests: set[Path] = set()
    for worker_index, ((gpu, worker), queue) in enumerate(zip(workers, queues)):
        suffix = (
            f"gpu{gpu}.txt"
            if args.workers_per_gpu == 1
            else f"gpu{gpu}_worker{worker}.txt"
        )
        manifest = MANIFEST_ROOT / suffix
        expected_manifests.add(manifest)
        manifest.write_text(
            "".join(f"{path.relative_to(REPO_ROOT)}\n" for path in queue),
            encoding="utf-8",
        )
    for stale_manifest in MANIFEST_ROOT.glob("gpu*.txt"):
        if stale_manifest not in expected_manifests:
            stale_manifest.unlink()

    write_smoke_configs()

    counts: dict[str, int] = {}
    for row in inventory:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Wrote {len(inventory)} configs and {len(workers)} GPU manifests.")
    print("Inventory:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for worker_index, ((gpu, worker), queue) in enumerate(zip(workers, queues)):
        print(
            f"  gpu{gpu}/worker{worker}: {len(queue)} pending configs "
            f"(weight={loads[worker_index]:.2f})"
        )


if __name__ == "__main__":
    main()
