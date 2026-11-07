#!/usr/bin/env python3
"""Create a machine-readable audit of the submitted and new evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

import analyze_review_round2 as analysis


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
OUTPUT = STUDY_ROOT / "outputs" / "reviewer_round2" / "evidence_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_old_primary() -> dict[str, Any]:
    inventory = analysis.load_inventory()
    records: list[dict[str, Any]] = []
    local_flat_values: list[float] = []
    for condition in (
        "matched_control",
        "full_per_domain",
        "frozen",
        "no_variance",
    ):
        for dataset in analysis.DATASETS:
            for seed in range(4):
                for target in range(4):
                    directory = analysis.run_dir(
                        condition, dataset, seed, target, inventory
                    )
                    metrics = analysis.read_metrics(directory / "metrics.jsonl")
                    complete = bool(
                        analysis.metrics_validation_error(metrics) is None
                        and (directory / "model_last.pt").is_file()
                    )
                    records.append(
                        {
                            "condition": condition,
                            "dataset": dataset,
                            "seed": seed,
                            "target": target,
                            "complete": complete,
                            "n_records": len(metrics),
                            "last_step": (
                                int(metrics[-1]["step"]) if metrics else None
                            ),
                        }
                    )
                    local_flat_values.extend(
                        float(row["loss_local_flat"])
                        for row in metrics
                        if "loss_local_flat" in row
                    )
    return {
        "expected_runs": 256,
        "complete_runs": sum(record["complete"] for record in records),
        "runs_with_25_eval_records": sum(
            record["n_records"] == 25 for record in records
        ),
        "records_with_local_flat_metric": len(local_flat_values),
        "max_abs_loss_local_flat": (
            max(map(abs, local_flat_values)) if local_flat_values else None
        ),
        "all_logged_local_flat_zero": bool(
            local_flat_values
            and all(value == 0.0 for value in local_flat_values)
        ),
    }


def resolved_environment(disable_user_site: bool) -> dict[str, Any]:
    """Resolve the packages the training interpreter imports in each mode."""
    probe = (
        "import json, sys, torch, torchvision, timm; "
        "print(json.dumps({"
        "'python': sys.version.split()[0], "
        "'executable': sys.executable, "
        "'torch': {'version': torch.__version__, 'path': torch.__file__}, "
        "'torchvision': {'version': torchvision.__version__, "
        "'path': torchvision.__file__}, "
        "'timm': {'version': timm.__version__, 'path': timm.__file__}"
        "}))"
    )
    command = [sys.executable]
    if disable_user_site:
        command.append("-s")
    command.extend(["-c", probe])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "error": completed.stderr.strip(),
            "returncode": completed.returncode,
        }
    return json.loads(completed.stdout)


def audit_legacy_shared_provenance() -> dict[str, Any]:
    """Record why the old seed-0 shared runs are excluded."""
    log_tokens = {
        "pacs": "pacs",
        "vlcs": "vlcs",
        "officehome": "officehome",
        "terraincognita": "terra",
    }
    records: list[dict[str, Any]] = []
    for dataset, token in log_tokens.items():
        for target in range(4):
            log_path = (
                REPO_ROOT
                / "logs"
                / "reviewer_hQnG"
                / f"single_curv_{token}_split{target}.log"
            )
            run_path = (
                REPO_ROOT
                / "experiments"
                / "reviewer_hQnG"
                / "single_curv"
                / f"{dataset}_split{target}"
            )
            metrics = analysis.read_metrics(run_path / "metrics.jsonl")
            complete = bool(
                analysis.metrics_validation_error(metrics) is None
                and (run_path / "model_last.pt").is_file()
            )
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file()
                else ""
            )
            if "/.local/" in log_text:
                import_root = "user_site"
            elif "/.conda/envs/" in log_text:
                import_root = "conda_environment"
            else:
                import_root = "unknown"
            records.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "complete": complete,
                    "last_step": int(metrics[-1]["step"]) if metrics else None,
                    "torchvision_import_root_from_log": import_root,
                    "log": str(log_path.relative_to(REPO_ROOT)),
                    "run": str(run_path.relative_to(REPO_ROOT)),
                    "confirmatory_inclusion": False,
                }
            )
    return {
        "policy": (
            "exclude all legacy shared-learnable runs and rerun all 64 cells "
            "under one resolved environment"
        ),
        "records": records,
        "complete_by_import_root": {
            root: sum(
                record["complete"]
                and record["torchvision_import_root_from_log"] == root
                for record in records
            )
            for root in ("conda_environment", "user_site", "unknown")
        },
    }


def audit_configs() -> dict[str, Any]:
    zero_config_path = (
        REPO_ROOT / "configs" / "batch2" / "erm" / "pacs_split0.yaml"
    )
    standard_config_path = (
        STUDY_ROOT
        / "configs"
        / "reviewer_round2"
        / "standard_erm"
        / "seed0"
        / "pacs_split0.yaml"
    )
    shared_config_path = (
        STUDY_ROOT
        / "configs"
        / "reviewer_round2"
        / "shared_learnable"
        / "seed0"
        / "pacs_split0.yaml"
    )
    zero = yaml.safe_load(zero_config_path.read_text(encoding="utf-8"))
    standard = yaml.safe_load(standard_config_path.read_text(encoding="utf-8"))
    shared = yaml.safe_load(shared_config_path.read_text(encoding="utf-8"))
    return {
        "zero_penalty_algorithm": zero["algorithm"],
        "standard_erm_algorithm": standard["algorithm"],
        "shared_algorithm": shared["algorithm"],
        "shared_single_curvature": shared["hparams"]["single_curvature"],
        "shared_curvature_lr": shared["hparams"]["curvature_lr"],
        "explicit_resnet50_augmix": standard["hparams"]["resnet50_augmix"],
        "explicit_data_augmentation": standard["hparams"]["data_augmentation"],
    }


def audit_smoke_checkpoints() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in ("shared_learnable", "standard_erm"):
        path = (
            STUDY_ROOT
            / "experiments"
            / "smoke"
            / condition
            / "model_last.pt"
        )
        if not path.is_file():
            result[condition] = {"checkpoint_present": False}
            continue
        state = torch.load(path, map_location="cpu")["state_dict"]
        curvature = {
            key: list(value.shape)
            for key, value in state.items()
            if "curvature" in key
        }
        result[condition] = {
            "checkpoint_present": True,
            "curvature_state_shapes": curvature,
        }
    return result


def main() -> None:
    code_paths = (
        REPO_ROOT / "scripts" / "train.py",
        REPO_ROOT / "src" / "sftdg" / "algorithms" / "__init__.py",
        REPO_ROOT / "src" / "sftdg" / "algorithms" / "base.py",
        REPO_ROOT / "src" / "sftdg" / "algorithms" / "registry.py",
        REPO_ROOT / "src" / "sftdg" / "algorithms" / "hyperdg.py",
        REPO_ROOT / "src" / "sftdg" / "models" / "__init__.py",
        REPO_ROOT / "src" / "sftdg" / "models" / "backbones.py",
        REPO_ROOT / "src" / "sftdg" / "utils" / "__init__.py",
        REPO_ROOT / "src" / "sftdg" / "utils" / "misc.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "algorithms.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "datasets.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "hparams_registry.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "networks.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "lib"
        / "misc.py",
        REPO_ROOT
        / "third_party"
        / "domainbed"
        / "domainbed"
        / "lib"
        / "fast_data_loader.py",
    )
    analysis_paths = (
        STUDY_ROOT / "scripts" / "generate_review_round2.py",
        STUDY_ROOT / "scripts" / "run_manifest.py",
        STUDY_ROOT / "scripts" / "validate_review_round2.py",
        STUDY_ROOT / "scripts" / "analyze_review_round2.py",
        STUDY_ROOT / "scripts" / "plot_review_round2.py",
        STUDY_ROOT / "scripts" / "render_review_results.py",
        STUDY_ROOT / "scripts" / "deep_validate_review_round2.py",
        STUDY_ROOT / "scripts" / "capture_review_provenance.py",
        STUDY_ROOT / "scripts" / "audit_existing_evidence.py",
        STUDY_ROOT / "tests" / "test_review_round2.py",
        STUDY_ROOT / "tests" / "test_render_review_results.py",
    )
    frozen_protocol_paths = (
        STUDY_ROOT / "EXPERIMENT_PROTOCOL.md",
        STUDY_ROOT / "README.md",
    )
    payload = {
        "old_primary": audit_old_primary(),
        "condition_routing": audit_configs(),
        "smoke_checkpoints": audit_smoke_checkpoints(),
        "resolved_environment": {
            "normal_training_mode": resolved_environment(False),
            "user_site_disabled_reference": resolved_environment(True),
        },
        "excluded_legacy_shared": audit_legacy_shared_provenance(),
        "frozen_code_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path) for path in code_paths
        },
        "analysis_pipeline_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in analysis_paths
        },
        "frozen_protocol_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in frozen_protocol_paths
        },
        "inventory_sha256": sha256(STUDY_ROOT / "manifests" / "inventory.csv"),
        "implementation_notes": {
            "norm_proxy": (
                "Var(||h/sqrt(c)||) * c algebraically cancels c away from "
                "clamp effects"
            ),
            "evaluation": (
                "HyperDG trains supervised logits on h/sqrt(c) but predict() "
                "evaluates classifier(h)"
            ),
            "backbone": (
                "DomainBed default resnet50_augmix=True routes to timm "
                "resnet50.ram_in1k after constructing torchvision ResNet-50"
            ),
            "source_validation": (
                "source out-splits inherit stochastic source augmentation; "
                "target accuracy is excluded from the selection score"
            ),
            "whole_package_utility": (
                "HyperDG and canonical ERM differ in more than curvature "
                "terms: HyperDG forwards source domains separately through "
                "batch normalization and its train-accuracy diagnostic makes "
                "a second no-grad train-mode forward. Full-versus-ERM is "
                "therefore a released-package utility contrast, not a "
                "curvature-only causal contrast."
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
