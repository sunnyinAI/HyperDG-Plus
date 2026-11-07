#!/usr/bin/env python3
"""Capture reproducibility provenance for the reviewer-round study."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
OUTPUT_ROOT = STUDY_ROOT / "outputs" / "reviewer_round2"
OUTPUT = OUTPUT_ROOT / "runtime_provenance.json"
DATASET_MANIFEST = OUTPUT_ROOT / "dataset_manifest.jsonl"

DATASET_ROOTS = {
    "pacs": REPO_ROOT / "data" / "PACS",
    "vlcs": REPO_ROOT / "data" / "VLCS",
    "officehome": REPO_ROOT / "data" / "officehome_clean" / "office_home",
    "terraincognita": REPO_ROOT / "data" / "terra_incognita",
}
PRETRAINED_ARTIFACTS = {
    "timm_resnet50_ram_in1k": (
        Path(
            os.environ.get(
                "HF_HUB_CACHE",
                str(Path.home() / ".cache" / "huggingface" / "hub"),
            )
        )
        / "models--timm--resnet50.ram_in1k"
        / "blobs"
        / "2bba9aa7bf8ba172d88f63f85ffac2e5a84671c80c3e73e8aecd42f78721e4d1"
    ),
    "torchvision_resnet50_imagenet1k_v1_constructed_then_replaced": (
        Path(
            os.environ.get(
                "TORCH_HOME",
                str(Path.home() / ".cache" / "torch"),
            )
        )
        / "hub"
        / "checkpoints"
        / "resnet50-0676ba61.pth"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_record(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {
        "version": getattr(module, "__version__", None),
        "path": getattr(module, "__file__", None),
    }


def runtime_environment() -> dict[str, Any]:
    packages = {
        name: package_record(name)
        for name in (
            "torch",
            "torchvision",
            "timm",
            "numpy",
            "pandas",
            "scipy",
            "yaml",
            "matplotlib",
        )
    }
    torch = importlib.import_module("torch")
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "user_site_enabled": bool(getattr(sys.flags, "no_user_site", 0) == 0),
        "packages": packages,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
    }


def hash_group(paths: list[Path]) -> dict[str, str]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing provenance inputs: " + ", ".join(map(str, missing))
        )
    return {
        str(path.relative_to(REPO_ROOT)): sha256(path)
        for path in sorted(paths)
    }


def source_hashes() -> dict[str, dict[str, str]]:
    training_paths = [
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
    ]
    study_paths = sorted((STUDY_ROOT / "scripts").glob("*.py"))
    study_paths += sorted((STUDY_ROOT / "tests").glob("*.py"))
    protocol_paths = [
        STUDY_ROOT / "EXPERIMENT_PROTOCOL.md",
        STUDY_ROOT / "IMPLEMENTATION_AND_EVIDENCE_AUDIT.md",
        STUDY_ROOT / "REVIEW_RESPONSE_DRAFT.md",
        STUDY_ROOT / "README.md",
    ]
    dataset_configs = [
        REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
        for dataset in DATASET_ROOTS
    ]
    generated_configs = sorted(
        (STUDY_ROOT / "configs" / "reviewer_round2").rglob("*.yaml")
    )
    manifests = sorted((STUDY_ROOT / "manifests").glob("*"))
    manifests = [path for path in manifests if path.is_file()]
    return {
        "training_code": hash_group(training_paths),
        "study_code_and_tests": hash_group(study_paths),
        "frozen_protocol_and_response": hash_group(protocol_paths),
        "dataset_configs": hash_group(dataset_configs),
        "generated_run_configs": hash_group(generated_configs),
        "manifests": hash_group(manifests),
    }


def iter_dataset_files(root: Path):
    """Yield logical paths while following the intentional domain symlinks."""
    for directory, directory_names, file_names in os.walk(
        root, followlinks=True
    ):
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        for name in file_names:
            path = base / name
            if path.is_file():
                yield path


def dataset_manifest() -> dict[str, Any]:
    DATASET_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    manifest_digest = hashlib.sha256()
    with DATASET_MANIFEST.open("w", encoding="utf-8") as handle:
        for dataset, root in DATASET_ROOTS.items():
            count = 0
            total_bytes = 0
            domain_counts: dict[str, int] = {}
            top_level_links = {
                child.name: os.readlink(child)
                for child in sorted(root.iterdir())
                if child.is_symlink()
            }
            for path in iter_dataset_files(root):
                logical = path.relative_to(root).as_posix()
                size = path.stat().st_size
                domain = logical.split("/", 1)[0]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                record = {
                    "dataset": dataset,
                    "logical_path": logical,
                    "bytes": size,
                    "resolved_path": str(path.resolve()),
                }
                line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                handle.write(line + "\n")
                manifest_digest.update(line.encode("utf-8"))
                manifest_digest.update(b"\n")
                count += 1
                total_bytes += size
            summaries[dataset] = {
                "logical_root": str(root.relative_to(REPO_ROOT)),
                "resolved_root": str(root.resolve()),
                "file_count": count,
                "total_bytes": total_bytes,
                "files_per_domain": domain_counts,
                "top_level_symlinks": top_level_links,
            }
    return {
        "manifest": str(DATASET_MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest_digest.hexdigest(),
        "manifest_identity_fields": (
            "dataset+logical_path+bytes+resolved_path"
        ),
        "datasets": summaries,
    }


def pretrained_artifacts() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in PRETRAINED_ARTIFACTS.items():
        records[name] = {
            "path": str(path),
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
    return records


def gpu_snapshot() -> dict[str, Any]:
    query = checked_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,"
            "memory.total,power.limit",
            "--format=csv,noheader",
        ]
    )
    return {
        "inventory_query": query,
        "requested_experiment_gpus": [2, 3],
    }


def git_snapshot() -> dict[str, Any]:
    return {
        "head": checked_command(["git", "rev-parse", "HEAD"]),
        "status": checked_command(["git", "status", "--short"]),
        "hyperdg2_diff_check": checked_command(
            ["git", "diff", "--check", "--", "HyperDG2"]
        ),
    }


def main() -> None:
    payload = {
        "runtime": runtime_environment(),
        "gpu": gpu_snapshot(),
        "git": git_snapshot(),
        "source_sha256": source_hashes(),
        "datasets": dataset_manifest(),
        "pretrained_artifacts": pretrained_artifacts(),
        "backbone_resolution": {
            "hparam_key": "resnet50_augmix",
            "hparam_value": True,
            "effective_timm_model": "resnet50.ram_in1k",
            "note": (
                "The code first constructs torchvision ResNet-50, then "
                "replaces it with the timm model."
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(REPO_ROOT)),
                "dataset_manifest": str(
                    DATASET_MANIFEST.relative_to(REPO_ROOT)
                ),
                "dataset_manifest_sha256": payload["datasets"][
                    "manifest_sha256"
                ],
                "generated_configs": len(
                    payload["source_sha256"]["generated_run_configs"]
                ),
                "gpus_requested": payload["gpu"][
                    "requested_experiment_gpus"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
