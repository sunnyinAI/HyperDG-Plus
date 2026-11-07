"""Training entry point for SFT Domain Generalization experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from domainbed import datasets as db_datasets
from domainbed import hparams_registry
from domainbed.algorithms import ALGORITHMS as DB_ALGORITHMS
from domainbed.lib import misc as db_misc
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader

from sftdg.algorithms import build_algorithm
from sftdg.utils import load_yaml_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT on DomainBed datasets.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config (e.g. configs/training/resnet50_sft_vlcs.yaml)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device from config (cuda/cpu)"
    )
    return parser.parse_args()


def resolve_dataset_config(root_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = root_cfg.get("dataset", {})
    nested_path = dataset_cfg.get("config")
    if nested_path:
        nested_cfg = load_yaml_config(nested_path)
        merged = {**nested_cfg, **{k: v for k, v in dataset_cfg.items() if k != "config"}}
        return merged
    return dataset_cfg


def default_hparams(algorithm: str, dataset: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Bootstraps hparams from DomainBed registry and applies overrides."""
    base_algo = algorithm if algorithm in DB_ALGORITHMS else "ERM"
    hparams = hparams_registry.default_hparams(base_algo, dataset)
    hparams.update(overrides)
    # DomainBed expects integer batch sizes
    hparams["batch_size"] = int(hparams.get("batch_size", 32))
    return hparams


def prepare_splits(
    dataset, test_envs: Sequence[int], hparams: Dict[str, Any], holdout_fraction: float, seed: int
) -> Tuple[List, List, List]:
    """Prepare DomainBed train/val splits per environment."""
    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset):
        env_split = int(len(env) * holdout_fraction)
        in_split, out_split = db_misc.split_dataset(
            env,
            len(env) - env_split,
            db_misc.seed_hash(seed, env_i),
        )
        if hparams.get("class_balanced", False):
            in_weights = db_misc.make_weights_for_balanced_classes(in_split)
            out_weights = db_misc.make_weights_for_balanced_classes(out_split)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_split, in_weights))
        out_splits.append((out_split, out_weights))

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=weights,
            batch_size=hparams["batch_size"],
            num_workers=dataset.N_WORKERS,
        )
        for env_i, (env, weights) in enumerate(in_splits)
        if env_i not in test_envs
    ]

    eval_loaders = [
        FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS,
        )
        for env, _ in (in_splits + out_splits)
    ]
    eval_weights = [weights for _, weights in (in_splits + out_splits)]
    eval_names = [f"env{idx}_in" for idx in range(len(in_splits))]
    eval_names += [f"env{idx}_out" for idx in range(len(out_splits))]
    return train_loaders, list(zip(eval_names, eval_loaders, eval_weights)), in_splits


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    dataset_cfg = resolve_dataset_config(cfg)

    output_dir = Path(cfg.get("output_dir", "experiments/unnamed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.jsonl"

    device = args.device or cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    seed = cfg.get("seed", 0)
    set_seed(seed)

    dataset_name = dataset_cfg["dataset"]
    test_envs = dataset_cfg.get("test_envs", [])
    holdout_fraction = dataset_cfg.get("holdout_fraction", 0.2)
    hparams = default_hparams(cfg.get("algorithm", "sft"), dataset_name, cfg.get("hparams", {}))

    dataset_class = db_datasets.get_dataset_class(dataset_name)
    dataset = dataset_class(dataset_cfg["data_dir"], test_envs, hparams)
    if "num_workers" in dataset_cfg:
        dataset.N_WORKERS = int(dataset_cfg["num_workers"])

    train_loaders, eval_loaders, _ = prepare_splits(
        dataset, test_envs, hparams, holdout_fraction, seed
    )

    algorithm = build_algorithm(
        cfg.get("algorithm", "sft"),
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    algorithm.to(device)

    train_iterator = zip(*train_loaders)
    n_steps = int(hparams.get("n_steps", dataset.N_STEPS))
    checkpoint_freq = int(hparams.get("checkpoint_freq", dataset.CHECKPOINT_FREQ))
    eval_on_start = bool(cfg.get("eval_on_start", True))

    checkpoint_path = output_dir / "model_last.pt"

    with results_path.open("w", encoding="utf-8") as metrics_f:
        for step in range(n_steps):
            minibatches = [
                (x.to(device), y.to(device)) for x, y in next(train_iterator)
            ]
            metrics = algorithm.update(minibatches)
            metrics["step"] = step

            should_eval = step % checkpoint_freq == 0 or step == n_steps - 1
            if step == 0 and not eval_on_start and n_steps > 1:
                should_eval = False
            if should_eval:
                algorithm.eval()
                for name, loader, weights in eval_loaders:
                    acc = db_misc.accuracy(algorithm, loader, weights, device)
                    metrics[f"{name}_acc"] = acc
                if hasattr(algorithm, "log_curvatures"):
                    with torch.no_grad():
                        curvatures = torch.nn.functional.softplus(
                            algorithm.log_curvatures
                        ) + getattr(algorithm, "curv_eps", 0.0)
                        metrics["curvature_mean"] = curvatures.mean().item()
                        metrics["curvature_std"] = curvatures.std(unbiased=False).item()
                        metrics["curvature_var"] = metrics["curvature_std"] ** 2
                        for d_idx, k_val in enumerate(curvatures.tolist()):
                            metrics[f"curvature_domain_{d_idx}"] = k_val
                algorithm.train()
                print(
                    f"[{step:05d}/{n_steps}] "
                    + " ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != "step")
                )
                metrics_f.write(json.dumps(metrics) + "\n")
                metrics_f.flush()

    torch.save({"state_dict": algorithm.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
