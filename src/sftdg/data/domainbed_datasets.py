"""Dataset plumbing that wraps DomainBed datasets for PyTorch training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from torch.utils.data import DataLoader

try:
    from domainbed import datasets as db_datasets
    from domainbed.lib import misc as db_misc
except ImportError as exc:  # pragma: no cover - placeholder until environment is built
    raise ImportError(
        "DomainBed is required. Install via `pip install git+https://github.com/facebookresearch/DomainBed.git`"
    ) from exc


@dataclass
class DataConfig:
    dataset: str
    data_dir: str
    batch_size: int
    num_workers: int
    holdout_fraction: float
    seed: int


def build_dataloaders(cfg: DataConfig) -> Tuple[List[DataLoader], List[DataLoader], Iterable]:
    """Create train/validation loaders for all domains.

    Returns
    -------
    tuple:
        - list of dataloaders (one per domain) used for Algorithm 1 updates
        - list of validation loaders (out-of-domain splits)
        - in_splits metadata from DomainBed (useful for evaluation hooks)
    """
    dataset_class = db_datasets.get_dataset_class(cfg.dataset)
    dataset = dataset_class(cfg.data_dir, cfg.holdout_fraction)

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        env_split = len(env) - int(len(env) * cfg.holdout_fraction)
        in_split, out_split = db_misc.split_dataset(
            env,
            int(env_split),
            cfg.seed,
        )
        in_splits.append(in_split)
        out_splits.append(out_split)

    train_loaders = [
        DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
        for subset, _ in in_splits
    ]

    val_loaders = [
        DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        for subset, _ in out_splits
    ]

    return train_loaders, val_loaders, in_splits
