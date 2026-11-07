#!/usr/bin/env python3
"""Domain-clustered analysis for the second TMLR reviewer round.

The inferential unit is the held-out target domain, not a seed-by-domain run.
For a contrast A-B, seed-paired differences are first averaged within target:

    d_domain = mean_seed(accuracy_A - accuracy_B)

The dataset estimand is the equal-target mean of the four domain effects.
Uncertainty is reported using a t interval with 3 degrees of freedom, an exact
two-sided sign-flip test over the four target-domain effects, and an exhaustive
domain bootstrap as a sensitivity analysis.

Source-validation selection maximizes mean source-domain ``env*_out_acc``.
The target environment is explicitly excluded from that score.  Ties select
the earliest checkpoint.  Target accuracy is read only after the selected step
has been fixed by source validation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
OUTPUT_ROOT = STUDY_ROOT / "outputs" / "reviewer_round2"
INVENTORY = STUDY_ROOT / "manifests" / "inventory.csv"
EXPECTED_LAST_STEP = 4999
EXPECTED_STEPS = tuple(range(200, 5000, 200)) + (EXPECTED_LAST_STEP,)
REQUIRED_EVAL_KEYS = tuple(
    f"env{env}_{split}_acc"
    for env in range(4)
    for split in ("in", "out")
)

DATASETS = ("pacs", "vlcs", "officehome", "terraincognita")
TARGET_NAMES = {
    "pacs": ("ArtPainting", "Cartoon", "Photo", "Sketch"),
    "vlcs": ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
    "officehome": ("Art", "Clipart", "Product", "RealWorld"),
    "terraincognita": ("L100", "L38", "L43", "L46"),
}
CONDITIONS = (
    "standard_erm",
    "matched_control",
    "full_per_domain",
    "frozen",
    "no_variance",
    "shared_learnable",
)
ENDPOINTS = ("source_selected", "final_step")


@dataclass(frozen=True)
class Contrast:
    key: str
    treatment: str
    control: str
    role: str


CONTRASTS = (
    Contrast(
        "full_minus_standard_erm",
        "full_per_domain",
        "standard_erm",
        "primary_utility",
    ),
    Contrast(
        "no_variance_minus_shared",
        "no_variance",
        "shared_learnable",
        "primary_domain_specificity",
    ),
    Contrast(
        "full_minus_matched_control",
        "full_per_domain",
        "matched_control",
        "secondary_auxiliary_objective",
    ),
    Contrast(
        "full_minus_no_variance",
        "full_per_domain",
        "no_variance",
        "secondary_variance_penalty",
    ),
    Contrast(
        "full_minus_frozen",
        "full_per_domain",
        "frozen",
        "secondary_learning_package",
    ),
    Contrast(
        "shared_minus_standard_erm",
        "shared_learnable",
        "standard_erm",
        "secondary_shared_utility",
    ),
    Contrast(
        "matched_control_minus_standard_erm",
        "matched_control",
        "standard_erm",
        "protocol_sanity",
    ),
    Contrast(
        "full_minus_shared",
        "full_per_domain",
        "shared_learnable",
        "secondary_parameterization",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sesoi",
        type=float,
        default=1.0,
        help="Primary symmetric equivalence margin in percentage points.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial descriptive outputs instead of exiting nonzero.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Analyze completed metric logs without requiring checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Write analysis files here (default: the study output directory).",
    )
    return parser.parse_args()


def load_inventory() -> dict[tuple[str, str, int, int], dict[str, str]]:
    if not INVENTORY.is_file():
        return {}
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["condition"],
            row["dataset"],
            int(row["seed"]),
            int(row["target"]),
        ): row
        for row in rows
    }


def old_run_dir(condition: str, dataset: str, seed: int, target: int) -> Path:
    suffix = f"{dataset}_split{target}"
    if condition == "matched_control":
        if seed == 0:
            return REPO_ROOT / "experiments" / "batch2" / "erm" / suffix
        return (
            REPO_ROOT
            / "experiments"
            / "batch4"
            / f"seed{seed}"
            / "erm"
            / suffix
        )
    if condition == "full_per_domain":
        if seed == 0:
            return REPO_ROOT / "experiments" / "batch3" / "baseline" / suffix
        return (
            REPO_ROOT
            / "experiments"
            / "batch4"
            / f"seed{seed}"
            / "baseline"
            / suffix
        )
    if condition == "frozen":
        if seed == 0:
            return (
                REPO_ROOT
                / "experiments"
                / "batch1"
                / "fixed"
                / "frozen"
                / suffix
            )
        return (
            REPO_ROOT
            / "experiments"
            / "batch4"
            / f"seed{seed}"
            / "fixed"
            / suffix
        )
    if condition == "no_variance":
        if seed == 0:
            return (
                REPO_ROOT
                / "experiments"
                / "batch1"
                / "novar"
                / "a0"
                / suffix
            )
        return (
            REPO_ROOT
            / "experiments"
            / "batch4"
            / f"seed{seed}"
            / "novar"
            / suffix
        )
    raise KeyError(condition)


def new_run_dir(
    condition: str,
    dataset: str,
    seed: int,
    target: int,
    inventory: dict[tuple[str, str, int, int], dict[str, str]],
    *,
    require_checkpoint: bool = True,
) -> Path:
    row = inventory[(condition, dataset, seed, target)]
    planned = REPO_ROOT / row["planned_output"]
    if is_complete_run(planned, require_checkpoint=require_checkpoint):
        return planned
    return REPO_ROOT / row["evidence_output"]


def run_dir(
    condition: str,
    dataset: str,
    seed: int,
    target: int,
    inventory: dict[tuple[str, str, int, int], dict[str, str]],
    *,
    require_checkpoint: bool = True,
) -> Path:
    if condition in {
        "matched_control",
        "full_per_domain",
        "frozen",
        "no_variance",
    }:
        return old_run_dir(condition, dataset, seed, target)
    return new_run_dir(
        condition, dataset, seed, target, inventory,
        require_checkpoint=require_checkpoint,
    )


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def metrics_validation_error(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return "missing"
    try:
        steps = tuple(int(row["step"]) for row in rows)
    except (KeyError, TypeError, ValueError):
        return "invalid_step_key"
    if steps != EXPECTED_STEPS:
        return f"invalid_checkpoint_schedule:{steps}"
    for row in rows:
        missing_keys = [key for key in REQUIRED_EVAL_KEYS if key not in row]
        if missing_keys:
            return (
                f"missing_eval_keys_at_step_{row.get('step')}:"
                + ",".join(missing_keys)
            )
        for key in REQUIRED_EVAL_KEYS:
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                return (
                    f"invalid_eval_value_at_step_{row.get('step')}:"
                    f"{key}={row[key]!r}"
                )
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return (
                    f"invalid_eval_value_at_step_{row.get('step')}:"
                    f"{key}={row[key]!r}"
                )
    return None


def is_complete_metrics(path: Path) -> bool:
    rows = read_metrics(path)
    return metrics_validation_error(rows) is None


def is_complete_run(directory: Path, *, require_checkpoint: bool = True) -> bool:
    return (
        is_complete_metrics(directory / "metrics.jsonl")
        and (not require_checkpoint or (directory / "model_last.pt").is_file())
    )


def source_validation_score(row: dict[str, Any], target: int) -> float:
    """Mean source out-split accuracy, with target rigorously excluded."""
    values = [
        float(row[f"env{env}_out_acc"])
        for env in range(4)
        if env != target
    ]
    return float(np.mean(values))


def select_source_checkpoint(
    rows: list[dict[str, Any]], target: int
) -> tuple[dict[str, Any], float]:
    candidates = [
        (source_validation_score(row, target), -int(row["step"]), row)
        for row in rows
        if all(
            f"env{env}_out_acc" in row
            for env in range(4)
            if env != target
        )
    ]
    if not candidates:
        raise ValueError("No checkpoint has all source-validation metrics")
    score, _, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return selected, score


def collect_runs(
    *, require_checkpoint: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    inventory = load_inventory()
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for dataset in DATASETS:
            for seed in range(4):
                for target in range(4):
                    try:
                        directory = run_dir(
                            condition, dataset, seed, target, inventory,
                            require_checkpoint=require_checkpoint,
                        )
                    except KeyError:
                        missing.append(
                            {
                                "condition": condition,
                                "dataset": dataset,
                                "seed": seed,
                                "target": target,
                                "reason": "not_in_inventory",
                            }
                        )
                        continue
                    metrics_path = directory / "metrics.jsonl"
                    rows = read_metrics(metrics_path)
                    missing_checkpoint = (
                        require_checkpoint and not (directory / "model_last.pt").is_file()
                    )
                    metrics_error = metrics_validation_error(rows)
                    if metrics_error is not None or missing_checkpoint:
                        missing.append(
                            {
                                "condition": condition,
                                "dataset": dataset,
                                "seed": seed,
                                "target": target,
                                "path": str(metrics_path.relative_to(REPO_ROOT)),
                                "reason": (
                                    "missing_checkpoint"
                                    if metrics_error is None
                                    else metrics_error
                                ),
                            }
                        )
                        continue

                    try:
                        selected, source_score = select_source_checkpoint(
                            rows, target
                        )
                    except (KeyError, TypeError, ValueError) as error:
                        missing.append(
                            {
                                "condition": condition,
                                "dataset": dataset,
                                "seed": seed,
                                "target": target,
                                "path": str(metrics_path.relative_to(REPO_ROOT)),
                                "reason": f"selection_error:{error}",
                            }
                        )
                        continue

                    final = rows[-1]
                    run_rows.append(
                        {
                            "condition": condition,
                            "dataset": dataset,
                            "seed": seed,
                            "target": target,
                            "target_name": TARGET_NAMES[dataset][target],
                            "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
                            "n_checkpoints": len(rows),
                            "last_step": int(final["step"]),
                            "source_selected_step": int(selected["step"]),
                            "source_val_acc_at_selection": 100.0 * source_score,
                            "source_selected_target_out_acc": 100.0
                            * float(selected[f"env{target}_out_acc"]),
                            "source_selected_target_in_acc": 100.0
                            * float(selected[f"env{target}_in_acc"]),
                            "final_step_target_out_acc": 100.0
                            * float(final[f"env{target}_out_acc"]),
                            "final_step_target_in_acc": 100.0
                            * float(final[f"env{target}_in_acc"]),
                        }
                    )
                    for row in rows:
                        if not all(
                            f"env{env}_out_acc" in row for env in range(4)
                        ):
                            continue
                        curve_rows.append(
                            {
                                "condition": condition,
                                "dataset": dataset,
                                "seed": seed,
                                "target": target,
                                "target_name": TARGET_NAMES[dataset][target],
                                "step": int(row["step"]),
                                "source_val_acc": 100.0
                                * source_validation_score(row, target),
                                "target_out_acc_posthoc": 100.0
                                * float(row[f"env{target}_out_acc"]),
                            }
                        )

    return pd.DataFrame(run_rows), pd.DataFrame(curve_rows), missing


def endpoint_long(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in run_df.to_dict("records"):
        common = {
            key: record[key]
            for key in (
                "condition",
                "dataset",
                "seed",
                "target",
                "target_name",
            )
        }
        rows.append(
            {
                **common,
                "endpoint": "source_selected",
                "accuracy": record["source_selected_target_out_acc"],
            }
        )
        rows.append(
            {
                **common,
                "endpoint": "final_step",
                "accuracy": record["final_step_target_out_acc"],
            }
        )
    return pd.DataFrame(rows)


def describe_conditions(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_domain_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    grouped = long_df.groupby(
        ["condition", "dataset", "endpoint", "target", "target_name"],
        sort=False,
    )
    for keys, group in grouped:
        condition, dataset, endpoint, target, target_name = keys
        values = group["accuracy"].to_numpy(float)
        per_domain_rows.append(
            {
                "condition": condition,
                "dataset": dataset,
                "endpoint": endpoint,
                "target": int(target),
                "target_name": target_name,
                "n_seeds": len(values),
                "mean_accuracy": float(values.mean()),
                "run_seed_sd": (
                    float(values.std(ddof=1)) if len(values) > 1 else math.nan
                ),
                **{
                    f"seed{int(row.seed)}_accuracy": float(row.accuracy)
                    for row in group.itertuples()
                },
            }
        )

    per_domain = pd.DataFrame(per_domain_rows)
    for keys, group in per_domain.groupby(
        ["condition", "dataset", "endpoint"], sort=False
    ):
        condition, dataset, endpoint = keys
        complete_targets = group[group["n_seeds"] == 4]
        means = complete_targets["mean_accuracy"].to_numpy(float)
        run_seed_sds = complete_targets["run_seed_sd"].to_numpy(float)
        summary_rows.append(
            {
                "condition": condition,
                "dataset": dataset,
                "endpoint": endpoint,
                "n_domains": len(complete_targets),
                "n_runs": int(complete_targets["n_seeds"].sum()),
                "macro_mean_accuracy": (
                    float(means.mean()) if len(means) else math.nan
                ),
                "between_target_sd_of_means": (
                    float(means.std(ddof=1)) if len(means) > 1 else math.nan
                ),
                "mean_within_target_run_seed_sd": (
                    float(run_seed_sds.mean())
                    if len(run_seed_sds)
                    else math.nan
                ),
                "rms_within_target_run_seed_sd": (
                    float(np.sqrt(np.mean(run_seed_sds**2)))
                    if len(run_seed_sds)
                    else math.nan
                ),
            }
        )
    return per_domain, pd.DataFrame(summary_rows)


def exact_sign_flip_p(domain_effects: np.ndarray) -> float:
    observed = abs(float(domain_effects.mean()))
    null = [
        abs(float(np.mean(domain_effects * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(domain_effects))
    ]
    tolerance = 1e-12
    return float(np.mean(np.asarray(null) >= observed - tolerance))


def exhaustive_cluster_bootstrap(
    domain_effects: np.ndarray, confidence: float
) -> tuple[float, float]:
    n = len(domain_effects)
    means = np.asarray(
        [
            float(np.mean(domain_effects[list(indices)]))
            for indices in itertools.product(range(n), repeat=n)
        ]
    )
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def t_interval(
    values: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, math.nan, math.nan
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    if se == 0.0:
        return mean, mean, mean
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean, mean - critical * se, mean + critical * se


def cluster_t_p(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.nan
    if float(values.std(ddof=1)) == 0.0:
        return 0.0 if float(values.mean()) != 0.0 else 1.0
    return float(stats.ttest_1samp(values, popmean=0.0).pvalue)


def tost(
    values: np.ndarray, margin: float
) -> tuple[float, float, float, float, bool]:
    mean, ci_low, ci_high = t_interval(values, 0.90)
    if len(values) < 2:
        return math.nan, ci_low, ci_high, math.nan, False
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    if se == 0.0:
        equivalent = -margin < mean < margin
        return 0.0 if equivalent else 1.0, ci_low, ci_high, mean, equivalent
    df = len(values) - 1
    p_lower = float(stats.t.sf((mean + margin) / se, df=df))
    p_upper = float(stats.t.cdf((mean - margin) / se, df=df))
    p_value = max(p_lower, p_upper)
    return p_value, ci_low, ci_high, mean, p_value < 0.05


def noninferiority(
    values: np.ndarray, margin: float
) -> tuple[float, float, bool]:
    """One-sided test that mean(values) is greater than ``-margin``."""
    if len(values) < 2:
        return math.nan, math.nan, False
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    if se == 0.0:
        passed = mean > -margin
        return 0.0 if passed else 1.0, mean, passed
    df = len(values) - 1
    lower_bound = mean - float(stats.t.ppf(0.95, df)) * se
    p_value = float(stats.t.sf((mean + margin) / se, df=df))
    return p_value, lower_bound, p_value < 0.05


def dataset_clustered_macro_stats(
    dataset_domain_effects: list[np.ndarray],
    margin: float,
) -> dict[str, Any]:
    """Primary suite inference on four equal-weight benchmark means.

    The 16 held-out targets are nested within four benchmarks, and leave-one-
    domain-out runs inside a benchmark reuse source data.  The suite-level
    cluster is therefore the benchmark dataset, not an individual target.
    """
    if len(dataset_domain_effects) != 4 or any(
        len(effects) != 4 for effects in dataset_domain_effects
    ):
        raise ValueError("Macro inference requires four domains in four datasets")
    dataset_means = np.asarray(
        [float(effects.mean()) for effects in dataset_domain_effects]
    )
    effect, ci95_low, ci95_high = t_interval(dataset_means, 0.95)
    boot_low, boot_high = exhaustive_cluster_bootstrap(dataset_means, 0.95)
    t_p = cluster_t_p(dataset_means)
    permutation_p = exact_sign_flip_p(dataset_means)
    tost_p, ci90_low, ci90_high, _, equivalent = tost(dataset_means, margin)
    return {
        "effect_pp": effect,
        "dataset_cluster_sd": float(dataset_means.std(ddof=1)),
        "ci95_t_low": ci95_low,
        "ci95_t_high": ci95_high,
        "ci95_dataset_boot_low": boot_low,
        "ci95_dataset_boot_high": boot_high,
        "exact_4_dataset_signflip_p": permutation_p,
        "dataset_cluster_t_p_descriptive": t_p,
        "sesoi_pp": margin,
        "ci90_t_low": ci90_low,
        "ci90_t_high": ci90_high,
        "tost_p": tost_p,
        "equivalent_at_0p05": equivalent,
    }


def stratified_domain_sensitivity_stats(
    dataset_domain_effects: list[np.ndarray],
    margin: float,
) -> dict[str, Any]:
    """Sensitivity only: treats the 16 targets as independent within strata."""
    if len(dataset_domain_effects) != 4 or any(
        len(effects) != 4 for effects in dataset_domain_effects
    ):
        raise ValueError("Macro sensitivity requires four domains in four datasets")
    n_datasets = len(dataset_domain_effects)
    dataset_means = np.asarray(
        [float(effects.mean()) for effects in dataset_domain_effects]
    )
    components = np.asarray(
        [
            float(effects.var(ddof=1) / len(effects))
            for effects in dataset_domain_effects
        ]
    )
    effect = float(dataset_means.mean())
    variance = float(components.sum() / (n_datasets**2))
    se = math.sqrt(max(variance, 0.0))
    denominator = float(
        np.sum(
            (components**2)
            / np.asarray([len(effects) - 1 for effects in dataset_domain_effects])
        )
    )
    df = (
        float(components.sum() ** 2 / denominator)
        if denominator > 0.0
        else math.inf
    )
    critical95 = float(stats.t.ppf(0.975, df)) if se > 0.0 else 0.0
    critical90 = float(stats.t.ppf(0.95, df)) if se > 0.0 else 0.0
    ci95_low, ci95_high = effect - critical95 * se, effect + critical95 * se
    ci90_low, ci90_high = effect - critical90 * se, effect + critical90 * se
    if se == 0.0:
        t_p = 0.0 if effect != 0.0 else 1.0
        tost_p = 0.0 if -margin < effect < margin else 1.0
    else:
        t_p = float(2.0 * stats.t.sf(abs(effect / se), df=df))
        p_lower = float(stats.t.sf((effect + margin) / se, df=df))
        p_upper = float(stats.t.cdf((effect - margin) / se, df=df))
        tost_p = max(p_lower, p_upper)
    all_domain_effects = np.concatenate(dataset_domain_effects)
    return {
        "effect_pp": effect,
        "stratified_se": se,
        "welch_satterthwaite_df": df,
        "ci95_t_low": ci95_low,
        "ci95_t_high": ci95_high,
        "stratified_t_p_descriptive": t_p,
        "exact_16_domain_signflip_p": exact_sign_flip_p(all_domain_effects),
        "independence_caveat": (
            "sensitivity_only_targets_nested_within_benchmark_and_source_data_reused"
        ),
        "sesoi_pp": margin,
        "ci90_t_low": ci90_low,
        "ci90_t_high": ci90_high,
        "tost_p": tost_p,
        "equivalent_at_0p05": tost_p < 0.05,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return result.tolist()
    ordered = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * values[index])
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def paired_domain_effects(
    long_df: pd.DataFrame,
    treatment: str,
    control: str,
    dataset: str,
    endpoint: str,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    subset = long_df[
        (long_df["dataset"] == dataset)
        & (long_df["endpoint"] == endpoint)
        & (long_df["condition"].isin([treatment, control]))
    ]
    pivot = subset.pivot_table(
        index=["target", "target_name", "seed"],
        columns="condition",
        values="accuracy",
        aggfunc="first",
    )
    if treatment not in pivot or control not in pivot:
        return None
    pivot = pivot.dropna(subset=[treatment, control]).copy()
    pivot["difference"] = pivot[treatment] - pivot[control]
    domain = (
        pivot.reset_index()
        .groupby(["target", "target_name"], as_index=False)
        .agg(
            n_seed_pairs=("difference", "size"),
            mean_difference=("difference", "mean"),
            run_seed_sd_difference=("difference", "std"),
        )
    )
    complete = domain[domain["n_seed_pairs"] == 4]
    if len(complete) != 4:
        return domain, np.asarray([], dtype=float)
    return domain, complete.sort_values("target")["mean_difference"].to_numpy(float)


def paired_seed_effect_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for dataset in DATASETS:
            for endpoint in ENDPOINTS:
                subset = long_df[
                    (long_df["dataset"] == dataset)
                    & (long_df["endpoint"] == endpoint)
                    & (
                        long_df["condition"].isin(
                            [contrast.treatment, contrast.control]
                        )
                    )
                ]
                pivot = subset.pivot_table(
                    index=["target", "target_name", "seed"],
                    columns="condition",
                    values="accuracy",
                    aggfunc="first",
                )
                if contrast.treatment not in pivot or contrast.control not in pivot:
                    continue
                pivot = pivot.dropna(
                    subset=[contrast.treatment, contrast.control]
                ).reset_index()
                for record in pivot.to_dict("records"):
                    rows.append(
                        {
                            "contrast": contrast.key,
                            "role": contrast.role,
                            "treatment": contrast.treatment,
                            "control": contrast.control,
                            "scope": "dataset",
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "target": int(record["target"]),
                            "target_name": record["target_name"],
                            "seed": int(record["seed"]),
                            "treatment_accuracy": float(
                                record[contrast.treatment]
                            ),
                            "control_accuracy": float(record[contrast.control]),
                            "difference": float(
                                record[contrast.treatment]
                                - record[contrast.control]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def model_selection_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame(rows)
    for keys, group in run_df.groupby(["condition", "dataset"], sort=False):
        condition, dataset = keys
        complete_targets = (
            group.groupby("target")["seed"].nunique().eq(4).all()
            and group["target"].nunique() == 4
        )
        selected_domain_means = group.groupby("target")[
            "source_selected_target_out_acc"
        ].mean()
        final_domain_means = group.groupby("target")[
            "final_step_target_out_acc"
        ].mean()
        selected_steps = group["source_selected_step"].to_numpy(float)
        rows.append(
            {
                "condition": condition,
                "dataset": dataset,
                "n_runs": len(group),
                "complete_four_by_four": bool(complete_targets),
                "selected_step_median": float(np.median(selected_steps)),
                "selected_step_q1": float(np.quantile(selected_steps, 0.25)),
                "selected_step_q3": float(np.quantile(selected_steps, 0.75)),
                "source_selected_macro_accuracy": float(
                    selected_domain_means.mean()
                ),
                "final_step_macro_accuracy": float(final_domain_means.mean()),
                "selected_minus_final_pp": float(
                    (selected_domain_means - final_domain_means).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_contrasts(
    long_df: pd.DataFrame, primary_sesoi: float
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    comparison_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    suite_sensitivity_rows: list[dict[str, Any]] = []

    for contrast in CONTRASTS:
        for endpoint in ENDPOINTS:
            dataset_domain_effects: list[np.ndarray] = []
            for dataset in DATASETS:
                paired = paired_domain_effects(
                    long_df,
                    contrast.treatment,
                    contrast.control,
                    dataset,
                    endpoint,
                )
                if paired is None:
                    continue
                domain, effects = paired
                for row in domain.to_dict("records"):
                    domain_rows.append(
                        {
                            "contrast": contrast.key,
                            "role": contrast.role,
                            "treatment": contrast.treatment,
                            "control": contrast.control,
                            "scope": "dataset",
                            "dataset": dataset,
                            "endpoint": endpoint,
                            **row,
                        }
                    )
                if len(effects) != 4:
                    comparison_rows.append(
                        {
                            "contrast": contrast.key,
                            "role": contrast.role,
                            "treatment": contrast.treatment,
                            "control": contrast.control,
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "n_domains": len(effects),
                            "complete_for_inference": False,
                        }
                    )
                    continue

                estimate, ci_low, ci_high = t_interval(effects, 0.95)
                boot_low, boot_high = exhaustive_cluster_bootstrap(effects, 0.95)
                t_p = cluster_t_p(effects)
                permutation_p = exact_sign_flip_p(effects)
                tost_p, ci90_low, ci90_high, _, equivalent = tost(
                    effects, primary_sesoi
                )
                dataset_domain_effects.append(effects)
                comparison = {
                    "contrast": contrast.key,
                    "role": contrast.role,
                    "treatment": contrast.treatment,
                    "control": contrast.control,
                    "scope": "dataset",
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_domains": 4,
                    "n_seed_pairs": 16,
                    "complete_for_inference": True,
                    "effect_pp": estimate,
                    "domain_sd": float(effects.std(ddof=1)),
                    "ci95_t_low": ci_low,
                    "ci95_t_high": ci_high,
                    "ci95_cluster_boot_low": boot_low,
                    "ci95_cluster_boot_high": boot_high,
                    "exact_signflip_p": permutation_p,
                    "cluster_t_p_descriptive": t_p,
                    "sesoi_pp": primary_sesoi,
                    "ci90_t_low": ci90_low,
                    "ci90_t_high": ci90_high,
                    "tost_p": tost_p,
                    "equivalent_at_0p05": equivalent,
                    **{
                        f"target_{TARGET_NAMES[dataset][idx]}_effect": float(
                            effects[idx]
                        )
                        for idx in range(4)
                    },
                }
                if contrast.key == "no_variance_minus_shared":
                    ni_p, ni_lower, ni_passed = noninferiority(
                        -effects, primary_sesoi
                    )
                    comparison.update(
                        {
                            "shared_minus_perdomain_effect_pp": -estimate,
                            "noninferiority_margin_pp": -primary_sesoi,
                            "shared_minus_perdomain_lower95": ni_lower,
                            "noninferiority_p": ni_p,
                            "shared_noninferior_at_0p05": ni_passed,
                        }
                    )
                comparison_rows.append(comparison)
                margins = sorted({0.5, primary_sesoi, 2.0})
                for margin in margins:
                    p_value, eq_low, eq_high, eq_mean, eq_result = tost(
                        effects, margin
                    )
                    equivalence_rows.append(
                        {
                            "contrast": contrast.key,
                            "role": contrast.role,
                            "scope": "dataset",
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "margin_pp": margin,
                            "effect_pp": eq_mean,
                            "ci90_t_low": eq_low,
                            "ci90_t_high": eq_high,
                            "tost_p": p_value,
                            "equivalent_at_0p05": eq_result,
                            "primary_equivalence_question": False,
                        }
                    )

            if len(dataset_domain_effects) == 4:
                suite_stats = dataset_clustered_macro_stats(
                    dataset_domain_effects, primary_sesoi
                )
                suite_row = {
                    "contrast": contrast.key,
                    "role": contrast.role,
                    "endpoint": endpoint,
                    "estimand": "equal_dataset_then_equal_target_macro_effect",
                    "inference_cluster": "benchmark_dataset",
                    "n_datasets": 4,
                    "n_dataset_clusters": 4,
                    "n_target_domains": 16,
                    **suite_stats,
                }
                dataset_means = np.asarray(
                    [float(effects.mean()) for effects in dataset_domain_effects]
                )
                if contrast.key == "no_variance_minus_shared":
                    ni_p, ni_lower, ni_passed = noninferiority(
                        -dataset_means, primary_sesoi
                    )
                    suite_row.update(
                        {
                            "shared_minus_perdomain_effect_pp": -float(
                                dataset_means.mean()
                            ),
                            "noninferiority_margin_pp": -primary_sesoi,
                            "shared_minus_perdomain_lower95": ni_lower,
                            "noninferiority_p": ni_p,
                            "shared_noninferior_at_0p05": ni_passed,
                        }
                    )
                suite_rows.append(suite_row)

                for margin in sorted({0.5, primary_sesoi, 2.0}):
                    p_value, eq_low, eq_high, eq_mean, eq_result = tost(
                        dataset_means, margin
                    )
                    equivalence_rows.append(
                        {
                            "contrast": contrast.key,
                            "role": contrast.role,
                            "scope": "suite",
                            "dataset": "suite_macro",
                            "endpoint": endpoint,
                            "margin_pp": margin,
                            "effect_pp": eq_mean,
                            "ci90_t_low": eq_low,
                            "ci90_t_high": eq_high,
                            "tost_p": p_value,
                            "equivalent_at_0p05": eq_result,
                            "primary_equivalence_question": bool(
                                contrast.key == "no_variance_minus_shared"
                                and endpoint == "source_selected"
                                and math.isclose(margin, primary_sesoi)
                            ),
                        }
                    )

                suite_sensitivity_rows.append(
                    {
                        "contrast": contrast.key,
                        "role": contrast.role,
                        "endpoint": endpoint,
                        "estimand": "equal_dataset_then_equal_target_macro_effect",
                        "n_datasets": 4,
                        "n_target_domains": 16,
                        **stratified_domain_sensitivity_stats(
                            dataset_domain_effects, primary_sesoi
                        ),
                    }
                )

    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["holm_p_dataset_8_sensitivity"] = np.nan
        primary_roles = {"primary_utility", "primary_domain_specificity"}
        for endpoint in ENDPOINTS:
            mask = (
                comparisons["role"].isin(primary_roles)
                & (comparisons["endpoint"] == endpoint)
                & comparisons["complete_for_inference"].fillna(False)
            )
            comparisons.loc[
                mask, "holm_p_dataset_8_sensitivity"
            ] = holm_adjust(
                comparisons.loc[mask, "exact_signflip_p"]
            )
    suite_frame = pd.DataFrame(suite_rows)
    if not suite_frame.empty:
        suite_frame["holm_p_primary_2"] = np.nan
        primary_roles = {"primary_utility", "primary_domain_specificity"}
        for endpoint in ENDPOINTS:
            mask = (
                suite_frame["role"].isin(primary_roles)
                & (suite_frame["endpoint"] == endpoint)
            )
            suite_frame.loc[mask, "holm_p_primary_2"] = holm_adjust(
                suite_frame.loc[mask, "exact_4_dataset_signflip_p"]
            )
    return (
        comparisons,
        pd.DataFrame(domain_rows),
        pd.DataFrame(equivalence_rows),
        suite_frame,
        pd.DataFrame(suite_sensitivity_rows),
    )


def two_way_variation(long_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive target/run-seed variation without pooling the two.

    A run seed changes the 80/20 split as well as initialization, minibatch
    order, and augmentation.  These components cannot be separated in this
    design.
    """
    rows: list[dict[str, Any]] = []
    for keys, group in long_df.groupby(
        ["condition", "dataset", "endpoint"], sort=False
    ):
        condition, dataset, endpoint = keys
        pivot = group.pivot(index="target", columns="seed", values="accuracy")
        if pivot.shape != (4, 4) or pivot.isna().any().any():
            continue
        values = pivot.to_numpy(float)
        grand = float(values.mean())
        domain_means = values.mean(axis=1)
        run_seed_means = values.mean(axis=0)
        residual = (
            values
            - domain_means[:, None]
            - run_seed_means[None, :]
            + grand
        )
        within_target_run_seed_sd = values.std(axis=1, ddof=1)
        rows.append(
            {
                "condition": condition,
                "dataset": dataset,
                "endpoint": endpoint,
                "grand_mean_accuracy": grand,
                "between_target_sd_of_means": float(domain_means.std(ddof=1)),
                "between_run_seed_sd_of_means": float(
                    run_seed_means.std(ddof=1)
                ),
                "rms_within_target_run_seed_sd": float(
                    np.sqrt(np.mean(within_target_run_seed_sd**2))
                ),
                "target_by_run_seed_residual_sd": float(
                    np.sqrt(np.sum(residual**2) / 9.0)
                ),
                "run_seed_components": (
                    "data_split+initialization+minibatch_order+augmentation"
                ),
            }
        )
    return pd.DataFrame(rows)


def effect_two_way_variation(paired_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same crossed decomposition to paired method effects."""
    rows: list[dict[str, Any]] = []
    if paired_seed_df.empty:
        return pd.DataFrame(rows)
    for keys, group in paired_seed_df.groupby(
        ["contrast", "role", "dataset", "endpoint"], sort=False
    ):
        contrast, role, dataset, endpoint = keys
        pivot = group.pivot(index="target", columns="seed", values="difference")
        if pivot.shape != (4, 4) or pivot.isna().any().any():
            continue
        values = pivot.to_numpy(float)
        grand = float(values.mean())
        target_means = values.mean(axis=1)
        run_seed_means = values.mean(axis=0)
        residual = (
            values
            - target_means[:, None]
            - run_seed_means[None, :]
            + grand
        )
        within_target_run_seed_sd = values.std(axis=1, ddof=1)
        rows.append(
            {
                "contrast": contrast,
                "role": role,
                "scope": "dataset",
                "dataset": dataset,
                "endpoint": endpoint,
                "grand_mean_effect_pp": grand,
                "between_target_sd_of_mean_effects": float(
                    target_means.std(ddof=1)
                ),
                "between_run_seed_sd_of_mean_effects": float(
                    run_seed_means.std(ddof=1)
                ),
                "rms_within_target_run_seed_sd_of_effects": float(
                    np.sqrt(np.mean(within_target_run_seed_sd**2))
                ),
                "target_by_run_seed_residual_sd_of_effects": float(
                    np.sqrt(np.sum(residual**2) / 9.0)
                ),
                "run_seed_components": (
                    "data_split+initialization+minibatch_order+augmentation"
                ),
            }
        )
    return pd.DataFrame(rows)


def target_in_sensitivity(run_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize the held-out target in-split as a declared sensitivity."""
    rows: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame(rows)

    endpoint_columns = {
        "source_selected_target_in": "source_selected_target_in_acc",
        "final_step_target_in": "final_step_target_in_acc",
    }
    for endpoint, column in endpoint_columns.items():
        for keys, group in run_df.groupby(
            ["condition", "dataset"], sort=False
        ):
            condition, dataset = keys
            target_means = group.groupby("target")[column].mean()
            target_run_seed_sds = group.groupby("target")[column].std(ddof=1)
            if len(target_means) != 4 or group["seed"].nunique() != 4:
                continue
            rows.append(
                {
                    "record_type": "condition_summary",
                    "condition": condition,
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "n_targets": 4,
                    "n_runs": len(group),
                    "macro_mean_accuracy": float(target_means.mean()),
                    "between_target_sd_of_means": float(
                        target_means.std(ddof=1)
                    ),
                    "rms_within_target_run_seed_sd": float(
                        np.sqrt(np.mean(target_run_seed_sds.to_numpy(float) ** 2))
                    ),
                }
            )

        for contrast in CONTRASTS[:2]:
            for dataset in DATASETS:
                subset = run_df[
                    (run_df["dataset"] == dataset)
                    & run_df["condition"].isin(
                        [contrast.treatment, contrast.control]
                    )
                ]
                pivot = subset.pivot_table(
                    index=["target", "seed"],
                    columns="condition",
                    values=column,
                    aggfunc="first",
                )
                if (
                    contrast.treatment not in pivot
                    or contrast.control not in pivot
                ):
                    continue
                pivot = pivot.dropna(
                    subset=[contrast.treatment, contrast.control]
                ).copy()
                if len(pivot) != 16:
                    continue
                pivot["difference"] = (
                    pivot[contrast.treatment] - pivot[contrast.control]
                )
                effects = (
                    pivot.reset_index()
                    .groupby("target")["difference"]
                    .mean()
                    .sort_index()
                    .to_numpy(float)
                )
                if len(effects) != 4:
                    continue
                estimate, ci_low, ci_high = t_interval(effects, 0.95)
                rows.append(
                    {
                        "record_type": "primary_contrast",
                        "contrast": contrast.key,
                        "role": contrast.role,
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "n_targets": 4,
                        "n_runs": 16,
                        "effect_pp": estimate,
                        "ci95_t_low": ci_low,
                        "ci95_t_high": ci_high,
                        "exact_signflip_p": exact_sign_flip_p(effects),
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No complete results yet._\n"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if not np.isfinite(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_summary(
    run_df: pd.DataFrame,
    missing: list[dict[str, Any]],
    condition_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    suite: pd.DataFrame,
    sesoi: float,
    output_dir: Path,
) -> None:
    source_summary = condition_summary[
        condition_summary["endpoint"] == "source_selected"
    ].copy()
    dataset_sensitivity = comparisons[
        (comparisons["endpoint"] == "source_selected")
        & comparisons["role"].isin(
            ["primary_utility", "primary_domain_specificity"]
        )
        & comparisons["complete_for_inference"].fillna(False)
    ].copy()
    primary_suite = suite[
        (suite["endpoint"] == "source_selected")
        & suite["role"].isin(
            ["primary_utility", "primary_domain_specificity"]
        )
    ].copy()

    lines = [
        "# Reviewer-round analysis status",
        "",
        f"- Complete runs available: **{len(run_df)}/384** "
        "(six conditions x four datasets x four targets x four seeds).",
        f"- Missing or incomplete runs: **{len(missing)}**.",
        "- Primary checkpoint rule: maximum mean source-domain out-split "
        "validation accuracy; the target domain is excluded; ties use the "
        "earliest checkpoint.",
        "- Inferential unit: held-out target domain after averaging the four "
        "paired run-seed differences within each benchmark; the suite-level "
        "unit is the benchmark dataset.",
        "- Exact two-sided sign-flip tests have only 16 assignments with four "
        "target domains or four benchmark datasets, so their smallest "
        "attainable p-value is 0.125.",
        f"- Frozen revision-stage SESOI: +/-{sesoi:.1f} percentage point.",
        "- A run seed changes the 80/20 data split as well as initialization, "
        "minibatch order, and augmentation.",
        "",
        "## Source-validation-selected condition summaries",
        "",
        markdown_table(
            source_summary,
            [
                "condition",
                "dataset",
                "n_domains",
                "macro_mean_accuracy",
                "between_target_sd_of_means",
                "rms_within_target_run_seed_sd",
            ],
        ),
        "## Primary suite-level reviewer-requested contrasts",
        "",
        markdown_table(
            primary_suite,
            [
                "contrast",
                "effect_pp",
                "ci95_t_low",
                "ci95_t_high",
                "exact_4_dataset_signflip_p",
                "holm_p_primary_2",
                "ci90_t_low",
                "ci90_t_high",
                "equivalent_at_0p05",
            ],
        ),
        "## Per-dataset estimates and Holm-8 sensitivity",
        "",
        markdown_table(
            dataset_sensitivity,
            [
                "contrast",
                "dataset",
                "effect_pp",
                "ci95_t_low",
                "ci95_t_high",
                "exact_signflip_p",
                "holm_p_dataset_8_sensitivity",
                "ci90_t_low",
                "ci90_t_high",
                "equivalent_at_0p05",
            ],
        ),
        "## Interpretation guardrails",
        "",
        "- A non-significant sign-flip test is not evidence of equivalence.",
        "- The primary equivalence question is the source-selected suite "
        "NoVariance-versus-SharedLearnable macro at the revision-stage "
        "margin. Dataset and alternate-margin TOSTs are sensitivities.",
        "- Macro equivalence concerns the equal-dataset/equal-target average, "
        "not every named target.",
        "- Target learning curves are post-hoc diagnostics and never enter "
        "checkpoint selection.",
        "- The clean domain-specificity contrast is NoVariance per-domain "
        "minus SharedLearnable, because the variance term is structurally zero "
        "for one shared scalar.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.sesoi <= 0:
        raise SystemExit("--sesoi must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df, curve_df, missing = collect_runs(require_checkpoint=not args.metrics_only)
    long_df = endpoint_long(run_df)
    per_domain, condition_summary = describe_conditions(long_df)
    (
        comparisons,
        domain_effects,
        equivalence,
        suite,
        suite_domain_sensitivity,
    ) = analyze_contrasts(long_df, args.sesoi)
    variation = two_way_variation(long_df)
    paired_seeds = paired_seed_effect_table(long_df)
    effect_variation = effect_two_way_variation(paired_seeds)
    target_in = target_in_sensitivity(run_df)
    selection_summary = model_selection_summary(run_df)

    run_df.to_csv(output_dir / "run_level_results.csv", index=False)
    curve_df.to_csv(output_dir / "learning_curve_points.csv", index=False)
    per_domain.to_csv(output_dir / "per_target_results.csv", index=False)
    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    variation.to_csv(output_dir / "variance_decomposition.csv", index=False)
    effect_variation.to_csv(
        output_dir / "effect_variance_decomposition.csv", index=False
    )
    paired_seeds.to_csv(output_dir / "paired_seed_effects.csv", index=False)
    selection_summary.to_csv(
        output_dir / "model_selection_summary.csv", index=False
    )
    comparisons.to_csv(output_dir / "clustered_comparisons.csv", index=False)
    domain_effects.to_csv(output_dir / "domain_effects.csv", index=False)
    equivalence.to_csv(output_dir / "equivalence_sensitivity.csv", index=False)
    suite.to_csv(output_dir / "suite_comparisons.csv", index=False)
    suite_domain_sensitivity.to_csv(
        output_dir / "suite_domain_sensitivity.csv", index=False
    )
    target_in.to_csv(output_dir / "target_in_sensitivity.csv", index=False)
    (output_dir / "missing_runs.json").write_text(
        json.dumps(missing, indent=2) + "\n", encoding="utf-8"
    )
    readiness = {
        "complete_runs": len(run_df),
        "expected_runs": 384,
        "missing_runs": len(missing),
        "conditions_complete": {
            condition: int((run_df["condition"] == condition).sum())
            for condition in CONDITIONS
        },
        "source_selection_target_keys_used": False,
        "source_selection_rule": (
            "maximize mean env*_out_acc over env != held-out target; "
            "break ties by earliest step"
        ),
        "per_dataset_inference_cluster": "held-out target domain",
        "suite_inference_cluster": "benchmark dataset",
        "primary_multiplicity_family": (
            "two source-selected suite contrasts with Holm correction"
        ),
        "run_seed_components": (
            "data_split+initialization+minibatch_order+augmentation"
        ),
        "primary_sesoi_pp": args.sesoi,
        "sesoi_timing": "frozen_during_revision_before_added_grid_completed",
    }
    (output_dir / "readiness.json").write_text(
        json.dumps(readiness, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(
        run_df, missing, condition_summary, comparisons, suite, args.sesoi, output_dir
    )
    print(json.dumps(readiness, indent=2))
    if missing and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
