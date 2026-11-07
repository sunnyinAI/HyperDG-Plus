#!/usr/bin/env python3
"""Render all-seed local-scale trajectories with semantic source labels.

The training log names the three local parameters ``curvature_domain_0`` through
``curvature_domain_2``.  Those suffixes are loader slots, not stable dataset
domains.  Within a leave-one-domain-out run, slot ``j`` corresponds to the
``j``-th original environment after removing the held-out target.  This script
performs that remapping before aggregating trajectories.

No checkpoint or GPU is needed: every plotted value comes from the finalized
``metrics.jsonl`` files referenced by ``run_level_results.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parent
RESULT_ROOT = STUDY_ROOT / "outputs" / "reviewer_round2"
DEFAULT_FIGURE_ROOT = (
    REPO_ROOT / "HyperDG_TMLR" / "revision_assets" / "figures"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "HyperDG_TMLR" / "revision_assets" / "csv"

DATASETS = ("pacs", "vlcs", "officehome", "terraincognita")
DATASET_LABELS = {
    "pacs": "PACS",
    "vlcs": "VLCS",
    "officehome": "OfficeHome",
    "terraincognita": "TerraIncognita",
}
TARGET_NAMES = {
    "pacs": ("ArtPainting", "Cartoon", "Photo", "Sketch"),
    "vlcs": ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
    "officehome": ("Art", "Clipart", "Product", "RealWorld"),
    "terraincognita": ("L100", "L38", "L43", "L46"),
}
CONDITIONS = ("no_variance", "full_per_domain")
CONDITION_LABELS = {
    "no_variance": r"NoVariance ($\lambda_{\mathrm{var}}=0$)",
    "full_per_domain": r"FullPerDomain ($\lambda_{\mathrm{var}}=0.2$)",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
EXPECTED_STEPS = tuple(range(200, 5000, 200)) + (4999,)
INITIAL_SCALE = float(np.log1p(np.exp(0.0)) + 1.0e-4)


class TrajectoryContractError(RuntimeError):
    """Raised when the finalized curvature-log contract is incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RESULT_ROOT,
        help="Directory containing the analysis run table.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_ROOT,
        help="Directory for the grid and dataset-specific PDF figures.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory for mapped point and summary CSV files.",
    )
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise TrajectoryContractError(f"cannot read {path}: {exc}") from exc


def collect_named_points(
    run_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "condition",
        "dataset",
        "seed",
        "target",
        "target_name",
        "metrics_path",
        "source_selected_step",
    }
    missing_columns = sorted(required_columns - set(run_table.columns))
    if missing_columns:
        raise TrajectoryContractError(
            "run_level_results.csv lacks columns: " + ", ".join(missing_columns)
        )

    selected = run_table[
        run_table["condition"].isin(CONDITIONS)
        & run_table["dataset"].isin(DATASETS)
    ].copy()
    expected_grid = {
        (condition, dataset, seed, target)
        for condition in CONDITIONS
        for dataset in DATASETS
        for seed in range(4)
        for target in range(4)
    }
    actual_grid = set(
        selected[["condition", "dataset", "seed", "target"]].itertuples(
            index=False, name=None
        )
    )
    if actual_grid != expected_grid:
        raise TrajectoryContractError(
            f"trajectory grid mismatch: missing={len(expected_grid-actual_grid)}, "
            f"extra={len(actual_grid-expected_grid)}"
        )

    point_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    for run in selected.itertuples(index=False):
        dataset = str(run.dataset)
        target = int(run.target)
        seed = int(run.seed)
        condition = str(run.condition)
        expected_target_name = TARGET_NAMES[dataset][target]
        if str(run.target_name) != expected_target_name:
            raise TrajectoryContractError(
                f"target-name mismatch for {condition}/{dataset}/seed{seed}/"
                f"target{target}: {run.target_name!r} != {expected_target_name!r}"
            )

        metrics_path = REPO_ROOT / str(run.metrics_path)
        records = read_metrics(metrics_path)
        steps = tuple(int(record.get("step", -1)) for record in records)
        if steps != EXPECTED_STEPS:
            raise TrajectoryContractError(
                f"unexpected checkpoint schedule in {metrics_path}: {steps}"
            )

        source_envs = [env for env in range(4) if env != target]
        if len(source_envs) != 3:
            raise AssertionError("leave-one-domain-out must have three sources")
        selected_step = int(run.source_selected_step)
        for record in records:
            step = int(record["step"])
            local_values: list[float] = []
            for loader_slot, source_env in enumerate(source_envs):
                key = f"curvature_domain_{loader_slot}"
                if key not in record:
                    raise TrajectoryContractError(
                        f"{metrics_path} step {step} lacks {key}"
                    )
                value = float(record[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise TrajectoryContractError(
                        f"invalid positive scale in {metrics_path} step {step}: "
                        f"{key}={value}"
                    )
                local_values.append(value)
                point_rows.append(
                    {
                        "condition": condition,
                        "dataset": dataset,
                        "seed": seed,
                        "held_out_target": target,
                        "held_out_target_name": expected_target_name,
                        "loader_slot": loader_slot,
                        "source_env": source_env,
                        "source_name": TARGET_NAMES[dataset][source_env],
                        "step": step,
                        "positive_scale": value,
                    }
                )
            for checkpoint, include in (
                ("source_selected", step == selected_step),
                ("final_step", step == 4999),
            ):
                if include:
                    spread_rows.append(
                        {
                            "condition": condition,
                            "dataset": dataset,
                            "seed": seed,
                            "held_out_target": target,
                            "held_out_target_name": expected_target_name,
                            "checkpoint": checkpoint,
                            "step": step,
                            "within_run_scale_range": max(local_values)
                            - min(local_values),
                        }
                    )

    points = pd.DataFrame(point_rows)
    spreads = pd.DataFrame(spread_rows)
    if len(points) != 2 * 4 * 4 * 4 * 25 * 3:
        raise TrajectoryContractError(
            f"expected 9,600 mapped scale points, found {len(points)}"
        )
    return points, spreads


def aggregate_points(points: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        points.groupby(
            ["condition", "dataset", "source_env", "source_name", "step"],
            sort=False,
        )["positive_scale"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
    )
    if len(aggregate) != 2 * 4 * 4 * 25:
        raise TrajectoryContractError(
            f"expected 800 aggregate points, found {len(aggregate)}"
        )
    if not aggregate["count"].eq(12).all():
        bad = aggregate.loc[~aggregate["count"].eq(12)].iloc[0].to_dict()
        raise TrajectoryContractError(
            f"each named source/step needs 12 target-context x seed values: {bad}"
        )
    return aggregate


def summarize_spreads(spreads: pd.DataFrame) -> pd.DataFrame:
    summary = (
        spreads.groupby(["condition", "dataset", "checkpoint"])[
            "within_run_scale_range"
        ]
        .agg(["count", "mean", "std", "median", "max"])
        .reset_index()
    )
    if not summary["count"].eq(16).all():
        bad = summary.loc[~summary["count"].eq(16)].iloc[0].to_dict()
        raise TrajectoryContractError(
            f"each spread summary needs 16 target x seed runs: {bad}"
        )
    return summary


def draw_panel(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    dataset: str,
    condition: str,
    show_legend: bool,
) -> None:
    subset = aggregate[
        (aggregate["dataset"] == dataset)
        & (aggregate["condition"] == condition)
    ]
    for source_env, source_name in enumerate(TARGET_NAMES[dataset]):
        curve = subset[subset["source_env"] == source_env].sort_values("step")
        if len(curve) != 25:
            raise TrajectoryContractError(
                f"missing curve for {condition}/{dataset}/{source_name}"
            )
        x = curve["step"].to_numpy(float)
        mean = curve["mean"].to_numpy(float)
        sd = curve["std"].to_numpy(float)
        ax.plot(
            x,
            mean,
            color=COLORS[source_env],
            linewidth=1.7,
            label=source_name,
        )
        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=COLORS[source_env],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(
        INITIAL_SCALE,
        color="#666666",
        linewidth=0.8,
        linestyle=":",
        label="initial scale" if show_legend else None,
    )
    ax.grid(alpha=0.20)
    ax.tick_params(labelsize=8)
    if show_legend:
        ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="best")


def plot_grid(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(CONDITIONS),
        figsize=(8.2, 9.7),
        sharex=True,
        sharey="row",
    )
    for row, dataset in enumerate(DATASETS):
        for column, condition in enumerate(CONDITIONS):
            ax = axes[row, column]
            draw_panel(ax, aggregate, dataset, condition, show_legend=column == 0)
            if row == 0:
                ax.set_title(CONDITION_LABELS[condition], fontsize=10)
            if column == 0:
                ax.set_ylabel(
                    f"{DATASET_LABELS[dataset]}\nPositive local scale $c$",
                    fontsize=9,
                )
            if row == len(DATASETS) - 1:
                ax.set_xlabel("Training update", fontsize=9)
    fig.suptitle(
        "Named-source local-scale trajectories across all seeds",
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Loader slot j is remapped to the j-th original environment after "
        "removing the held-out target. Lines and bands are descriptive mean ± "
        "SD over 3 target contexts × 4 seeds (n=12; overlapping source data).",
        ha="center",
        va="bottom",
        fontsize=7.1,
    )
    fig.tight_layout(rect=(0.02, 0.045, 1.0, 0.98))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(
    aggregate: pd.DataFrame, dataset: str, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.45), sharex=True, sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        draw_panel(ax, aggregate, dataset, condition, show_legend=True)
        ax.set_title(CONDITION_LABELS[condition], fontsize=9.5)
        ax.set_xlabel("Training update", fontsize=8.5)
    axes[0].set_ylabel("Positive local scale $c$", fontsize=8.5)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: named-source local-scale trajectories "
        "across all seeds",
        fontsize=10.5,
        y=1.01,
    )
    fig.text(
        0.5,
        0.01,
        "Semantic source names are reconstructed from loader slots; mean ± SD "
        "over 3 target contexts × 4 seeds (n=12; descriptive, overlapping).",
        ha="center",
        va="bottom",
        fontsize=7.1,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_path = args.input_dir / "run_level_results.csv"
    if not run_path.is_file():
        raise SystemExit(f"missing finalized run table: {run_path}")
    runs = pd.read_csv(run_path)
    points, spreads = collect_named_points(runs)
    aggregate = aggregate_points(points)
    spread_summary = summarize_spreads(spreads)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.figure_dir / "curvature_scale_trajectories_named_all_seeds.pdf"
    plot_grid(aggregate, grid_path)
    for dataset in DATASETS:
        plot_dataset(
            aggregate,
            dataset,
            args.figure_dir
            / f"curvature_scale_trajectories_named_all_seeds_{dataset}.pdf",
        )

    points.to_csv(
        args.data_dir / "curvature_scale_trajectory_named_points.csv", index=False
    )
    aggregate.to_csv(
        args.data_dir / "curvature_scale_trajectory_named_summary.csv", index=False
    )
    spread_summary.to_csv(
        args.data_dir / "curvature_scale_spread_summary.csv", index=False
    )
    print(
        f"Validated {len(points):,} mapped values from 128 runs; "
        f"wrote {grid_path} and four dataset figures."
    )


if __name__ == "__main__":
    main()
