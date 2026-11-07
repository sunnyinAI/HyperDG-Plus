#!/usr/bin/env python3
"""Plot reviewer-requested learning curves and selection diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        # Embed searchable TrueType outlines in publication PDFs.  Matplotlib's
        # default PDF font type is Type 3, which many submission checkers reject.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


STUDY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = STUDY_ROOT / "outputs" / "reviewer_round2"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
DATASETS = ("pacs", "vlcs", "officehome", "terraincognita")
DATASET_LABELS = {
    "pacs": "PACS",
    "vlcs": "VLCS",
    "officehome": "OfficeHome",
    "terraincognita": "TerraIncognita",
}
CONDITION_ORDER = (
    "standard_erm",
    "matched_control",
    "full_per_domain",
    "no_variance",
    "shared_learnable",
    "frozen",
)
LABELS = {
    "standard_erm": "StandardERM",
    "matched_control": "MatchedControl",
    "full_per_domain": "FullPerDomain",
    "no_variance": "NoVariance",
    "shared_learnable": "SharedLearnable",
    "frozen": "Frozen",
}
COLORS = {
    "standard_erm": "#111111",
    "matched_control": "#7f7f7f",
    "full_per_domain": "#0072B2",
    "no_variance": "#009E73",
    "shared_learnable": "#D55E00",
    "frozen": "#CC79A7",
}


def available_conditions(frame: pd.DataFrame) -> list[str]:
    present = set(frame["condition"].unique())
    expected_targets = int(frame["target"].nunique()) if "target" in frame else 0
    complete: list[str] = []
    for condition in CONDITION_ORDER:
        if condition not in present:
            continue
        subset = frame[frame["condition"] == condition]
        if "target" not in subset or "seed" not in subset:
            complete.append(condition)
            continue
        seed_counts = subset.groupby("target")["seed"].nunique()
        if len(seed_counts) == expected_targets and bool((seed_counts == 4).all()):
            complete.append(condition)
    return complete


def plot_dataset_macro(curves: pd.DataFrame, dataset: str) -> None:
    subset = curves[curves["dataset"] == dataset]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharex=True)
    metrics = (
        ("source_val_acc", "Mean source-validation accuracy (%)"),
        (
            "target_out_acc_posthoc",
            "Held-out target accuracy (%) (post-hoc)",
        ),
    )
    for metric, ylabel in metrics:
        ax = axes[0] if metric == "source_val_acc" else axes[1]
        for condition in available_conditions(subset):
            condition_frame = subset[subset["condition"] == condition]
            # Average seeds inside each target before averaging target domains.
            by_target = (
                condition_frame.groupby(["step", "target"], as_index=False)[metric]
                .mean()
            )
            macro = by_target.groupby("step")[metric].agg(["mean", "min", "max"])
            ax.plot(
                macro.index,
                macro["mean"],
                color=COLORS[condition],
                linewidth=2.0,
                label=LABELS[condition],
            )
            ax.fill_between(
                macro.index,
                macro["min"],
                macro["max"],
                color=COLORS[condition],
                alpha=0.08,
                linewidth=0,
            )
        ax.set_xlabel("Training update")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: target-balanced learning curves "
        "(source validation uses one stochastic pass per checkpoint; "
        "target panel is post-hoc)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / f"learning_curves_{dataset}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_targets(
    curves: pd.DataFrame, runs: pd.DataFrame, dataset: str
) -> None:
    subset = curves[curves["dataset"] == dataset]
    if subset.empty:
        return
    targets = (
        subset[["target", "target_name"]]
        .drop_duplicates()
        .sort_values("target")
        .to_records(index=False)
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), sharex=True)
    for ax, (target, target_name) in zip(axes.flat, targets):
        target_frame = subset[subset["target"] == target]
        for condition in available_conditions(target_frame):
            condition_frame = target_frame[target_frame["condition"] == condition]
            curve = condition_frame.groupby("step")[
                "target_out_acc_posthoc"
            ].agg(["mean", "std", "count"])
            ax.plot(
                curve.index,
                curve["mean"],
                color=COLORS[condition],
                linewidth=1.8,
                label=LABELS[condition],
            )
            if int(curve["count"].max()) > 1:
                sd = curve["std"].fillna(0.0)
                ax.fill_between(
                    curve.index,
                    curve["mean"] - sd,
                    curve["mean"] + sd,
                    color=COLORS[condition],
                    alpha=0.09,
                    linewidth=0,
                )
            selected = runs[
                (runs["dataset"] == dataset)
                & (runs["target"] == target)
                & (runs["condition"] == condition)
            ]
            if not selected.empty:
                ax.scatter(
                    selected["source_selected_step"],
                    selected["source_selected_target_out_acc"],
                    marker="x",
                    s=28,
                    linewidths=1.1,
                    color=COLORS[condition],
                    zorder=5,
                )
        ax.set_title(f"Target {target_name}")
        ax.set_xlabel("Training update")
        ax.set_ylabel("Target accuracy (%)")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        f"{DATASET_LABELS[dataset]}: post-hoc target curves "
        "(mean ± run-seed SD); "
        "x marks source-validation-selected checkpoints",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_ROOT / f"learning_curves_by_target_{dataset}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_selected_steps(run_df: pd.DataFrame) -> None:
    if run_df.empty:
        return
    datasets = [dataset for dataset in DATASETS if dataset in set(run_df["dataset"])]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=True)
    for ax, dataset in zip(axes.flat, datasets):
        subset = run_df[run_df["dataset"] == dataset]
        conditions = available_conditions(subset)
        values = [
            subset[subset["condition"] == condition][
                "source_selected_step"
            ].to_numpy(float)
            for condition in conditions
        ]
        ax.boxplot(values, tick_labels=[LABELS[c] for c in conditions], showfliers=True)
        ax.set_title(DATASET_LABELS[dataset])
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_ylabel("Source-selected update")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Checkpoint steps selected without target-domain data")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "source_selected_steps.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_primary_domain_forests(input_dir: Path = OUTPUT_ROOT) -> None:
    domain_path = input_dir / "domain_effects.csv"
    comparison_path = input_dir / "clustered_comparisons.csv"
    suite_path = input_dir / "suite_comparisons.csv"
    if not all(path.is_file() for path in (domain_path, comparison_path, suite_path)):
        return
    domains = pd.read_csv(domain_path)
    comparisons = pd.read_csv(comparison_path)
    suites = pd.read_csv(suite_path)
    primary = (
        ("full_minus_standard_erm", "FullPerDomain − StandardERM"),
        (
            "no_variance_minus_shared",
            "NoVariance − SharedLearnable",
        ),
    )
    for contrast, title in primary:
        points = domains[
            (domains["contrast"] == contrast)
            & (domains["endpoint"] == "source_selected")
            & (domains["n_seed_pairs"] == 4)
        ].copy()
        summaries = comparisons[
            (comparisons["contrast"] == contrast)
            & (comparisons["endpoint"] == "source_selected")
            & comparisons["complete_for_inference"].fillna(False)
        ]
        macro = suites[
            (suites["contrast"] == contrast)
            & (suites["endpoint"] == "source_selected")
        ]
        if len(points) != 16 or len(summaries) != 4 or len(macro) != 1:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 9.2))
        ax.axvspan(-1.0, 1.0, color="#999999", alpha=0.12, label="±1 pp SESOI")
        ax.axvline(0.0, color="black", linewidth=1.0)
        y = 0
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for dataset in DATASETS:
            dataset_points = points[points["dataset"] == dataset].sort_values(
                "target"
            )
            for row in dataset_points.itertuples():
                seed_ci = (
                    float(stats.t.ppf(0.975, 3))
                    * float(row.run_seed_sd_difference)
                    / 2.0
                )
                ax.errorbar(
                    float(row.mean_difference),
                    y,
                    xerr=seed_ci,
                    fmt="o",
                    color="#4C78A8",
                    ecolor="#9ecae1",
                    capsize=2,
                    markersize=4,
                )
                tick_positions.append(y)
                tick_labels.append(
                    f"{DATASET_LABELS[dataset]} / {row.target_name}"
                )
                y += 1
            summary = summaries[summaries["dataset"] == dataset].iloc[0]
            lower = float(summary["effect_pp"] - summary["ci95_t_low"])
            upper = float(summary["ci95_t_high"] - summary["effect_pp"])
            ax.errorbar(
                float(summary["effect_pp"]),
                y,
                xerr=np.asarray([[lower], [upper]]),
                fmt="D",
                color="#D55E00",
                capsize=4,
                markersize=6,
                linewidth=2,
            )
            tick_positions.append(y)
            tick_labels.append(
                f"{DATASET_LABELS[dataset]} macro (target-cluster CI)"
            )
            y += 1.1

        macro_row = macro.iloc[0]
        lower = float(macro_row["effect_pp"] - macro_row["ci95_t_low"])
        upper = float(macro_row["ci95_t_high"] - macro_row["effect_pp"])
        ax.errorbar(
            float(macro_row["effect_pp"]),
            y,
            xerr=np.asarray([[lower], [upper]]),
            fmt="s",
            color="black",
            capsize=5,
            markersize=7,
            linewidth=2.2,
        )
        tick_positions.append(y)
        tick_labels.append("Four-dataset macro (dataset-cluster CI)")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Source-selected paired accuracy effect (percentage points)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(
            FIGURE_ROOT / f"domain_effect_forest_{contrast}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_primary_seed_heatmaps(input_dir: Path = OUTPUT_ROOT) -> None:
    path = input_dir / "paired_seed_effects.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    primary = (
        ("full_minus_standard_erm", "FullPerDomain − StandardERM"),
        ("no_variance_minus_shared", "NoVariance − SharedLearnable"),
    )
    for contrast, title in primary:
        subset = frame[
            (frame["contrast"] == contrast)
            & (frame["endpoint"] == "source_selected")
        ].copy()
        pivot = subset.pivot_table(
            index=["dataset", "target", "target_name"],
            columns="seed",
            values="difference",
            aggfunc="first",
        )
        if pivot.shape != (16, 4) or pivot.isna().any().any():
            continue
        values = pivot.to_numpy(float)
        limit = max(1.0, float(np.nanmax(np.abs(values))))
        fig, ax = plt.subplots(figsize=(6.8, 7.0))
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xticks(range(4), labels=[f"seed {seed}" for seed in range(4)])
        ax.set_yticks(
            range(16),
            labels=[
                f"{DATASET_LABELS[dataset]} / {target_name}"
                for dataset, _, target_name in pivot.index
            ],
            fontsize=8,
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        ax.set_title(f"{title}: paired seed effects (pp)")
        fig.colorbar(image, ax=ax, label="Accuracy difference (pp)")
        fig.tight_layout()
        fig.savefig(
            FIGURE_ROOT / f"paired_seed_heatmap_{contrast}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory containing the analysis CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_ROOT,
        help="Directory for generated PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    global FIGURE_ROOT
    args = parse_args()
    FIGURE_ROOT = args.output_dir
    curves_path = args.input_dir / "learning_curve_points.csv"
    runs_path = args.input_dir / "run_level_results.csv"
    if not curves_path.is_file() or not runs_path.is_file():
        raise SystemExit("Run analyze_review_round2.py before plotting.")
    curves = pd.read_csv(curves_path)
    runs = pd.read_csv(runs_path)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        plot_dataset_macro(curves, dataset)
        plot_dataset_targets(curves, runs, dataset)
    plot_selected_steps(runs)
    plot_primary_domain_forests(args.input_dir)
    plot_primary_seed_heatmaps(args.input_dir)
    print(f"Wrote figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
