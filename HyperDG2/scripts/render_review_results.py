#!/usr/bin/env python3
"""Render the final reviewer-round analysis into paper-ready tables.

This script is deliberately downstream-only: it reads the analysis contract
(``readiness.json`` and finalized CSV files) and never reads raw metrics,
checkpoints, configs, or training logs.  It performs no statistical
recomputation.  Numerical fields are formatted, tables are reshaped, and
claim labels are derived only from the analyzer's reported adjusted p-values
and equivalence flags.

Rendering is fail-closed.  No output is written unless all 384 runs are ready,
all six conditions are complete, and both primary contrasts have complete
dataset- and suite-level results for source-selected and final-step endpoints.
The suite-level primary family must use the four dataset means as inference
clusters; the per-dataset eight-test family is rendered only as a sensitivity.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = STUDY_ROOT / "outputs" / "reviewer_round2"
DEFAULT_OUTPUT = DEFAULT_INPUT / "rendered"

DATASETS = ("pacs", "vlcs", "officehome", "terraincognita")
DATASET_LABELS = {
    "pacs": "PACS",
    "vlcs": "VLCS",
    "officehome": "OfficeHome",
    "terraincognita": "TerraIncognita",
}
CONDITIONS = (
    "standard_erm",
    "matched_control",
    "full_per_domain",
    "frozen",
    "no_variance",
    "shared_learnable",
)
CONDITION_LABELS = {
    "standard_erm": "Standard ERM",
    "matched_control": "Matched control",
    "full_per_domain": "Full per-domain",
    "frozen": "Frozen",
    "no_variance": "No-variance per-domain",
    "shared_learnable": "Shared learnable",
}
ENDPOINTS = ("source_selected", "final_step")
PRIMARY_CONTRASTS = (
    "full_minus_standard_erm",
    "no_variance_minus_shared",
)
PRIMARY_ROLES = {
    "full_minus_standard_erm": "primary_utility",
    "no_variance_minus_shared": "primary_domain_specificity",
}
CONTRAST_LABELS = {
    "full_minus_standard_erm": "Full per-domain - Standard ERM",
    "no_variance_minus_shared": (
        "No-variance per-domain - Shared learnable"
    ),
}
TARGET_NAMES = {
    "pacs": ("ArtPainting", "Cartoon", "Photo", "Sketch"),
    "vlcs": ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
    "officehome": ("Art", "Clipart", "Product", "RealWorld"),
    "terraincognita": ("L100", "L38", "L43", "L46"),
}

CSV_FILES = {
    "condition_summary": "condition_summary.csv",
    "comparisons": "clustered_comparisons.csv",
    "suite": "suite_comparisons.csv",
    "selection": "model_selection_summary.csv",
    "targets": "per_target_results.csv",
    "domain_effects": "domain_effects.csv",
    "effect_variation": "effect_variance_decomposition.csv",
    "target_in": "target_in_sensitivity.csv",
    "equivalence": "equivalence_sensitivity.csv",
}


class RenderContractError(RuntimeError):
    """Raised when finalized analysis artifacts violate the render contract."""


@dataclass(frozen=True)
class AnalysisBundle:
    readiness: Mapping[str, Any]
    frames: Mapping[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing readiness.json and finalized analysis CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for RESULTS_READY.md and LaTeX fragments.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RenderContractError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RenderContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RenderContractError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise RenderContractError(f"missing required artifact: {path}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise RenderContractError(f"invalid CSV in {path}: {exc}") from exc
    if frame.empty:
        raise RenderContractError(f"required artifact is empty: {path}")
    return frame


def load_bundle(input_dir: Path) -> AnalysisBundle:
    """Load only the finalized analysis contract, never raw experiment data."""
    readiness = _read_json(input_dir / "readiness.json")
    frames = {
        key: _read_csv(input_dir / filename)
        for key, filename in CSV_FILES.items()
    }
    return AnalysisBundle(readiness=readiness, frames=frames)


def _require_columns(
    frame: pd.DataFrame, name: str, columns: Iterable[str]
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RenderContractError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )


def _require_unique(
    frame: pd.DataFrame, name: str, columns: Iterable[str]
) -> None:
    keys = list(columns)
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        example = frame.loc[duplicate, keys].iloc[0].to_dict()
        raise RenderContractError(
            f"{name} contains duplicate key {example}"
        )


def _require_finite(
    frame: pd.DataFrame, name: str, columns: Iterable[str]
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise RenderContractError(
                f"{name}.{column} contains a missing or non-finite value"
            )


def _bool_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RenderContractError(f"{name} contains a non-boolean value")
    return normalized.eq("true")


def _expected_grid(*axes: Iterable[Any]) -> set[tuple[Any, ...]]:
    result: set[tuple[Any, ...]] = {()}
    for axis in axes:
        result = {prefix + (value,) for prefix in result for value in axis}
    return result


def _require_grid(
    frame: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    expected: set[tuple[Any, ...]],
) -> None:
    actual = set(frame.loc[:, list(columns)].itertuples(index=False, name=None))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} keys (first: {missing[0]})")
        if extra:
            details.append(f"unexpected {len(extra)} keys (first: {extra[0]})")
        raise RenderContractError(f"{name} grid mismatch: {'; '.join(details)}")


def validate_bundle(bundle: AnalysisBundle) -> None:
    """Validate completeness and schemas before any rendered file is written."""
    readiness = bundle.readiness
    expected_readiness = {
        "complete_runs": 384,
        "expected_runs": 384,
        "missing_runs": 0,
        "source_selection_target_keys_used": False,
        "per_dataset_inference_cluster": "held-out target domain",
        "suite_inference_cluster": "benchmark dataset",
        "primary_multiplicity_family": (
            "two source-selected suite contrasts with Holm correction"
        ),
        "run_seed_components": (
            "data_split+initialization+minibatch_order+augmentation"
        ),
        "sesoi_timing": (
            "frozen_during_revision_before_added_grid_completed"
        ),
    }
    for key, expected in expected_readiness.items():
        if readiness.get(key) != expected:
            raise RenderContractError(
                f"readiness.{key} must be {expected!r}, "
                f"found {readiness.get(key)!r}"
            )
    if not math.isclose(
        float(readiness.get("primary_sesoi_pp", math.nan)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RenderContractError(
            "readiness.primary_sesoi_pp must equal the frozen 1.0 pp SESOI"
        )
    condition_counts = readiness.get("conditions_complete")
    if not isinstance(condition_counts, dict):
        raise RenderContractError(
            "readiness.conditions_complete must be an object"
        )
    if condition_counts != {condition: 64 for condition in CONDITIONS}:
        raise RenderContractError(
            "readiness.conditions_complete must contain exactly 64 runs "
            "for each of the six conditions"
        )

    frames = bundle.frames
    summary = frames["condition_summary"]
    _require_columns(
        summary,
        "condition_summary",
        (
            "condition",
            "dataset",
            "endpoint",
            "n_domains",
            "n_runs",
            "macro_mean_accuracy",
            "between_target_sd_of_means",
            "rms_within_target_run_seed_sd",
        ),
    )
    _require_unique(
        summary, "condition_summary", ("condition", "dataset", "endpoint")
    )
    _require_grid(
        summary,
        "condition_summary",
        ("condition", "dataset", "endpoint"),
        _expected_grid(CONDITIONS, DATASETS, ENDPOINTS),
    )
    if not (summary["n_domains"].eq(4) & summary["n_runs"].eq(16)).all():
        raise RenderContractError(
            "condition_summary must report 4 domains and 16 runs in every cell"
        )
    _require_finite(
        summary,
        "condition_summary",
        (
            "macro_mean_accuracy",
            "between_target_sd_of_means",
            "rms_within_target_run_seed_sd",
        ),
    )

    comparisons = frames["comparisons"]
    _require_columns(
        comparisons,
        "clustered_comparisons",
        (
            "contrast",
            "role",
            "scope",
            "dataset",
            "endpoint",
            "n_domains",
            "n_seed_pairs",
            "complete_for_inference",
            "effect_pp",
            "ci95_t_low",
            "ci95_t_high",
            "exact_signflip_p",
            "holm_p_dataset_8_sensitivity",
            "sesoi_pp",
            "ci90_t_low",
            "ci90_t_high",
            "tost_p",
            "equivalent_at_0p05",
        ),
    )
    primary_comparisons = comparisons[
        comparisons["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    _require_unique(
        primary_comparisons,
        "primary clustered_comparisons",
        ("contrast", "dataset", "endpoint"),
    )
    _require_grid(
        primary_comparisons,
        "primary clustered_comparisons",
        ("contrast", "dataset", "endpoint"),
        _expected_grid(PRIMARY_CONTRASTS, DATASETS, ENDPOINTS),
    )
    if not _bool_series(
        primary_comparisons["complete_for_inference"],
        "clustered_comparisons.complete_for_inference",
    ).all():
        raise RenderContractError(
            "both primary contrasts must be complete for inference"
        )
    if not (
        primary_comparisons["n_domains"].eq(4)
        & primary_comparisons["n_seed_pairs"].eq(16)
    ).all():
        raise RenderContractError(
            "primary contrasts must contain 4 domains and 16 seed pairs"
        )
    expected_roles = primary_comparisons["contrast"].map(PRIMARY_ROLES)
    if not primary_comparisons["role"].eq(expected_roles).all():
        raise RenderContractError("primary contrast roles do not match protocol")
    _require_finite(
        primary_comparisons,
        "primary clustered_comparisons",
        (
            "effect_pp",
            "ci95_t_low",
            "ci95_t_high",
            "exact_signflip_p",
            "holm_p_dataset_8_sensitivity",
            "sesoi_pp",
            "ci90_t_low",
            "ci90_t_high",
            "tost_p",
        ),
    )

    suite = frames["suite"]
    _require_columns(
        suite,
        "suite_comparisons",
        (
            "contrast",
            "role",
            "endpoint",
            "inference_cluster",
            "n_datasets",
            "n_dataset_clusters",
            "n_target_domains",
            "effect_pp",
            "ci95_t_low",
            "ci95_t_high",
            "exact_4_dataset_signflip_p",
            "holm_p_primary_2",
            "sesoi_pp",
            "ci90_t_low",
            "ci90_t_high",
            "tost_p",
            "equivalent_at_0p05",
        ),
    )
    primary_suite = suite[suite["contrast"].isin(PRIMARY_CONTRASTS)].copy()
    _require_unique(
        primary_suite, "primary suite_comparisons", ("contrast", "endpoint")
    )
    _require_grid(
        primary_suite,
        "primary suite_comparisons",
        ("contrast", "endpoint"),
        _expected_grid(PRIMARY_CONTRASTS, ENDPOINTS),
    )
    if not (
        primary_suite["n_datasets"].eq(4)
        & primary_suite["n_dataset_clusters"].eq(4)
        & primary_suite["n_target_domains"].eq(16)
    ).all():
        raise RenderContractError(
            "primary suite contrasts must use 4 dataset inference clusters "
            "while retaining 16 contributing target domains"
        )
    if not primary_suite["inference_cluster"].eq(
        "benchmark_dataset"
    ).all():
        raise RenderContractError(
            "primary suite inference_cluster must be benchmark_dataset"
        )
    expected_suite_roles = primary_suite["contrast"].map(PRIMARY_ROLES)
    if not primary_suite["role"].eq(expected_suite_roles).all():
        raise RenderContractError(
            "primary suite contrast roles do not match protocol"
        )
    _require_finite(
        primary_suite,
        "primary suite_comparisons",
        (
            "effect_pp",
            "ci95_t_low",
            "ci95_t_high",
            "exact_4_dataset_signflip_p",
            "holm_p_primary_2",
            "sesoi_pp",
            "ci90_t_low",
            "ci90_t_high",
            "tost_p",
        ),
    )

    selection = frames["selection"]
    _require_columns(
        selection,
        "model_selection_summary",
        (
            "condition",
            "dataset",
            "n_runs",
            "complete_four_by_four",
            "source_selected_macro_accuracy",
            "final_step_macro_accuracy",
            "selected_minus_final_pp",
        ),
    )
    _require_unique(
        selection, "model_selection_summary", ("condition", "dataset")
    )
    _require_grid(
        selection,
        "model_selection_summary",
        ("condition", "dataset"),
        _expected_grid(CONDITIONS, DATASETS),
    )
    if not (
        selection["n_runs"].eq(16)
        & _bool_series(
            selection["complete_four_by_four"],
            "model_selection_summary.complete_four_by_four",
        )
    ).all():
        raise RenderContractError(
            "model_selection_summary must be complete four-by-four"
        )
    _require_finite(
        selection,
        "model_selection_summary",
        (
            "source_selected_macro_accuracy",
            "final_step_macro_accuracy",
            "selected_minus_final_pp",
        ),
    )

    targets = frames["targets"]
    target_seed_columns = tuple(
        f"seed{seed}_accuracy" for seed in range(4)
    )
    _require_columns(
        targets,
        "per_target_results",
        (
            "condition",
            "dataset",
            "endpoint",
            "target",
            "target_name",
            "n_seeds",
            "mean_accuracy",
            "run_seed_sd",
            *target_seed_columns,
        ),
    )
    _require_unique(
        targets,
        "per_target_results",
        ("condition", "dataset", "endpoint", "target"),
    )
    _require_grid(
        targets,
        "per_target_results",
        ("condition", "dataset", "endpoint", "target"),
        _expected_grid(CONDITIONS, DATASETS, ENDPOINTS, range(4)),
    )
    if not targets["n_seeds"].eq(4).all():
        raise RenderContractError(
            "per_target_results must contain four seeds in every cell"
        )
    expected_target_names = targets.apply(
        lambda row: TARGET_NAMES[row["dataset"]][int(row["target"])], axis=1
    )
    if not targets["target_name"].eq(expected_target_names).all():
        raise RenderContractError("per_target_results target names are invalid")
    _require_finite(
        targets,
        "per_target_results",
        ("mean_accuracy", "run_seed_sd", *target_seed_columns),
    )

    effects = frames["domain_effects"]
    _require_columns(
        effects,
        "domain_effects",
        (
            "contrast",
            "role",
            "scope",
            "dataset",
            "endpoint",
            "target",
            "target_name",
            "n_seed_pairs",
            "mean_difference",
            "run_seed_sd_difference",
        ),
    )
    primary_effects = effects[
        effects["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    _require_unique(
        primary_effects,
        "primary domain_effects",
        ("contrast", "dataset", "endpoint", "target"),
    )
    _require_grid(
        primary_effects,
        "primary domain_effects",
        ("contrast", "dataset", "endpoint", "target"),
        _expected_grid(PRIMARY_CONTRASTS, DATASETS, ENDPOINTS, range(4)),
    )
    if not primary_effects["n_seed_pairs"].eq(4).all():
        raise RenderContractError(
            "primary domain effects must contain four paired seeds"
        )
    expected_effect_names = primary_effects.apply(
        lambda row: TARGET_NAMES[row["dataset"]][int(row["target"])], axis=1
    )
    if not primary_effects["target_name"].eq(expected_effect_names).all():
        raise RenderContractError("domain_effects target names are invalid")
    _require_finite(
        primary_effects,
        "primary domain_effects",
        ("mean_difference", "run_seed_sd_difference"),
    )

    effect_variation = frames["effect_variation"]
    _require_columns(
        effect_variation,
        "effect_variance_decomposition",
        (
            "contrast",
            "role",
            "scope",
            "dataset",
            "endpoint",
            "grand_mean_effect_pp",
            "between_target_sd_of_mean_effects",
            "between_run_seed_sd_of_mean_effects",
            "rms_within_target_run_seed_sd_of_effects",
            "target_by_run_seed_residual_sd_of_effects",
            "run_seed_components",
        ),
    )
    primary_effect_variation = effect_variation[
        effect_variation["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    _require_unique(
        primary_effect_variation,
        "primary effect_variance_decomposition",
        ("contrast", "dataset", "endpoint"),
    )
    _require_grid(
        primary_effect_variation,
        "primary effect_variance_decomposition",
        ("contrast", "dataset", "endpoint"),
        _expected_grid(PRIMARY_CONTRASTS, DATASETS, ENDPOINTS),
    )
    expected_effect_roles = primary_effect_variation["contrast"].map(
        PRIMARY_ROLES
    )
    if not primary_effect_variation["role"].eq(expected_effect_roles).all():
        raise RenderContractError(
            "primary effect variability roles do not match protocol"
        )
    if not primary_effect_variation["run_seed_components"].eq(
        "data_split+initialization+minibatch_order+augmentation"
    ).all():
        raise RenderContractError(
            "effect variability must describe the complete run-seed components"
        )
    _require_finite(
        primary_effect_variation,
        "primary effect_variance_decomposition",
        (
            "grand_mean_effect_pp",
            "between_target_sd_of_mean_effects",
            "between_run_seed_sd_of_mean_effects",
            "rms_within_target_run_seed_sd_of_effects",
            "target_by_run_seed_residual_sd_of_effects",
        ),
    )

    target_in = frames["target_in"]
    _require_columns(
        target_in,
        "target_in_sensitivity",
        (
            "record_type",
            "condition",
            "contrast",
            "role",
            "dataset",
            "endpoint",
            "n_targets",
            "n_runs",
            "macro_mean_accuracy",
            "between_target_sd_of_means",
            "rms_within_target_run_seed_sd",
            "effect_pp",
            "ci95_t_low",
            "ci95_t_high",
            "exact_signflip_p",
        ),
    )
    target_in_endpoints = (
        "source_selected_target_in",
        "final_step_target_in",
    )
    target_in_conditions = target_in[
        target_in["record_type"].eq("condition_summary")
    ].copy()
    _require_unique(
        target_in_conditions,
        "target_in condition summaries",
        ("condition", "dataset", "endpoint"),
    )
    _require_grid(
        target_in_conditions,
        "target_in condition summaries",
        ("condition", "dataset", "endpoint"),
        _expected_grid(CONDITIONS, DATASETS, target_in_endpoints),
    )
    if not (
        target_in_conditions["n_targets"].eq(4)
        & target_in_conditions["n_runs"].eq(16)
    ).all():
        raise RenderContractError(
            "target_in condition summaries must contain 4 targets and 16 runs"
        )
    _require_finite(
        target_in_conditions,
        "target_in condition summaries",
        (
            "macro_mean_accuracy",
            "between_target_sd_of_means",
            "rms_within_target_run_seed_sd",
        ),
    )
    target_in_primary = target_in[
        target_in["record_type"].eq("primary_contrast")
        & target_in["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    _require_unique(
        target_in_primary,
        "target_in primary contrasts",
        ("contrast", "dataset", "endpoint"),
    )
    _require_grid(
        target_in_primary,
        "target_in primary contrasts",
        ("contrast", "dataset", "endpoint"),
        _expected_grid(PRIMARY_CONTRASTS, DATASETS, target_in_endpoints),
    )
    if not (
        target_in_primary["n_targets"].eq(4)
        & target_in_primary["n_runs"].eq(16)
    ).all():
        raise RenderContractError(
            "target_in primary contrasts must contain 4 targets and 16 runs"
        )
    expected_target_in_roles = target_in_primary["contrast"].map(
        PRIMARY_ROLES
    )
    if not target_in_primary["role"].eq(expected_target_in_roles).all():
        raise RenderContractError(
            "target_in primary contrast roles do not match protocol"
        )
    _require_finite(
        target_in_primary,
        "target_in primary contrasts",
        ("effect_pp", "ci95_t_low", "ci95_t_high", "exact_signflip_p"),
    )

    equivalence = frames["equivalence"]
    _require_columns(
        equivalence,
        "equivalence_sensitivity",
        (
            "contrast",
            "role",
            "scope",
            "dataset",
            "endpoint",
            "margin_pp",
            "effect_pp",
            "ci90_t_low",
            "ci90_t_high",
            "tost_p",
            "equivalent_at_0p05",
            "primary_equivalence_question",
        ),
    )
    primary_equivalence = equivalence[
        equivalence["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    allowed_scopes = {"dataset", "suite"}
    actual_scopes = set(primary_equivalence["scope"].dropna().astype(str))
    if actual_scopes != allowed_scopes:
        raise RenderContractError(
            "primary equivalence_sensitivity must contain exactly "
            "scope={dataset,suite}"
        )
    dataset_equivalence = primary_equivalence[
        primary_equivalence["scope"].eq("dataset")
    ].copy()
    _require_unique(
        dataset_equivalence,
        "dataset primary equivalence_sensitivity",
        ("contrast", "dataset", "endpoint", "margin_pp"),
    )
    _require_grid(
        dataset_equivalence,
        "dataset primary equivalence_sensitivity",
        ("contrast", "dataset", "endpoint", "margin_pp"),
        _expected_grid(
            PRIMARY_CONTRASTS, DATASETS, ENDPOINTS, (0.5, 1.0, 2.0)
        ),
    )
    suite_equivalence = primary_equivalence[
        primary_equivalence["scope"].eq("suite")
    ].copy()
    _require_unique(
        suite_equivalence,
        "suite primary equivalence_sensitivity",
        ("contrast", "endpoint", "margin_pp"),
    )
    _require_grid(
        suite_equivalence,
        "suite primary equivalence_sensitivity",
        ("contrast", "endpoint", "margin_pp"),
        _expected_grid(PRIMARY_CONTRASTS, ENDPOINTS, (0.5, 1.0, 2.0)),
    )
    _require_finite(
        primary_equivalence,
        "primary equivalence_sensitivity",
        ("effect_pp", "ci90_t_low", "ci90_t_high", "tost_p"),
    )
    primary_flags = _bool_series(
        primary_equivalence["primary_equivalence_question"],
        "equivalence_sensitivity.primary_equivalence_question",
    )
    if int(primary_flags.sum()) != 1:
        raise RenderContractError(
            "equivalence_sensitivity must identify exactly one primary "
            "equivalence question"
        )
    flagged = primary_equivalence.loc[primary_flags].iloc[0]
    if not (
        flagged["scope"] == "suite"
        and flagged["contrast"] == "no_variance_minus_shared"
        and flagged["endpoint"] == "source_selected"
        and math.isclose(float(flagged["margin_pp"]), 1.0, abs_tol=1e-12)
    ):
        raise RenderContractError(
            "the primary equivalence question must be the source-selected "
            "suite NoVariance-minus-Shared contrast at +/-1.0 pp"
        )


def _ordered(
    frame: pd.DataFrame,
    columns_and_orders: tuple[tuple[str, tuple[str, ...]], ...],
) -> pd.DataFrame:
    result = frame.copy()
    sort_columns = []
    helper_columns = []
    for index, (column, order) in enumerate(columns_and_orders):
        helper = f"__order_{index}"
        mapping = {value: rank for rank, value in enumerate(order)}
        result[helper] = result[column].map(mapping)
        sort_columns.append(helper)
        helper_columns.append(helper)
    return result.sort_values(sort_columns).drop(columns=helper_columns)


def _number(value: Any, digits: int = 2) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def _p_value(value: Any) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return "--"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def _interval(low: Any, high: Any) -> str:
    return f"[{_number(low)}, {_number(high)}]"


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise RenderContractError(f"cannot interpret boolean value {value!r}")
    return normalized == "true"


def _zero_effect_label(adjusted_p: Any) -> str:
    if float(adjusted_p) < 0.05:
        return "Adjusted evidence against zero"
    return "No adjusted evidence against zero"


def _equivalence_label(
    equivalent: Any, margin: Any, scope: str = "suite"
) -> str:
    prefix = "Suite macro" if scope == "suite" else "Dataset macro"
    if _truth(equivalent):
        return (
            f"{prefix} equivalent within +/-{_number(margin, 1)} pp"
        )
    return f"{prefix} equivalence not demonstrated"


def _equivalence_role(primary_equivalence_question: Any) -> str:
    if _truth(primary_equivalence_question):
        return "Primary revision-stage equivalence"
    return "Sensitivity"


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(_md_escape(value) for value in headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(_md_escape(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_tabular(
    alignment: str, headers: list[str], rows: list[list[Any]]
) -> str:
    lines = [
        "% Generated by HyperDG2/scripts/render_review_results.py.",
        "% Requires \\usepackage{booktabs}.",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        "\\midrule",
    ]
    lines.extend(
        " & ".join(_latex_escape(value) for value in row) + r" \\"
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def latex_longtable(
    alignment: str, headers: list[str], rows: list[list[Any]]
) -> str:
    header = " & ".join(_latex_escape(value) for value in headers) + r" \\"
    lines = [
        "% Generated by HyperDG2/scripts/render_review_results.py.",
        "% Requires \\usepackage{booktabs,longtable}.",
        f"\\begin{{longtable}}{{{alignment}}}",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    lines.extend(
        " & ".join(_latex_escape(value) for value in row) + r" \\"
        for row in rows
    )
    lines.extend(["\\end{longtable}", ""])
    return "\n".join(lines)


def _source_summary_rows(frame: pd.DataFrame) -> list[list[str]]:
    source = frame[frame["endpoint"] == "source_selected"]
    source = _ordered(
        source,
        (
            ("condition", CONDITIONS),
            ("dataset", DATASETS),
        ),
    )
    return [
        [
            CONDITION_LABELS[row.condition],
            DATASET_LABELS[row.dataset],
            _number(row.macro_mean_accuracy),
            _number(row.between_target_sd_of_means),
            _number(row.rms_within_target_run_seed_sd),
        ]
        for row in source.itertuples(index=False)
    ]


def _primary_rows(
    comparisons: pd.DataFrame, suite: pd.DataFrame
) -> list[list[str]]:
    rows: list[list[str]] = []
    macro_rows = suite[
        suite["contrast"].isin(PRIMARY_CONTRASTS)
        & suite["endpoint"].eq("source_selected")
    ]
    macro_rows = _ordered(macro_rows, (("contrast", PRIMARY_CONTRASTS),))
    for row in macro_rows.itertuples(index=False):
        rows.append(
            [
                CONTRAST_LABELS[row.contrast],
                "Suite macro",
                "Primary Holm-2",
                _number(row.effect_pp),
                _interval(row.ci95_t_low, row.ci95_t_high),
                _p_value(row.exact_4_dataset_signflip_p),
                _p_value(row.holm_p_primary_2),
                _zero_effect_label(row.holm_p_primary_2),
                _interval(row.ci90_t_low, row.ci90_t_high),
                _p_value(row.tost_p),
                _equivalence_label(
                    row.equivalent_at_0p05, row.sesoi_pp, "suite"
                ),
            ]
        )
    dataset_rows = comparisons[
        comparisons["contrast"].isin(PRIMARY_CONTRASTS)
        & comparisons["endpoint"].eq("source_selected")
    ]
    dataset_rows = _ordered(
        dataset_rows,
        (("contrast", PRIMARY_CONTRASTS), ("dataset", DATASETS)),
    )
    for row in dataset_rows.itertuples(index=False):
        rows.append(
            [
                CONTRAST_LABELS[row.contrast],
                DATASET_LABELS[row.dataset],
                "Sensitivity Holm-8",
                _number(row.effect_pp),
                _interval(row.ci95_t_low, row.ci95_t_high),
                _p_value(row.exact_signflip_p),
                _p_value(row.holm_p_dataset_8_sensitivity),
                _zero_effect_label(row.holm_p_dataset_8_sensitivity),
                _interval(row.ci90_t_low, row.ci90_t_high),
                _p_value(row.tost_p),
                _equivalence_label(
                    row.equivalent_at_0p05, row.sesoi_pp, "dataset"
                ),
            ]
        )
    return rows


def _selection_rows(frame: pd.DataFrame) -> list[list[str]]:
    ordered = _ordered(
        frame, (("condition", CONDITIONS), ("dataset", DATASETS))
    )
    return [
        [
            CONDITION_LABELS[row.condition],
            DATASET_LABELS[row.dataset],
            _number(row.source_selected_macro_accuracy),
            _number(row.final_step_macro_accuracy),
            _number(row.selected_minus_final_pp),
        ]
        for row in ordered.itertuples(index=False)
    ]


def _target_accuracy_rows(frame: pd.DataFrame) -> list[list[str]]:
    source = frame[frame["endpoint"] == "source_selected"].copy()
    source = _ordered(
        source,
        (("condition", CONDITIONS), ("dataset", DATASETS)),
    ).sort_values(
        ["condition", "dataset", "target"],
        key=lambda column: (
            column.map({value: i for i, value in enumerate(CONDITIONS)})
            if column.name == "condition"
            else (
                column.map({value: i for i, value in enumerate(DATASETS)})
                if column.name == "dataset"
                else column
            )
        ),
    )
    return [
        [
            CONDITION_LABELS[row.condition],
            DATASET_LABELS[row.dataset],
            row.target_name,
            _number(row.seed0_accuracy),
            _number(row.seed1_accuracy),
            _number(row.seed2_accuracy),
            _number(row.seed3_accuracy),
            _number(row.mean_accuracy),
            _number(row.run_seed_sd),
        ]
        for row in source.itertuples(index=False)
    ]


def _target_effect_rows(frame: pd.DataFrame) -> list[list[str]]:
    source = frame[
        frame["contrast"].isin(PRIMARY_CONTRASTS)
        & frame["endpoint"].eq("source_selected")
    ]
    source = _ordered(
        source,
        (("contrast", PRIMARY_CONTRASTS), ("dataset", DATASETS)),
    ).sort_values(
        ["contrast", "dataset", "target"],
        key=lambda column: (
            column.map({value: i for i, value in enumerate(PRIMARY_CONTRASTS)})
            if column.name == "contrast"
            else (
                column.map({value: i for i, value in enumerate(DATASETS)})
                if column.name == "dataset"
                else column
            )
        ),
    )
    return [
        [
            CONTRAST_LABELS[row.contrast],
            DATASET_LABELS[row.dataset],
            row.target_name,
            _number(row.mean_difference),
            _number(row.run_seed_sd_difference),
        ]
        for row in source.itertuples(index=False)
    ]


def _effect_variability_rows(frame: pd.DataFrame) -> list[list[str]]:
    source = frame[
        frame["contrast"].isin(PRIMARY_CONTRASTS)
        & frame["endpoint"].eq("source_selected")
    ]
    source = _ordered(
        source,
        (("contrast", PRIMARY_CONTRASTS), ("dataset", DATASETS)),
    )
    return [
        [
            CONTRAST_LABELS[row.contrast],
            DATASET_LABELS[row.dataset],
            _number(row.grand_mean_effect_pp),
            _number(row.between_target_sd_of_mean_effects),
            _number(row.between_run_seed_sd_of_mean_effects),
            _number(row.rms_within_target_run_seed_sd_of_effects),
            _number(row.target_by_run_seed_residual_sd_of_effects),
        ]
        for row in source.itertuples(index=False)
    ]


def _target_in_rows(frame: pd.DataFrame) -> list[list[str]]:
    source = frame[
        frame["record_type"].eq("primary_contrast")
        & frame["contrast"].isin(PRIMARY_CONTRASTS)
        & frame["endpoint"].eq("source_selected_target_in")
    ]
    source = _ordered(
        source,
        (("contrast", PRIMARY_CONTRASTS), ("dataset", DATASETS)),
    )
    return [
        [
            CONTRAST_LABELS[row.contrast],
            DATASET_LABELS[row.dataset],
            _number(row.effect_pp),
            _interval(row.ci95_t_low, row.ci95_t_high),
            _p_value(row.exact_signflip_p),
        ]
        for row in source.itertuples(index=False)
    ]


def _equivalence_rows(equivalence: pd.DataFrame) -> list[list[str]]:
    source = equivalence[
        equivalence["contrast"].isin(PRIMARY_CONTRASTS)
        & equivalence["endpoint"].eq("source_selected")
    ]
    suite_rows = source[source["scope"].eq("suite")]
    suite_rows = _ordered(
        suite_rows,
        (("contrast", PRIMARY_CONTRASTS),),
    ).sort_values(
        ["contrast", "margin_pp"],
        key=lambda column: (
            column.map({value: i for i, value in enumerate(PRIMARY_CONTRASTS)})
            if column.name == "contrast"
            else column
        ),
    )
    rows = [
        [
            CONTRAST_LABELS[row.contrast],
            "Suite macro",
            _equivalence_role(row.primary_equivalence_question),
            _number(row.margin_pp, 1),
            _number(row.effect_pp),
            _interval(row.ci90_t_low, row.ci90_t_high),
            _p_value(row.tost_p),
            _equivalence_label(
                row.equivalent_at_0p05, row.margin_pp, "suite"
            ),
        ]
        for row in suite_rows.itertuples(index=False)
    ]
    dataset_rows = source[source["scope"].eq("dataset")]
    dataset_rows = _ordered(
        dataset_rows,
        (("contrast", PRIMARY_CONTRASTS), ("dataset", DATASETS)),
    ).sort_values(
        ["contrast", "dataset", "margin_pp"],
        key=lambda column: (
            column.map({value: i for i, value in enumerate(PRIMARY_CONTRASTS)})
            if column.name == "contrast"
            else (
                column.map({value: i for i, value in enumerate(DATASETS)})
                if column.name == "dataset"
                else column
            )
        ),
    )
    rows.extend(
        [
            CONTRAST_LABELS[row.contrast],
            DATASET_LABELS[row.dataset],
            _equivalence_role(row.primary_equivalence_question),
            _number(row.margin_pp, 1),
            _number(row.effect_pp),
            _interval(row.ci90_t_low, row.ci90_t_high),
            _p_value(row.tost_p),
            _equivalence_label(
                row.equivalent_at_0p05, row.margin_pp, "dataset"
            ),
        ]
        for row in dataset_rows.itertuples(index=False)
    )
    return rows


def _suite_narrative(suite: pd.DataFrame) -> list[str]:
    source = suite[
        suite["contrast"].isin(PRIMARY_CONTRASTS)
        & suite["endpoint"].eq("source_selected")
    ]
    source = _ordered(source, (("contrast", PRIMARY_CONTRASTS),))
    lines = []
    for row in source.itertuples(index=False):
        zero_claim = _zero_effect_label(row.holm_p_primary_2).lower()
        practical_claim = _equivalence_label(
            row.equivalent_at_0p05, row.sesoi_pp, "suite"
        ).lower()
        lines.append(
            f"- **{CONTRAST_LABELS[row.contrast]}:** "
            f"{_number(row.effect_pp)} pp "
            f"(95% CI {_interval(row.ci95_t_low, row.ci95_t_high)}; "
            f"Holm-adjusted p={_p_value(row.holm_p_primary_2)}). "
            f"Evidence relative to zero: {zero_claim}. Practical-effect "
            f"classification: {practical_claim}."
        )
    return lines


def build_rendered_files(bundle: AnalysisBundle) -> dict[str, str]:
    frames = bundle.frames
    summary_headers = [
        "Condition",
        "Dataset",
        "Accuracy (%)",
        "Between-target SD",
        "Within-target run-seed SD (RMS)",
    ]
    summary_rows = _source_summary_rows(frames["condition_summary"])
    primary_headers = [
        "Contrast",
        "Scope",
        "Multiplicity role",
        "Effect (pp)",
        "95% CI",
        "Exact p",
        "Holm p",
        "Zero-effect conclusion",
        "90% CI",
        "TOST p",
        "SESOI conclusion",
    ]
    primary_rows = _primary_rows(frames["comparisons"], frames["suite"])
    selection_headers = [
        "Condition",
        "Dataset",
        "Source-selected (%)",
        "Final-step (%)",
        "Selected - final (pp)",
    ]
    selection_rows = _selection_rows(frames["selection"])
    target_headers = [
        "Condition",
        "Dataset",
        "Target",
        "Seed 0",
        "Seed 1",
        "Seed 2",
        "Seed 3",
        "Mean",
        "Run-seed SD",
    ]
    target_rows = _target_accuracy_rows(frames["targets"])
    target_effect_headers = [
        "Contrast",
        "Dataset",
        "Target",
        "Mean paired effect (pp)",
        "Run-seed-pair SD",
    ]
    target_effect_rows = _target_effect_rows(frames["domain_effects"])
    effect_variability_headers = [
        "Contrast",
        "Dataset",
        "Mean effect (pp)",
        "Between-target SD",
        "Between-run-seed SD",
        "Within-target run-seed SD (RMS)",
        "Target x run-seed residual SD",
    ]
    effect_variability_rows = _effect_variability_rows(
        frames["effect_variation"]
    )
    target_in_headers = [
        "Contrast",
        "Dataset",
        "Target-in effect (pp)",
        "95% CI",
        "Exact p",
    ]
    target_in_rows = _target_in_rows(frames["target_in"])
    equivalence_headers = [
        "Contrast",
        "Scope",
        "Analysis role",
        "Margin (+/- pp)",
        "Effect (pp)",
        "90% CI",
        "TOST p",
        "Macro-equivalence conclusion",
    ]
    equivalence_rows = _equivalence_rows(frames["equivalence"])

    markdown = [
        "# Final reviewer-round results",
        "",
        "This report is rendered only after the finalized analysis records "
        "384/384 complete runs (six conditions, four datasets, four targets, "
        "and four seeds). Source-validation selection excludes the held-out "
        "target; target-domain accuracy is read only after selection. Effects "
        "are percentage-point treatment-minus-control differences.",
        "",
        "The held-out target is the inferential cluster for each dataset. "
        "Dataset estimates average four paired run-seed differences within "
        "each of four named targets. The suite primary analysis uses the four "
        "dataset means as its inference clusters while retaining all 16 "
        "target-domain contributions. Non-significance is not interpreted as "
        "equivalence.",
        "",
        "## Primary suite-level conclusions",
        "",
        *_suite_narrative(frames["suite"]),
        "",
        "## Six-condition source-selected summary",
        "",
        "Accuracy is the equal-target macro mean. The two SD columns keep "
        "between-target and within-target run-seed variability separate. A "
        "run seed combines data-split and optimization randomness.",
        "",
        markdown_table(summary_headers, summary_rows),
        "",
        "## Primary suite contrasts and per-dataset sensitivity",
        "",
        "The two suite tests are the primary Holm family. Their exact p-values "
        "sign-flip four dataset-mean effects. Per-dataset exact p-values "
        "sign-flip four target effects; their Holm-8 family is a sensitivity, "
        "not a second primary family. Evidence against zero and practical "
        "equivalence are separate conclusions and may both hold.",
        "",
        markdown_table(primary_headers, primary_rows),
        "",
        "## Source-selected versus final-step sensitivity",
        "",
        markdown_table(selection_headers, selection_rows),
        "",
        "## Named-target source-selected accuracies",
        "",
        "Every cell reports all four run-seed accuracies plus their mean and "
        "run-seed SD. Run seeds vary both the data split and optimization.",
        "",
        markdown_table(target_headers, target_rows),
        "",
        "## Named-target effects for the primary contrasts",
        "",
        markdown_table(target_effect_headers, target_effect_rows),
        "",
        "## Paired-effect variability decomposition",
        "",
        "These descriptive fields keep target, run-seed, and interaction "
        "variability separate for each primary paired effect. They are not "
        "additional inferential tests.",
        "",
        markdown_table(effect_variability_headers, effect_variability_rows),
        "",
        "## Held-out target in-split sensitivity",
        "",
        "The target out-split remains primary. These source-selected target "
        "in-split contrasts are sensitivity results and never enter checkpoint "
        "selection.",
        "",
        markdown_table(target_in_headers, target_in_rows),
        "",
        "## SESOI and equivalence sensitivity",
        "",
        "Equivalence is reported only when the analyzer's relevant clustered "
        "90% interval lies inside the stated symmetric margin and its TOST "
        "p-value is below 0.05. The +/-1.0 pp SESOI was frozen at the revision "
        "stage; it is not represented as preregistered or prespecified. "
        "Margins of +/-0.5 and +/-2.0 pp are sensitivity analyses. A suite "
        "conclusion concerns the equal-dataset macro effect, not every target.",
        "",
        markdown_table(equivalence_headers, equivalence_rows),
        "",
    ]

    return {
        "RESULTS_READY.md": "\n".join(markdown),
        "latex/six_condition_source_selected.tex": latex_tabular(
            "llrrr", summary_headers, summary_rows
        ),
        "latex/primary_contrasts.tex": latex_tabular(
            "llp{2.2cm}rrrrp{3.0cm}rrp{3.5cm}",
            primary_headers,
            primary_rows,
        ),
        "latex/checkpoint_sensitivity.tex": latex_tabular(
            "llrrr", selection_headers, selection_rows
        ),
        "latex/target_level_accuracies.tex": latex_longtable(
            "lllrrrrrr", target_headers, target_rows
        ),
        "latex/primary_target_effects.tex": latex_tabular(
            "lllrr", target_effect_headers, target_effect_rows
        ),
        "latex/effect_variability.tex": latex_tabular(
            "llrrrrr", effect_variability_headers, effect_variability_rows
        ),
        "latex/target_in_sensitivity.tex": latex_tabular(
            "llrrr", target_in_headers, target_in_rows
        ),
        "latex/equivalence_sensitivity.tex": latex_tabular(
            "llp{2.8cm}rrrrp{4.0cm}",
            equivalence_headers,
            equivalence_rows,
        ),
    }


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def render_results(input_dir: Path, output_dir: Path) -> list[Path]:
    bundle = load_bundle(input_dir)
    validate_bundle(bundle)
    rendered = build_rendered_files(bundle)
    paths = []
    for relative, content in rendered.items():
        path = output_dir / relative
        _write_atomic(path, content)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    try:
        paths = render_results(args.input_dir, args.output_dir)
    except RenderContractError as exc:
        raise SystemExit(f"refusing to render: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "rendered",
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "files": [str(path) for path in paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
