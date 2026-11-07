from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "scripts"))

import render_review_results as renderer  # noqa: E402


def write_complete_analysis(root: Path) -> None:
    readiness = {
        "complete_runs": 384,
        "expected_runs": 384,
        "missing_runs": 0,
        "conditions_complete": {
            condition: 64 for condition in renderer.CONDITIONS
        },
        "source_selection_target_keys_used": False,
        "per_dataset_inference_cluster": "held-out target domain",
        "suite_inference_cluster": "benchmark dataset",
        "primary_multiplicity_family": (
            "two source-selected suite contrasts with Holm correction"
        ),
        "run_seed_components": (
            "data_split+initialization+minibatch_order+augmentation"
        ),
        "primary_sesoi_pp": 1.0,
        "sesoi_timing": (
            "frozen_during_revision_before_added_grid_completed"
        ),
    }
    (root / "readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )

    summary = []
    for condition_index, condition in enumerate(renderer.CONDITIONS):
        for dataset_index, dataset in enumerate(renderer.DATASETS):
            for endpoint in renderer.ENDPOINTS:
                summary.append(
                    {
                        "condition": condition,
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "n_domains": 4,
                        "n_runs": 16,
                        "macro_mean_accuracy": (
                            70.0 + condition_index + dataset_index
                        ),
                        "between_target_sd_of_means": 2.0,
                        "rms_within_target_run_seed_sd": 1.0,
                    }
                )
    pd.DataFrame(summary).to_csv(
        root / "condition_summary.csv", index=False
    )

    comparisons = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for dataset in renderer.DATASETS:
            for endpoint in renderer.ENDPOINTS:
                comparisons.append(
                    {
                        "contrast": contrast,
                        "role": renderer.PRIMARY_ROLES[contrast],
                        "scope": "dataset",
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "n_domains": 4,
                        "n_seed_pairs": 16,
                        "complete_for_inference": True,
                        "effect_pp": 0.25,
                        "ci95_t_low": -0.75,
                        "ci95_t_high": 1.25,
                        "exact_signflip_p": 0.5,
                        "holm_p_dataset_8_sensitivity": 1.0,
                        "sesoi_pp": 1.0,
                        "ci90_t_low": -0.60,
                        "ci90_t_high": 0.90,
                        "tost_p": 0.04,
                        "equivalent_at_0p05": True,
                    }
                )
    pd.DataFrame(comparisons).to_csv(
        root / "clustered_comparisons.csv", index=False
    )

    suite = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for endpoint in renderer.ENDPOINTS:
            suite.append(
                {
                    "contrast": contrast,
                    "role": renderer.PRIMARY_ROLES[contrast],
                    "endpoint": endpoint,
                    "inference_cluster": "benchmark_dataset",
                    "n_datasets": 4,
                    "n_dataset_clusters": 4,
                    "n_target_domains": 16,
                    "effect_pp": 0.20,
                    "ci95_t_low": -0.40,
                    "ci95_t_high": 0.80,
                    "exact_4_dataset_signflip_p": 0.25,
                    "holm_p_primary_2": 0.25,
                    "sesoi_pp": 1.0,
                    "ci90_t_low": -0.30,
                    "ci90_t_high": 0.70,
                    "tost_p": 0.03,
                    "equivalent_at_0p05": True,
                }
            )
    pd.DataFrame(suite).to_csv(
        root / "suite_comparisons.csv", index=False
    )

    selection = []
    for condition in renderer.CONDITIONS:
        for dataset in renderer.DATASETS:
            selection.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "n_runs": 16,
                    "complete_four_by_four": True,
                    "source_selected_macro_accuracy": 75.0,
                    "final_step_macro_accuracy": 74.5,
                    "selected_minus_final_pp": 0.5,
                }
            )
    pd.DataFrame(selection).to_csv(
        root / "model_selection_summary.csv", index=False
    )

    targets = []
    for condition in renderer.CONDITIONS:
        for dataset in renderer.DATASETS:
            for endpoint in renderer.ENDPOINTS:
                for target, target_name in enumerate(
                    renderer.TARGET_NAMES[dataset]
                ):
                    targets.append(
                        {
                            "condition": condition,
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "target": target,
                            "target_name": target_name,
                            "n_seeds": 4,
                            "mean_accuracy": 75.0 + target,
                            "run_seed_sd": 1.0,
                            "seed0_accuracy": 74.0 + target,
                            "seed1_accuracy": 75.0 + target,
                            "seed2_accuracy": 76.0 + target,
                            "seed3_accuracy": 75.0 + target,
                        }
                    )
    pd.DataFrame(targets).to_csv(
        root / "per_target_results.csv", index=False
    )

    effects = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for dataset in renderer.DATASETS:
            for endpoint in renderer.ENDPOINTS:
                for target, target_name in enumerate(
                    renderer.TARGET_NAMES[dataset]
                ):
                    effects.append(
                        {
                            "contrast": contrast,
                            "role": renderer.PRIMARY_ROLES[contrast],
                            "scope": "dataset",
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "target": target,
                            "target_name": target_name,
                            "n_seed_pairs": 4,
                            "mean_difference": 0.25,
                            "run_seed_sd_difference": 0.5,
                        }
                    )
    pd.DataFrame(effects).to_csv(root / "domain_effects.csv", index=False)

    effect_variation = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for dataset in renderer.DATASETS:
            for endpoint in renderer.ENDPOINTS:
                effect_variation.append(
                    {
                        "contrast": contrast,
                        "role": renderer.PRIMARY_ROLES[contrast],
                        "scope": "dataset",
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "grand_mean_effect_pp": 0.25,
                        "between_target_sd_of_mean_effects": 0.30,
                        "between_run_seed_sd_of_mean_effects": 0.20,
                        "rms_within_target_run_seed_sd_of_effects": 0.50,
                        "target_by_run_seed_residual_sd_of_effects": 0.40,
                        "run_seed_components": (
                            "data_split+initialization+minibatch_order+"
                            "augmentation"
                        ),
                    }
                )
    pd.DataFrame(effect_variation).to_csv(
        root / "effect_variance_decomposition.csv", index=False
    )

    target_in_condition_rows = []
    target_in_endpoints = (
        "source_selected_target_in",
        "final_step_target_in",
    )
    for condition in renderer.CONDITIONS:
        for dataset in renderer.DATASETS:
            for endpoint in target_in_endpoints:
                target_in_condition_rows.append(
                    {
                        "record_type": "condition_summary",
                        "condition": condition,
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "n_targets": 4,
                        "n_runs": 16,
                        "macro_mean_accuracy": 76.0,
                        "between_target_sd_of_means": 2.0,
                        "rms_within_target_run_seed_sd": 1.0,
                    }
                )
    target_in_contrast_rows = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for dataset in renderer.DATASETS:
            for endpoint in target_in_endpoints:
                target_in_contrast_rows.append(
                    {
                        "record_type": "primary_contrast",
                        "contrast": contrast,
                        "role": renderer.PRIMARY_ROLES[contrast],
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "n_targets": 4,
                        "n_runs": 16,
                        "effect_pp": 0.20,
                        "ci95_t_low": -0.40,
                        "ci95_t_high": 0.80,
                        "exact_signflip_p": 0.25,
                    }
                )
    pd.concat(
        [
            pd.DataFrame(target_in_condition_rows),
            pd.DataFrame(target_in_contrast_rows),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(root / "target_in_sensitivity.csv", index=False)

    equivalence = []
    for contrast in renderer.PRIMARY_CONTRASTS:
        for dataset in renderer.DATASETS:
            for endpoint in renderer.ENDPOINTS:
                for margin in (0.5, 1.0, 2.0):
                    equivalence.append(
                        {
                            "contrast": contrast,
                            "role": renderer.PRIMARY_ROLES[contrast],
                            "scope": "dataset",
                            "dataset": dataset,
                            "endpoint": endpoint,
                            "margin_pp": margin,
                            "effect_pp": 0.25,
                            "ci90_t_low": -0.30,
                            "ci90_t_high": 0.70,
                            "tost_p": 0.03 if margin >= 1.0 else 0.2,
                            "equivalent_at_0p05": margin >= 1.0,
                            "primary_equivalence_question": False,
                        }
                    )
        for endpoint in renderer.ENDPOINTS:
            for margin in (0.5, 1.0, 2.0):
                equivalence.append(
                    {
                        "contrast": contrast,
                        "role": renderer.PRIMARY_ROLES[contrast],
                        "scope": "suite",
                        "dataset": "suite_macro",
                        "endpoint": endpoint,
                        "margin_pp": margin,
                        "effect_pp": 0.20,
                        "ci90_t_low": -0.30,
                        "ci90_t_high": 0.70,
                        "tost_p": 0.03 if margin >= 1.0 else 0.2,
                        "equivalent_at_0p05": margin >= 1.0,
                        "primary_equivalence_question": (
                            contrast == "no_variance_minus_shared"
                            and endpoint == "source_selected"
                            and margin == 1.0
                        ),
                    }
                )
    pd.DataFrame(equivalence).to_csv(
        root / "equivalence_sensitivity.csv", index=False
    )


class RenderReviewResultsTests(unittest.TestCase):
    def test_refuses_incomplete_readiness_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "input"
            output_dir = Path(temporary) / "output"
            input_dir.mkdir()
            write_complete_analysis(input_dir)
            readiness_path = input_dir / "readiness.json"
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            readiness["complete_runs"] = 383
            readiness["missing_runs"] = 1
            readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

            with self.assertRaisesRegex(
                renderer.RenderContractError, "complete_runs"
            ):
                renderer.render_results(input_dir, output_dir)
            self.assertFalse(output_dir.exists())

    def test_refuses_missing_primary_contrast_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "input"
            output_dir = Path(temporary) / "output"
            input_dir.mkdir()
            write_complete_analysis(input_dir)
            path = input_dir / "clustered_comparisons.csv"
            frame = pd.read_csv(path)
            frame.iloc[1:].to_csv(path, index=False)

            with self.assertRaisesRegex(
                renderer.RenderContractError, "grid mismatch"
            ):
                renderer.render_results(input_dir, output_dir)
            self.assertFalse(output_dir.exists())

    def test_complete_contract_writes_report_and_all_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary) / "input"
            output_dir = Path(temporary) / "output"
            input_dir.mkdir()
            write_complete_analysis(input_dir)

            paths = renderer.render_results(input_dir, output_dir)

            self.assertEqual(len(paths), 9)
            self.assertTrue((output_dir / "RESULTS_READY.md").is_file())
            self.assertTrue(
                (output_dir / "latex" / "primary_contrasts.tex").is_file()
            )
            report = (output_dir / "RESULTS_READY.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("384/384 complete runs", report)
            self.assertIn("Named-target source-selected accuracies", report)
            self.assertIn("adjusted evidence against zero", report)
            self.assertIn("suite macro equivalent within +/-1.0 pp", report)
            self.assertIn("| 74.00 | 75.00 | 76.00 | 75.00 |", report)
            latex = (
                output_dir / "latex" / "primary_contrasts.tex"
            ).read_text(encoding="utf-8")
            self.assertIn(r"\begin{tabular}", latex)
            self.assertIn(r"\bottomrule", latex)
            target_latex = (
                output_dir / "latex" / "target_level_accuracies.tex"
            ).read_text(encoding="utf-8")
            self.assertIn(r"\begin{longtable}", target_latex)


if __name__ == "__main__":
    unittest.main()
