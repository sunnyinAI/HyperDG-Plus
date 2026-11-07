from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "scripts"))

import analyze_review_round2 as analysis  # noqa: E402
import generate_review_round2 as generator  # noqa: E402


def metric_row(step: int, source_value: float, target_value: float) -> dict:
    row = {"step": step}
    for env in range(4):
        row[f"env{env}_out_acc"] = target_value if env == 2 else source_value
        row[f"env{env}_in_acc"] = target_value if env == 2 else source_value
    return row


class ReviewRound2Tests(unittest.TestCase):
    def test_source_selection_excludes_target_and_breaks_ties_early(self) -> None:
        rows = [
            metric_row(200, source_value=0.80, target_value=0.01),
            metric_row(400, source_value=0.79, target_value=0.99),
        ]
        selected, score = analysis.select_source_checkpoint(rows, target=2)
        self.assertEqual(selected["step"], 200)
        self.assertTrue(math.isclose(score, 0.80))

        # Mutating target accuracy cannot alter the source-only selection.
        rows[1]["env2_out_acc"] = 1000.0
        selected_again, _ = analysis.select_source_checkpoint(rows, target=2)
        self.assertEqual(selected_again["step"], 200)

        tied = [
            metric_row(200, source_value=0.80, target_value=0.2),
            metric_row(400, source_value=0.80, target_value=0.9),
        ]
        selected_tie, _ = analysis.select_source_checkpoint(tied, target=2)
        self.assertEqual(selected_tie["step"], 200)

    def test_four_domain_signflip_minimum_two_sided_p_is_point_125(self) -> None:
        value = analysis.exact_sign_flip_p(np.asarray([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(value, 0.125)

    def test_generated_inventory_and_true_erm_routing(self) -> None:
        inventory_path = STUDY_ROOT / "manifests" / "inventory.csv"
        with inventory_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 128)
        self.assertEqual(
            sum(row["condition"] == "standard_erm" for row in rows), 64
        )
        self.assertEqual(
            sum(row["condition"] == "shared_learnable" for row in rows), 64
        )
        self.assertEqual(sum(row["status"] == "reuse_legacy" for row in rows), 0)

        erm_path = (
            STUDY_ROOT
            / "configs"
            / "reviewer_round2"
            / "standard_erm"
            / "seed0"
            / "pacs_split0.yaml"
        )
        shared_path = (
            STUDY_ROOT
            / "configs"
            / "reviewer_round2"
            / "shared_learnable"
            / "seed0"
            / "pacs_split0.yaml"
        )
        erm = yaml.safe_load(erm_path.read_text(encoding="utf-8"))
        shared = yaml.safe_load(shared_path.read_text(encoding="utf-8"))
        self.assertEqual(erm["algorithm"], "ERM")
        self.assertNotIn("single_curvature", erm["hparams"])
        self.assertEqual(shared["algorithm"], "hyperdg")
        self.assertIs(shared["hparams"]["single_curvature"], True)
        self.assertGreater(shared["hparams"]["curvature_lr"], 0)
        self.assertIs(erm["hparams"]["resnet50_augmix"], True)
        self.assertIs(shared["hparams"]["resnet50_augmix"], True)

    def test_existing_clustered_reanalysis_matches_artifacts(self) -> None:
        path = (
            STUDY_ROOT
            / "outputs"
            / "reviewer_round2"
            / "clustered_comparisons.csv"
        )
        frame = pd.read_csv(path)
        row = frame[
            (frame["contrast"] == "full_minus_matched_control")
            & (frame["dataset"] == "pacs")
            & (frame["endpoint"] == "final_step")
        ].iloc[0]
        self.assertTrue(math.isclose(row["effect_pp"], 0.454067, abs_tol=1e-5))
        self.assertTrue(
            math.isclose(row["ci95_t_low"], -2.366027, abs_tol=1e-5)
        )
        self.assertTrue(
            math.isclose(row["ci95_t_high"], 3.274160, abs_tol=1e-5)
        )
        self.assertTrue(
            math.isclose(row["exact_signflip_p"], 0.75, abs_tol=1e-12)
        )

    def test_completion_requires_final_step_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            truncated = {"step": generator.EXPECTED_LAST_STEP}
            truncated.update(
                {key: 0.5 for key in generator.REQUIRED_EVAL_KEYS}
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps(truncated) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(generator.is_complete(run_dir))

            rows = []
            for step in generator.EXPECTED_STEPS:
                row = {"step": step}
                row.update({key: 0.5 for key in generator.REQUIRED_EVAL_KEYS})
                rows.append(row)
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertFalse(generator.is_complete(run_dir))
            (run_dir / "model_last.pt").write_bytes(b"checkpoint")
            self.assertTrue(generator.is_complete(run_dir))

            rows[0][generator.REQUIRED_EVAL_KEYS[0]] = float("nan")
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertFalse(generator.is_complete(run_dir))
            self.assertIn(
                "invalid_eval_value_at_step_200",
                analysis.metrics_validation_error(rows),
            )

            rows[0][generator.REQUIRED_EVAL_KEYS[0]] = 1.01
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertFalse(generator.is_complete(run_dir))
            self.assertIn(
                "invalid_eval_value_at_step_200",
                analysis.metrics_validation_error(rows),
            )


if __name__ == "__main__":
    unittest.main()
