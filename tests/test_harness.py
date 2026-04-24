from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.harness.aggregate import _collect_model_rows
from sdr_bench.harness.sweep import load_models_file
from sdr_bench.io_utils import dump_json


class HarnessTests(unittest.TestCase):
    def test_models_file_parser_reads_expected_specs(self) -> None:
        models = load_models_file(ROOT_DIR / "configs" / "models.yaml")
        self.assertEqual(len(models), 4)
        self.assertEqual(models[0]["spec"], "anthropic:claude-opus-4-7")

    def test_aggregate_builds_ranked_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir)
            model_a = {
                "artifact": {"usage_log": {"input_tokens": 10, "output_tokens": 5, "latency_ms": 3}},
                "evaluation": {"metrics": {"OfflineScore": 80.0, "FitScore": 75.0, "TimingScore": 70.0, "ContactScore": 65.0, "GroundingScore": 90.0}},
                "random_baseline_report": {"metrics": {"OfflineScore": 40.0, "FitScore": 40.0, "TimingScore": 40.0, "ContactScore": 40.0, "GroundingScore": 40.0}},
                "oracle_report": {"metrics": {"OfflineScore": 100.0, "FitScore": 100.0, "TimingScore": 100.0, "ContactScore": 100.0, "GroundingScore": 100.0}},
                "cost_usd": 0.01,
                "split": "test",
            }
            model_b = {
                "artifact": {"usage_log": {"input_tokens": 10, "output_tokens": 5, "latency_ms": 3}},
                "evaluation": {"metrics": {"OfflineScore": 60.0, "FitScore": 55.0, "TimingScore": 50.0, "ContactScore": 45.0, "GroundingScore": 80.0}},
                "random_baseline_report": {"metrics": {"OfflineScore": 40.0, "FitScore": 40.0, "TimingScore": 40.0, "ContactScore": 40.0, "GroundingScore": 40.0}},
                "oracle_report": {"metrics": {"OfflineScore": 100.0, "FitScore": 100.0, "TimingScore": 100.0, "ContactScore": 100.0, "GroundingScore": 100.0}},
                "cost_usd": 0.02,
                "split": "test",
            }
            dump_json(results_dir / "offline" / "model_a" / "run.json", model_a, pretty=True)
            dump_json(results_dir / "offline" / "model_b" / "run.json", model_b, pretty=True)
            dump_json(
                results_dir / "index.json",
                {
                    "offline_runs": [
                        {"model_spec": "model_a", "path": "offline/model_a/run.json"},
                        {"model_spec": "model_b", "path": "offline/model_b/run.json"},
                    ],
                    "policy_runs": [],
                    "robustness_runs": [],
                },
                pretty=True,
            )

            rows = _collect_model_rows({"offline_runs": [{"model_spec": "model_a", "path": "offline/model_a/run.json"}, {"model_spec": "model_b", "path": "offline/model_b/run.json"}], "policy_runs": [], "robustness_runs": []}, results_dir)
            self.assertEqual(rows[0]["model_spec"], "model_a")
            self.assertGreater(rows[0]["public_score"], rows[1]["public_score"])

    def test_aggregate_keeps_robustness_diagnostic_out_of_public_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir)
            offline = {
                "artifact": {"usage_log": {"input_tokens": 10, "output_tokens": 5, "latency_ms": 3}},
                "evaluation": {"metrics": {"OfflineScore": 80.0, "FitScore": 75.0, "TimingScore": 70.0, "ContactScore": 65.0, "GroundingScore": 90.0}},
                "random_baseline_report": {"metrics": {"OfflineScore": 40.0, "FitScore": 40.0, "TimingScore": 40.0, "ContactScore": 40.0, "GroundingScore": 40.0}},
                "oracle_report": {"metrics": {"OfflineScore": 100.0, "FitScore": 100.0, "TimingScore": 100.0, "ContactScore": 100.0, "GroundingScore": 100.0}},
                "cost_usd": 0.01,
                "split": "test",
            }
            robustness = {
                "artifact": {"usage_log": {"input_tokens": 7, "output_tokens": 4, "latency_ms": 2}},
                "evaluation": {"summary": {"average_metrics": {"EnterpriseAllocationScore": 40.0}}},
                "random_baseline_report": {"summary": {"average_metrics": {"EnterpriseAllocationScore": 40.0}}},
                "oracle_report": {"summary": {"average_metrics": {"EnterpriseAllocationScore": 100.0}}},
                "cost_usd": 0.01,
            }
            dump_json(results_dir / "offline" / "model_a" / "run.json", offline, pretty=True)
            dump_json(results_dir / "robustness" / "model_a" / "run.json", robustness, pretty=True)

            rows = _collect_model_rows(
                {
                    "offline_runs": [{"model_spec": "model_a", "path": "offline/model_a/run.json"}],
                    "policy_runs": [],
                    "robustness_runs": [{"model_spec": "model_a", "path": "robustness/model_a/run.json"}],
                },
                results_dir,
            )

            self.assertAlmostEqual(rows[0]["robustness_score"], 0.0)
            self.assertAlmostEqual(rows[0]["public_score"], rows[0]["offline_score"])


if __name__ == "__main__":
    unittest.main()
