from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.baselines import generate_oracle_window_submission
from sdr_bench.evaluator import evaluate_window
from sdr_bench.evaluator import evaluate_window_baseline
from sdr_bench.evaluator import validate_instance
from sdr_bench.simulator.corpus import build_corpus
from sdr_bench.simulator.corpus import corpus_files
from sdr_bench.simulator.generator import generate_window


class SimulatorTests(unittest.TestCase):
    def test_generate_window_is_deterministic(self) -> None:
        first_window, first_labels = generate_window(11, 24, "wk_det")
        second_window, second_labels = generate_window(11, 24, "wk_det")

        self.assertEqual(
            json.dumps(first_window, sort_keys=True),
            json.dumps(second_window, sort_keys=True),
        )
        self.assertEqual(
            json.dumps(first_labels, sort_keys=True),
            json.dumps(second_labels, sort_keys=True),
        )

    def test_generate_window_matches_schemas(self) -> None:
        window, labels = generate_window(7, 20, "wk_schema")
        self.assertEqual(validate_instance(window, "evaluation_window"), [])
        self.assertEqual(validate_instance(labels, "hidden_labels"), [])

    def test_oracle_beats_random_and_propensity(self) -> None:
        window, labels = generate_window(19, 40, "wk_quality")
        oracle_report = evaluate_window(
            window,
            generate_oracle_window_submission(window, labels),
            labels_data=labels,
        )
        random_report = evaluate_window_baseline(
            window,
            "random_within_icp",
            labels_data=labels,
            seed=19,
        )["report"]
        propensity_report = evaluate_window_baseline(
            window,
            "propensity_only_lead_score",
            labels_data=labels,
            seed=19,
        )["report"]

        self.assertGreater(
            oracle_report["metrics"]["OfflineScore"],
            random_report["metrics"]["OfflineScore"] + 20.0,
        )
        self.assertGreater(
            oracle_report["metrics"]["OfflineScore"],
            propensity_report["metrics"]["OfflineScore"] + 10.0,
        )

    def test_corpus_files_validate(self) -> None:
        corpus = build_corpus(seed=5, n_windows=4, accounts_per_window=12)
        files = corpus_files(corpus)

        for relative_path, payload in files.items():
            if "/windows/" in relative_path:
                self.assertEqual(validate_instance(payload, "evaluation_window"), [], msg=relative_path)
            elif "/labels/" in relative_path and not relative_path.startswith("episode_labels/"):
                self.assertEqual(validate_instance(payload, "hidden_labels"), [], msg=relative_path)
            elif relative_path.startswith("episodes/"):
                self.assertEqual(validate_instance(payload, "policy_episode"), [], msg=relative_path)
            elif relative_path.startswith("episode_labels/"):
                self.assertEqual(validate_instance(payload, "policy_episode_labels"), [], msg=relative_path)
            elif relative_path == "robustness/suite.json":
                self.assertEqual(validate_instance(payload, "robustness_suite"), [], msg=relative_path)


if __name__ == "__main__":
    unittest.main()
