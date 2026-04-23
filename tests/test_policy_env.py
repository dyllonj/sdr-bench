from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.baselines import build_account_contexts
from sdr_bench.baselines import generate_episode_submission
from sdr_bench.baselines import generate_oracle_episode_submission
from sdr_bench.baselines import make_human_touch_decision
from sdr_bench.evaluator import evaluate_episode
from sdr_bench.simulator.corpus import build_corpus
from sdr_bench.simulator.env import SDRBenchEnv


class PolicyEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        corpus = build_corpus(seed=9, n_windows=6, accounts_per_window=16)
        self.episode = corpus["episodes"][0]
        self.episode_labels = corpus["episode_labels"][0]

    def test_env_is_deterministic_for_fixed_wait_policy(self) -> None:
        def rollout() -> list[dict[str, object]]:
            env = SDRBenchEnv(self.episode, seed=3, labels_data=self.episode_labels)
            state = env.reset()
            outcomes = []
            while state is not None:
                submission = {
                    "window_id": state["window_id"],
                    "decisions": [
                        {
                            "account_id": account["account_id"],
                            "chosen_action": "wait",
                            "action_score": 0.0,
                        }
                        for account in state["window"]["accounts"]
                    ],
                }
                state, outcome, done = env.step(submission)
                outcomes.append(
                    {
                        "window_id": outcome["window_id"],
                        "metrics": outcome["metrics"],
                        "issue_count": len(outcome["issues"]),
                    }
                )
                if done:
                    break
            return outcomes

        self.assertEqual(rollout(), rollout())

    def test_human_touch_removal_affects_next_state(self) -> None:
        first_window = self.episode["windows"][0]
        first_labels_window = self.episode_labels["windows"][0]
        removable_accounts = [
            row["account_id"]
            for row in first_labels_window["policy_transitions"]
            if row["actions"]["human_touch"]["remove_from_episode"]
        ]
        self.assertTrue(removable_accounts)
        contexts = build_account_contexts(first_window)
        context_by_account = {
            context["account"]["account_id"]: context
            for context in contexts
        }
        target_account = next(
            account_id
            for account_id in removable_accounts
            if context_by_account[account_id]["human_touch_feasible"]
        )
        submission = {
            "window_id": first_window["window_id"],
            "decisions": [
                make_human_touch_decision(
                    context_by_account[target_account],
                    human_touch_rank=1,
                    baseline_score=1.0,
                )
            ],
        }

        env = SDRBenchEnv(self.episode, seed=3, labels_data=self.episode_labels)
        env.reset()
        next_state, _, done = env.step(submission)
        self.assertFalse(done)
        self.assertIsNotNone(next_state)
        self.assertIn(target_account, next_state["ineligible_account_ids"])

    def test_oracle_episode_outperforms_random_episode(self) -> None:
        oracle_submission = generate_oracle_episode_submission(self.episode, self.episode_labels)
        random_submission = generate_episode_submission(self.episode, "random_within_icp", seed=9)
        oracle_report = evaluate_episode(
            self.episode,
            oracle_submission,
            self.episode_labels,
        )
        random_report = evaluate_episode(
            self.episode,
            random_submission,
            self.episode_labels,
        )

        self.assertGreater(
            oracle_report["metrics"]["PolicyScore"],
            random_report["metrics"]["PolicyScore"] + 15.0,
        )


if __name__ == "__main__":
    unittest.main()
