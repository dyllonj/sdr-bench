from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.agent import AgentToolCall
from sdr_bench.agent import AgentTurnResponse
from sdr_bench.agent import run_policy_episode_agent_model
from sdr_bench.evaluator import load_json
from sdr_bench.runner.episode_runner import run_policy_episode_model_with_tools


class EpisodeQueueAdapter:
    def __init__(self, submissions: list[dict[str, Any]]) -> None:
        self.name = "mock:episode-agent"
        self.submissions = list(submissions)
        self.turn_messages: list[list[dict[str, Any]]] = []

    def create_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentTurnResponse:
        self.turn_messages.append(list(messages))
        if not self.submissions:
            raise AssertionError("EpisodeQueueAdapter ran out of submissions")
        submission = self.submissions.pop(0)
        return AgentTurnResponse(
            text="Submit week.",
            tool_calls=[
                AgentToolCall(
                    id=f"submit_{len(self.turn_messages)}",
                    name="submit_weekly_decisions",
                    arguments={"decisions": submission["decisions"]},
                )
            ],
            input_tokens=10,
            output_tokens=5,
            latency_ms=2,
            raw={},
        )


class AgentPolicyRunnerTests(unittest.TestCase):
    def test_policy_agent_runner_uses_public_history_without_metrics(self) -> None:
        episode = load_json(ROOT_DIR / "examples" / "sample_episode.json")
        labels = load_json(ROOT_DIR / "examples" / "sample_policy_labels.json")
        policy_submission = load_json(ROOT_DIR / "examples" / "sample_policy_submission.json")
        adapter = EpisodeQueueAdapter(policy_submission["submissions"])

        artifact = run_policy_episode_agent_model(
            episode,
            labels,
            adapter=adapter,
            model_spec="mock:episode-agent",
            seed=5,
        )

        self.assertEqual("tools", artifact["runner_mode"])
        self.assertEqual(episode["episode_id"], artifact["policy_submission"]["episode_id"])
        self.assertEqual(2, len(artifact["policy_submission"]["submissions"]))
        self.assertEqual(2, len(artifact["usage_log"]["windows"]))
        self.assertEqual(20, artifact["usage_log"]["input_tokens"])
        self.assertEqual(10, artifact["usage_log"]["output_tokens"])
        self.assertEqual(4, artifact["usage_log"]["latency_ms"])

        public_history = artifact["usage_log"]["public_history"]
        self.assertEqual(2, len(public_history))
        self.assertIn("compliance", public_history[0])
        self.assertNotIn("metrics", public_history[0])
        self.assertNotIn("agent_incremental_values", str(public_history))
        self.assertNotIn("oracle_incremental_values", str(public_history))

        second_turn_context = "\n".join(
            str(message.get("content", ""))
            for message in adapter.turn_messages[1]
        )
        self.assertIn("Public episode history", second_turn_context)
        self.assertNotIn("metrics", second_turn_context)
        self.assertNotIn("pipeline_value", second_turn_context)

    def test_episode_runner_tool_mode_uses_adapter_registry(self) -> None:
        episode = load_json(ROOT_DIR / "examples" / "sample_episode.json")
        labels = load_json(ROOT_DIR / "examples" / "sample_policy_labels.json")
        policy_submission = load_json(ROOT_DIR / "examples" / "sample_policy_submission.json")
        adapter = EpisodeQueueAdapter(policy_submission["submissions"])

        with mock.patch("sdr_bench.runner.episode_runner.load_adapter", return_value=adapter):
            artifact = run_policy_episode_model_with_tools(
                episode,
                labels,
                model_spec="mock:episode-agent",
                seed=6,
            )

        self.assertEqual("tools", artifact["runner_mode"])
        self.assertEqual("mock:episode-agent", artifact["model_spec"])
        self.assertEqual(episode["episode_id"], artifact["policy_submission"]["episode_id"])
        self.assertEqual(2, len(artifact["policy_submission"]["submissions"]))


if __name__ == "__main__":
    unittest.main()
