from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.evaluator import load_json
from sdr_bench.runner.adapters.base import AdapterResponse
from sdr_bench.runner.prompts import build_window_prompt
from sdr_bench.runner.repair import generate_with_repair
from sdr_bench.runner.run import run_window_model


class QueueAdapter:
    def __init__(self, responses: list[AdapterResponse]) -> None:
        self.responses = list(responses)
        self.name = "mock:queue"

    def generate(self, system: str, user: str, json_schema=None, max_tokens: int = 4096, temperature: float = 0.0) -> AdapterResponse:
        if not self.responses:
            raise AssertionError("QueueAdapter ran out of responses")
        return self.responses.pop(0)


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = load_json(ROOT_DIR / "examples" / "sample_window.json")
        self.wait_submission = {
            "window_id": self.window["window_id"],
            "decisions": [
                {
                    "account_id": account["account_id"],
                    "chosen_action": "wait",
                    "action_score": 0.0,
                }
                for account in self.window["accounts"]
            ],
        }

    def test_build_window_prompt_includes_budget_override_and_codes(self) -> None:
        system, user, schema = build_window_prompt(self.window, budget=2)

        self.assertIn("enterprise_icp_fit", system)
        self.assertIn('"human_sdr_actions": 2', user)
        self.assertEqual(schema["required"], ["window_id", "decisions"])

    def test_repair_extracts_fenced_json_without_retry(self) -> None:
        adapter = QueueAdapter(
            [
                AdapterResponse(
                    text="```json\n" + str(self.wait_submission).replace("'", '"') + "\n```",
                    parsed=None,
                    input_tokens=11,
                    output_tokens=7,
                    latency_ms=3,
                    raw={},
                )
            ]
        )

        result = generate_with_repair(
            adapter,
            window_data=self.window,
            system="system",
            user="user",
            json_schema=build_window_prompt(self.window)[2],
        )

        self.assertEqual(result.submission["window_id"], self.window["window_id"])
        self.assertEqual(result.usage_log["retry_count"], 0)
        self.assertEqual(result.usage_log["repair_outcome"], "initial_success")

    def test_repair_retries_after_invalid_json(self) -> None:
        adapter = QueueAdapter(
            [
                AdapterResponse(
                    text="not valid json",
                    parsed=None,
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=2,
                    raw={},
                ),
                AdapterResponse(
                    text=str(self.wait_submission).replace("'", '"'),
                    parsed=None,
                    input_tokens=12,
                    output_tokens=6,
                    latency_ms=2,
                    raw={},
                ),
            ]
        )

        result = generate_with_repair(
            adapter,
            window_data=self.window,
            system="system",
            user="user",
            json_schema=build_window_prompt(self.window)[2],
        )

        self.assertEqual(result.usage_log["retry_count"], 1)
        self.assertEqual(result.usage_log["repair_outcome"], "retry_success")
        self.assertEqual(result.submission["window_id"], self.window["window_id"])

    def test_run_window_model_uses_adapter_registry(self) -> None:
        adapter = QueueAdapter(
            [
                AdapterResponse(
                    text=str(self.wait_submission).replace("'", '"'),
                    parsed=None,
                    input_tokens=14,
                    output_tokens=9,
                    latency_ms=4,
                    raw={},
                )
            ]
        )
        with mock.patch("sdr_bench.runner.run.load_adapter", return_value=adapter):
            artifact = run_window_model(self.window, model_spec="mock:test", seed=7)

        self.assertEqual(artifact["submission"]["window_id"], self.window["window_id"])
        self.assertEqual(artifact["model_spec"], "mock:test")
        self.assertTrue(artifact["prompt_hash"])

    @unittest.skipUnless(os.getenv("SDR_BENCH_RUN_OLLAMA_TEST"), "set SDR_BENCH_RUN_OLLAMA_TEST=1 to enable")
    def test_openai_compatible_integration_with_ollama(self) -> None:
        from sdr_bench.runner.adapters import load_adapter

        adapter = load_adapter(
            os.getenv(
                "SDR_BENCH_OLLAMA_SPEC",
                "openai_compatible:llama3.2@http://localhost:11434/v1",
            )
        )
        result = generate_with_repair(
            adapter,
            window_data=self.window,
            system=build_window_prompt(self.window)[0],
            user=build_window_prompt(self.window)[1],
            json_schema=build_window_prompt(self.window)[2],
            max_tokens=512,
        )
        self.assertEqual(result.submission["window_id"], self.window["window_id"])


if __name__ == "__main__":
    unittest.main()
