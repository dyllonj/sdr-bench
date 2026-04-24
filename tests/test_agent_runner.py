from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.agent import AgentToolCall
from sdr_bench.agent import AgentTurnResponse
from sdr_bench.agent import agent_tool_definitions
from sdr_bench.agent import run_window_agent_model
from sdr_bench.evaluator import load_json


class QueueAgentAdapter:
    def __init__(self, responses: list[AgentTurnResponse]) -> None:
        self.responses = list(responses)
        self.name = "mock:agent"
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
        if not self.responses:
            raise AssertionError("QueueAgentAdapter ran out of responses")
        return self.responses.pop(0)


class AgentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = load_json(ROOT_DIR / "examples" / "sample_window.json")
        self.submission = load_json(ROOT_DIR / "examples" / "sample_submission.json")

    def test_agent_tool_definitions_include_seller_knowledge(self) -> None:
        tool_names = {tool["name"] for tool in agent_tool_definitions()}

        self.assertIn("list_accounts", tool_names)
        self.assertIn("get_account_context", tool_names)
        self.assertIn("get_seller_knowledge", tool_names)
        self.assertIn("submit_weekly_decisions", tool_names)

    def test_run_window_agent_model_executes_tool_loop_until_submission(self) -> None:
        adapter = QueueAgentAdapter(
            [
                AgentTurnResponse(
                    text="Inspect the book.",
                    tool_calls=[
                        AgentToolCall(
                            id="call_1",
                            name="list_accounts",
                            arguments={"limit": 1},
                        )
                    ],
                    input_tokens=10,
                    output_tokens=4,
                    latency_ms=3,
                    raw={},
                ),
                AgentTurnResponse(
                    text="Inspect the top account.",
                    tool_calls=[
                        AgentToolCall(
                            id="call_2",
                            name="get_account_context",
                            arguments={"account_id": "acct_123"},
                        )
                    ],
                    input_tokens=12,
                    output_tokens=5,
                    latency_ms=4,
                    raw={},
                ),
                AgentTurnResponse(
                    text="Submit final routing.",
                    tool_calls=[
                        AgentToolCall(
                            id="call_3",
                            name="submit_weekly_decisions",
                            arguments={"decisions": self.submission["decisions"]},
                        )
                    ],
                    input_tokens=15,
                    output_tokens=7,
                    latency_ms=6,
                    raw={},
                ),
            ]
        )

        artifact = run_window_agent_model(
            self.window,
            adapter=adapter,
            model_spec="mock:agent",
            seed=3,
        )

        self.assertEqual(self.window["window_id"], artifact["submission"]["window_id"])
        self.assertEqual("tools", artifact["runner_mode"])
        self.assertEqual("mock:agent", artifact["model_spec"])
        self.assertEqual(3, artifact["seed"])
        self.assertTrue(artifact["prompt_hash"])

        usage = artifact["usage_log"]
        self.assertEqual(3, usage["turn_count"])
        self.assertTrue(usage["finalized"])
        self.assertEqual(37, usage["input_tokens"])
        self.assertEqual(16, usage["output_tokens"])
        self.assertEqual(13, usage["latency_ms"])
        self.assertEqual(1, len(usage["tool_trace"]))
        self.assertEqual("submit_weekly_decisions", usage["tool_trace"][0]["tool_name"])

        self.assertGreaterEqual(len(adapter.turn_messages), 3)
        last_turn_messages = adapter.turn_messages[-1]
        self.assertTrue(
            any(
                message.get("role") == "tool"
                and message.get("name") == "get_account_context"
                for message in last_turn_messages
            )
        )

    def test_run_window_agent_model_falls_back_to_wait_when_no_submission(self) -> None:
        adapter = QueueAgentAdapter(
            [
                AgentTurnResponse(
                    text="No tools.",
                    tool_calls=[],
                    input_tokens=5,
                    output_tokens=2,
                    latency_ms=1,
                    raw={},
                )
            ]
        )

        artifact = run_window_agent_model(
            self.window,
            adapter=adapter,
            model_spec="mock:agent",
        )

        self.assertFalse(artifact["usage_log"]["finalized"])
        self.assertEqual(self.window["window_id"], artifact["submission"]["window_id"])
        self.assertTrue(
            all(
                decision["chosen_action"] == "wait"
                for decision in artifact["submission"]["decisions"]
            )
        )


if __name__ == "__main__":
    unittest.main()
