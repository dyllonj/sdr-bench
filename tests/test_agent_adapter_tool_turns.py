from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.runner.adapters.anthropic import AnthropicAdapter
from sdr_bench.runner.adapters.openai import OpenAIAdapter


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class ToolTurnAdapterTests(unittest.TestCase):
    def test_openai_create_turn_normalizes_tool_calls(self) -> None:
        message = Obj(
            content="Inspecting.",
            tool_calls=[
                Obj(
                    id="call_1",
                    function=Obj(name="list_accounts", arguments='{"limit": 2}'),
                )
            ],
        )
        response = Obj(
            choices=[Obj(message=message)],
            usage=Obj(prompt_tokens=11, completion_tokens=7),
        )
        fake_client = Obj(chat=Obj(completions=Obj(create=mock.Mock(return_value=response))))

        with mock.patch("openai.OpenAI", return_value=fake_client):
            adapter = OpenAIAdapter("gpt-test", api_key="test-key")
            result = adapter.create_turn(
                [{"role": "user", "content": "start"}],
                [
                    {
                        "name": "list_accounts",
                        "description": "List accounts",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            )

        request = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual("gpt-test", request["model"])
        self.assertEqual("function", request["tools"][0]["type"])
        self.assertEqual("list_accounts", request["tools"][0]["function"]["name"])
        self.assertEqual("Inspecting.", result.text)
        self.assertEqual(11, result.input_tokens)
        self.assertEqual(7, result.output_tokens)
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("call_1", result.tool_calls[0].id)
        self.assertEqual("list_accounts", result.tool_calls[0].name)
        self.assertEqual({"limit": 2}, result.tool_calls[0].arguments)

    def test_openai_create_turn_serializes_normalized_assistant_tool_calls(self) -> None:
        response = Obj(
            choices=[Obj(message=Obj(content="", tool_calls=[]))],
            usage=Obj(prompt_tokens=1, completion_tokens=1),
        )
        fake_client = Obj(chat=Obj(completions=Obj(create=mock.Mock(return_value=response))))

        with mock.patch("openai.OpenAI", return_value=fake_client):
            adapter = OpenAIAdapter("gpt-test", api_key="test-key")
            adapter.create_turn(
                [
                    {
                        "role": "assistant",
                        "content": "tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "list_accounts",
                                "arguments": {"limit": 1},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "list_accounts",
                        "content": "{}",
                    },
                ],
                [],
            )

        messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual("assistant", messages[0]["role"])
        self.assertEqual("function", messages[0]["tool_calls"][0]["type"])
        self.assertEqual('{"limit":1}', messages[0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual("tool", messages[1]["role"])
        self.assertEqual("call_1", messages[1]["tool_call_id"])

    def test_anthropic_create_turn_normalizes_tool_use_blocks(self) -> None:
        response = Obj(
            content=[
                Obj(type="text", text="Inspecting."),
                Obj(
                    type="tool_use",
                    id="toolu_1",
                    name="get_account_context",
                    input={"account_id": "acct_123"},
                ),
            ],
            usage=Obj(input_tokens=13, output_tokens=8),
        )
        fake_client = Obj(messages=Obj(create=mock.Mock(return_value=response)))
        fake_module = types.SimpleNamespace(Anthropic=mock.Mock(return_value=fake_client))

        with mock.patch.dict(sys.modules, {"anthropic": fake_module}):
            adapter = AnthropicAdapter("claude-test", api_key="test-key")
            result = adapter.create_turn(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "start"},
                ],
                [
                    {
                        "name": "get_account_context",
                        "description": "Get account",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
            )

        request = fake_client.messages.create.call_args.kwargs
        self.assertEqual("claude-test", request["model"])
        self.assertEqual("system", request["system"])
        self.assertEqual("get_account_context", request["tools"][0]["name"])
        self.assertEqual("Inspecting.", result.text)
        self.assertEqual(13, result.input_tokens)
        self.assertEqual(8, result.output_tokens)
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("toolu_1", result.tool_calls[0].id)
        self.assertEqual("get_account_context", result.tool_calls[0].name)
        self.assertEqual({"account_id": "acct_123"}, result.tool_calls[0].arguments)


if __name__ == "__main__":
    unittest.main()
