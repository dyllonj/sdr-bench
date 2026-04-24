"""Provider-neutral tool-loop runner for SDR Bench agent mode."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sdr_bench.agent.sandbox import AgentSandbox
from sdr_bench.agent.types import AgentTurnAdapter
from sdr_bench.agent.types import AgentToolCall
from sdr_bench.runner.repair import wait_everything_submission


def agent_tool_definitions() -> list[dict[str, Any]]:
    """Return the stable public tool contract for the first sandbox slice."""

    return [
        {
            "name": "list_accounts",
            "description": "Page through model-visible public account snapshots.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "cursor": {"type": ["string", "null"]},
                },
            },
        },
        {
            "name": "get_account_context",
            "description": "Fetch account-local public account, contact, trigger, and evidence context.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["account_id"],
                "properties": {
                    "account_id": {"type": "string", "minLength": 1},
                },
            },
        },
        {
            "name": "submit_weekly_decisions",
            "description": "Finalize the weekly SDR routing decisions.",
            "input_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["decisions"],
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
        },
    ]


def _prompt_hash(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "messages": messages,
            "tools": tools,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _initial_messages(window_id: str, budget: int) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are participating in SDR Bench agent mode. Use only the provided "
                "public tools. Hidden labels, oracle outcomes, and scorer metrics are not "
                "available. Submit final routing through submit_weekly_decisions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Window {window_id} has a human-touch budget of {budget}. "
                "Inspect public account context as needed, then submit weekly decisions."
            ),
        },
    ]


def _tool_call_to_dict(call: AgentToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    }


def run_window_agent_model(
    window_data: dict[str, Any],
    *,
    adapter: AgentTurnAdapter,
    model_spec: str,
    max_turns: int = 8,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one offline window through an agent tool loop.

    This runner is provider-neutral and intentionally small: provider-specific
    adapters only need to return normalized tool calls. The session log is kept
    in the returned artifact, not in model context.
    """

    sandbox = AgentSandbox(window_data)
    tools = agent_tool_definitions()
    messages = _initial_messages(
        sandbox.window_id,
        window_data["capacity_budget"]["human_sdr_actions"],
    )
    initial_prompt_hash = _prompt_hash(messages, tools)

    turns: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0

    for turn_index in range(max_turns):
        response = adapter.create_turn(
            messages,
            tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_latency_ms += response.latency_ms

        turn_record = {
            "turn_index": turn_index,
            "text": response.text,
            "tool_calls": [_tool_call_to_dict(call) for call in response.tool_calls],
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }
        turns.append(turn_record)

        messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": turn_record["tool_calls"],
            }
        )

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_result = sandbox.execute(tool_call.name, tool_call.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps(tool_result, sort_keys=True),
                }
            )
            if tool_call.name == "submit_weekly_decisions" and tool_result.get("ok"):
                break
        if sandbox.finalized:
            break

    submission = sandbox.submission or wait_everything_submission(window_data)
    return {
        "submission": submission,
        "usage_log": {
            "adapter_name": adapter.name,
            "runner_mode": "tools",
            "turn_count": len(turns),
            "turns": turns,
            "tool_trace": sandbox.trace_events,
            "finalized": sandbox.finalized,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_ms": total_latency_ms,
        },
        "prompt_hash": initial_prompt_hash,
        "model_spec": model_spec,
        "runner_mode": "tools",
        "seed": seed,
    }
