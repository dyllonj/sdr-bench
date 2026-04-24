"""OpenAI adapter implementation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sdr_bench.agent.types import AgentToolCall
from sdr_bench.agent.types import AgentTurnResponse
from sdr_bench.runner.adapters.base import AdapterResponse


class OpenAIAdapter:
    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.name = f"openai:{model}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AdapterResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI adapter.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sdr_bench_submission",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        started = time.perf_counter()
        response = client.chat.completions.create(**request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = response.choices[0].message
        text = message.content or ""
        parsed = None
        if json_schema is not None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

        usage = getattr(response, "usage", None)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return AdapterResponse(
            text=text,
            parsed=parsed,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            raw=raw,
        )

    def create_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentTurnResponse:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI adapter.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
            "tools": [_to_openai_tool(tool) for tool in tools],
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        started = time.perf_counter()
        response = client.chat.completions.create(**request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = response.choices[0].message
        text = getattr(message, "content", None) or ""
        tool_calls = [
            _from_openai_tool_call(tool_call)
            for tool_call in (getattr(message, "tool_calls", None) or [])
        ]

        usage = getattr(response, "usage", None)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return AgentTurnResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            raw=raw,
        )


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        },
    }


def _to_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role == "assistant" and message.get("tool_calls"):
        converted = {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call.get("arguments", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    },
                }
                for call in message["tool_calls"]
            ],
        }
        return converted
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message["tool_call_id"],
            "content": message.get("content", ""),
        }
    return {
        "role": role,
        "content": message.get("content", ""),
    }


def _from_openai_tool_call(tool_call: Any) -> AgentToolCall:
    function = getattr(tool_call, "function", None)
    raw_arguments = getattr(function, "arguments", "{}") if function else "{}"
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}
    return AgentToolCall(
        id=str(getattr(tool_call, "id", "")),
        name=str(getattr(function, "name", "")),
        arguments=arguments if isinstance(arguments, dict) else {},
    )
