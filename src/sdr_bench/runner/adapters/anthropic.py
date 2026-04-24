"""Anthropic adapter implementation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sdr_bench.agent.types import AgentToolCall
from sdr_bench.agent.types import AgentTurnResponse
from sdr_bench.runner.adapters.base import AdapterResponse


class AnthropicAdapter:
    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.name = f"anthropic:{model}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AdapterResponse:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic adapter.")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        request: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            request["tools"] = [
                {
                    "name": "submit_json",
                    "description": "Return the SDR Bench submission as structured JSON.",
                    "input_schema": json_schema,
                }
            ]
            request["tool_choice"] = {"type": "tool", "name": "submit_json"}

        started = time.perf_counter()
        response = client.messages.create(**request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        text_chunks: list[str] = []
        parsed = None
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(block.text)
            if block_type == "tool_use" and getattr(block, "name", "") == "submit_json":
                parsed = dict(block.input)

        text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
        if parsed is not None and not text:
            text = json.dumps(parsed)

        usage = getattr(response, "usage", None)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return AdapterResponse(
            text=text,
            parsed=parsed,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
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
            raise RuntimeError("ANTHROPIC_API_KEY is required for the Anthropic adapter.")

        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        system, anthropic_messages = _to_anthropic_messages(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "tools": [_to_anthropic_tool(tool) for tool in tools],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            request["system"] = system

        started = time.perf_counter()
        response = client.messages.create(**request)
        latency_ms = int((time.perf_counter() - started) * 1000)

        text_chunks: list[str] = []
        tool_calls: list[AgentToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_chunks.append(getattr(block, "text", ""))
            if block_type == "tool_use":
                tool_input = getattr(block, "input", {})
                tool_calls.append(
                    AgentToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=tool_input if isinstance(tool_input, dict) else {},
                    )
                )

        usage = getattr(response, "usage", None)
        raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return AgentTurnResponse(
            text="\n".join(chunk for chunk in text_chunks if chunk).strip(),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            latency_ms=latency_ms,
            raw=raw,
        )


def _to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("input_schema", {"type": "object"}),
    }


def _to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if content:
                system_chunks.append(str(content))
            continue
        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            if message.get("content"):
                content_blocks.append({"type": "text", "text": str(message["content"])})
            for call in message.get("tool_calls", []):
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("arguments", {}),
                    }
                )
            converted.append({"role": "assistant", "content": content_blocks or ""})
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": message.get("content", ""),
                        }
                    ],
                }
            )
            continue
        converted.append({"role": "user", "content": str(message.get("content", ""))})
    return "\n\n".join(system_chunks), converted
