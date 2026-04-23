"""Anthropic adapter implementation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

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
