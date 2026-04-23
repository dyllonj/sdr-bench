"""OpenAI adapter implementation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

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
