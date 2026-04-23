"""OpenAI-compatible adapter for local or hosted OSS endpoints."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sdr_bench.runner.adapters.base import AdapterResponse

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OpenAICompatibleAdapter:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.api_key = (
            api_key
            or os.getenv("OPENAI_COMPATIBLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "sk-local"
        )
        self.name = f"openai_compatible:{model}@{self.base_url}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AdapterResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
