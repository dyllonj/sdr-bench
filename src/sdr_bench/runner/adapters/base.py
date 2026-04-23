"""Provider-agnostic model adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class AdapterResponse:
    text: str
    parsed: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw: dict[str, Any]


class ModelAdapter(Protocol):
    name: str

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AdapterResponse: ...
