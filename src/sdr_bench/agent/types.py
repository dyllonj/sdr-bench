"""Shared lightweight types for the SDR Bench agent sandbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class WindowIndexes:
    """Deterministic lookup tables over a public evaluation window."""

    accounts_by_id: dict[str, JsonObject]
    contacts_by_id: dict[str, JsonObject]
    triggers_by_id: dict[str, JsonObject]
    evidence_by_id: dict[str, JsonObject]
    contacts_by_account: dict[str, list[JsonObject]]
    triggers_by_account: dict[str, list[JsonObject]]
    evidence_by_account: dict[str, list[JsonObject]]


@dataclass(slots=True)
class PublicWindowView:
    """Public, model-visible window payload plus deterministic indexes."""

    window_id: str
    window: JsonObject
    indexes: WindowIndexes


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Append-only trace event shape used by tool-mode harnesses."""

    event_type: str
    payload: Any
    event_index: int | None = None
    public_only: bool = True


@dataclass(slots=True)
class AgentToolCall:
    """Provider-neutral model-requested tool call."""

    id: str
    name: str
    arguments: JsonObject


@dataclass(slots=True)
class AgentTurnResponse:
    """One provider-neutral agent turn response."""

    text: str
    tool_calls: list[AgentToolCall]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw: JsonObject


class AgentTurnAdapter(Protocol):
    """Minimal protocol for tool-mode model adapters."""

    name: str

    def create_turn(
        self,
        messages: list[JsonObject],
        tools: list[JsonObject],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AgentTurnResponse: ...
