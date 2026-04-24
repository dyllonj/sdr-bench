"""Stable serialization and hashing helpers for agent traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import is_dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Mapping

from sdr_bench.agent.types import TraceEvent


def normalize_trace_payload(payload: Any) -> Any:
    """Return a JSON-compatible payload with deterministic container shapes."""

    if is_dataclass(payload) and not isinstance(payload, type):
        return normalize_trace_payload(asdict(payload))
    if isinstance(payload, Mapping):
        return {
            str(key): normalize_trace_payload(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list | tuple):
        return [normalize_trace_payload(value) for value in payload]
    if isinstance(payload, set | frozenset):
        normalized_items = [normalize_trace_payload(value) for value in payload]
        return sorted(normalized_items, key=canonical_json)
    if isinstance(payload, datetime | date):
        return payload.isoformat()
    if isinstance(payload, Path):
        return str(payload)
    return payload


def canonical_json(payload: Any) -> str:
    """Serialize payload to stable canonical JSON for trace hashes."""

    return json.dumps(
        normalize_trace_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_hash(payload: Any) -> str:
    """Return the SHA-256 hash of the payload's canonical JSON."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def trace_hash(payload: Any) -> str:
    """Alias for hashing trace payloads with the canonical trace format."""

    return canonical_json_hash(payload)


def hash_payload(payload: Any) -> str:
    """Compatibility alias for callers that hash arbitrary JSON payloads."""

    return canonical_json_hash(payload)


def trace_event_to_dict(event: TraceEvent) -> dict[str, Any]:
    """Materialize a trace event with stable payload and event hashes."""

    event_dict: dict[str, Any] = {
        "event_type": event.event_type,
        "payload": normalize_trace_payload(event.payload),
        "public_only": event.public_only,
    }
    if event.event_index is not None:
        event_dict["event_index"] = event.event_index
    event_dict["payload_hash"] = canonical_json_hash(event_dict["payload"])
    event_dict["event_hash"] = canonical_json_hash(event_dict)
    return event_dict


def build_trace_event(
    event_type: str,
    payload: Any,
    *,
    event_index: int | None = None,
    public_only: bool = True,
) -> dict[str, Any]:
    """Build a serializable trace event dictionary."""

    return trace_event_to_dict(
        TraceEvent(
            event_type=event_type,
            payload=payload,
            event_index=event_index,
            public_only=public_only,
        )
    )
