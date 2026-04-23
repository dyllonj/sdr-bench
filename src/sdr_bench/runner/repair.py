"""Repair loop for invalid or malformed model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from json import JSONDecoder
from typing import Any

from jsonschema import RefResolver
from jsonschema.validators import validator_for

from sdr_bench.evaluator import load_schemas
from sdr_bench.runner.adapters.base import AdapterResponse
from sdr_bench.runner.adapters.base import ModelAdapter


@dataclass(slots=True)
class RepairResult:
    submission: dict[str, Any]
    usage_log: dict[str, Any]


def _validator_for_schema(schema: dict[str, Any]):
    schemas, store = load_schemas()
    store.update(
        {
            schema.get("$id"): schema
            for schema in schemas.values()
            if schema.get("$id")
        }
    )
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    resolver = RefResolver.from_schema(schema, store=store)
    return validator_cls(schema, resolver=resolver)


def _format_errors(validator, payload: Any) -> list[str]:
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    formatted: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path)
        if path:
            formatted.append(f"{path}: {error.message}")
        else:
            formatted.append(error.message)
    return formatted


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_first_json_value(text: str) -> Any | None:
    decoder = JSONDecoder()
    normalized = _strip_markdown_fences(text)
    try:
        return json.loads(normalized)
    except JSONDecodeError:
        pass

    for index, character in enumerate(normalized):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(normalized[index:])
            return value
        except JSONDecodeError:
            continue
    return None


def _parse_candidate(response: AdapterResponse) -> dict[str, Any] | None:
    if response.parsed is not None:
        return response.parsed
    extracted = _extract_first_json_value(response.text)
    if isinstance(extracted, dict):
        return extracted
    return None


def wait_everything_submission(window_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": window_data["window_id"],
        "decisions": [
            {
                "account_id": account["account_id"],
                "chosen_action": "wait",
                "action_score": 0.0,
            }
            for account in window_data["accounts"]
        ],
    }


def generate_with_repair(
    adapter: ModelAdapter,
    *,
    window_data: dict[str, Any],
    system: str,
    user: str,
    json_schema: dict[str, Any],
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> RepairResult:
    validator = _validator_for_schema(json_schema)
    attempts: list[dict[str, Any]] = []
    retry_count = 0
    repair_outcome = "initial_success"
    validation_errors: list[str] = []

    for attempt_index in range(2):
        response = adapter.generate(
            system,
            user,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        candidate = _parse_candidate(response)
        errors = ["No JSON object could be parsed from the adapter response."]
        if candidate is not None:
            errors = _format_errors(validator, candidate)

        attempts.append(
            {
                "attempt_index": attempt_index,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "parsed": candidate is not None,
                "validation_errors": errors,
            }
        )

        if candidate is not None and not errors:
            if attempt_index == 1:
                repair_outcome = "retry_success"
                retry_count = 1
            usage_log = {
                "adapter_name": adapter.name,
                "retry_count": retry_count,
                "repair_outcome": repair_outcome,
                "attempts": attempts,
                "input_tokens": sum(item["input_tokens"] for item in attempts),
                "output_tokens": sum(item["output_tokens"] for item in attempts),
                "latency_ms": sum(item["latency_ms"] for item in attempts),
                "validation_errors": [],
            }
            return RepairResult(submission=candidate, usage_log=usage_log)

        validation_errors = errors
        if attempt_index == 0:
            retry_count = 1
            repair_outcome = "retry_after_validation_error"
            user = (
                f"{user}\n\n"
                "The previous response was invalid against the required JSON schema.\n"
                f"Validation errors:\n- " + "\n- ".join(errors) + "\n"
                "Return only one valid JSON object that satisfies the schema."
            )

    fallback = wait_everything_submission(window_data)
    usage_log = {
        "adapter_name": adapter.name,
        "retry_count": retry_count,
        "repair_outcome": "fallback_wait_submission",
        "attempts": attempts,
        "input_tokens": sum(item["input_tokens"] for item in attempts),
        "output_tokens": sum(item["output_tokens"] for item in attempts),
        "latency_ms": sum(item["latency_ms"] for item in attempts),
        "validation_errors": validation_errors,
    }
    return RepairResult(submission=fallback, usage_log=usage_log)
