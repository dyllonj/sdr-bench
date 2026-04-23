"""Prompt builders for offline and policy runs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from sdr_bench.evaluator import load_schemas
from sdr_bench.rationale_codes import build_rationale_catalog

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _read_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _action_semantics_table() -> str:
    return "\n".join(
        [
            "| Action | Meaning |",
            "| --- | --- |",
            "| `human_touch` | Spend one scarce personalized SDR slot on the account this week. |",
            "| `automated_outbound` | Put the account into a low-cost outbound sequence. |",
            "| `nurture` | Keep the account warm via lower-touch nurture. |",
            "| `recycle` | Snooze the account because fit exists but timing is weak right now. |",
            "| `disqualify` | Mark the account as out of scope or unreachable. |",
            "| `wait` | Take no action this week. |",
        ]
    )


def _rationale_code_markdown(rationale_codes: dict[str, dict[str, dict[str, Any]]] | None = None) -> str:
    catalog = rationale_codes or build_rationale_catalog()
    title_map = {
        "why_account_codes": "Why Account",
        "why_now_codes": "Why Now",
        "why_persona_codes": "Why Persona",
        "why_channel_codes": "Why Channel",
    }
    sections: list[str] = []
    for category in (
        "why_account_codes",
        "why_now_codes",
        "why_persona_codes",
        "why_channel_codes",
    ):
        sections.append(f"### {title_map[category]}")
        for code, definition in catalog[category].items():
            sections.append(
                f"- `{code}`: {definition['description']} "
                f"(fact types: {', '.join(definition['allowed_fact_types'])}; "
                f"entities: {', '.join(definition['allowed_entities'])})"
            )
        sections.append("")
    return "\n".join(sections).strip()


def _window_payload(window: dict[str, Any], budget: int | None) -> dict[str, Any]:
    payload = deepcopy(window)
    if budget is not None:
        payload["capacity_budget"]["human_sdr_actions"] = budget
    return payload


def _base_schema() -> dict[str, Any]:
    schemas, _ = load_schemas()
    return deepcopy(schemas["model_output"])


def build_window_prompt(
    window: dict[str, Any],
    budget: int | None = None,
    rationale_codes: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    schema = _base_schema()
    window_payload = _window_payload(window, budget)
    budget_value = window_payload["capacity_budget"]["human_sdr_actions"]
    system = _read_template("window_system.md").format(
        action_semantics_table=_action_semantics_table(),
        rationale_code_catalog=_rationale_code_markdown(rationale_codes),
    )
    user = _read_template("window_user.md").format(
        budget=budget_value,
        max_contacts=window_payload["capacity_budget"]["max_contacts_per_account"],
        output_schema=json.dumps(schema, indent=2, sort_keys=True),
        window_json=json.dumps(window_payload, indent=2, sort_keys=True),
    )
    return system, user, schema


def build_episode_prompt(
    episode_state: dict[str, Any],
    history: list[dict[str, Any]],
    rationale_codes: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    schema = _base_schema()
    system = _read_template("episode_system.md").format(
        action_semantics_table=_action_semantics_table(),
        rationale_code_catalog=_rationale_code_markdown(rationale_codes),
    )
    user = _read_template("episode_user.md").format(
        week_index=episode_state["week_index"],
        total_weeks=episode_state["total_weeks"],
        history_json=json.dumps(history, indent=2, sort_keys=True),
        state_json=json.dumps(episode_state, indent=2, sort_keys=True),
        output_schema=json.dumps(schema, indent=2, sort_keys=True),
    )
    return system, user, schema
