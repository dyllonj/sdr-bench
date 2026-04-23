"""Canonical structured rationale codes for grounded SDR decisions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

GROUNDING_CODE_FIELDS = (
    ("why_account_codes", "why_account_codes", True),
    ("why_now_code", "why_now_codes", False),
    ("why_persona_code", "why_persona_codes", False),
    ("why_channel_code", "why_channel_codes", False),
)

WHY_NOW_EVENT_CODE_MAP = {
    "leadership_change": "leadership_change_recent",
    "compliance_deadline": "compliance_deadline",
    "usage_change": "usage_change_recent",
    "competitive_displacement": "competitive_displacement",
    "product_launch": "product_launch_recent",
    "hiring": "hiring_recent",
    "expansion": "expansion_recent",
    "funding": "funding_recent",
}


def _freeze_category(raw_codes: dict[str, dict[str, Any]]) -> MappingProxyType[str, MappingProxyType[str, Any]]:
    frozen: dict[str, MappingProxyType[str, Any]] = {}
    for code, entry in raw_codes.items():
        frozen[code] = MappingProxyType(
            {
                "code": code,
                "description": entry["description"],
                "allowed_fact_types": tuple(entry["allowed_fact_types"]),
                "allowed_entities": tuple(entry["allowed_entities"]),
            }
        )
    return MappingProxyType(frozen)


WHY_ACCOUNT_CODES = _freeze_category(
    {
        "enterprise_icp_fit": {
            "description": "The account is a strong named-account enterprise ICP match.",
            "allowed_fact_types": ["source:crm", "source:news"],
            "allowed_entities": ["account"],
        },
        "mid_market_expansion_fit": {
            "description": "The account fits an expansion motion even if it is not tier-one enterprise.",
            "allowed_fact_types": ["source:crm", "source:product"],
            "allowed_entities": ["account"],
        },
        "intent_surge": {
            "description": "Recent high-intent web or product activity indicates active evaluation.",
            "allowed_fact_types": ["source:web", "source:product", "event:usage_change"],
            "allowed_entities": ["account", "trigger"],
        },
        "product_usage_growth": {
            "description": "Product adoption or seat growth suggests expansion potential.",
            "allowed_fact_types": ["source:product", "event:usage_change"],
            "allowed_entities": ["account", "trigger"],
        },
        "competitor_present": {
            "description": "Competitive footprint creates a displacement angle.",
            "allowed_fact_types": ["source:crm", "source:news", "event:competitive_displacement"],
            "allowed_entities": ["account", "trigger"],
        },
        "security_stack_match": {
            "description": "Observed security tooling or needs align with the offered solution.",
            "allowed_fact_types": ["source:crm", "source:contact"],
            "allowed_entities": ["account", "contact"],
        },
        "cloud_modernization_fit": {
            "description": "The cloud and data stack fit a modernization or migration message.",
            "allowed_fact_types": ["source:crm", "source:news"],
            "allowed_entities": ["account"],
        },
        "named_account_priority": {
            "description": "The account is explicitly prioritized by territory or account strategy.",
            "allowed_fact_types": ["source:crm"],
            "allowed_entities": ["account"],
        },
    }
)

WHY_NOW_CODES = _freeze_category(
    {
        "leadership_change_recent": {
            "description": "A fresh leadership change creates a near-term buying window.",
            "allowed_fact_types": ["source:news", "event:leadership_change"],
            "allowed_entities": ["trigger", "account"],
        },
        "compliance_deadline": {
            "description": "A compliance deadline creates a time-bounded need to act.",
            "allowed_fact_types": ["source:news", "source:crm", "event:compliance_deadline"],
            "allowed_entities": ["trigger", "account"],
        },
        "usage_change_recent": {
            "description": "Recent product-usage movement makes the timing attractive now.",
            "allowed_fact_types": ["source:product", "event:usage_change"],
            "allowed_entities": ["trigger", "account"],
        },
        "competitive_displacement": {
            "description": "A live competitive event opens a replacement window.",
            "allowed_fact_types": ["source:news", "event:competitive_displacement"],
            "allowed_entities": ["trigger", "account"],
        },
        "product_launch_recent": {
            "description": "A recent launch or roadmap event changes the relevance of outreach.",
            "allowed_fact_types": ["source:news", "event:product_launch"],
            "allowed_entities": ["trigger", "account"],
        },
        "hiring_recent": {
            "description": "New hiring implies active budget, project momentum, or team formation.",
            "allowed_fact_types": ["source:jobs", "event:hiring"],
            "allowed_entities": ["trigger", "account"],
        },
        "expansion_recent": {
            "description": "Expansion signals suggest a new budget or operational moment.",
            "allowed_fact_types": ["source:news", "event:expansion"],
            "allowed_entities": ["trigger", "account"],
        },
        "funding_recent": {
            "description": "New funding changes the urgency or affordability of a project.",
            "allowed_fact_types": ["source:news", "event:funding"],
            "allowed_entities": ["trigger", "account"],
        },
        "timing_signal_recent": {
            "description": "A generic but fresh trigger justifies action even without a narrower why-now code.",
            "allowed_fact_types": ["source:news", "source:product", "source:jobs"],
            "allowed_entities": ["trigger", "account"],
        },
    }
)

WHY_PERSONA_CODES = _freeze_category(
    {
        "technical_buyer_plus_security_champion": {
            "description": "A technical buyer and a security champion together cover the buying center.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "technical_buyer": {
            "description": "The main target should be the technical buying owner.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "economic_buyer": {
            "description": "The main target should be an economic or budget owner.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "champion": {
            "description": "An internal user or operational champion is the best entry point.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "security_buyer": {
            "description": "Security ownership is central to deal progression.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "technical_champion_pair": {
            "description": "A technical buyer and a likely champion should be worked together.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "multithreaded_buying_center": {
            "description": "Multiple complementary stakeholders should be contacted in parallel.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "end_user_champion": {
            "description": "A high-engagement end user is the best near-term persona target.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
    }
)

WHY_CHANNEL_CODES = _freeze_category(
    {
        "email_valid_plus_recent_web_intent": {
            "description": "Email is viable and recent web or product intent supports immediate outreach.",
            "allowed_fact_types": ["source:web", "source:product", "source:contact"],
            "allowed_entities": ["account", "contact", "trigger"],
        },
        "email_valid": {
            "description": "Valid email reachability makes direct outbound appropriate.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "linkedin_present": {
            "description": "LinkedIn is available and is the best reliable reachable channel.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "phone_valid": {
            "description": "Phone reachability supports call-first or multichannel outreach.",
            "allowed_fact_types": ["source:contact"],
            "allowed_entities": ["contact"],
        },
        "multichannel_exec_outreach": {
            "description": "The account supports a coordinated multichannel outreach pattern.",
            "allowed_fact_types": ["source:contact", "source:web"],
            "allowed_entities": ["contact", "account"],
        },
        "nurture_until_signal": {
            "description": "The account should remain in lower-touch nurture until a stronger signal appears.",
            "allowed_fact_types": ["source:web", "source:crm"],
            "allowed_entities": ["account"],
        },
        "manual_research_required": {
            "description": "Existing channels are weak enough that more research is required before outreach.",
            "allowed_fact_types": ["source:crm", "source:contact"],
            "allowed_entities": ["account", "contact"],
        },
        "warm_intro_available": {
            "description": "Relationship evidence suggests a warm introduction channel.",
            "allowed_fact_types": ["source:crm", "source:contact"],
            "allowed_entities": ["account", "contact"],
        },
    }
)

RATIONALE_CODE_MAPS = MappingProxyType(
    {
        "why_account_codes": WHY_ACCOUNT_CODES,
        "why_now_codes": WHY_NOW_CODES,
        "why_persona_codes": WHY_PERSONA_CODES,
        "why_channel_codes": WHY_CHANNEL_CODES,
    }
)


def get_code_definition(category: str, code: str) -> dict[str, Any] | None:
    definitions = RATIONALE_CODE_MAPS.get(category)
    if definitions is None:
        return None
    definition = definitions.get(code)
    if definition is None:
        return None
    return {
        "code": definition["code"],
        "description": definition["description"],
        "allowed_fact_types": list(definition["allowed_fact_types"]),
        "allowed_entities": list(definition["allowed_entities"]),
    }


def list_all_codes() -> dict[str, list[str]]:
    return {
        category: sorted(definitions)
        for category, definitions in RATIONALE_CODE_MAPS.items()
    }


def build_rationale_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for category, definitions in RATIONALE_CODE_MAPS.items():
        catalog[category] = {
            code: get_code_definition(category, code) or {}
            for code in sorted(definitions)
        }
    return catalog


def derive_document_fact_types(
    document: dict[str, Any],
    trigger_by_id: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    fact_types = {f"source:{document['source_type']}"}
    support = document.get("grounding_support", {})
    if trigger_by_id:
        for event_id in support.get("related_event_ids", []):
            trigger = trigger_by_id.get(event_id)
            if trigger:
                fact_types.add(f"event:{trigger['event_type']}")
    return fact_types


def derive_document_entities(document: dict[str, Any]) -> set[str]:
    entities = {"account"}
    support = document.get("grounding_support", {})
    if document["source_type"] == "contact" or support.get("why_persona_codes"):
        entities.add("contact")
    if support.get("related_event_ids"):
        entities.add("trigger")
    return entities


def supports_grounding_claim(
    category: str,
    code: str,
    document: dict[str, Any],
    trigger_by_id: dict[str, dict[str, Any]] | None = None,
) -> bool:
    definition = get_code_definition(category, code)
    if definition is None:
        return False

    support = document.get("grounding_support", {})
    if code not in support.get(category, []):
        return False

    fact_types = derive_document_fact_types(document, trigger_by_id)
    entities = derive_document_entities(document)
    if not set(definition["allowed_fact_types"]).intersection(fact_types):
        return False
    if not set(definition["allowed_entities"]).intersection(entities):
        return False
    return True


def format_rationale_codes_markdown() -> str:
    lines = [
        "# Rationale Codes",
        "",
        "Canonical structured rationale codes used by the evaluator, prompts, and synthetic generator.",
        "",
    ]

    titles = {
        "why_account_codes": "Why Account",
        "why_now_codes": "Why Now",
        "why_persona_codes": "Why Persona",
        "why_channel_codes": "Why Channel",
    }

    for category, title in titles.items():
        lines.extend(
            [
                f"## {title}",
                "",
                "| Code | Description | Allowed Fact Types | Allowed Entities |",
                "| --- | --- | --- | --- |",
            ]
        )
        for code, definition in sorted(RATIONALE_CODE_MAPS[category].items()):
            fact_types = ", ".join(definition["allowed_fact_types"])
            entities = ", ".join(definition["allowed_entities"])
            lines.append(
                f"| `{code}` | {definition['description']} | `{fact_types}` | `{entities}` |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
