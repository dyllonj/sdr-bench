"""Public seller-knowledge artifacts for full-cycle SDR agent tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sdr_bench.agent.public_views import redact_scoring_fields


SELLER_KNOWLEDGE_SECTIONS = (
    "value_props",
    "case_studies",
    "product_docs",
    "objection_answers",
    "handoff_criteria",
    "qualification_criteria",
)

DEFAULT_SELLER_PROFILE: dict[str, Any] = {
    "seller_profile_id": "neutral_enterprise_tech_v1",
    "seller": {
        "company_name_redacted": True,
        "category": "enterprise data security and workflow automation platform",
        "target_segments": ["mid_market", "enterprise"],
        "primary_buyers": ["IT", "Security", "Engineering", "Operations"],
    },
    "value_props": [
        {
            "knowledge_id": "vp_secure_workflows",
            "title": "Secure workflow automation",
            "content": (
                "Helps IT and security teams automate approval-heavy workflows while "
                "keeping audit trails, access policy, and operational ownership clear."
            ),
            "tags": ["security", "automation", "audit"],
        },
        {
            "knowledge_id": "vp_data_governance",
            "title": "Governed data access",
            "content": (
                "Centralizes data access requests, policy checks, and exception handling "
                "for teams using cloud data warehouses and analytics platforms."
            ),
            "tags": ["data", "governance", "compliance"],
        },
    ],
    "case_studies": [
        {
            "knowledge_id": "case_financial_services_controls",
            "title": "Financial services access review",
            "content": (
                "A regulated financial services customer reduced manual access-review "
                "work by standardizing approval flows and evidence capture."
            ),
            "tags": ["financial_services", "compliance", "access_review"],
        },
        {
            "knowledge_id": "case_platform_team_growth",
            "title": "Platform team request automation",
            "content": (
                "A fast-growing software company used the platform to route internal "
                "data and infrastructure requests without increasing platform-team headcount."
            ),
            "tags": ["software", "platform", "scale"],
        },
    ],
    "product_docs": [
        {
            "knowledge_id": "doc_integrations",
            "title": "Core integrations",
            "content": (
                "The platform integrates with common identity providers, cloud data "
                "warehouses, ticketing systems, and collaboration tools."
            ),
            "tags": ["integrations", "identity", "data_platform"],
        },
        {
            "knowledge_id": "doc_security_controls",
            "title": "Security controls",
            "content": (
                "Supported controls include role-based access, policy approval steps, "
                "audit logging, reviewer assignment, and escalation rules."
            ),
            "tags": ["security", "rbac", "audit"],
        },
    ],
    "objection_answers": [
        {
            "knowledge_id": "obj_existing_ticketing",
            "title": "We already use a ticketing system",
            "content": (
                "Position the platform as the policy and workflow layer that coordinates "
                "ticketing, identity, and data systems rather than replacing every ticket."
            ),
            "tags": ["ticketing", "workflow", "positioning"],
        },
        {
            "knowledge_id": "obj_timing",
            "title": "This is not a priority this quarter",
            "content": (
                "Ask whether compliance deadlines, access-review cycles, headcount "
                "constraints, or platform growth create a reason to revisit timing."
            ),
            "tags": ["timing", "discovery", "priority"],
        },
    ],
    "handoff_criteria": [
        {
            "knowledge_id": "handoff_ae_ready",
            "title": "AE handoff readiness",
            "content": (
                "Hand off when there is a relevant business pain, a plausible buying "
                "center contact, a current initiative or deadline, and agreement to meet."
            ),
            "criteria": ["pain", "authority_or_influence", "timing", "meeting_intent"],
            "tags": ["handoff", "qualification"],
        }
    ],
    "qualification_criteria": [
        {
            "knowledge_id": "qual_enterprise_sdr",
            "title": "Enterprise SDR qualification",
            "content": (
                "Qualify need, authority or influence, timing, current workflow, data "
                "or security environment, and expected business impact."
            ),
            "criteria": ["need", "authority", "timing", "environment", "impact"],
            "tags": ["qualification", "discovery"],
        }
    ],
}


class SellerKnowledgeError(ValueError):
    """Structured validation error for seller-knowledge queries."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def default_seller_profile() -> dict[str, Any]:
    """Return a public neutral enterprise-tech seller profile."""

    return deepcopy(DEFAULT_SELLER_PROFILE)


def query_seller_knowledge(
    seller_profile: dict[str, Any],
    *,
    section: str | None = None,
    query: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Return public seller knowledge items matching optional section/query filters."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 20:
        raise SellerKnowledgeError(
            "invalid_limit",
            "limit must be an integer between 1 and 20",
            limit=limit,
            max_limit=20,
        )
    if section is not None and section not in SELLER_KNOWLEDGE_SECTIONS:
        raise SellerKnowledgeError(
            "unknown_section",
            "section is not part of the public seller profile",
            section=section,
            allowed_sections=list(SELLER_KNOWLEDGE_SECTIONS),
        )

    public_profile = redact_scoring_fields(seller_profile)
    selected_sections = (section,) if section else SELLER_KNOWLEDGE_SECTIONS
    query_text = query.casefold().strip() if isinstance(query, str) else ""
    items = []
    for section_name in selected_sections:
        raw_items = public_profile.get(section_name, [])
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = deepcopy(raw_item)
            item["section"] = section_name
            if query_text and query_text not in _search_text(item):
                continue
            items.append(item)

    return {
        "seller_profile_id": public_profile.get("seller_profile_id"),
        "seller": public_profile.get("seller", {}),
        "section": section,
        "query": query,
        "items": items[:limit],
        "total_matches": len(items),
    }


def _search_text(item: dict[str, Any]) -> str:
    chunks: list[str] = []
    for value in item.values():
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            chunks.extend(str(child) for child in value)
    return " ".join(chunks).casefold()
