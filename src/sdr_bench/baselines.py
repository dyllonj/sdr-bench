"""Deterministic reference baselines for the SDR benchmark."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from sdr_bench.rationale_codes import WHY_NOW_EVENT_CODE_MAP

BASELINE_NAMES = (
    "random_within_icp",
    "rules_trigger_queue",
    "propensity_only_lead_score",
    "last_touch_recency",
    "bau_enterprise_sdr_policy",
)
ORACLE_BASELINE_NAME = "oracle_hidden_labels"

IN_SCOPE_RELATIONSHIP_MOTIONS = {
    "net_new",
    "product_led_pre_opportunity",
}

SENIORITY_SCORES = {
    "C": 1.00,
    "SVP": 0.92,
    "VP": 0.86,
    "Head": 0.78,
    "Director": 0.66,
    "Manager": 0.50,
    "Lead": 0.44,
    "IC": 0.34,
}

BUYING_ROLE_SCORES = {
    "economic_buyer": 1.00,
    "technical_buyer": 0.94,
    "champion": 0.82,
    "security_buyer": 0.80,
    "user": 0.48,
}

REPLY_BUCKET_SCORES = {
    "high": 1.00,
    "medium": 0.65,
    "low": 0.30,
}

TRIGGER_TYPE_WEIGHTS = {
    "leadership_change": 1.00,
    "compliance_deadline": 0.98,
    "usage_change": 0.90,
    "competitive_displacement": 0.90,
    "product_launch": 0.80,
    "hiring": 0.72,
    "expansion": 0.72,
    "funding": 0.60,
}

REVENUE_BAND_SCORES = {
    "1b_5b": 1.00,
    "250m_1b": 0.82,
    "100m_250m": 0.64,
    "25m_100m": 0.46,
}

NAMED_TIER_SCORES = {
    "tier_1_named": 1.00,
    "tier_2_named": 0.84,
    "tier_3_named": 0.68,
}


def list_baselines() -> tuple[str, ...]:
    return BASELINE_NAMES


def ensure_baseline_name(baseline_name: str) -> None:
    if baseline_name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline: {baseline_name}")


def stable_text_seed(text: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(text))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def account_id_sort_key(context: dict[str, Any]) -> str:
    return context["account"]["account_id"]


def freshness_score(recency_days: int) -> float:
    return clamp(1.0 - (min(recency_days, 30) / 30.0))


def company_scale_score(account: dict[str, Any]) -> float:
    employee_score = clamp(account["employee_count"] / 4000.0)
    revenue_score = REVENUE_BAND_SCORES.get(account["revenue_band"], 0.45)
    return 0.55 * employee_score + 0.45 * revenue_score


def compute_fit_score(account: dict[str, Any]) -> float:
    tier_score = NAMED_TIER_SCORES.get(account["account_tier"], 0.45)
    tech_completeness = 0.0
    technographics = account["technographics"]
    tech_fields = (
        technographics.get("cloud_provider"),
        technographics.get("identity_provider"),
        technographics.get("data_platform"),
    )
    tech_completeness = sum(1 for field in tech_fields if field) / len(tech_fields)

    score = 0.0
    score += 0.28 if account["segment"] == "enterprise" else 0.04
    score += 0.18 if account["relationship_motion"] in IN_SCOPE_RELATIONSHIP_MOTIONS else 0.0
    score += 0.20 * tier_score
    score += 0.12 if account["sales_geo"] == "NA_enterprise" else 0.05
    score += 0.12 * company_scale_score(account)
    score += 0.10 * tech_completeness
    if account["crm_state"]["has_open_opportunity"]:
        score -= 0.20
    if account["crm_state"]["open_support_escalation"]:
        score -= 0.08
    return clamp(score)


def compute_web_intent_score(account: dict[str, Any]) -> float:
    engagement = account["web_engagement"]
    pricing = clamp(engagement["pricing_visits_30d"] / 5.0)
    product_pages = clamp(engagement["product_pages_30d"] / 12.0)
    downloads = clamp(engagement["high_intent_content_downloads_30d"] / 2.0)
    trials = clamp(engagement["trial_signups_30d"] / 1.0)
    return clamp(
        0.35 * pricing
        + 0.25 * product_pages
        + 0.20 * downloads
        + 0.20 * trials
    )


def compute_product_usage_score(account: dict[str, Any]) -> float:
    usage = account["product_or_usage_signals"]
    seat_growth = clamp(usage["seat_growth_30d"] / 0.18)
    active_growth = clamp(usage["active_users_growth_30d"] / 0.12)
    feature_score = clamp(len(usage["feature_adoption_flags"]) / 2.0)
    return clamp(
        0.45 * seat_growth
        + 0.35 * active_growth
        + 0.20 * feature_score
    )


def compute_trigger_score(trigger: dict[str, Any]) -> float:
    return clamp(
        0.40 * TRIGGER_TYPE_WEIGHTS.get(trigger["event_type"], 0.50)
        + 0.35 * trigger["confidence"]
        + 0.25 * freshness_score(trigger["recency_days"])
    )


def rank_contacts(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        contacts,
        key=lambda contact: (
            1.0 if contact["channel_reachability"]["email_valid"] else 0.0,
            BUYING_ROLE_SCORES.get(contact["likely_buying_role"], 0.40),
            SENIORITY_SCORES.get(contact["seniority"], 0.40),
            REPLY_BUCKET_SCORES.get(contact["historical_reply_rate_bucket"], 0.40),
            clamp(contact["recent_activity"]["content_engagement_30d"] / 5.0),
            contact["contact_id"],
        ),
        reverse=True,
    )


def select_contacts(context: dict[str, Any], *, prefer_coverage: bool = True) -> list[str]:
    ranked = rank_contacts(context["contacts"])
    if not ranked:
        return []

    selection_limit = min(context["max_contacts"], 2 if len(ranked) > 1 else 1)
    selected_ids: list[str] = [ranked[0]["contact_id"]]
    selected_roles = {ranked[0].get("likely_buying_role")}
    selected_departments = {ranked[0].get("department")}

    if prefer_coverage:
        for contact in ranked[1:]:
            if len(selected_ids) >= selection_limit:
                break
            role = contact.get("likely_buying_role")
            department = contact.get("department")
            if role not in selected_roles or department not in selected_departments:
                selected_ids.append(contact["contact_id"])
                selected_roles.add(role)
                selected_departments.add(department)

    for contact in ranked[1:]:
        if len(selected_ids) >= selection_limit:
            break
        if contact["contact_id"] not in selected_ids:
            selected_ids.append(contact["contact_id"])

    return selected_ids


def infer_persona_code(context: dict[str, Any], selected_contact_ids: list[str]) -> str:
    contacts_by_id = {
        contact["contact_id"]: contact
        for contact in context["contacts"]
    }
    selected_contacts = [
        contacts_by_id[contact_id]
        for contact_id in selected_contact_ids
        if contact_id in contacts_by_id
    ]
    roles = {
        contact.get("likely_buying_role")
        for contact in selected_contacts
    }
    functions = {
        contact.get("function")
        for contact in selected_contacts
    }
    if "technical_buyer" in roles and ("champion" in roles or "Security" in functions):
        return "technical_buyer_plus_security_champion"
    if "technical_buyer" in roles:
        return "technical_buyer"
    if "economic_buyer" in roles:
        return "economic_buyer"
    if "champion" in roles:
        return "champion"
    return "technical_buyer"


def infer_channel_code(context: dict[str, Any], selected_contact_ids: list[str]) -> str:
    contacts_by_id = {
        contact["contact_id"]: contact
        for contact in context["contacts"]
    }
    selected_contacts = [
        contacts_by_id[contact_id]
        for contact_id in selected_contact_ids
        if contact_id in contacts_by_id
    ]
    has_email = any(contact["channel_reachability"]["email_valid"] for contact in selected_contacts)
    has_linkedin = any(contact["channel_reachability"]["linkedin_present"] for contact in selected_contacts)
    if has_email and context["web_intent_score"] >= 0.35:
        return "email_valid_plus_recent_web_intent"
    if has_email:
        return "email_valid"
    if has_linkedin:
        return "linkedin_present"
    return "manual_research_required"


def infer_account_codes(context: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if context["fit_score"] >= 0.70:
        codes.append("enterprise_icp_fit")
    if context["web_intent_score"] >= 0.45:
        codes.append("intent_surge")
    if context["product_usage_score"] >= 0.45:
        codes.append("product_usage_growth")
    if not codes:
        codes.append("enterprise_icp_fit")
    return codes[:2]


def infer_why_now_code(trigger: dict[str, Any]) -> str:
    return WHY_NOW_EVENT_CODE_MAP.get(trigger["event_type"], "timing_signal_recent")


def group_window_entities(window_data: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    contacts_by_account: dict[str, list[dict[str, Any]]] = {}
    triggers_by_account: dict[str, list[dict[str, Any]]] = {}
    evidence_by_account: dict[str, list[dict[str, Any]]] = {}

    for account in window_data["accounts"]:
        account_id = account["account_id"]
        contacts_by_account[account_id] = []
        triggers_by_account[account_id] = []
        evidence_by_account[account_id] = []

    for contact in window_data["contacts"]:
        contacts_by_account.setdefault(contact["account_id"], []).append(contact)
    for trigger in window_data["triggers"]:
        triggers_by_account.setdefault(trigger["account_id"], []).append(trigger)
    for document in window_data["evidence"]:
        evidence_by_account.setdefault(document["account_id"], []).append(document)

    return contacts_by_account, triggers_by_account, evidence_by_account


def build_account_contexts(window_data: dict[str, Any]) -> list[dict[str, Any]]:
    contacts_by_account, triggers_by_account, evidence_by_account = group_window_entities(window_data)
    max_contacts = window_data["capacity_budget"]["max_contacts_per_account"]
    contexts: list[dict[str, Any]] = []

    for account in window_data["accounts"]:
        account_id = account["account_id"]
        contacts = contacts_by_account.get(account_id, [])
        triggers = triggers_by_account.get(account_id, [])
        evidence = evidence_by_account.get(account_id, [])
        fit_score = compute_fit_score(account)
        web_intent_score = compute_web_intent_score(account)
        product_usage_score = compute_product_usage_score(account)
        best_trigger_score = max((compute_trigger_score(trigger) for trigger in triggers), default=0.0)
        best_trigger_freshness = max((freshness_score(trigger["recency_days"]) for trigger in triggers), default=0.0)
        contactability_score = max(
            (
                (
                    0.40 * (1.0 if contact["channel_reachability"]["email_valid"] else 0.0)
                    + 0.20 * BUYING_ROLE_SCORES.get(contact["likely_buying_role"], 0.40)
                    + 0.15 * SENIORITY_SCORES.get(contact["seniority"], 0.40)
                    + 0.15 * REPLY_BUCKET_SCORES.get(contact["historical_reply_rate_bucket"], 0.40)
                    + 0.10 * clamp(contact["recent_activity"]["content_engagement_30d"] / 5.0)
                )
                for contact in contacts
            ),
            default=0.0,
        )
        cooldown_ok = account["crm_state"]["days_since_last_human_touch"] >= 21
        sequence_ready = account["crm_state"]["days_since_last_outbound_sequence"] >= 7
        in_scope_icp = (
            account["segment"] == "enterprise"
            and account["relationship_motion"] in IN_SCOPE_RELATIONSHIP_MOTIONS
            and not account["crm_state"]["has_open_opportunity"]
        )

        contexts.append(
            {
                "account": account,
                "contacts": contacts,
                "triggers": triggers,
                "evidence": evidence,
                "max_contacts": max_contacts,
                "fit_score": fit_score,
                "web_intent_score": web_intent_score,
                "product_usage_score": product_usage_score,
                "trigger_score": best_trigger_score,
                "freshness_score": best_trigger_freshness,
                "contactability_score": contactability_score,
                "cooldown_ok": cooldown_ok,
                "sequence_ready": sequence_ready,
                "in_scope_icp": in_scope_icp,
                "human_touch_feasible": bool(contacts and triggers and evidence),
                "automation_feasible": any(
                    contact["channel_reachability"]["email_valid"]
                    for contact in contacts
                ),
                "episode_eligible": True,
                "episode_removed": False,
            }
        )

    return contexts


def choose_primary_trigger(context: dict[str, Any]) -> dict[str, Any] | None:
    if not context["triggers"]:
        return None
    return sorted(
        context["triggers"],
        key=lambda trigger: (
            compute_trigger_score(trigger),
            freshness_score(trigger["recency_days"]),
            trigger["confidence"],
            trigger["event_id"],
        ),
        reverse=True,
    )[0]


def build_grounding_maps(context: dict[str, Any]) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]], dict[str, dict[str, Any]]]:
    code_support = {
        "why_account_codes": {},
        "why_now_codes": {},
        "why_persona_codes": {},
        "why_channel_codes": {},
    }
    event_support: dict[str, list[str]] = {}
    docs_by_id: dict[str, dict[str, Any]] = {}

    for document in context["evidence"]:
        if not document["allowed_for_grounding"]:
            continue
        docs_by_id[document["doc_id"]] = document
        support = document.get("grounding_support", {})
        for support_key in code_support:
            for code in support.get(support_key, []):
                code_support[support_key].setdefault(code, []).append(document["doc_id"])
        for event_id in support.get("related_event_ids", []):
            event_support.setdefault(event_id, []).append(document["doc_id"])

    return code_support, event_support, docs_by_id


def choose_code_with_citations(
    preferred_codes: list[str],
    supported_codes: dict[str, list[str]],
    fallback_code: str,
) -> tuple[str, list[str]]:
    for code in preferred_codes:
        citations = supported_codes.get(code)
        if citations:
            return code, citations

    if supported_codes:
        chosen_code = sorted(supported_codes)[0]
        return chosen_code, supported_codes[chosen_code]

    return fallback_code, []


def build_evidence_brief(
    context: dict[str, Any],
    selected_contacts: list[str],
    trigger: dict[str, Any],
) -> dict[str, Any]:
    code_support, event_support, docs_by_id = build_grounding_maps(context)

    preferred_account_codes = infer_account_codes(context)
    chosen_account_codes: list[str] = []
    citations: list[str] = []
    for code in preferred_account_codes:
        if code in chosen_account_codes:
            continue
        if len(chosen_account_codes) >= 2:
            break
        if code in code_support["why_account_codes"]:
            chosen_account_codes.append(code)
            citations.extend(code_support["why_account_codes"][code])
    if not chosen_account_codes:
        chosen_account_codes = preferred_account_codes[:1]

    inferred_why_now = infer_why_now_code(trigger)
    why_now_code, why_now_citations = choose_code_with_citations(
        [inferred_why_now],
        {
            code: doc_ids
            for code, doc_ids in code_support["why_now_codes"].items()
            if set(doc_ids).intersection(event_support.get(trigger["event_id"], []))
        },
        inferred_why_now,
    )
    if not why_now_citations:
        why_now_citations = event_support.get(trigger["event_id"], [])
    citations.extend(why_now_citations)

    inferred_persona = infer_persona_code(context, selected_contacts)
    why_persona_code, persona_citations = choose_code_with_citations(
        [inferred_persona],
        code_support["why_persona_codes"],
        inferred_persona,
    )
    citations.extend(persona_citations)

    inferred_channel = infer_channel_code(context, selected_contacts)
    why_channel_code, channel_citations = choose_code_with_citations(
        [inferred_channel],
        code_support["why_channel_codes"],
        inferred_channel,
    )
    citations.extend(channel_citations)

    trigger_doc_ids = [
        doc_id
        for doc_id in trigger["evidence_refs"]
        if doc_id in docs_by_id
    ]
    citations.extend(trigger_doc_ids)

    if not citations:
        citations = [
            document["doc_id"]
            for document in context["evidence"]
            if document["allowed_for_grounding"]
        ]
    if not citations:
        citations = [document["doc_id"] for document in context["evidence"][:1]]

    deduped_citations: list[str] = []
    for doc_id in citations:
        if doc_id not in deduped_citations:
            deduped_citations.append(doc_id)

    return {
        "why_account_codes": chosen_account_codes[:2],
        "why_now_code": why_now_code,
        "why_persona_code": why_persona_code,
        "why_channel_code": why_channel_code,
        "citations": deduped_citations[:4],
    }


def human_touch_action_score(context: dict[str, Any], baseline_score: float) -> float:
    return round(clamp(0.55 * baseline_score + 0.25 * context["fit_score"] + 0.20 * context["trigger_score"]), 6)


def make_human_touch_decision(
    context: dict[str, Any],
    *,
    human_touch_rank: int,
    baseline_score: float,
    prefer_coverage: bool = True,
) -> dict[str, Any]:
    selected_contacts = select_contacts(context, prefer_coverage=prefer_coverage)
    trigger = choose_primary_trigger(context)
    if not selected_contacts or trigger is None:
        return {
            "account_id": context["account"]["account_id"],
            "chosen_action": "wait",
            "action_score": 0.0,
        }

    return {
        "account_id": context["account"]["account_id"],
        "human_touch_rank": human_touch_rank,
        "chosen_action": "human_touch",
        "action_score": human_touch_action_score(context, baseline_score),
        "selected_contacts": selected_contacts,
        "primary_trigger_event_id": trigger["event_id"],
        "evidence_brief": build_evidence_brief(context, selected_contacts, trigger),
    }


def make_nonhuman_decision(
    account_id: str,
    action: str,
    action_score: float,
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "chosen_action": action,
        "action_score": round(clamp(action_score), 6),
    }


def generic_fallback_action(context: dict[str, Any]) -> tuple[str, float]:
    if context["episode_removed"] or not context["episode_eligible"]:
        return "wait", 0.95
    if not context["in_scope_icp"]:
        return "disqualify", 0.95
    if context["fit_score"] >= 0.72 and not context["cooldown_ok"]:
        return "recycle", 0.82
    if context["fit_score"] >= 0.72 and context["automation_feasible"]:
        return "automated_outbound", max(context["web_intent_score"], context["trigger_score"])
    if context["fit_score"] >= 0.52:
        return "nurture", max(context["fit_score"], context["web_intent_score"] * 0.8)
    return "wait", max(context["fit_score"] * 0.5, 0.1)


def finalize_window_submission(
    window_data: dict[str, Any],
    contexts: list[dict[str, Any]],
    selected_contexts: list[tuple[dict[str, Any], float]],
    fallback_actions: dict[str, tuple[str, float]],
    *,
    prefer_coverage: bool = True,
) -> dict[str, Any]:
    selected_ids = {
        context["account"]["account_id"]
        for context, _ in selected_contexts
    }
    decisions: list[dict[str, Any]] = []

    for rank, (context, score) in enumerate(selected_contexts, start=1):
        decisions.append(
            make_human_touch_decision(
                context,
                human_touch_rank=rank,
                baseline_score=score,
                prefer_coverage=prefer_coverage,
            )
        )

    for context in contexts:
        account_id = context["account"]["account_id"]
        if account_id in selected_ids:
            continue
        action, score = fallback_actions.get(account_id, ("wait", 0.0))
        decisions.append(make_nonhuman_decision(account_id, action, score))

    return {
        "window_id": window_data["window_id"],
        "decisions": decisions,
    }


def compute_rules_queue_score(context: dict[str, Any]) -> float:
    return clamp(
        0.45 * context["trigger_score"]
        + 0.35 * context["fit_score"]
        + 0.20 * context["web_intent_score"]
    )


def compute_propensity_score(context: dict[str, Any]) -> float:
    return clamp(
        0.32 * context["fit_score"]
        + 0.24 * context["web_intent_score"]
        + 0.20 * context["product_usage_score"]
        + 0.14 * context["trigger_score"]
        + 0.10 * company_scale_score(context["account"])
    )


def compute_last_touch_recency_score(context: dict[str, Any]) -> float:
    return clamp(
        0.60 * context["freshness_score"]
        + 0.25 * context["web_intent_score"]
        + 0.15 * (1.0 if context["cooldown_ok"] else 0.0)
    )


def compute_bau_priority_score(context: dict[str, Any]) -> float:
    timing_score = clamp(
        0.55 * context["trigger_score"]
        + 0.25 * context["web_intent_score"]
        + 0.20 * context["product_usage_score"]
    )
    return clamp(
        0.40 * context["fit_score"]
        + 0.35 * timing_score
        + 0.15 * context["contactability_score"]
        + 0.10 * company_scale_score(context["account"])
    )


def build_random_baseline(window_data: dict[str, Any], contexts: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + stable_text_seed(f"{window_data['window_id']}:random_within_icp"))
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    candidates: list[tuple[float, dict[str, Any]]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}

    for context in contexts:
        account_id = context["account"]["account_id"]
        fallback_actions[account_id] = generic_fallback_action(context)
        if context["in_scope_icp"] and context["human_touch_feasible"] and context["episode_eligible"]:
            candidates.append((rng.random(), context))

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (item[0], item[1]["account"]["account_id"]),
        reverse=True,
    )
    selected_contexts = [
        (context, score)
        for score, context in ranked_candidates[:budget]
    ]
    return finalize_window_submission(window_data, contexts, selected_contexts, fallback_actions)


def build_rules_trigger_queue(window_data: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    ranked_candidates: list[tuple[dict[str, Any], float]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}

    for context in contexts:
        score = compute_rules_queue_score(context)
        account_id = context["account"]["account_id"]
        if context["episode_removed"] or not context["episode_eligible"]:
            fallback_actions[account_id] = ("wait", 0.95)
        elif not context["in_scope_icp"]:
            fallback_actions[account_id] = ("disqualify", 0.95)
        elif context["fit_score"] >= 0.72 and not context["cooldown_ok"]:
            fallback_actions[account_id] = ("recycle", 0.84)
        elif context["fit_score"] >= 0.68 and context["automation_feasible"]:
            fallback_actions[account_id] = ("automated_outbound", score)
        elif context["fit_score"] >= 0.50:
            fallback_actions[account_id] = ("nurture", max(score, context["fit_score"]))
        else:
            fallback_actions[account_id] = ("wait", score)

        if context["episode_eligible"] and context["human_touch_feasible"] and context["cooldown_ok"] and score >= 0.45:
            ranked_candidates.append((context, score))

    selected_contexts = sorted(
        ranked_candidates,
        key=lambda item: (item[1], account_id_sort_key(item[0])),
        reverse=True,
    )[:budget]
    return finalize_window_submission(window_data, contexts, selected_contexts, fallback_actions)


def build_propensity_baseline(window_data: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    ranked_candidates: list[tuple[dict[str, Any], float]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}

    for context in contexts:
        score = compute_propensity_score(context)
        account_id = context["account"]["account_id"]
        if context["episode_removed"] or not context["episode_eligible"]:
            fallback_actions[account_id] = ("wait", 0.95)
        elif not context["in_scope_icp"]:
            fallback_actions[account_id] = ("disqualify", 0.95)
        elif context["fit_score"] >= 0.70 and context["automation_feasible"]:
            fallback_actions[account_id] = ("automated_outbound", score)
        elif context["fit_score"] >= 0.52:
            fallback_actions[account_id] = ("nurture", max(score, context["fit_score"]))
        else:
            fallback_actions[account_id] = ("wait", score)

        if context["episode_eligible"] and context["human_touch_feasible"] and context["cooldown_ok"] and score >= 0.42:
            ranked_candidates.append((context, score))

    selected_contexts = sorted(
        ranked_candidates,
        key=lambda item: (item[1], account_id_sort_key(item[0])),
        reverse=True,
    )[:budget]
    return finalize_window_submission(window_data, contexts, selected_contexts, fallback_actions)


def build_last_touch_recency(window_data: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    ranked_candidates: list[tuple[dict[str, Any], float]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}

    for context in contexts:
        score = compute_last_touch_recency_score(context)
        account_id = context["account"]["account_id"]
        if context["episode_removed"] or not context["episode_eligible"]:
            fallback_actions[account_id] = ("wait", 0.95)
        elif not context["in_scope_icp"]:
            fallback_actions[account_id] = ("disqualify", 0.95)
        elif context["fit_score"] >= 0.72 and not context["cooldown_ok"]:
            fallback_actions[account_id] = ("recycle", 0.86)
        elif context["automation_feasible"] and max(context["freshness_score"], context["web_intent_score"]) >= 0.35:
            fallback_actions[account_id] = ("automated_outbound", score)
        elif context["fit_score"] >= 0.52:
            fallback_actions[account_id] = ("nurture", score)
        else:
            fallback_actions[account_id] = ("wait", score)

        if context["episode_eligible"] and context["human_touch_feasible"] and context["cooldown_ok"] and score >= 0.40:
            ranked_candidates.append((context, score))

    selected_contexts = sorted(
        ranked_candidates,
        key=lambda item: (item[1], account_id_sort_key(item[0])),
        reverse=True,
    )[:budget]
    return finalize_window_submission(window_data, contexts, selected_contexts, fallback_actions)


def build_bau_baseline(window_data: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    ranked_candidates: list[tuple[dict[str, Any], float]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}

    for context in contexts:
        account_id = context["account"]["account_id"]
        timing_score = clamp(
            0.55 * context["trigger_score"]
            + 0.25 * context["web_intent_score"]
            + 0.20 * context["product_usage_score"]
        )
        high_fit = context["fit_score"] >= 0.72
        medium_fit = context["fit_score"] >= 0.52
        strong_timing = timing_score >= 0.58
        moderate_timing = timing_score >= 0.38
        enough_coverage = len(context["contacts"]) >= 2

        if context["episode_removed"] or not context["episode_eligible"]:
            fallback_actions[account_id] = ("wait", 0.95)
        elif not context["in_scope_icp"]:
            fallback_actions[account_id] = ("disqualify", 0.97)
        elif high_fit and not context["cooldown_ok"]:
            fallback_actions[account_id] = ("recycle", 0.90)
        elif high_fit and moderate_timing and context["automation_feasible"]:
            fallback_actions[account_id] = ("automated_outbound", compute_bau_priority_score(context))
        elif medium_fit and (not context["automation_feasible"] or context["contactability_score"] < 0.42):
            fallback_actions[account_id] = ("nurture", max(timing_score, context["fit_score"]))
        elif medium_fit:
            fallback_actions[account_id] = ("wait", max(timing_score, context["fit_score"] * 0.8))
        else:
            fallback_actions[account_id] = ("wait", context["fit_score"])

        if (
            high_fit
            and strong_timing
            and context["human_touch_feasible"]
            and context["episode_eligible"]
            and context["cooldown_ok"]
            and (enough_coverage or context["contactability_score"] >= 0.58)
        ):
            ranked_candidates.append((context, compute_bau_priority_score(context)))

    selected_contexts = sorted(
        ranked_candidates,
        key=lambda item: (item[1], account_id_sort_key(item[0])),
        reverse=True,
    )[:budget]
    return finalize_window_submission(
        window_data,
        contexts,
        selected_contexts,
        fallback_actions,
    )


def generate_window_submission(
    window_data: dict[str, Any],
    baseline_name: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    ensure_baseline_name(baseline_name)
    contexts = build_account_contexts(window_data)

    if baseline_name == "random_within_icp":
        return build_random_baseline(window_data, contexts, seed=seed)
    if baseline_name == "rules_trigger_queue":
        return build_rules_trigger_queue(window_data, contexts)
    if baseline_name == "propensity_only_lead_score":
        return build_propensity_baseline(window_data, contexts)
    if baseline_name == "last_touch_recency":
        return build_last_touch_recency(window_data, contexts)
    if baseline_name == "bau_enterprise_sdr_policy":
        return build_bau_baseline(window_data, contexts)

    raise ValueError(f"Unhandled baseline: {baseline_name}")


def generate_all_window_submissions(
    window_data: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    return {
        baseline_name: generate_window_submission(window_data, baseline_name, seed=seed)
        for baseline_name in BASELINE_NAMES
    }


def apply_episode_state_to_contexts(
    contexts: list[dict[str, Any]],
    state: dict[str, dict[str, Any]],
    window_index: int,
) -> list[dict[str, Any]]:
    adjusted_contexts: list[dict[str, Any]] = []
    for context in contexts:
        account_id = context["account"]["account_id"]
        account_state = state.setdefault(
            account_id,
            {
                "removed": False,
                "available_from_index": 0,
            },
        )
        adjusted = {
            **context,
            "episode_removed": account_state["removed"],
            "episode_eligible": (
                not account_state["removed"]
                and account_state["available_from_index"] <= window_index
            ),
        }
        if not adjusted["episode_eligible"]:
            adjusted["cooldown_ok"] = False
        adjusted_contexts.append(adjusted)
    return adjusted_contexts


def update_episode_state_from_submission(
    submission: dict[str, Any],
    state: dict[str, dict[str, Any]],
    window_index: int,
) -> None:
    for decision in submission["decisions"]:
        account_id = decision["account_id"]
        account_state = state.setdefault(
            account_id,
            {
                "removed": False,
                "available_from_index": 0,
            },
        )
        if account_state["removed"]:
            continue
        if decision["chosen_action"] == "disqualify":
            account_state["removed"] = True
        elif decision["chosen_action"] == "human_touch":
            account_state["available_from_index"] = max(
                account_state["available_from_index"],
                window_index + 2,
            )
        elif decision["chosen_action"] == "recycle":
            account_state["available_from_index"] = max(
                account_state["available_from_index"],
                window_index + 1,
            )


def generate_episode_submission(
    episode_data: dict[str, Any],
    baseline_name: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    ensure_baseline_name(baseline_name)
    state: dict[str, dict[str, Any]] = {}
    submissions: list[dict[str, Any]] = []
    for window_index, window in enumerate(episode_data["windows"]):
        contexts = apply_episode_state_to_contexts(
            build_account_contexts(window),
            state,
            window_index,
        )
        if baseline_name == "random_within_icp":
            submission = build_random_baseline(window, contexts, seed=seed)
        elif baseline_name == "rules_trigger_queue":
            submission = build_rules_trigger_queue(window, contexts)
        elif baseline_name == "propensity_only_lead_score":
            submission = build_propensity_baseline(window, contexts)
        elif baseline_name == "last_touch_recency":
            submission = build_last_touch_recency(window, contexts)
        elif baseline_name == "bau_enterprise_sdr_policy":
            submission = build_bau_baseline(window, contexts)
        else:
            raise ValueError(f"Unhandled baseline: {baseline_name}")
        submissions.append(submission)
        update_episode_state_from_submission(submission, state, window_index)

    return {
        "episode_id": episode_data["episode_id"],
        "submissions": submissions,
    }


def generate_all_episode_submissions(
    episode_data: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    return {
        baseline_name: generate_episode_submission(episode_data, baseline_name, seed=seed)
        for baseline_name in BASELINE_NAMES
    }


def best_nonhuman_action_from_labels(potential_outcomes: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return max(
        (
            (action_name, potential_outcomes[action_name])
            for action_name in ("automated_outbound", "nurture", "recycle", "disqualify", "wait")
        ),
        key=lambda item: (
            item[1]["pipeline_value"],
            item[1]["opp_prob"],
            item[1]["meeting_prob"],
            item[0],
        ),
    )


def generate_oracle_window_submission(
    window_data: dict[str, Any],
    labels_data: dict[str, Any],
    *,
    contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    contexts = contexts or build_account_contexts(window_data)
    labels_by_account = {
        row["account_id"]: row["potential_outcomes"]
        for row in labels_data["account_outcomes"]
    }
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    context_by_account = {
        context["account"]["account_id"]: context
        for context in contexts
    }

    candidates: list[tuple[float, float, float, str]] = []
    fallback_actions: dict[str, tuple[str, float]] = {}
    for account_id, context in context_by_account.items():
        outcomes = labels_by_account[account_id]
        best_nonhuman_name, best_nonhuman = best_nonhuman_action_from_labels(outcomes)
        fallback_actions[account_id] = (best_nonhuman_name, 1.0)
        if not context["human_touch_feasible"] or not context["episode_eligible"]:
            continue
        gain = outcomes["human_touch"]["pipeline_value"] - best_nonhuman["pipeline_value"]
        candidates.append(
            (
                gain,
                outcomes["human_touch"]["opp_prob"] - best_nonhuman["opp_prob"],
                outcomes["human_touch"]["meeting_prob"] - best_nonhuman["meeting_prob"],
                account_id,
            )
        )

    selected_accounts = {
        account_id
        for gain, _, _, account_id in sorted(candidates, reverse=True)[:budget]
        if gain > 0
    }
    selected_contexts = [
        (context_by_account[account_id], 1.0)
        for _, _, _, account_id in sorted(candidates, reverse=True)
        if account_id in selected_accounts
    ]
    return finalize_window_submission(
        window_data,
        contexts,
        selected_contexts,
        fallback_actions,
    )


def _clone_hidden_labels(labels_data: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(labels_data)


def _apply_outcome_modifiers_to_hidden_labels(
    labels_data: dict[str, Any],
    outcome_modifiers_by_account: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    adjusted = _clone_hidden_labels(labels_data)
    for account_row in adjusted["account_outcomes"]:
        modifiers = outcome_modifiers_by_account.get(account_row["account_id"])
        if not modifiers:
            continue
        for action_name, action_modifiers in modifiers.items():
            if action_name not in account_row["potential_outcomes"]:
                continue
            outcome = account_row["potential_outcomes"][action_name]
            outcome["meeting_prob"] *= action_modifiers["meeting_prob_multiplier"]
            outcome["opp_prob"] *= action_modifiers["opp_prob_multiplier"]
            outcome["pipeline_value"] *= action_modifiers["pipeline_value_multiplier"]
    return adjusted


def _extract_pending_outcome_modifiers(
    state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    modifiers: dict[str, dict[str, Any]] = {}
    for account_id, account_state in state.items():
        pending = account_state.get("pending_outcome_modifiers")
        if pending:
            modifiers[account_id] = deepcopy(pending)
            account_state["pending_outcome_modifiers"] = None
    return modifiers


def _apply_policy_transition_rows(
    window_index: int,
    submission: dict[str, Any],
    transition_map: dict[str, dict[str, Any]],
    state: dict[str, dict[str, Any]],
) -> None:
    for decision in submission["decisions"]:
        account_id = decision["account_id"]
        account_state = state.setdefault(
            account_id,
            {
                "removed": False,
                "available_from_index": 0,
                "pending_outcome_modifiers": None,
            },
        )
        transition = transition_map[account_id][decision["chosen_action"]]
        if transition["remove_from_episode"]:
            account_state["removed"] = True
            continue
        cooldown_weeks = transition["cooldown_weeks"]
        if cooldown_weeks > 0:
            account_state["available_from_index"] = max(
                account_state["available_from_index"],
                window_index + cooldown_weeks + 1,
            )
        account_state["pending_outcome_modifiers"] = transition.get("next_window_outcome_modifiers")


def generate_oracle_episode_submission(
    episode_data: dict[str, Any],
    labels_data: dict[str, Any],
) -> dict[str, Any]:
    labels_by_window_id = {
        entry["window_id"]: entry
        for entry in labels_data["windows"]
    }
    state: dict[str, dict[str, Any]] = {}
    submissions: list[dict[str, Any]] = []

    for window_index, window in enumerate(episode_data["windows"]):
        label_window = labels_by_window_id[window["window_id"]]
        current_modifiers = _extract_pending_outcome_modifiers(state)
        adjusted_labels = _apply_outcome_modifiers_to_hidden_labels(
            label_window["labels"],
            current_modifiers,
        )
        contexts = apply_episode_state_to_contexts(
            build_account_contexts(window),
            state,
            window_index,
        )
        submission = generate_oracle_window_submission(
            window,
            adjusted_labels,
            contexts=contexts,
        )
        submissions.append(submission)
        transition_map = {
            row["account_id"]: row["actions"]
            for row in label_window["policy_transitions"]
        }
        _apply_policy_transition_rows(
            window_index,
            submission,
            transition_map,
            state,
        )

    return {
        "episode_id": episode_data["episode_id"],
        "submissions": submissions,
    }
