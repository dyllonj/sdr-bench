"""Synthetic offline window generator."""

from __future__ import annotations

import copy
import datetime as dt
import random
from typing import Any

from sdr_bench.rationale_codes import WHY_NOW_EVENT_CODE_MAP
from sdr_bench.simulator.causal import ACCOUNT_TIERS
from sdr_bench.simulator.causal import CONTACT_ROLE_WEIGHTS
from sdr_bench.simulator.causal import EVENT_SOURCE_MAP
from sdr_bench.simulator.causal import EVENT_WEIGHT_BASES
from sdr_bench.simulator.causal import HOLDOUT_INDUSTRIES
from sdr_bench.simulator.causal import HOLDOUT_REGION
from sdr_bench.simulator.causal import INDUSTRIES
from sdr_bench.simulator.causal import REGIONS
from sdr_bench.simulator.causal import RELATIONSHIP_MOTIONS
from sdr_bench.simulator.causal import SEGMENTS
from sdr_bench.simulator.causal import best_nonhuman_action
from sdr_bench.simulator.causal import build_policy_transition_row
from sdr_bench.simulator.causal import choose_weighted
from sdr_bench.simulator.causal import clamp
from sdr_bench.simulator.causal import compute_action_outcomes
from sdr_bench.simulator.causal import deal_size_for_account
from sdr_bench.simulator.causal import revenue_band_for_employees

DEFAULT_START_DATE = dt.datetime(2026, 1, 5, tzinfo=dt.UTC)


def _slug(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_")


def _week_timestamp(week_index: int) -> dt.datetime:
    return DEFAULT_START_DATE + dt.timedelta(days=7 * week_index)


def _iso_timestamp(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sample_latents(rng: random.Random, *, segment: str) -> dict[str, float]:
    enterprise_bias = 0.08 if segment == "enterprise" else -0.03
    return {
        "structural_fit": clamp(rng.betavariate(4.2, 2.3) + enterprise_bias),
        "timing_heat": clamp(rng.betavariate(2.4, 3.7)),
        "persona_match": clamp(rng.betavariate(3.3, 2.7)),
        "channel_reach": clamp(rng.betavariate(3.5, 2.5)),
        "fatigue": clamp(rng.betavariate(2.0, 5.8)),
    }


def _contact_blueprints(
    rng: random.Random,
    account_id: str,
    latents: dict[str, float],
) -> list[dict[str, Any]]:
    roles: list[tuple[str, str, str, str, str]] = [
        ("technical_buyer", "IT", "VP of Infrastructure", "VP", "Infrastructure"),
        ("security_buyer", "Security", "Security Director", "Director", "Security"),
        ("economic_buyer", "Finance", "CIO", "C", "Executive"),
        ("champion", "Engineering", "Head of Platform", "Head", "Platform"),
        ("user", "Operations", "Operations Manager", "Manager", "Operations"),
    ]
    count = 3 + (1 if latents["persona_match"] >= 0.62 else 0) + (1 if latents["structural_fit"] >= 0.78 else 0)
    count = max(2, min(count, len(roles)))
    selected_roles = roles[:count]
    contacts: list[dict[str, Any]] = []
    for index, (buying_role, function, title, seniority, department) in enumerate(selected_roles, start=1):
        contact_id = f"{account_id}_ct_{index:02d}"
        reach = clamp(latents["channel_reach"] + rng.uniform(-0.18, 0.18))
        proxy = clamp(CONTACT_ROLE_WEIGHTS[buying_role] * 0.78 + rng.uniform(-0.08, 0.08))
        reply_bucket = "high" if reach >= 0.72 else "medium" if reach >= 0.42 else "low"
        if index == count and reply_bucket == "low" and reach < 0.25:
            reply_bucket = "unknown"
        contacts.append(
            {
                "contact_id": contact_id,
                "role": buying_role,
                "function": function,
                "title": title,
                "seniority": seniority,
                "department": department,
                "base_reach": reach,
                "is_decision_maker_proxy": proxy,
                "historical_reply_rate_bucket": reply_bucket,
            }
        )
    return contacts


def sample_account_blueprint(
    rng: random.Random,
    account_index: int,
    *,
    scenario: dict[str, Any] | None = None,
    allow_holdouts: bool = True,
) -> dict[str, Any]:
    scenario = scenario or {}
    enterprise_share = scenario.get("enterprise_share", 0.68)
    segment = "enterprise" if rng.random() < enterprise_share else "mid_market"
    industry_choices = [
        industry
        for industry in INDUSTRIES
        if allow_holdouts or industry not in HOLDOUT_INDUSTRIES
    ]
    region_choices = [
        region
        for region in REGIONS
        if allow_holdouts or region != HOLDOUT_REGION
    ]
    industry = rng.choice(industry_choices)
    region = rng.choice(region_choices)
    latents = _sample_latents(rng, segment=segment)
    employee_count = int(
        rng.triangular(120, 4200 if segment == "enterprise" else 1500, 1800 if segment == "enterprise" else 650)
    )
    account_tier = ACCOUNT_TIERS[0] if latents["structural_fit"] >= 0.82 else ACCOUNT_TIERS[1] if latents["structural_fit"] >= 0.62 else ACCOUNT_TIERS[2]
    account_id = f"acct_{account_index:06d}"
    blueprint = {
        "account_id": account_id,
        "segment": segment,
        "relationship_motion": (
            "product_led_pre_opportunity"
            if latents["timing_heat"] >= 0.58 and rng.random() < 0.45
            else RELATIONSHIP_MOTIONS[0]
        ),
        "account_tier": account_tier,
        "industry": industry,
        "hq_region": region,
        "sales_geo": f"{region}_{segment}",
        "employee_count": max(employee_count, 50),
        "revenue_band": revenue_band_for_employees(max(employee_count, 50)),
        "technographics": {
            "cloud_provider": rng.choice(["aws", "gcp", "azure"]),
            "identity_provider": rng.choice(["okta", "entra", "ping"]),
            "data_platform": rng.choice(["snowflake", "databricks", "bigquery", "redshift"]),
            "security_tools": [rng.choice(["crowdstrike", "wiz", "snyk", "sentinelone"])],
            "competitors_present": [rng.choice(["vendor_x", "vendor_y"])] if rng.random() < 0.34 else [],
        },
        "latents": latents,
    }
    blueprint["deal_size"] = deal_size_for_account(blueprint, rng)
    blueprint["contact_blueprints"] = _contact_blueprints(rng, account_id, latents)
    return blueprint


def evolve_account_blueprint(
    blueprint: dict[str, Any],
    rng: random.Random,
    *,
    scenario: dict[str, Any] | None = None,
) -> None:
    scenario = scenario or {}
    trigger_shift = scenario.get("trigger_shift", 0.0)
    fatigue_shift = scenario.get("fatigue_shift", 0.0)
    latents = blueprint["latents"]
    latents["structural_fit"] = clamp(0.92 * latents["structural_fit"] + 0.08 * rng.betavariate(3.8, 2.8))
    latents["timing_heat"] = clamp(
        0.62 * latents["timing_heat"] + 0.38 * rng.betavariate(2.5, 3.4) + trigger_shift
    )
    latents["persona_match"] = clamp(0.86 * latents["persona_match"] + 0.14 * rng.betavariate(3.1, 2.9))
    latents["channel_reach"] = clamp(0.88 * latents["channel_reach"] + 0.12 * rng.betavariate(3.4, 2.7))
    latents["fatigue"] = clamp(0.72 * latents["fatigue"] + 0.28 * rng.betavariate(2.1, 5.2) + fatigue_shift)


def _event_weights(blueprint: dict[str, Any], scenario: dict[str, Any]) -> dict[str, float]:
    latents = blueprint["latents"]
    weights = dict(EVENT_WEIGHT_BASES)
    if blueprint["relationship_motion"] == "product_led_pre_opportunity":
        weights["usage_change"] += 0.18
    if blueprint["technographics"]["competitors_present"]:
        weights["competitive_displacement"] += 0.10
    if blueprint["industry"] in {"financial_services", "healthcare"}:
        weights["compliance_deadline"] += 0.10
    if latents["timing_heat"] >= 0.66:
        weights["leadership_change"] += 0.04
        weights["funding"] += 0.03
    multiplier = scenario.get("trigger_multiplier", 1.0)
    return {event_type: weight * multiplier for event_type, weight in weights.items()}


def _build_triggers(
    blueprint: dict[str, Any],
    *,
    week_index: int,
    window_id: str,
    rng: random.Random,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    latents = blueprint["latents"]
    expected = 0.25 + 2.4 * latents["timing_heat"] + scenario.get("trigger_shift", 0.0) * 2.0
    trigger_count = min(3, max(0, int(round(expected + rng.uniform(-0.6, 0.6)))))
    if latents["timing_heat"] >= 0.74 and trigger_count == 0:
        trigger_count = 1
    weights = _event_weights(blueprint, scenario)
    week_ts = _week_timestamp(week_index)
    triggers: list[dict[str, Any]] = []
    used_types: set[str] = set()
    for index in range(trigger_count):
        event_type = choose_weighted(rng, weights)
        if event_type in used_types:
            alternatives = {key: value for key, value in weights.items() if key not in used_types}
            if alternatives:
                event_type = choose_weighted(rng, alternatives)
        used_types.add(event_type)
        recency = max(1, min(28, int(round((1.0 - latents["timing_heat"]) * 20 + rng.uniform(1.0, 6.0)))))
        event_ts = week_ts - dt.timedelta(days=recency)
        event_id = f"{window_id}_{blueprint['account_id']}_evt_{index + 1:02d}"
        triggers.append(
            {
                "event_id": event_id,
                "account_id": blueprint["account_id"],
                "event_ts": _iso_timestamp(event_ts),
                "event_type": event_type,
                "source_type": EVENT_SOURCE_MAP[event_type],
                "confidence": round(clamp(0.48 + 0.40 * latents["timing_heat"] + rng.uniform(-0.08, 0.12)), 6),
                "recency_days": recency,
                "evidence_refs": [f"{window_id}_{blueprint['account_id']}_doc_trigger_{index + 1:02d}"],
            }
        )
    return triggers


def _build_contacts(
    blueprint: dict[str, Any],
    *,
    rng: random.Random,
    timing_heat: float,
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for contact in blueprint["contact_blueprints"]:
        reach = clamp(contact["base_reach"] + rng.uniform(-0.08, 0.08))
        contacts.append(
            {
                "contact_id": contact["contact_id"],
                "account_id": blueprint["account_id"],
                "name_redacted": True,
                "title": contact["title"],
                "function": contact["function"],
                "seniority": contact["seniority"],
                "department": contact["department"],
                "likely_buying_role": contact["role"],
                "is_decision_maker_proxy": round(contact["is_decision_maker_proxy"], 6),
                "historical_reply_rate_bucket": contact["historical_reply_rate_bucket"],
                "channel_reachability": {
                    "email_valid": reach >= 0.28,
                    "phone_valid": reach >= 0.62,
                    "linkedin_present": reach >= 0.18,
                },
                "recent_activity": {
                    "job_change_90d": timing_heat >= 0.72 and rng.random() < 0.28,
                    "content_engagement_30d": max(0, int(round(6 * reach + 5 * timing_heat + rng.uniform(-2.0, 2.0)))),
                },
            }
        )
    return contacts


def _infer_persona_code(contacts: list[dict[str, Any]]) -> str:
    roles = {contact["likely_buying_role"] for contact in contacts}
    functions = {contact["function"] for contact in contacts}
    if "technical_buyer" in roles and ("champion" in roles or "Security" in functions):
        return "technical_buyer_plus_security_champion"
    if "economic_buyer" in roles:
        return "economic_buyer"
    if "champion" in roles:
        return "champion"
    return "technical_buyer"


def _infer_channel_code(
    contacts: list[dict[str, Any]],
    web_engagement: dict[str, Any],
) -> str:
    has_email = any(contact["channel_reachability"]["email_valid"] for contact in contacts)
    has_linkedin = any(contact["channel_reachability"]["linkedin_present"] for contact in contacts)
    high_intent = (
        web_engagement["pricing_visits_30d"] >= 3
        or web_engagement["product_pages_30d"] >= 7
        or web_engagement["high_intent_content_downloads_30d"] >= 1
    )
    if has_email and high_intent:
        return "email_valid_plus_recent_web_intent"
    if has_email:
        return "email_valid"
    if has_linkedin:
        return "linkedin_present"
    return "manual_research_required"


def _account_codes(
    blueprint: dict[str, Any],
    web_engagement: dict[str, Any],
    product_signals: dict[str, Any],
) -> list[str]:
    codes: list[str] = []
    if blueprint["segment"] == "enterprise" or blueprint["account_tier"] in {"tier_1_named", "tier_2_named"}:
        codes.append("enterprise_icp_fit")
    else:
        codes.append("mid_market_expansion_fit")
    if web_engagement["pricing_visits_30d"] >= 3 or web_engagement["product_pages_30d"] >= 7:
        codes.append("intent_surge")
    if product_signals["seat_growth_30d"] >= 0.10 or product_signals["active_users_growth_30d"] >= 0.08:
        codes.append("product_usage_growth")
    if blueprint["technographics"]["competitors_present"] and len(codes) < 3:
        codes.append("competitor_present")
    return codes[:3]


def _build_observables(
    blueprint: dict[str, Any],
    *,
    week_index: int,
    triggers: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    latents = blueprint["latents"]
    week_ts = _week_timestamp(week_index)
    web_engagement = {
        "pricing_visits_30d": max(0, int(round(8 * latents["timing_heat"] + 3 * latents["channel_reach"] + rng.uniform(-2.0, 2.0)))),
        "product_pages_30d": max(0, int(round(12 * latents["timing_heat"] + 5 * latents["structural_fit"] + rng.uniform(-3.0, 3.0)))),
        "high_intent_content_downloads_30d": max(0, int(round(2.4 * latents["timing_heat"] + rng.uniform(-0.8, 0.8)))),
        "trial_signups_30d": max(0, int(round(1.8 * latents["timing_heat"] * latents["channel_reach"] + rng.uniform(-0.7, 0.7)))),
    }
    product_signals = {
        "seat_growth_30d": round(max(0.0, 0.18 * latents["timing_heat"] + 0.12 * latents["structural_fit"] + rng.uniform(-0.03, 0.05)), 6),
        "active_users_growth_30d": round(max(0.0, 0.16 * latents["timing_heat"] + 0.08 * latents["structural_fit"] + rng.uniform(-0.02, 0.04)), 6),
        "feature_adoption_flags": [
            flag
            for flag in ("security_module", "governance_module", "workflow_automation")
            if rng.random() < 0.25 + 0.30 * latents["structural_fit"]
        ],
    }
    if not product_signals["feature_adoption_flags"]:
        product_signals["feature_adoption_flags"] = [rng.choice(["security_module", "governance_module"])]

    crm_state = {
        "current_stage": "none",
        "owner_role": "SDR_pool",
        "past_meetings_365d": max(0, int(round(4 * latents["structural_fit"] + rng.uniform(-1.0, 1.0)))),
        "past_opps_365d": max(0, int(round(2 * latents["structural_fit"] + rng.uniform(-0.5, 0.5)))),
        "days_since_last_human_touch": max(0, int(round(7 + 70 * latents["fatigue"] + rng.uniform(-6.0, 6.0)))),
        "days_since_last_outbound_sequence": max(0, int(round(5 + 35 * latents["fatigue"] + rng.uniform(-4.0, 4.0)))),
        "has_open_opportunity": False,
        "open_support_escalation": rng.random() < 0.08,
    }
    account_snapshot = {
        "account_id": blueprint["account_id"],
        "snapshot_ts": _iso_timestamp(week_ts),
        "segment": blueprint["segment"],
        "relationship_motion": blueprint["relationship_motion"],
        "account_tier": blueprint["account_tier"],
        "industry": blueprint["industry"],
        "employee_count": blueprint["employee_count"],
        "revenue_band": blueprint["revenue_band"],
        "hq_region": blueprint["hq_region"],
        "sales_geo": blueprint["sales_geo"],
        "technographics": copy.deepcopy(blueprint["technographics"]),
        "crm_state": crm_state,
        "web_engagement": web_engagement,
        "product_or_usage_signals": product_signals,
        "trigger_events": [trigger["event_id"] for trigger in triggers],
        "available_contacts": [contact["contact_id"] for contact in contacts],
        "label_window_id": "",
    }
    return account_snapshot, web_engagement, product_signals


def _build_evidence(
    blueprint: dict[str, Any],
    *,
    window_id: str,
    week_index: int,
    triggers: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    web_engagement: dict[str, Any],
    product_signals: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    week_ts = _week_timestamp(week_index)
    persona_code = _infer_persona_code(contacts)
    channel_code = _infer_channel_code(contacts, web_engagement)
    account_codes = _account_codes(blueprint, web_engagement, product_signals)
    evidence: list[dict[str, Any]] = []

    evidence.append(
        {
            "doc_id": f"{window_id}_{blueprint['account_id']}_doc_fit",
            "account_id": blueprint["account_id"],
            "source_type": "crm",
            "published_ts": _iso_timestamp(week_ts),
            "excerpt": "CRM and territory data indicate the account is in the active named-account motion.",
            "allowed_for_grounding": True,
            "grounding_support": {
                "why_account_codes": [account_codes[0]],
            },
        }
    )

    if "intent_surge" in account_codes:
        evidence.append(
            {
                "doc_id": f"{window_id}_{blueprint['account_id']}_doc_intent",
                "account_id": blueprint["account_id"],
                "source_type": "web",
                "published_ts": _iso_timestamp(week_ts - dt.timedelta(days=1)),
                "excerpt": "Web activity shows pricing and product-page interest spiking recently.",
                "allowed_for_grounding": True,
                "grounding_support": {
                    "why_account_codes": ["intent_surge"],
                    "why_channel_codes": ["email_valid_plus_recent_web_intent"],
                },
            }
        )

    if "product_usage_growth" in account_codes:
        evidence.append(
            {
                "doc_id": f"{window_id}_{blueprint['account_id']}_doc_usage",
                "account_id": blueprint["account_id"],
                "source_type": "product",
                "published_ts": _iso_timestamp(week_ts - dt.timedelta(days=2)),
                "excerpt": "Usage telemetry shows recent seat and active-user growth.",
                "allowed_for_grounding": True,
                "grounding_support": {
                    "why_account_codes": ["product_usage_growth"],
                    "why_channel_codes": ["email_valid_plus_recent_web_intent"],
                },
            }
        )

    evidence.append(
        {
            "doc_id": f"{window_id}_{blueprint['account_id']}_doc_persona",
            "account_id": blueprint["account_id"],
            "source_type": "contact",
            "published_ts": _iso_timestamp(week_ts),
            "excerpt": "The surfaced contacts cover the likely buying center for this account.",
            "allowed_for_grounding": True,
            "grounding_support": {
                "why_persona_codes": [persona_code],
                "why_channel_codes": [channel_code],
            },
        }
    )

    for index, trigger in enumerate(triggers, start=1):
        evidence.append(
            {
                "doc_id": f"{window_id}_{blueprint['account_id']}_doc_trigger_{index:02d}",
                "account_id": blueprint["account_id"],
                "source_type": trigger["source_type"],
                "published_ts": trigger["event_ts"],
                "excerpt": f"Observed trigger: {_slug(trigger['event_type'])} in the current evaluation week.",
                "allowed_for_grounding": True,
                "grounding_support": {
                    "why_now_codes": [WHY_NOW_EVENT_CODE_MAP.get(trigger["event_type"], "timing_signal_recent")],
                    "related_event_ids": [trigger["event_id"]],
                },
            }
        )

    evidence.append(
        {
            "doc_id": f"{window_id}_{blueprint['account_id']}_doc_noise",
            "account_id": blueprint["account_id"],
            "source_type": "news",
            "published_ts": _iso_timestamp(week_ts - dt.timedelta(days=12)),
            "excerpt": "General market news not directly tied to the current outreach decision.",
            "allowed_for_grounding": rng.random() < 0.15,
        }
    )
    return evidence


def _build_contact_outcomes(
    blueprint: dict[str, Any],
    contacts: list[dict[str, Any]],
) -> dict[str, Any]:
    latents = blueprint["latents"]
    default_contact = min(
        contacts,
        key=lambda contact: (
            CONTACT_ROLE_WEIGHTS.get(contact["likely_buying_role"], 0.5),
            contact["contact_id"],
        ),
    )
    rows: list[dict[str, Any]] = []
    for contact in contacts:
        role_weight = CONTACT_ROLE_WEIGHTS.get(contact["likely_buying_role"], 0.5)
        reach_bonus = 0.08 if contact["channel_reachability"]["email_valid"] else 0.02
        meeting_gain = clamp(0.015 + 0.14 * role_weight * latents["persona_match"] + reach_bonus - 0.03 * latents["fatigue"])
        opp_gain = clamp(meeting_gain * (0.32 + 0.30 * latents["structural_fit"]))
        pipeline_gain = round(opp_gain * blueprint["deal_size"] * 0.8, 2)
        rows.append(
            {
                "contact_id": contact["contact_id"],
                "meeting_gain": round(meeting_gain, 6),
                "opp_gain": round(opp_gain, 6),
                "pipeline_gain": pipeline_gain,
            }
        )
    return {
        "account_id": blueprint["account_id"],
        "default_contact_id": default_contact["contact_id"],
        "contacts": rows,
    }


def _build_trigger_outcomes(triggers: list[dict[str, Any]]) -> dict[str, Any]:
    event_weight = {
        "leadership_change": 1.00,
        "compliance_deadline": 0.95,
        "usage_change": 0.90,
        "competitive_displacement": 0.86,
        "product_launch": 0.72,
        "hiring": 0.70,
        "expansion": 0.66,
        "funding": 0.62,
    }
    rows = []
    for trigger in triggers:
        recency_bonus = max(0.1, 1.0 - trigger["recency_days"] / 30.0)
        relevance = clamp(
            0.45 * event_weight.get(trigger["event_type"], 0.55)
            + 0.35 * trigger["confidence"]
            + 0.20 * recency_bonus
        )
        rows.append(
            {
                "event_id": trigger["event_id"],
                "relevance_gain": round(relevance, 6),
            }
        )
    return {
        "account_id": triggers[0]["account_id"] if triggers else "",
        "triggers": rows or [{"event_id": "", "relevance_gain": 0.0}],
    }


def generate_window_bundle_from_blueprints(
    blueprints: list[dict[str, Any]],
    *,
    window_id: str,
    week_index: int = 0,
    scenario: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    scenario = scenario or {}
    rng = random.Random(seed)
    window_accounts: list[dict[str, Any]] = []
    window_contacts: list[dict[str, Any]] = []
    window_triggers: list[dict[str, Any]] = []
    window_evidence: list[dict[str, Any]] = []
    account_outcomes: list[dict[str, Any]] = []
    contact_outcomes: list[dict[str, Any]] = []
    trigger_outcomes: list[dict[str, Any]] = []
    policy_transitions: list[dict[str, Any]] = []

    for blueprint in sorted(blueprints, key=lambda item: item["account_id"]):
        local_rng = random.Random(rng.randint(0, 10_000_000))
        triggers = _build_triggers(
            blueprint,
            week_index=week_index,
            window_id=window_id,
            rng=local_rng,
            scenario=scenario,
        )
        contacts = _build_contacts(blueprint, rng=local_rng, timing_heat=blueprint["latents"]["timing_heat"])
        account_snapshot, web_engagement, product_signals = _build_observables(
            blueprint,
            week_index=week_index,
            triggers=triggers,
            contacts=contacts,
            rng=local_rng,
        )
        account_snapshot["label_window_id"] = window_id
        evidence = _build_evidence(
            blueprint,
            window_id=window_id,
            week_index=week_index,
            triggers=triggers,
            contacts=contacts,
            web_engagement=web_engagement,
            product_signals=product_signals,
            rng=local_rng,
        )

        outcomes = compute_action_outcomes(blueprint["latents"], deal_size=blueprint["deal_size"])
        account_outcomes.append(
            {
                "account_id": blueprint["account_id"],
                "potential_outcomes": outcomes,
            }
        )
        contact_outcomes.append(_build_contact_outcomes(blueprint, contacts))
        trigger_outcomes.append(_build_trigger_outcomes(triggers))
        policy_transitions.append(build_policy_transition_row(blueprint["account_id"], blueprint["latents"], outcomes))

        window_accounts.append(account_snapshot)
        window_contacts.extend(contacts)
        window_triggers.extend(triggers)
        window_evidence.extend(evidence)

    human_budget = scenario.get("human_budget")
    if human_budget is None:
        human_budget = max(1, int(round(len(window_accounts) * scenario.get("human_budget_ratio", 0.08))))

    window = {
        "window_id": window_id,
        "accounts": window_accounts,
        "contacts": window_contacts,
        "triggers": window_triggers,
        "evidence": window_evidence,
        "capacity_budget": {
            "window_id": window_id,
            "human_sdr_actions": min(human_budget, len(window_accounts)),
            "max_contacts_per_account": 3,
            "channel_costs": {
                "human_touch": 1.0,
                "automated_outbound": 0.15,
                "nurture": 0.05,
                "recycle": 0.02,
                "wait": 0.0,
                "disqualify": 0.0,
            },
        },
    }
    labels = {
        "window_id": window_id,
        "account_outcomes": account_outcomes,
        "contact_outcomes": contact_outcomes,
        "trigger_outcomes": [
            row
            for row in trigger_outcomes
            if row["triggers"] and row["triggers"][0]["event_id"]
        ],
    }
    return {
        "window": window,
        "labels": labels,
        "policy_transitions": policy_transitions,
    }


def generate_window(seed: int, n_accounts: int, window_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    blueprints = [
        sample_account_blueprint(rng, account_index=index + 1)
        for index in range(n_accounts)
    ]
    bundle = generate_window_bundle_from_blueprints(
        blueprints,
        window_id=window_id,
        week_index=0,
        seed=seed,
    )
    return bundle["window"], bundle["labels"]
