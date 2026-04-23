"""Shared causal surface for synthetic SDR Bench data."""

from __future__ import annotations

import math
import random
from typing import Any

ACTIONS = (
    "human_touch",
    "automated_outbound",
    "nurture",
    "recycle",
    "disqualify",
    "wait",
)

NON_HUMAN_ACTIONS = tuple(action for action in ACTIONS if action != "human_touch")

INDUSTRIES = (
    "software",
    "financial_services",
    "healthcare",
    "manufacturing",
    "retail",
    "public_sector",
)
HOLDOUT_INDUSTRIES = ("healthcare", "public_sector")
REGIONS = ("NA", "EMEA", "APAC")
HOLDOUT_REGION = "APAC"
SEGMENTS = ("enterprise", "mid_market")
RELATIONSHIP_MOTIONS = ("net_new", "product_led_pre_opportunity")
ACCOUNT_TIERS = ("tier_1_named", "tier_2_named", "tier_3_named")
REVENUE_BANDS = ("25m_100m", "100m_250m", "250m_1b", "1b_5b")

EVENT_SOURCE_MAP = {
    "leadership_change": "news",
    "compliance_deadline": "news",
    "usage_change": "product",
    "competitive_displacement": "news",
    "product_launch": "news",
    "hiring": "jobs",
    "expansion": "news",
    "funding": "news",
}

EVENT_WEIGHT_BASES = {
    "leadership_change": 0.14,
    "compliance_deadline": 0.10,
    "usage_change": 0.22,
    "competitive_displacement": 0.08,
    "product_launch": 0.08,
    "hiring": 0.16,
    "expansion": 0.10,
    "funding": 0.12,
}

CONTACT_ROLE_WEIGHTS = {
    "economic_buyer": 1.00,
    "technical_buyer": 0.96,
    "champion": 0.84,
    "security_buyer": 0.88,
    "user": 0.50,
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def revenue_band_for_employees(employee_count: int) -> str:
    if employee_count >= 3500:
        return "1b_5b"
    if employee_count >= 1400:
        return "250m_1b"
    if employee_count >= 500:
        return "100m_250m"
    return "25m_100m"


def deal_size_for_account(account_state: dict[str, Any], rng: random.Random) -> float:
    tier_multiplier = {
        "tier_1_named": 1.30,
        "tier_2_named": 1.05,
        "tier_3_named": 0.90,
    }[account_state["account_tier"]]
    segment_multiplier = 1.25 if account_state["segment"] == "enterprise" else 0.72
    base = 42000.0 + account_state["employee_count"] * 18.0
    variance = rng.uniform(0.88, 1.14)
    return round(base * tier_multiplier * segment_multiplier * variance, 2)


def choose_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(max(weight, 0.0) for weight in weights.values())
    if total <= 0:
        return sorted(weights)[0]
    threshold = rng.uniform(0.0, total)
    running = 0.0
    for key, value in sorted(weights.items()):
        running += max(value, 0.0)
        if running >= threshold:
            return key
    return sorted(weights)[-1]


def best_nonhuman_action(potential_outcomes: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    best_action = max(
        NON_HUMAN_ACTIONS,
        key=lambda action: (
            potential_outcomes[action]["pipeline_value"],
            potential_outcomes[action]["opp_prob"],
            potential_outcomes[action]["meeting_prob"],
            action,
        ),
    )
    return best_action, potential_outcomes[best_action]


def compute_action_outcomes(
    latents: dict[str, float],
    *,
    deal_size: float,
) -> dict[str, dict[str, float]]:
    fit = latents["structural_fit"]
    timing = latents["timing_heat"]
    persona = latents["persona_match"]
    channel = latents["channel_reach"]
    fatigue = latents["fatigue"]

    human_lift = timing * persona * (1.0 - fatigue)
    base_logit = -2.9 + 1.15 * fit + 0.95 * timing + 0.55 * channel - 1.10 * fatigue
    action_logits = {
        "human_touch": base_logit + 0.72 * persona + 2.55 * human_lift,
        "automated_outbound": base_logit - 0.18 + 0.78 * channel - 0.20 * persona,
        "nurture": base_logit - 0.55 + 0.46 * fit + 0.12 * (1.0 - fatigue),
        "recycle": -3.8 + 0.35 * fit - 1.25 * timing + 0.75 * fatigue,
        "disqualify": -9.0,
        "wait": -3.25 + 0.55 * fit + 0.32 * timing - 0.25 * fatigue,
    }

    pipeline_multipliers = {
        "human_touch": 1.18 + 0.55 * human_lift,
        "automated_outbound": 0.76 + 0.18 * channel,
        "nurture": 0.54 + 0.12 * fit,
        "recycle": 0.16,
        "disqualify": 0.0,
        "wait": 0.38 + 0.10 * timing,
    }

    outcomes: dict[str, dict[str, float]] = {}
    for action in ACTIONS:
        meeting_prob = clamp(sigmoid(action_logits[action]))
        opp_logit = -1.55 + 1.10 * fit + 0.88 * timing + 0.72 * human_lift
        if action != "human_touch":
            opp_logit -= 0.25
        opp_prob = clamp(meeting_prob * sigmoid(opp_logit))
        if action == "recycle":
            opp_prob *= 0.55
        pipeline_value = max(0.0, opp_prob * deal_size * pipeline_multipliers[action])
        outcomes[action] = {
            "meeting_prob": round(meeting_prob, 6),
            "opp_prob": round(opp_prob, 6),
            "pipeline_value": round(pipeline_value, 2),
        }

    outcomes["disqualify"] = {
        "meeting_prob": 0.0,
        "opp_prob": 0.0,
        "pipeline_value": 0.0,
    }
    return outcomes


def build_policy_transition_row(
    account_id: str,
    latents: dict[str, float],
    outcomes: dict[str, dict[str, float]],
) -> dict[str, Any]:
    timing = latents["timing_heat"]
    fatigue = latents["fatigue"]
    fit = latents["structural_fit"]

    human_convert = outcomes["human_touch"]["opp_prob"] >= 0.18 or (
        fit >= 0.72 and timing >= 0.72
    )
    automation_warm = 1.02 + 0.16 * latents["channel_reach"] * timing
    nurture_warm = 1.04 + 0.10 * (1.0 - fatigue)
    wait_decay = 0.94 if timing >= 0.70 else 1.02

    def modifier(multiplier: float) -> dict[str, float]:
        return {
            "meeting_prob_multiplier": round(multiplier, 6),
            "opp_prob_multiplier": round(multiplier, 6),
            "pipeline_value_multiplier": round(multiplier, 6),
        }

    return {
        "account_id": account_id,
        "actions": {
            "human_touch": {
                "cooldown_weeks": 1,
                "remove_from_episode": human_convert,
                "next_window_outcome_modifiers": {
                    "human_touch": modifier(0.72 + 0.08 * (1.0 - fatigue)),
                },
            },
            "automated_outbound": {
                "cooldown_weeks": 0,
                "remove_from_episode": False,
                "next_window_outcome_modifiers": {
                    "human_touch": modifier(automation_warm),
                },
            },
            "nurture": {
                "cooldown_weeks": 0,
                "remove_from_episode": False,
                "next_window_outcome_modifiers": {
                    "human_touch": modifier(nurture_warm),
                },
            },
            "recycle": {
                "cooldown_weeks": 1,
                "remove_from_episode": False,
                "next_window_outcome_modifiers": {
                    "human_touch": modifier(1.08 + 0.04 * fit),
                },
            },
            "disqualify": {
                "cooldown_weeks": 0,
                "remove_from_episode": True,
            },
            "wait": {
                "cooldown_weeks": 0,
                "remove_from_episode": False,
                "next_window_outcome_modifiers": {
                    "human_touch": modifier(wait_decay),
                },
            },
        },
    }
