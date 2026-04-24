"""Stub evaluator for the SDR benchmark.

This CLI validates benchmark inputs, applies structural policy rules,
and emits a report skeleton for future hidden-label scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import RefResolver
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from sdr_bench.baselines import BASELINE_NAMES
from sdr_bench.baselines import generate_all_episode_submissions
from sdr_bench.baselines import generate_all_window_submissions
from sdr_bench.baselines import generate_episode_submission
from sdr_bench.baselines import generate_window_submission
from sdr_bench.rationale_codes import GROUNDING_CODE_FIELDS
from sdr_bench.rationale_codes import supports_grounding_claim

ROOT_DIR = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT_DIR / "schemas"

SCHEMA_FILES = {
    "evaluation_window": "evaluation_window.schema.json",
    "account_snapshot": "account_snapshot.schema.json",
    "contact_snapshot": "contact_snapshot.schema.json",
    "trigger_event": "trigger_event.schema.json",
    "document_evidence": "document_evidence.schema.json",
    "capacity_budget": "capacity_budget.schema.json",
    "model_output": "model_output.schema.json",
    "hidden_labels": "hidden_labels.schema.json",
    "policy_episode": "policy_episode.schema.json",
    "policy_submission": "policy_submission.schema.json",
    "policy_episode_labels": "policy_episode_labels.schema.json",
    "robustness_suite": "robustness_suite.schema.json",
    "robustness_submission": "robustness_submission.schema.json",
    "rationale_codes": "rationale_codes.schema.json",
}

ACTION_SPACE = (
    "human_touch",
    "automated_outbound",
    "nurture",
    "recycle",
    "disqualify",
    "wait",
)

NON_HUMAN_ACTIONS = tuple(action for action in ACTION_SPACE if action != "human_touch")
NORMALIZED_SCORE_METRICS = (
    "EnterpriseAllocationScore",
    "OfflineScore",
    "PolicyScore",
    "FitScore",
    "TimingScore",
    "ContactScore",
    "GroundingScore",
    "LiftScore",
)
NORMALIZATION_CLIP_RANGE = (-25.0, 125.0)
SLICE_DIMENSIONS = (
    "segment",
    "industry",
    "region",
    "intent_presence",
    "relationship_motion",
    "data_density",
)
TOUCHED_ONLY_SLICE_METRICS = {
    "TimingScore",
    "trigger_accuracy_at_1",
    "trigger_weighted_gain",
    "oracle_trigger_weighted_gain",
    "trigger_gain_ratio",
    "timing_score_ratio",
    "ContactScore",
    "contact_precision_at_selected",
    "contact_mrr",
    "contact_meeting_lift_delta",
    "oracle_contact_meeting_lift_delta",
    "contact_lift_ratio",
    "contact_score_ratio",
    "GroundingScore",
    "grounded_claim_ratio",
    "citation_relevance_ratio",
    "trigger_citation_alignment",
    "evidence_schema_accuracy",
    "unsupported_claim_rate",
    "grounding_score_ratio",
}
ROBUSTNESS_CASE_TYPES = (
    "heldout_slice",
    "distribution_shift",
)
ROBUSTNESS_SUMMARY_METRICS = NORMALIZED_SCORE_METRICS
ACTION_TERMINOLOGY = {
    "human_touch": {
        "sales_label": "Personalized SDR Outreach",
        "sales_term": "personalized_outreach",
        "description": "A rep spends real time on tailored email, calls, or LinkedIn outreach for this account this week.",
    },
    "automated_outbound": {
        "sales_label": "Sequence Enrollment",
        "sales_term": "sequence_enrollment",
        "description": "The account goes into a light-touch outbound sequence with approved messaging and targeting.",
    },
    "nurture": {
        "sales_label": "Marketing Nurture",
        "sales_term": "marketing_nurture",
        "description": "The account stays warm through lower-touch nurture rather than direct rep effort this week.",
    },
    "recycle": {
        "sales_label": "Recycle / Snooze",
        "sales_term": "recycle_snooze",
        "description": "The account comes out of the active SDR queue for a cooldown period and can be revisited later.",
    },
    "disqualify": {
        "sales_label": "Disqualify",
        "sales_term": "disqualify",
        "description": "The account is treated as out of ICP or operationally unreachable for the benchmark horizon.",
    },
    "wait": {
        "sales_label": "Monitor / No Action",
        "sales_term": "monitor_no_action",
        "description": "The account stays visible in the book, but no action is taken this week.",
    },
}
METRIC_TERMINOLOGY = {
    "EnterpriseAllocationScore": {
        "sales_label": "Named-Account Book Score",
        "description": "The public top-line score for how well the model manages the SDR book.",
    },
    "OfflineScore": {
        "sales_label": "Weekly Queue Prioritization Score",
        "description": "How well the model prioritizes and routes a single weekly book.",
    },
    "PolicyScore": {
        "sales_label": "Multi-Week Book Management Score",
        "description": "How well the model manages the book over time as actions change future outcomes.",
    },
    "FitScore": {
        "sales_label": "ICP / Account Selection Score",
        "description": "How well the model identifies the right accounts to work.",
    },
    "TimingScore": {
        "sales_label": "Why-Now Score",
        "description": "How well the model picks the strongest current buying signal or trigger.",
    },
    "ContactScore": {
        "sales_label": "Buying-Center Coverage Score",
        "description": "How well the model picks the best personas and contact mix for outreach.",
    },
    "GroundingScore": {
        "sales_label": "Account Research Grounding Score",
        "description": "How well the structured rationale is supported by cited account research.",
    },
    "LiftScore": {
        "sales_label": "Incremental Pipeline Score",
        "description": "How much incremental value the chosen rep effort creates over the next-best alternative.",
    },
}
SUBMISSION_FIELD_TERMINOLOGY = {
    "human_touch_rank": {
        "sales_label": "Outreach Priority Rank",
        "description": "The account's position in the personalized outreach queue for the week.",
    },
    "selected_contacts": {
        "sales_label": "Target Contacts",
        "description": "The people the rep should work for this account.",
    },
    "primary_trigger_event_id": {
        "sales_label": "Primary Why-Now Signal",
        "description": "The main signal or event that justifies working the account now.",
    },
    "evidence_brief": {
        "sales_label": "Account Research Brief",
        "description": "The structured SDR rationale and supporting account research citations.",
    },
}
BASELINE_TERMINOLOGY = {
    "random_within_icp": {
        "sales_label": "Random Named-Account Picklist",
        "description": "Uniform random picks from eligible ICP accounts.",
    },
    "rules_trigger_queue": {
        "sales_label": "Rules-Based Trigger Queue",
        "description": "A trigger-and-fit rules engine that prioritizes accounts with strong recent signals.",
    },
    "propensity_only_lead_score": {
        "sales_label": "Lead-Score-Only Ranking",
        "description": "A classic propensity ranking that ignores incremental lift.",
    },
    "last_touch_recency": {
        "sales_label": "Recent-Signal Follow-Up Queue",
        "description": "A recency-driven queue built mostly on fresh activity and signals.",
    },
    "bau_enterprise_sdr_policy": {
        "sales_label": "Business-as-Usual SDR Playbook",
        "description": "A deterministic enterprise SDR routing and prioritization playbook.",
    },
}
SUPPLEMENTAL_METRIC_TERMINOLOGY = {
    "precision_at_capacity": {
        "sales_label": "Positive Opportunity Precision @ Capacity",
        "description": "Share of prioritized outreach slots that land on accounts with positive opportunity lift.",
    },
    "ndcg_at_capacity": {
        "sales_label": "Queue Ranking Quality @ Capacity",
        "description": "How closely the outreach queue matches the ideal ranking at the current rep-capacity level.",
    },
    "uplift_at_capacity": {
        "sales_label": "Incremental Pipeline per Outreach Slot",
        "description": "Incremental pipeline created per personalized outreach slot at the current weekly capacity.",
    },
    "oracle_uplift_at_capacity": {
        "sales_label": "Oracle Incremental Pipeline per Outreach Slot",
        "description": "Best-achievable incremental pipeline per outreach slot at the same weekly capacity.",
    },
    "uplift_ratio_at_capacity": {
        "sales_label": "Incremental Pipeline Attainment",
        "description": "Share of oracle incremental pipeline captured by the prioritized outreach queue.",
    },
    "action_policy_value": {
        "sales_label": "Book-Level Incremental Pipeline",
        "description": "Total incremental pipeline generated by the chosen routing plan versus taking no action.",
    },
    "oracle_action_policy_value": {
        "sales_label": "Oracle Book-Level Incremental Pipeline",
        "description": "Best-achievable book-level incremental pipeline versus taking no action.",
    },
    "action_policy_ratio": {
        "sales_label": "Book-Level Pipeline Attainment",
        "description": "Share of oracle book-level incremental pipeline captured by the chosen routing plan.",
    },
    "trigger_accuracy_at_1": {
        "sales_label": "Primary Why-Now Pick Accuracy @1",
        "description": "How often the selected primary why-now signal matches the oracle signal for touched accounts.",
    },
    "trigger_weighted_gain": {
        "sales_label": "Why-Now Weighted Gain",
        "description": "Recency-weighted value of the selected why-now signals.",
    },
    "oracle_trigger_weighted_gain": {
        "sales_label": "Oracle Why-Now Weighted Gain",
        "description": "Best-achievable recency-weighted why-now value on the touched accounts.",
    },
    "trigger_gain_ratio": {
        "sales_label": "Why-Now Gain Attainment",
        "description": "Share of oracle why-now signal value captured by the model's selected signals.",
    },
    "timing_score_ratio": {
        "sales_label": "Why-Now Composite Ratio",
        "description": "Underlying ratio used to build the top-line Why-Now Score.",
    },
    "grounded_claim_ratio": {
        "sales_label": "Supported Research Claim Rate",
        "description": "Share of structured SDR claims supported by cited account research.",
    },
    "citation_relevance_ratio": {
        "sales_label": "Research Citation Relevance",
        "description": "Share of cited documents that actually support the structured rationale.",
    },
    "trigger_citation_alignment": {
        "sales_label": "Why-Now Citation Alignment",
        "description": "How often the cited research directly supports the selected why-now signal.",
    },
    "evidence_schema_accuracy": {
        "sales_label": "Research Brief Completeness",
        "description": "Whether the account research brief includes the required structured evidence fields.",
    },
    "unsupported_claim_rate": {
        "sales_label": "Unsupported Research Claim Rate",
        "description": "Share of structured SDR claims not supported by the cited account research.",
    },
    "grounding_score_ratio": {
        "sales_label": "Research Grounding Composite Ratio",
        "description": "Underlying ratio used to build the top-line Account Research Grounding Score.",
    },
    "contact_precision_at_selected": {
        "sales_label": "Target Contact Precision",
        "description": "Share of selected contacts that match the oracle top-contact set for the same outreach size.",
    },
    "contact_mrr": {
        "sales_label": "Best-Contact MRR",
        "description": "How highly the best available contact appears in the selected contact list.",
    },
    "contact_meeting_lift_delta": {
        "sales_label": "Meeting Lift over Default Contact",
        "description": "Meeting gain from the selected contact mix over the default contact choice.",
    },
    "oracle_contact_meeting_lift_delta": {
        "sales_label": "Oracle Meeting Lift over Default Contact",
        "description": "Best-achievable meeting gain over the default contact choice.",
    },
    "contact_lift_ratio": {
        "sales_label": "Contact Lift Attainment",
        "description": "Share of oracle contact-selection lift captured by the selected contact mix.",
    },
    "contact_score_ratio": {
        "sales_label": "Buying-Center Composite Ratio",
        "description": "Underlying ratio used to build the top-line Buying-Center Coverage Score.",
    },
    "cumulative_incremental_pipeline": {
        "sales_label": "Incremental Pipeline Generated",
        "description": "Total incremental pipeline generated across the full multi-week book-management run.",
    },
    "cumulative_incremental_opps": {
        "sales_label": "Incremental Opportunities Generated",
        "description": "Total incremental opportunities generated across the multi-week run.",
    },
    "cumulative_incremental_meetings": {
        "sales_label": "Incremental Meetings Generated",
        "description": "Total incremental meetings generated across the multi-week run.",
    },
    "oracle_cumulative_incremental_pipeline": {
        "sales_label": "Oracle Incremental Pipeline",
        "description": "Best-achievable incremental pipeline across the multi-week run.",
    },
    "oracle_cumulative_incremental_opps": {
        "sales_label": "Oracle Incremental Opportunities",
        "description": "Best-achievable incremental opportunities across the multi-week run.",
    },
    "oracle_cumulative_incremental_meetings": {
        "sales_label": "Oracle Incremental Meetings",
        "description": "Best-achievable incremental meetings across the multi-week run.",
    },
    "policy_pipeline_ratio": {
        "sales_label": "Pipeline Attainment vs Oracle",
        "description": "Share of oracle pipeline captured over the multi-week run.",
    },
    "policy_opp_ratio": {
        "sales_label": "Opportunity Attainment vs Oracle",
        "description": "Share of oracle opportunity creation captured over the multi-week run.",
    },
    "policy_meeting_ratio": {
        "sales_label": "Meeting Attainment vs Oracle",
        "description": "Share of oracle meeting creation captured over the multi-week run.",
    },
    "policy_regret": {
        "sales_label": "Pipeline Regret",
        "description": "Pipeline left on the table versus the multi-week oracle book-management policy.",
    },
    "policy_negative_regret_ratio": {
        "sales_label": "Negative Regret Score",
        "description": "Inverse regret component used inside the top-line Multi-Week Book Management Score.",
    },
}
COMPLIANCE_TERMINOLOGY = {
    "total_accounts": {
        "sales_label": "Total Accounts in Book",
        "description": "Number of accounts in the weekly book.",
    },
    "submitted_decisions": {
        "sales_label": "Accounts with Explicit Routing",
        "description": "Number of accounts explicitly routed by the submission.",
    },
    "omitted_accounts_defaulted_to_wait": {
        "sales_label": "Accounts Defaulted to Monitor / No Action",
        "description": "Accounts omitted from the submission and therefore defaulted to no action.",
    },
    "effective_human_touch_count": {
        "sales_label": "Personalized Outreach Slots Used",
        "description": "Number of personalized outreach slots that remained after compliance cleanup.",
    },
    "budgeted_human_touch_count": {
        "sales_label": "Personalized Outreach Capacity",
        "description": "Number of personalized outreach slots available this week.",
    },
    "over_budget_downgraded_count": {
        "sales_label": "Over-Capacity Outreach Requests Downgraded",
        "description": "Personalized outreach decisions downgraded because they fell outside the feasible weekly capacity.",
    },
    "compliance_multiplier": {
        "sales_label": "Compliance Multiplier",
        "description": "Multiplier reported for over-capacity behavior.",
    },
}
WINDOW_SUPPLEMENTAL_METRIC_ORDER = (
    "precision_at_capacity",
    "ndcg_at_capacity",
    "uplift_at_capacity",
    "oracle_uplift_at_capacity",
    "uplift_ratio_at_capacity",
    "action_policy_value",
    "oracle_action_policy_value",
    "action_policy_ratio",
    "trigger_accuracy_at_1",
    "trigger_weighted_gain",
    "oracle_trigger_weighted_gain",
    "trigger_gain_ratio",
    "timing_score_ratio",
    "contact_precision_at_selected",
    "contact_mrr",
    "contact_meeting_lift_delta",
    "oracle_contact_meeting_lift_delta",
    "contact_lift_ratio",
    "contact_score_ratio",
    "grounded_claim_ratio",
    "citation_relevance_ratio",
    "trigger_citation_alignment",
    "evidence_schema_accuracy",
    "unsupported_claim_rate",
    "grounding_score_ratio",
)
EPISODE_SUPPLEMENTAL_METRIC_ORDER = WINDOW_SUPPLEMENTAL_METRIC_ORDER + (
    "cumulative_incremental_pipeline",
    "cumulative_incremental_opps",
    "cumulative_incremental_meetings",
    "oracle_cumulative_incremental_pipeline",
    "oracle_cumulative_incremental_opps",
    "oracle_cumulative_incremental_meetings",
    "policy_pipeline_ratio",
    "policy_opp_ratio",
    "policy_meeting_ratio",
    "policy_regret",
    "policy_negative_regret_ratio",
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    schemas: dict[str, Any] = {}
    store: dict[str, Any] = {}

    for name, filename in SCHEMA_FILES.items():
        path = SCHEMA_DIR / filename
        schema = load_json(path)
        schemas[name] = schema

        schema_id = schema.get("$id")
        if schema_id:
            store[schema_id] = schema

        store[filename] = schema
        store[path.resolve().as_uri()] = schema

    return schemas, store


def validate_instance(instance: Any, schema_name: str) -> list[str]:
    schemas, store = load_schemas()
    schema = schemas[schema_name]
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    resolver = RefResolver.from_schema(schema, store=store)
    validator = validator_cls(schema, resolver=resolver)
    errors = sorted(validator.iter_errors(instance), key=lambda err: list(err.path))
    return [format_validation_error(error) for error in errors]


def format_validation_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    if path:
        return f"{path}: {error.message}"
    return error.message


def build_issue(
    code: str,
    message: str,
    *,
    severity: str = "error",
    account_id: str | None = None,
    window_id: str | None = None,
) -> dict[str, Any]:
    issue = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if account_id is not None:
        issue["account_id"] = account_id
    if window_id is not None:
        issue["window_id"] = window_id
    return issue


def build_sales_terminology() -> dict[str, Any]:
    return {
        "audience": "sdr_sales_ops",
        "notes": [
            "Canonical machine keys stay stable for schemas, submissions, and report parsing.",
            "Sales-facing labels are interpretive aliases only; submissions must still use the canonical machine keys.",
        ],
        "actions": deepcopy(ACTION_TERMINOLOGY),
        "metrics": deepcopy(METRIC_TERMINOLOGY),
        "submission_fields": deepcopy(SUBMISSION_FIELD_TERMINOLOGY),
        "baselines": deepcopy(BASELINE_TERMINOLOGY),
    }


def humanize_internal_name(name: str) -> str:
    replacements = {
        "ndcg": "nDCG",
        "mrr": "MRR",
        "icp": "ICP",
        "sdr": "SDR",
        "auuc": "AUUC",
    }
    words = [replacements.get(part, part.capitalize()) for part in name.split("_")]
    label = " ".join(words)
    label = label.replace("At 1", "@1")
    return label


def build_sales_metric_entries(
    metric_map: dict[str, Any],
    *,
    metric_names: tuple[str, ...] | None = None,
    omit_none: bool = True,
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    names = metric_names if metric_names is not None else tuple(metric_map)
    for metric_name in names:
        if metric_name not in metric_map:
            continue
        value = metric_map.get(metric_name)
        if omit_none and value is None:
            continue
        terminology = METRIC_TERMINOLOGY.get(metric_name) or SUPPLEMENTAL_METRIC_TERMINOLOGY.get(metric_name)
        sales_label = (
            terminology["sales_label"]
            if terminology
            else humanize_internal_name(metric_name)
        )
        entry = {
            "canonical_key": metric_name,
            "value": value,
        }
        if terminology and terminology.get("description"):
            entry["description"] = terminology["description"]
        entries[sales_label] = entry
    return entries


def build_sales_action_counts(action_counts: dict[str, int]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for action_name in ACTION_SPACE:
        if action_name not in action_counts:
            continue
        entries[ACTION_TERMINOLOGY[action_name]["sales_label"]] = {
            "canonical_key": action_name,
            "count": action_counts[action_name],
        }
    return entries


def build_sales_compliance_view(compliance: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field_name, terminology in COMPLIANCE_TERMINOLOGY.items():
        if field_name not in compliance:
            continue
        summary[terminology["sales_label"]] = {
            "canonical_key": field_name,
            "value": compliance[field_name],
            "description": terminology["description"],
        }
    return {
        "summary": summary,
        "requested_routing_mix": build_sales_action_counts(
            compliance.get("requested_action_counts", {})
        ),
        "effective_routing_mix": build_sales_action_counts(
            compliance.get("effective_action_counts", {})
        ),
    }


def build_sales_normalization_view(
    normalization: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not normalization:
        return None

    baseline_name = normalization["baseline_name"]
    baseline_terminology = BASELINE_TERMINOLOGY.get(baseline_name)
    return {
        "anchor_baseline": {
            "canonical_key": baseline_name,
            "sales_label": (
                baseline_terminology["sales_label"]
                if baseline_terminology
                else humanize_internal_name(baseline_name)
            ),
            "description": baseline_terminology.get("description") if baseline_terminology else None,
        },
        "clip_range": normalization["clip_range"],
        "normalized_scorecard": build_sales_metric_entries(
            normalization.get("normalized_metrics", {}),
            metric_names=NORMALIZED_SCORE_METRICS,
        ),
        "anchor_baseline_scorecard": build_sales_metric_entries(
            normalization.get("baseline_metrics", {}),
            metric_names=NORMALIZED_SCORE_METRICS,
        ),
        "oracle_scorecard": build_sales_metric_entries(
            normalization.get("oracle_metrics", {}),
            metric_names=NORMALIZED_SCORE_METRICS,
        ),
    }


def build_window_sales_view(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_mode": "Weekly Queue Prioritization Benchmark",
        "benchmark_entity_id": report.get("window_id"),
        "status": report.get("status"),
        "scorecard": build_sales_metric_entries(
            report.get("metrics", {}),
            metric_names=NORMALIZED_SCORE_METRICS,
        ),
        "supplemental_metrics": build_sales_metric_entries(
            report.get("metrics", {}),
            metric_names=WINDOW_SUPPLEMENTAL_METRIC_ORDER,
        ),
        "routing_summary": build_sales_compliance_view(report.get("compliance", {})),
        "benchmark_vs_anchor": build_sales_normalization_view(report.get("normalization")),
    }


def build_episode_sales_view(report: dict[str, Any]) -> dict[str, Any]:
    window_reports = report.get("window_reports")
    return {
        "benchmark_mode": "Multi-Week Book Management Benchmark",
        "benchmark_entity_id": report.get("episode_id"),
        "status": report.get("status"),
        "window_count": len(window_reports) if isinstance(window_reports, list) else None,
        "scorecard": build_sales_metric_entries(
            report.get("metrics", {}),
            metric_names=NORMALIZED_SCORE_METRICS,
        ),
        "supplemental_metrics": build_sales_metric_entries(
            report.get("metrics", {}),
            metric_names=EPISODE_SUPPLEMENTAL_METRIC_ORDER,
        ),
        "benchmark_vs_anchor": build_sales_normalization_view(report.get("normalization")),
    }


def build_robustness_sales_view(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    cases = report.get("cases", [])
    underperforming_cases = [
        {
            "case_id": case["case_id"],
            "robustness_type": case["robustness_type"],
            "named_account_book_score_vs_anchor": case["normalized_enterprise_allocation_score"],
        }
        for case in summary.get("underperforming_cases", [])
    ]
    case_scoreboard = [
        {
            "case_id": case["case_id"],
            "robustness_type": case["robustness_type"],
            "status": case["status"],
            "Named-Account Book Score": case.get("metrics", {}).get("EnterpriseAllocationScore"),
            "Named-Account Book Score vs Anchor": case.get("normalized_metrics", {}).get("EnterpriseAllocationScore"),
        }
        for case in cases
    ]
    return {
        "benchmark_mode": "Robustness Diagnostics",
        "benchmark_entity_id": report.get("suite_id"),
        "status": report.get("status"),
        "summary": {
            "Robustness Score": {
                "value": summary.get("robustness_score"),
                "basis": summary.get("robustness_score_basis"),
                "description": "Blend of average and worst-case Named-Account Book Score across robustness cases.",
            },
            "average_scorecard": build_sales_metric_entries(
                summary.get("average_metrics", {}),
                metric_names=NORMALIZED_SCORE_METRICS,
            ),
            "worst_case_scorecard": build_sales_metric_entries(
                summary.get("worst_case_metrics", {}),
                metric_names=NORMALIZED_SCORE_METRICS,
            ),
            "average_scorecard_vs_anchor": build_sales_metric_entries(
                summary.get("average_normalized_metrics", {}),
                metric_names=NORMALIZED_SCORE_METRICS,
            ),
            "worst_case_scorecard_vs_anchor": build_sales_metric_entries(
                summary.get("worst_case_normalized_metrics", {}),
                metric_names=NORMALIZED_SCORE_METRICS,
            ),
            "underperforming_cases": underperforming_cases,
        },
        "case_scoreboard": case_scoreboard,
    }


def finalize_window_report(report: dict[str, Any]) -> dict[str, Any]:
    report["sales_view"] = build_window_sales_view(report)
    return report


def finalize_episode_report(report: dict[str, Any]) -> dict[str, Any]:
    report["sales_view"] = build_episode_sales_view(report)
    return report


def finalize_robustness_report(report: dict[str, Any]) -> dict[str, Any]:
    report["sales_view"] = build_robustness_sales_view(report)
    return report


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def average_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def ratio_against_oracle(observed: float, oracle: float) -> float:
    if math.isclose(observed, 0.0) and math.isclose(oracle, 0.0):
        return 1.0
    return safe_divide(observed, oracle)


def clip_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def normalize_score_against_baseline(
    observed: float | None,
    baseline: float | None,
    *,
    oracle: float = 100.0,
    clip_range: tuple[float, float] = NORMALIZATION_CLIP_RANGE,
) -> float | None:
    if observed is None or baseline is None:
        return None

    if math.isclose(oracle, baseline):
        if math.isclose(observed, oracle):
            return 100.0
        if observed > oracle:
            return clip_range[1]
        return 0.0

    normalized = 100.0 * (observed - baseline) / (oracle - baseline)
    return clip_value(normalized, clip_range[0], clip_range[1])


def build_score_normalization(
    raw_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    baseline_name: str,
) -> dict[str, Any]:
    oracle_metrics: dict[str, float] = {}
    normalized_metrics: dict[str, float] = {}

    for metric_name in NORMALIZED_SCORE_METRICS:
        raw_value = raw_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        normalized_value = normalize_score_against_baseline(raw_value, baseline_value)
        if normalized_value is None:
            continue
        oracle_metrics[metric_name] = 100.0
        normalized_metrics[metric_name] = normalized_value

    return {
        "baseline_name": baseline_name,
        "clip_range": list(NORMALIZATION_CLIP_RANGE),
        "baseline_metrics": {
            metric_name: baseline_metrics.get(metric_name)
            for metric_name in normalized_metrics
        },
        "oracle_metrics": oracle_metrics,
        "normalized_metrics": normalized_metrics,
    }


COMPLIANCE_MULTIPLIED_METRICS = {
    "EnterpriseAllocationScore",
    "OfflineScore",
    "FitScore",
    "LiftScore",
    "precision_at_capacity",
    "ndcg_at_capacity",
    "uplift_ratio_at_capacity",
    "action_policy_ratio",
}


def apply_compliance_multiplier(
    metrics: dict[str, Any],
    compliance: dict[str, Any],
) -> dict[str, Any]:
    """Apply budget-compliance penalties to score/ranking metrics only."""
    multiplier = compliance.get("compliance_multiplier", 1.0)
    if not isinstance(multiplier, (int, float)) or math.isclose(multiplier, 1.0):
        return metrics

    adjusted = deepcopy(metrics)
    for metric_name in COMPLIANCE_MULTIPLIED_METRICS:
        value = adjusted.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            adjusted[metric_name] = value * multiplier
    return adjusted


def has_inbound_intent(account: dict[str, Any]) -> bool:
    web_engagement = account["web_engagement"]
    return (
        web_engagement["pricing_visits_30d"] > 0
        or web_engagement["high_intent_content_downloads_30d"] > 0
        or web_engagement["trial_signups_30d"] > 0
    )


def classify_data_density(
    account: dict[str, Any],
    *,
    contact_count: int,
    trigger_count: int,
    evidence_count: int,
) -> str:
    return (
        "dense"
        if contact_count >= 2 and trigger_count >= 2 and evidence_count >= 3
        else "sparse"
    )


def derive_account_slice_values(
    window_data: dict[str, Any],
) -> dict[str, dict[str, str]]:
    contact_counts = Counter(
        contact["account_id"]
        for contact in window_data["contacts"]
    )
    trigger_counts = Counter(
        trigger["account_id"]
        for trigger in window_data["triggers"]
    )
    evidence_counts = Counter(
        document["account_id"]
        for document in window_data["evidence"]
    )

    slice_values_by_account: dict[str, dict[str, str]] = {}
    for account in window_data["accounts"]:
        account_id = account["account_id"]
        slice_values_by_account[account_id] = {
            "segment": account["segment"],
            "industry": account["industry"],
            "region": account["hq_region"],
            "intent_presence": "present" if has_inbound_intent(account) else "absent",
            "relationship_motion": account["relationship_motion"],
            "data_density": classify_data_density(
                account,
                contact_count=contact_counts.get(account_id, 0),
                trigger_count=trigger_counts.get(account_id, 0),
                evidence_count=evidence_counts.get(account_id, 0),
            ),
        }

    return slice_values_by_account


def build_slice_diagnostic_notes() -> list[str]:
    return [
        "Slice diagnostics are diagnostic only and do not change leaderboard scoring.",
        (
            "intent_presence is present when the account shows pricing visits, "
            "high-intent content downloads, or trial signups in the current window."
        ),
        (
            "data_density is dense when the account has at least 2 surfaced contacts, "
            "2 triggers, and 3 evidence documents in the current window; otherwise sparse."
        ),
    ]


def subset_effective_decisions_to_accounts(
    effective_decisions: list[dict[str, Any]],
    account_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        decision
        for decision in effective_decisions
        if decision["account_id"] in account_ids
    ]


def build_window_slice_diagnostics(
    window_data: dict[str, Any],
    effective_decisions: list[dict[str, Any]],
    *,
    labels_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slice_values_by_account = derive_account_slice_values(window_data)
    dimensions: dict[str, dict[str, Any]] = {}

    for dimension in SLICE_DIMENSIONS:
        grouped_account_ids: dict[str, set[str]] = {}
        for account in window_data["accounts"]:
            account_id = account["account_id"]
            slice_value = slice_values_by_account[account_id][dimension]
            grouped_account_ids.setdefault(slice_value, set()).add(account_id)

        dimension_entries: dict[str, Any] = {}
        for slice_value in sorted(grouped_account_ids):
            account_ids = grouped_account_ids[slice_value]
            subset_window = subset_window_to_accounts(window_data, account_ids)
            subset_labels = subset_hidden_labels_to_accounts(labels_data, account_ids)
            subset_decisions = subset_effective_decisions_to_accounts(
                effective_decisions,
                account_ids,
            )
            subset_metrics, _ = compute_window_metrics(
                subset_window,
                subset_decisions,
                labels_data=subset_labels,
            )
            dimension_entries[slice_value] = {
                "observation_count": len(account_ids),
                "window_count": 1,
                "human_touch_count": sum(
                    1
                    for decision in subset_decisions
                    if decision["chosen_action"] == "human_touch"
                ),
                "metrics": subset_metrics,
                "normalization": None,
            }

        dimensions[dimension] = dimension_entries

    return {
        "notes": build_slice_diagnostic_notes(),
        "dimensions": dimensions,
    }


def attach_slice_normalization(
    slice_diagnostics: dict[str, Any],
    baseline_slice_diagnostics: dict[str, Any] | None,
    *,
    baseline_name: str,
) -> dict[str, Any]:
    normalized = deepcopy(slice_diagnostics)
    if not baseline_slice_diagnostics:
        return normalized

    baseline_dimensions = baseline_slice_diagnostics.get("dimensions", {})
    for dimension, slices in normalized["dimensions"].items():
        baseline_slices = baseline_dimensions.get(dimension, {})
        for slice_value, slice_entry in slices.items():
            baseline_entry = baseline_slices.get(slice_value)
            if not baseline_entry:
                continue
            slice_entry["normalization"] = build_score_normalization(
                slice_entry["metrics"],
                baseline_entry["metrics"],
                baseline_name=baseline_name,
            )

    normalized["notes"] = normalized["notes"] + [
        f"Normalized slice metrics are reported relative to the {baseline_name} baseline."
    ]
    return normalized


def initialize_episode_slice_aggregates() -> dict[str, dict[str, Any]]:
    return {
        dimension: {}
        for dimension in SLICE_DIMENSIONS
    }


def update_episode_slice_aggregates(
    aggregates: dict[str, dict[str, Any]],
    window_slice_diagnostics: dict[str, Any],
    *,
    agent_slice_incrementals: dict[tuple[str, str], dict[str, float]] | None = None,
    oracle_slice_incrementals: dict[tuple[str, str], dict[str, float]] | None = None,
) -> None:
    for dimension, slices in window_slice_diagnostics["dimensions"].items():
        for slice_value, slice_entry in slices.items():
            aggregate_entry = aggregates[dimension].setdefault(
                slice_value,
                {
                    "observation_count": 0,
                    "window_count": 0,
                    "human_touch_count": 0,
                    "metric_weighted_sums": {},
                    "metric_weights": {},
                    "metric_seen": set(),
                    "agent_incremental_pipeline": 0.0,
                    "agent_incremental_opps": 0.0,
                    "agent_incremental_meetings": 0.0,
                    "oracle_incremental_pipeline": 0.0,
                    "oracle_incremental_opps": 0.0,
                    "oracle_incremental_meetings": 0.0,
                },
            )

            observation_count = slice_entry["observation_count"]
            human_touch_count = slice_entry["human_touch_count"]
            aggregate_entry["observation_count"] += observation_count
            aggregate_entry["window_count"] += slice_entry["window_count"]
            aggregate_entry["human_touch_count"] += human_touch_count

            for metric_name, value in slice_entry["metrics"].items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                aggregate_entry["metric_seen"].add(metric_name)
                metric_weight = (
                    human_touch_count
                    if metric_name in TOUCHED_ONLY_SLICE_METRICS
                    else observation_count
                )
                if metric_weight == 0 and metric_name in TOUCHED_ONLY_SLICE_METRICS:
                    continue
                aggregate_entry["metric_weighted_sums"][metric_name] = (
                    aggregate_entry["metric_weighted_sums"].get(metric_name, 0.0)
                    + value * metric_weight
                )
                aggregate_entry["metric_weights"][metric_name] = (
                    aggregate_entry["metric_weights"].get(metric_name, 0)
                    + metric_weight
                )

            if agent_slice_incrementals:
                agent_values = agent_slice_incrementals.get((dimension, slice_value))
                if agent_values:
                    aggregate_entry["agent_incremental_pipeline"] += agent_values["pipeline"]
                    aggregate_entry["agent_incremental_opps"] += agent_values["opps"]
                    aggregate_entry["agent_incremental_meetings"] += agent_values["meetings"]
            if oracle_slice_incrementals:
                oracle_values = oracle_slice_incrementals.get((dimension, slice_value))
                if oracle_values:
                    aggregate_entry["oracle_incremental_pipeline"] += oracle_values["pipeline"]
                    aggregate_entry["oracle_incremental_opps"] += oracle_values["opps"]
                    aggregate_entry["oracle_incremental_meetings"] += oracle_values["meetings"]


def finalize_episode_slice_diagnostics(
    aggregates: dict[str, dict[str, Any]],
    *,
    include_policy_metrics: bool,
) -> dict[str, Any]:
    dimensions: dict[str, dict[str, Any]] = {}

    for dimension, slices in aggregates.items():
        dimension_entries: dict[str, Any] = {}
        for slice_value in sorted(slices):
            aggregate_entry = slices[slice_value]
            metrics: dict[str, Any] = {}
            for metric_name in sorted(aggregate_entry["metric_seen"]):
                metric_weight = aggregate_entry["metric_weights"].get(metric_name, 0)
                if metric_weight:
                    metrics[metric_name] = safe_divide(
                        aggregate_entry["metric_weighted_sums"][metric_name],
                        metric_weight,
                    )
                elif metric_name in TOUCHED_ONLY_SLICE_METRICS:
                    metrics[metric_name] = 0.0

            if include_policy_metrics:
                agent_pipeline = aggregate_entry["agent_incremental_pipeline"]
                agent_opps = aggregate_entry["agent_incremental_opps"]
                agent_meetings = aggregate_entry["agent_incremental_meetings"]
                oracle_pipeline = aggregate_entry["oracle_incremental_pipeline"]
                oracle_opps = aggregate_entry["oracle_incremental_opps"]
                oracle_meetings = aggregate_entry["oracle_incremental_meetings"]

                policy_pipeline_ratio = ratio_against_oracle(agent_pipeline, oracle_pipeline)
                policy_opp_ratio = ratio_against_oracle(agent_opps, oracle_opps)
                policy_meeting_ratio = ratio_against_oracle(agent_meetings, oracle_meetings)
                policy_regret = max(oracle_pipeline - agent_pipeline, 0.0)
                policy_negative_regret_ratio = ratio_against_oracle(
                    oracle_pipeline - policy_regret,
                    oracle_pipeline,
                )
                policy_score = 100.0 * (
                    0.50 * policy_pipeline_ratio
                    + 0.25 * policy_opp_ratio
                    + 0.15 * policy_meeting_ratio
                    + 0.10 * policy_negative_regret_ratio
                )
                enterprise_allocation_score = policy_score
                if metrics.get("OfflineScore") is not None:
                    enterprise_allocation_score = (
                        0.60 * metrics["OfflineScore"]
                        + 0.40 * policy_score
                    )

                metrics.update(
                    {
                        "EnterpriseAllocationScore": enterprise_allocation_score,
                        "PolicyScore": policy_score,
                        "cumulative_incremental_pipeline": agent_pipeline,
                        "cumulative_incremental_opps": agent_opps,
                        "cumulative_incremental_meetings": agent_meetings,
                        "oracle_cumulative_incremental_pipeline": oracle_pipeline,
                        "oracle_cumulative_incremental_opps": oracle_opps,
                        "oracle_cumulative_incremental_meetings": oracle_meetings,
                        "policy_pipeline_ratio": policy_pipeline_ratio,
                        "policy_opp_ratio": policy_opp_ratio,
                        "policy_meeting_ratio": policy_meeting_ratio,
                        "policy_regret": policy_regret,
                        "policy_negative_regret_ratio": policy_negative_regret_ratio,
                    }
                )

            dimension_entries[slice_value] = {
                "observation_count": aggregate_entry["observation_count"],
                "window_count": aggregate_entry["window_count"],
                "human_touch_count": aggregate_entry["human_touch_count"],
                "metrics": metrics,
                "normalization": None,
            }

        dimensions[dimension] = dimension_entries

    return {
        "notes": build_slice_diagnostic_notes(),
        "dimensions": dimensions,
    }


def sanitize_human_touch_decision(
    decision: dict[str, Any],
    *,
    account_id: str,
    account_contacts: set[str],
    account_triggers: set[str],
    account_evidence: set[str],
    account_grounding_evidence: set[str],
    max_contacts_per_account: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    sanitized = deepcopy(decision)

    original_contacts = sanitized["selected_contacts"]
    valid_contacts = [contact_id for contact_id in original_contacts if contact_id in account_contacts]
    if len(valid_contacts) != len(original_contacts):
        dropped = sorted(set(original_contacts) - set(valid_contacts))
        issues.append(
            build_issue(
                "invalid_contact",
                f"Dropped contacts not surfaced for the account: {dropped}",
                account_id=account_id,
            )
        )

    if len(valid_contacts) > max_contacts_per_account:
        issues.append(
            build_issue(
                "contact_limit_exceeded",
                (
                    f"Trimmed selected contacts from {len(valid_contacts)} to "
                    f"{max_contacts_per_account}"
                ),
                account_id=account_id,
            )
        )
        valid_contacts = valid_contacts[:max_contacts_per_account]

    sanitized["selected_contacts"] = valid_contacts

    evidence_brief = deepcopy(sanitized["evidence_brief"])
    original_citations = evidence_brief["citations"]
    valid_citations = [doc_id for doc_id in original_citations if doc_id in account_evidence]
    if len(valid_citations) != len(original_citations):
        dropped = sorted(set(original_citations) - set(valid_citations))
        issues.append(
            build_issue(
                "invalid_citation",
                f"Dropped citations outside the account window: {dropped}",
                severity="warning",
                account_id=account_id,
            )
        )
    grounding_citations = [
        doc_id
        for doc_id in valid_citations
        if doc_id in account_grounding_evidence
    ]
    if len(grounding_citations) != len(valid_citations):
        dropped = sorted(set(valid_citations) - set(grounding_citations))
        issues.append(
            build_issue(
                "disallowed_grounding_citation",
                f"Dropped citations that are not allowed for grounding: {dropped}",
                severity="warning",
                account_id=account_id,
            )
        )
    evidence_brief["citations"] = grounding_citations
    sanitized["evidence_brief"] = evidence_brief

    if sanitized["primary_trigger_event_id"] not in account_triggers:
        issues.append(
            build_issue(
                "invalid_trigger",
                "Coerced decision to wait because the primary trigger does not belong to the account.",
                account_id=account_id,
            )
        )
        return coerce_to_wait(sanitized, reason="invalid_trigger")

    if not sanitized["selected_contacts"]:
        issues.append(
            build_issue(
                "missing_valid_contacts",
                "Coerced decision to wait because no valid selected contacts remained.",
                account_id=account_id,
            )
        )
        return coerce_to_wait(sanitized, reason="missing_valid_contacts")

    return sanitized


def coerce_to_wait(decision: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "account_id": decision["account_id"],
        "chosen_action": "wait",
        "action_score": decision.get("action_score", 0.0),
        "coerced_from": decision.get("chosen_action"),
        "coerce_reason": reason,
    }


def normalize_submission(
    window_data: dict[str, Any],
    submission_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accounts = window_data["accounts"]
    contacts = window_data["contacts"]
    triggers = window_data["triggers"]
    evidence = window_data["evidence"]
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    max_contacts = window_data["capacity_budget"]["max_contacts_per_account"]

    account_ids = {account["account_id"] for account in accounts}
    contacts_by_account: dict[str, set[str]] = {account_id: set() for account_id in account_ids}
    triggers_by_account: dict[str, set[str]] = {account_id: set() for account_id in account_ids}
    evidence_by_account: dict[str, set[str]] = {account_id: set() for account_id in account_ids}
    grounding_evidence_by_account: dict[str, set[str]] = {
        account_id: set()
        for account_id in account_ids
    }

    for contact in contacts:
        contacts_by_account.setdefault(contact["account_id"], set()).add(contact["contact_id"])

    for trigger in triggers:
        triggers_by_account.setdefault(trigger["account_id"], set()).add(trigger["event_id"])

    for document in evidence:
        evidence_by_account.setdefault(document["account_id"], set()).add(document["doc_id"])
        if document["allowed_for_grounding"]:
            grounding_evidence_by_account.setdefault(document["account_id"], set()).add(document["doc_id"])

    issues: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    seen_accounts: set[str] = set()

    for raw_decision in submission_data["decisions"]:
        account_id = raw_decision["account_id"]
        decision = deepcopy(raw_decision)

        if account_id in seen_accounts:
            issues.append(
                build_issue(
                    "duplicate_account_decision",
                    "Coerced duplicate account decision to wait.",
                    account_id=account_id,
                )
            )
            normalized.append(coerce_to_wait(decision, reason="duplicate_account_decision"))
            continue

        seen_accounts.add(account_id)

        if account_id not in account_ids:
            issues.append(
                build_issue(
                    "unknown_account",
                    "Coerced decision to wait because the account is not present in the window.",
                    account_id=account_id,
                )
            )
            normalized.append(coerce_to_wait(decision, reason="unknown_account"))
            continue

        if decision["chosen_action"] == "human_touch":
            normalized.append(
                sanitize_human_touch_decision(
                    decision,
                    account_id=account_id,
                    account_contacts=contacts_by_account.get(account_id, set()),
                    account_triggers=triggers_by_account.get(account_id, set()),
                    account_evidence=evidence_by_account.get(account_id, set()),
                    account_grounding_evidence=grounding_evidence_by_account.get(account_id, set()),
                    max_contacts_per_account=max_contacts,
                    issues=issues,
                )
            )
        else:
            normalized.append(decision)

    human_touch_decisions = sorted(
        (
            decision
            for decision in normalized
            if decision["chosen_action"] == "human_touch"
        ),
        key=lambda decision: (decision["human_touch_rank"], decision["account_id"]),
    )

    rank_to_account: dict[int, str] = {}
    duplicate_ranks: list[int] = []
    for decision in human_touch_decisions:
        rank = decision["human_touch_rank"]
        if rank in rank_to_account:
            duplicate_ranks.append(rank)
        else:
            rank_to_account[rank] = decision["account_id"]

    if duplicate_ranks:
        issues.append(
            build_issue(
                "duplicate_human_touch_rank",
                f"Duplicate human touch ranks found: {sorted(set(duplicate_ranks))}",
            )
        )
        seen_ranks: set[int] = set()
        deduped: list[dict[str, Any]] = []
        for decision in human_touch_decisions:
            rank = decision["human_touch_rank"]
            if rank in seen_ranks:
                deduped.append(coerce_to_wait(decision, reason="duplicate_human_touch_rank"))
            else:
                seen_ranks.add(rank)
                deduped.append(decision)
        human_touch_decisions = deduped

    effective_order = [
        decision
        for decision in human_touch_decisions
        if decision["chosen_action"] == "human_touch"
    ]

    expected_ranks = list(range(1, len(effective_order) + 1))
    actual_ranks = [decision["human_touch_rank"] for decision in effective_order]
    if actual_ranks and actual_ranks != expected_ranks:
        issues.append(
            build_issue(
                "noncontiguous_human_touch_rank",
                (
                    "Human touch ranks are not contiguous from 1. "
                    f"Observed ranks: {actual_ranks}"
                ),
                severity="warning",
            )
        )

    over_budget_count = 0
    allowed_human_touch_accounts = {
        decision["account_id"]
        for index, decision in enumerate(effective_order)
        if index < budget
    }

    final_decisions: list[dict[str, Any]] = []
    for decision in normalized:
        if decision["chosen_action"] != "human_touch":
            final_decisions.append(decision)
            continue

        if decision["account_id"] not in allowed_human_touch_accounts:
            over_budget_count += 1
            issues.append(
                build_issue(
                    "over_budget_human_touch",
                    "Coerced human touch decision to wait because it fell outside the feasible budget prefix.",
                    severity="warning",
                    account_id=decision["account_id"],
                )
            )
            final_decisions.append(coerce_to_wait(decision, reason="over_budget_human_touch"))
            continue

        final_decisions.append(decision)

    requested_counts = Counter(decision["chosen_action"] for decision in submission_data["decisions"])
    effective_counts = Counter(decision["chosen_action"] for decision in final_decisions)

    omitted_count = len(account_ids - seen_accounts)
    compliance = {
        "total_accounts": len(account_ids),
        "submitted_decisions": len(submission_data["decisions"]),
        "omitted_accounts_defaulted_to_wait": omitted_count,
        "requested_action_counts": dict(requested_counts),
        "effective_action_counts": dict(effective_counts),
        "effective_human_touch_count": effective_counts.get("human_touch", 0),
        "budgeted_human_touch_count": budget,
        "over_budget_downgraded_count": over_budget_count,
        "compliance_multiplier": 0.95 if over_budget_count else 1.0,
    }

    return final_decisions, issues, compliance


def materialize_effective_decisions(
    window_data: dict[str, Any],
    final_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decision_by_account = {
        decision["account_id"]: decision
        for decision in final_decisions
    }

    materialized: list[dict[str, Any]] = []
    for account in window_data["accounts"]:
        account_id = account["account_id"]
        materialized.append(
            deepcopy(
                decision_by_account.get(
                    account_id,
                    {
                        "account_id": account_id,
                        "chosen_action": "wait",
                        "action_score": 0.0,
                        "defaulted_to_wait": True,
                    },
                )
            )
        )

    return materialized


def subset_window_to_accounts(
    window_data: dict[str, Any],
    account_ids: set[str],
) -> dict[str, Any]:
    subset = deepcopy(window_data)
    subset["accounts"] = [
        account
        for account in window_data["accounts"]
        if account["account_id"] in account_ids
    ]
    subset["contacts"] = [
        contact
        for contact in window_data["contacts"]
        if contact["account_id"] in account_ids
    ]
    subset["triggers"] = [
        trigger
        for trigger in window_data["triggers"]
        if trigger["account_id"] in account_ids
    ]
    subset["evidence"] = [
        document
        for document in window_data["evidence"]
        if document["account_id"] in account_ids
    ]
    subset["capacity_budget"]["human_sdr_actions"] = min(
        subset["capacity_budget"]["human_sdr_actions"],
        len(subset["accounts"]),
    )
    return subset


def subset_hidden_labels_to_accounts(
    labels_data: dict[str, Any] | None,
    account_ids: set[str],
) -> dict[str, Any] | None:
    if labels_data is None:
        return None

    subset = deepcopy(labels_data)
    subset["account_outcomes"] = [
        row
        for row in labels_data["account_outcomes"]
        if row["account_id"] in account_ids
    ]
    if "contact_outcomes" in labels_data:
        subset["contact_outcomes"] = [
            row
            for row in labels_data["contact_outcomes"]
            if row["account_id"] in account_ids
        ]
    if "trigger_outcomes" in labels_data:
        subset["trigger_outcomes"] = [
            row
            for row in labels_data["trigger_outcomes"]
            if row["account_id"] in account_ids
        ]
    return subset


def clone_labels(labels_data: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(labels_data)


def apply_outcome_modifiers_to_labels(
    labels_data: dict[str, Any],
    outcome_modifiers_by_account: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    adjusted_labels = clone_labels(labels_data)
    for account_row in adjusted_labels["account_outcomes"]:
        account_id = account_row["account_id"]
        modifiers = outcome_modifiers_by_account.get(account_id)
        if not modifiers:
            continue

        for action_name, action_modifiers in modifiers.items():
            if action_name not in account_row["potential_outcomes"]:
                continue
            outcome = account_row["potential_outcomes"][action_name]
            outcome["meeting_prob"] *= action_modifiers["meeting_prob_multiplier"]
            outcome["opp_prob"] *= action_modifiers["opp_prob_multiplier"]
            outcome["pipeline_value"] *= action_modifiers["pipeline_value_multiplier"]

    return adjusted_labels


def is_account_eligible(
    state: dict[str, dict[str, Any]],
    account_id: str,
    window_index: int,
) -> bool:
    account_state = state.setdefault(
        account_id,
        {
            "removed": False,
            "available_from_index": 0,
            "pending_outcome_modifiers": None,
        },
    )
    return (
        not account_state["removed"]
        and account_state["available_from_index"] <= window_index
    )


def apply_policy_state_constraints(
    window_id: str,
    window_index: int,
    effective_decisions: list[dict[str, Any]],
    state: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    adjusted_decisions: list[dict[str, Any]] = []

    for decision in effective_decisions:
        account_id = decision["account_id"]
        if is_account_eligible(state, account_id, window_index):
            adjusted_decisions.append(decision)
            continue

        if decision["chosen_action"] != "wait":
            issues.append(
                build_issue(
                    "policy_ineligible_account",
                    "Coerced decision to wait because the account is in cooldown or removed from the episode.",
                    severity="warning",
                    account_id=account_id,
                    window_id=window_id,
                )
            )
            adjusted_decisions.append(coerce_to_wait(decision, reason="policy_ineligible_account"))
        else:
            adjusted_decisions.append(decision)

    return adjusted_decisions, issues


def compute_window_incremental_values(
    effective_decisions: list[dict[str, Any]],
    labels_data: dict[str, Any],
) -> dict[str, float]:
    labels_by_account = {
        row["account_id"]: row["potential_outcomes"]
        for row in labels_data["account_outcomes"]
    }

    totals = {
        "pipeline": 0.0,
        "opps": 0.0,
        "meetings": 0.0,
    }
    for decision in effective_decisions:
        outcomes = labels_by_account[decision["account_id"]]
        chosen = outcomes[decision["chosen_action"]]
        wait = outcomes["wait"]
        totals["pipeline"] += chosen["pipeline_value"] - wait["pipeline_value"]
        totals["opps"] += chosen["opp_prob"] - wait["opp_prob"]
        totals["meetings"] += chosen["meeting_prob"] - wait["meeting_prob"]

    return totals


def compute_greedy_oracle_window_decisions(
    window_data: dict[str, Any],
    labels_data: dict[str, Any],
    state: dict[str, dict[str, Any]],
    window_index: int,
) -> list[dict[str, Any]]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    labels_by_account = {
        row["account_id"]: row["potential_outcomes"]
        for row in labels_data["account_outcomes"]
    }

    oracle_decisions: list[dict[str, Any]] = []
    human_touch_candidates: list[tuple[float, float, float, str]] = []
    nonhuman_actions: dict[str, str] = {}

    for account in window_data["accounts"]:
        account_id = account["account_id"]
        if not is_account_eligible(state, account_id, window_index):
            oracle_decisions.append(
                {
                    "account_id": account_id,
                    "chosen_action": "wait",
                    "action_score": 0.0,
                }
            )
            continue

        outcomes = labels_by_account[account_id]
        best_nonhuman_name, best_nonhuman = best_nonhuman_action(outcomes)
        nonhuman_actions[account_id] = best_nonhuman_name
        human_touch_candidates.append(
            (
                outcomes["human_touch"]["pipeline_value"] - best_nonhuman["pipeline_value"],
                outcomes["human_touch"]["opp_prob"] - best_nonhuman["opp_prob"],
                outcomes["human_touch"]["meeting_prob"] - best_nonhuman["meeting_prob"],
                account_id,
            )
        )

    selected_accounts = {
        account_id
        for gain, _, _, account_id in sorted(human_touch_candidates, reverse=True)[:budget]
        if gain > 0
    }

    selected_rank = 1
    selected_window_accounts = {
        decision["account_id"]
        for decision in oracle_decisions
    }

    for account in window_data["accounts"]:
        account_id = account["account_id"]
        if account_id in selected_window_accounts:
            continue

        if account_id in selected_accounts:
            oracle_decisions.append(
                {
                    "account_id": account_id,
                    "chosen_action": "human_touch",
                    "human_touch_rank": selected_rank,
                    "action_score": 1.0,
                }
            )
            selected_rank += 1
        else:
            oracle_decisions.append(
                {
                    "account_id": account_id,
                    "chosen_action": nonhuman_actions.get(account_id, "wait"),
                    "action_score": 1.0,
                }
            )

    return oracle_decisions


def apply_policy_transitions(
    window_index: int,
    effective_decisions: list[dict[str, Any]],
    transition_map: dict[str, dict[str, Any]],
    state: dict[str, dict[str, Any]],
) -> None:
    for decision in effective_decisions:
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


def extract_current_outcome_modifiers(
    state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    outcome_modifiers_by_account: dict[str, dict[str, Any]] = {}
    for account_id, account_state in state.items():
        pending_modifiers = account_state.get("pending_outcome_modifiers")
        if pending_modifiers:
            outcome_modifiers_by_account[account_id] = deepcopy(pending_modifiers)
            account_state["pending_outcome_modifiers"] = None

    return outcome_modifiers_by_account


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


def compute_dcg(gains: list[float]) -> float:
    return sum(
        gain / math.log2(index + 2)
        for index, gain in enumerate(gains)
    )


def compute_trigger_label_metrics(
    window_data: dict[str, Any],
    effective_decisions: list[dict[str, Any]],
    labels_data: dict[str, Any],
    *,
    recency_lambda: float = 0.08,
) -> tuple[dict[str, Any] | None, list[str]]:
    trigger_rows = labels_data.get("trigger_outcomes")
    if not trigger_rows:
        return None, ["Trigger-level hidden labels are not available for this window."]

    notes = [
        "Timing metrics score only accounts assigned human_touch.",
        "Trigger quality uses recency-weighted relevance gain with exponential time decay.",
    ]

    trigger_event_map = {
        trigger["event_id"]: trigger
        for trigger in window_data["triggers"]
    }
    trigger_labels_by_account = {
        row["account_id"]: row
        for row in trigger_rows
    }
    touched_decisions = [
        decision
        for decision in effective_decisions
        if decision["chosen_action"] == "human_touch"
    ]

    if not touched_decisions:
        return {
            "TimingScore": 0.0,
            "trigger_accuracy_at_1": 0.0,
            "trigger_weighted_gain": 0.0,
            "oracle_trigger_weighted_gain": 0.0,
            "trigger_gain_ratio": 0.0,
            "timing_score_ratio": 0.0,
        }, notes

    accuracy_sum = 0.0
    selected_weighted_gain_total = 0.0
    oracle_weighted_gain_total = 0.0
    scored_accounts = 0

    for decision in touched_decisions:
        account_id = decision["account_id"]
        account_row = trigger_labels_by_account.get(account_id)
        if not account_row:
            continue

        weighted_trigger_rows = []
        for row in account_row["triggers"]:
            event = trigger_event_map[row["event_id"]]
            weighted_gain = row["relevance_gain"] * math.exp(-recency_lambda * event["recency_days"])
            weighted_trigger_rows.append(
                {
                    "event_id": row["event_id"],
                    "weighted_gain": weighted_gain,
                    "relevance_gain": row["relevance_gain"],
                }
            )

        ranked_triggers = sorted(
            weighted_trigger_rows,
            key=lambda row: (
                row["weighted_gain"],
                row["relevance_gain"],
                row["event_id"],
            ),
            reverse=True,
        )
        oracle_event_id = ranked_triggers[0]["event_id"]
        oracle_weighted_gain = ranked_triggers[0]["weighted_gain"]
        selected_event_id = decision["primary_trigger_event_id"]
        selected_weighted_gain = next(
            row["weighted_gain"]
            for row in weighted_trigger_rows
            if row["event_id"] == selected_event_id
        )

        accuracy_sum += 1.0 if selected_event_id == oracle_event_id else 0.0
        selected_weighted_gain_total += selected_weighted_gain
        oracle_weighted_gain_total += oracle_weighted_gain
        scored_accounts += 1

    if scored_accounts == 0:
        return None, ["No touched accounts had trigger-level hidden labels."]

    trigger_accuracy = safe_divide(accuracy_sum, scored_accounts)
    trigger_weighted_gain = safe_divide(selected_weighted_gain_total, scored_accounts)
    oracle_trigger_weighted_gain = safe_divide(oracle_weighted_gain_total, scored_accounts)
    trigger_gain_ratio = ratio_against_oracle(
        selected_weighted_gain_total,
        oracle_weighted_gain_total,
    )
    timing_score_ratio = 0.60 * trigger_accuracy + 0.40 * trigger_gain_ratio

    return {
        "TimingScore": 100.0 * timing_score_ratio,
        "trigger_accuracy_at_1": trigger_accuracy,
        "trigger_weighted_gain": trigger_weighted_gain,
        "oracle_trigger_weighted_gain": oracle_trigger_weighted_gain,
        "trigger_gain_ratio": trigger_gain_ratio,
        "timing_score_ratio": timing_score_ratio,
    }, notes


def compute_evidence_packet_metrics(
    window_data: dict[str, Any],
    effective_decisions: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    touched_decisions = [
        decision
        for decision in effective_decisions
        if decision["chosen_action"] == "human_touch"
    ]
    if not touched_decisions:
        return {
            "GroundingScore": 0.0,
            "grounded_claim_ratio": 0.0,
            "citation_relevance_ratio": 0.0,
            "trigger_citation_alignment": 0.0,
            "evidence_schema_accuracy": 0.0,
            "unsupported_claim_rate": 1.0,
            "grounding_score_ratio": 0.0,
        }, [
            "Grounding metrics score only accounts assigned human_touch.",
            "Optional free-text rationale fields are ignored for leaderboard scoring.",
        ]

    evidence_by_id = {
        document["doc_id"]: document
        for document in window_data["evidence"]
    }
    if not any(document.get("grounding_support") for document in evidence_by_id.values()):
        return None, [
            "Grounding annotations are not available for this window.",
            "Optional free-text rationale fields are ignored for leaderboard scoring.",
        ]

    trigger_by_id = {
        trigger["event_id"]: trigger
        for trigger in window_data["triggers"]
    }
    notes = [
        "Grounding metrics score only accounts assigned human_touch.",
        "Optional free-text rationale fields are ignored for leaderboard scoring.",
    ]

    grounded_claim_ratios: list[float] = []
    citation_relevance_ratios: list[float] = []
    trigger_alignments: list[float] = []
    schema_accuracies: list[float] = []

    for decision in touched_decisions:
        brief = decision["evidence_brief"]
        cited_docs = [
            evidence_by_id[doc_id]
            for doc_id in brief["citations"]
            if doc_id in evidence_by_id and evidence_by_id[doc_id]["allowed_for_grounding"]
        ]

        total_claims = 0
        supported_claims = 0
        claimed_pairs: list[tuple[str, str]] = []
        for brief_key, support_key, is_multi in GROUNDING_CODE_FIELDS:
            codes = brief[brief_key] if is_multi else [brief[brief_key]]
            total_claims += len(codes)
            for code in codes:
                claimed_pairs.append((support_key, code))
                if any(
                    supports_grounding_claim(
                        support_key,
                        code,
                        document,
                        trigger_by_id=trigger_by_id,
                    )
                    for document in cited_docs
                ):
                    supported_claims += 1

        grounded_claim_ratios.append(safe_divide(supported_claims, total_claims))

        relevant_citations = 0
        for document in cited_docs:
            if any(
                supports_grounding_claim(
                    support_key,
                    code,
                    document,
                    trigger_by_id=trigger_by_id,
                )
                for support_key, code in claimed_pairs
            ):
                relevant_citations += 1
        citation_relevance_ratios.append(safe_divide(relevant_citations, len(cited_docs)))

        selected_trigger_id = decision["primary_trigger_event_id"]
        trigger_refs = set(trigger_by_id[selected_trigger_id]["evidence_refs"])
        trigger_alignment = 1.0 if trigger_refs.intersection(brief["citations"]) else 0.0
        if not trigger_alignment and any(
            selected_trigger_id in document.get("grounding_support", {}).get("related_event_ids", [])
            for document in cited_docs
        ):
            trigger_alignment = 1.0
        trigger_alignments.append(trigger_alignment)

        schema_accuracies.append(1.0 if brief["citations"] and total_claims == supported_claims else 0.0)

    grounded_claim_ratio = average_or_none(grounded_claim_ratios) or 0.0
    citation_relevance_ratio = average_or_none(citation_relevance_ratios) or 0.0
    trigger_citation_alignment = average_or_none(trigger_alignments) or 0.0
    evidence_schema_accuracy = average_or_none(schema_accuracies) or 0.0
    relevance_ratio = 0.60 * citation_relevance_ratio + 0.40 * trigger_citation_alignment
    grounding_score_ratio = (
        0.60 * grounded_claim_ratio
        + 0.25 * relevance_ratio
        + 0.15 * evidence_schema_accuracy
    )

    return {
        "GroundingScore": 100.0 * grounding_score_ratio,
        "grounded_claim_ratio": grounded_claim_ratio,
        "citation_relevance_ratio": citation_relevance_ratio,
        "trigger_citation_alignment": trigger_citation_alignment,
        "evidence_schema_accuracy": evidence_schema_accuracy,
        "unsupported_claim_rate": 1.0 - grounded_claim_ratio,
        "grounding_score_ratio": grounding_score_ratio,
    }, notes


def compute_account_hidden_label_metrics(
    window_data: dict[str, Any],
    effective_decisions: list[dict[str, Any]],
    labels_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    budget = window_data["capacity_budget"]["human_sdr_actions"]
    labels_by_account = {
        row["account_id"]: row["potential_outcomes"]
        for row in labels_data["account_outcomes"]
    }

    notes: list[str] = [
        "Offline metrics use hidden account-level potential outcomes only.",
        "Human-touch uplift is measured against the best available non-human action.",
    ]

    selection_by_rank = sorted(
        (
            decision
            for decision in effective_decisions
            if decision["chosen_action"] == "human_touch"
        ),
        key=lambda decision: decision["human_touch_rank"],
    )
    top_k_selection = selection_by_rank[:budget]

    per_account_gains: list[dict[str, Any]] = []
    for account in window_data["accounts"]:
        account_id = account["account_id"]
        outcomes = labels_by_account[account_id]
        best_nonhuman_name, best_nonhuman = best_nonhuman_action(outcomes)
        per_account_gains.append(
            {
                "account_id": account_id,
                "best_nonhuman_action": best_nonhuman_name,
                "human_touch_pipeline_gain": outcomes["human_touch"]["pipeline_value"]
                - best_nonhuman["pipeline_value"],
                "human_touch_opp_gain": outcomes["human_touch"]["opp_prob"]
                - best_nonhuman["opp_prob"],
                "human_touch_meeting_gain": outcomes["human_touch"]["meeting_prob"]
                - best_nonhuman["meeting_prob"],
            }
        )

    gains_by_account = {
        item["account_id"]: item
        for item in per_account_gains
    }

    top_k_pipeline_gains = [
        gains_by_account[decision["account_id"]]["human_touch_pipeline_gain"]
        for decision in top_k_selection
    ]
    top_k_positive_opp = [
        1.0
        if gains_by_account[decision["account_id"]]["human_touch_opp_gain"] > 0
        else 0.0
        for decision in top_k_selection
    ]

    if len(top_k_pipeline_gains) < budget:
        top_k_pipeline_gains.extend([0.0] * (budget - len(top_k_pipeline_gains)))
        top_k_positive_opp.extend([0.0] * (budget - len(top_k_positive_opp)))

    oracle_accounts = sorted(
        per_account_gains,
        key=lambda item: (
            item["human_touch_pipeline_gain"],
            item["human_touch_opp_gain"],
            item["human_touch_meeting_gain"],
            item["account_id"],
        ),
        reverse=True,
    )
    oracle_top_k = oracle_accounts[:budget]
    oracle_pipeline_gains = [item["human_touch_pipeline_gain"] for item in oracle_top_k]

    uplift_at_capacity = safe_divide(sum(top_k_pipeline_gains), budget)
    oracle_uplift_at_capacity = safe_divide(sum(oracle_pipeline_gains), budget)
    uplift_ratio = ratio_against_oracle(uplift_at_capacity, oracle_uplift_at_capacity)

    precision_at_capacity = safe_divide(sum(top_k_positive_opp), budget)

    selected_dcg = compute_dcg([max(gain, 0.0) for gain in top_k_pipeline_gains])
    ideal_dcg = compute_dcg([max(gain, 0.0) for gain in oracle_pipeline_gains])
    ndcg_at_capacity = safe_divide(selected_dcg, ideal_dcg)

    decision_by_account = {
        decision["account_id"]: decision
        for decision in effective_decisions
    }

    action_policy_value = 0.0
    oracle_policy_value = 0.0
    for account in window_data["accounts"]:
        account_id = account["account_id"]
        outcomes = labels_by_account[account_id]
        wait_pipeline = outcomes["wait"]["pipeline_value"]

        chosen_action = decision_by_account[account_id]["chosen_action"]
        action_policy_value += outcomes[chosen_action]["pipeline_value"] - wait_pipeline

        best_nonhuman_name, best_nonhuman = best_nonhuman_action(outcomes)
        oracle_best_value = best_nonhuman["pipeline_value"]
        if account_id in {item["account_id"] for item in oracle_top_k}:
            oracle_best_value = max(
                oracle_best_value,
                outcomes["human_touch"]["pipeline_value"],
            )

        oracle_policy_value += oracle_best_value - wait_pipeline

    action_policy_ratio = ratio_against_oracle(action_policy_value, oracle_policy_value)

    metrics = {
        "precision_at_capacity": precision_at_capacity,
        "ndcg_at_capacity": ndcg_at_capacity,
        "uplift_at_capacity": uplift_at_capacity,
        "oracle_uplift_at_capacity": oracle_uplift_at_capacity,
        "uplift_ratio_at_capacity": uplift_ratio,
        "action_policy_value": action_policy_value,
        "oracle_action_policy_value": oracle_policy_value,
        "action_policy_ratio": action_policy_ratio,
    }

    return metrics, notes


def compute_contact_label_metrics(
    effective_decisions: list[dict[str, Any]],
    labels_data: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    contact_rows = labels_data.get("contact_outcomes")
    if not contact_rows:
        return None, ["Contact-level hidden labels are not available for this window."]

    notes = [
        "Contact metrics score only accounts assigned human_touch.",
        "Contact precision compares selected contacts to the oracle top-M contact set for each touched account.",
    ]

    contact_labels_by_account = {
        row["account_id"]: row
        for row in contact_rows
    }
    touched_decisions = [
        decision
        for decision in effective_decisions
        if decision["chosen_action"] == "human_touch"
    ]

    if not touched_decisions:
        return {
            "ContactScore": 0.0,
            "contact_precision_at_selected": 0.0,
            "contact_mrr": 0.0,
            "contact_meeting_lift_delta": 0.0,
            "oracle_contact_meeting_lift_delta": 0.0,
            "contact_lift_ratio": 0.0,
            "contact_score_ratio": 0.0,
        }, notes

    precision_sum = 0.0
    mrr_sum = 0.0
    selected_lift_total = 0.0
    oracle_lift_total = 0.0
    scored_accounts = 0

    for decision in touched_decisions:
        account_id = decision["account_id"]
        account_row = contact_labels_by_account.get(account_id)
        if not account_row:
            continue

        contacts = account_row["contacts"]
        gains_by_contact = {
            row["contact_id"]: row
            for row in contacts
        }
        ranked_contacts = sorted(
            contacts,
            key=lambda row: (
                row["pipeline_gain"],
                row["opp_gain"],
                row["meeting_gain"],
                row["contact_id"],
            ),
            reverse=True,
        )
        best_contact_id = ranked_contacts[0]["contact_id"]
        selected_contacts = decision["selected_contacts"]
        selection_size = len(selected_contacts)
        oracle_top_contact_ids = {
            row["contact_id"]
            for row in ranked_contacts[:selection_size]
        }

        precision_sum += safe_divide(
            sum(1 for contact_id in selected_contacts if contact_id in oracle_top_contact_ids),
            selection_size,
        )

        if best_contact_id in selected_contacts:
            mrr_sum += 1.0 / (selected_contacts.index(best_contact_id) + 1)

        selected_mean_meeting_gain = safe_divide(
            sum(gains_by_contact[contact_id]["meeting_gain"] for contact_id in selected_contacts),
            selection_size,
        )
        oracle_mean_meeting_gain = safe_divide(
            sum(row["meeting_gain"] for row in ranked_contacts[:selection_size]),
            selection_size,
        )
        default_contact_id = account_row["default_contact_id"]
        default_meeting_gain = gains_by_contact[default_contact_id]["meeting_gain"]

        selected_lift_total += max(selected_mean_meeting_gain - default_meeting_gain, 0.0)
        oracle_lift_total += max(oracle_mean_meeting_gain - default_meeting_gain, 0.0)
        scored_accounts += 1

    if scored_accounts == 0:
        return None, ["No touched accounts had contact-level hidden labels."]

    contact_precision = safe_divide(precision_sum, scored_accounts)
    contact_mrr = safe_divide(mrr_sum, scored_accounts)
    contact_meeting_lift_delta = safe_divide(selected_lift_total, scored_accounts)
    oracle_contact_meeting_lift_delta = safe_divide(oracle_lift_total, scored_accounts)
    contact_lift_ratio = ratio_against_oracle(selected_lift_total, oracle_lift_total)
    contact_score_ratio = (
        0.50 * contact_precision
        + 0.25 * contact_mrr
        + 0.25 * contact_lift_ratio
    )

    return {
        "ContactScore": 100.0 * contact_score_ratio,
        "contact_precision_at_selected": contact_precision,
        "contact_mrr": contact_mrr,
        "contact_meeting_lift_delta": contact_meeting_lift_delta,
        "oracle_contact_meeting_lift_delta": oracle_contact_meeting_lift_delta,
        "contact_lift_ratio": contact_lift_ratio,
        "contact_score_ratio": contact_score_ratio,
    }, notes


def compute_window_metrics(
    window_data: dict[str, Any],
    effective_decisions: list[dict[str, Any]],
    *,
    labels_data: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    metrics: dict[str, Any] = {}
    notes: list[str] = []

    evidence_metrics, evidence_notes = compute_evidence_packet_metrics(
        window_data,
        effective_decisions,
    )
    notes.extend(evidence_notes)
    if evidence_metrics is not None:
        metrics.update(evidence_metrics)

    if not labels_data:
        return metrics, notes

    account_metrics, metric_notes = compute_account_hidden_label_metrics(
        window_data,
        effective_decisions,
        labels_data,
    )
    metrics.update(account_metrics)
    notes.extend(metric_notes)

    trigger_metrics, trigger_notes = compute_trigger_label_metrics(
        window_data,
        effective_decisions,
        labels_data,
    )
    notes.extend(trigger_notes)
    if trigger_metrics is not None:
        metrics.update(trigger_metrics)

    contact_metrics, contact_notes = compute_contact_label_metrics(
        effective_decisions,
        labels_data,
    )
    notes.extend(contact_notes)
    if contact_metrics is not None:
        metrics.update(contact_metrics)

    fit_score = 100.0 * metrics["ndcg_at_capacity"]
    lift_score = 100.0 * metrics["uplift_ratio_at_capacity"]
    contact_score_ratio = metrics.get("contact_score_ratio")
    grounding_score_ratio = metrics.get("grounding_score_ratio")
    offline_component_weights = {
        "uplift": 0.45,
        "ndcg": 0.20,
        "action": 0.15,
        "contact": 0.10 if contact_score_ratio is not None else 0.0,
        "grounding": 0.10 if grounding_score_ratio is not None else 0.0,
    }
    present_weight = sum(offline_component_weights.values())
    offline_score = 100.0 * safe_divide(
        (
            offline_component_weights["uplift"] * metrics["uplift_ratio_at_capacity"]
            + offline_component_weights["ndcg"] * metrics["ndcg_at_capacity"]
            + offline_component_weights["action"] * metrics["action_policy_ratio"]
            + offline_component_weights["contact"] * (contact_score_ratio or 0.0)
            + offline_component_weights["grounding"] * (grounding_score_ratio or 0.0)
        ),
        present_weight,
    )
    metrics.update(
        {
            "EnterpriseAllocationScore": offline_score,
            "OfflineScore": offline_score,
            "FitScore": fit_score,
            "LiftScore": lift_score,
            "PolicyScore": None,
        }
    )

    return metrics, notes


def evaluate_window(
    window_data: dict[str, Any],
    submission_data: dict[str, Any],
    *,
    labels_data: dict[str, Any] | None = None,
    include_effective_decisions: bool = False,
    include_normalization: bool = True,
    normalization_seed: int = 0,
) -> dict[str, Any]:
    window_errors = validate_instance(window_data, "evaluation_window")
    submission_errors = validate_instance(submission_data, "model_output")
    hidden_label_errors = validate_instance(labels_data, "hidden_labels") if labels_data else []

    window_schema_valid = not window_errors
    submission_schema_valid = not submission_errors
    hidden_labels_schema_valid = not hidden_label_errors if labels_data else True
    window_id_match = window_data.get("window_id") == submission_data.get("window_id")
    labels_window_id_match = (
        window_data.get("window_id") == labels_data.get("window_id")
        if labels_data
        else True
    )

    report: dict[str, Any] = {
        "window_id": window_data.get("window_id"),
        "status": "ok",
        "scorable": False,
        "validation": {
            "window_schema_valid": window_schema_valid,
            "submission_schema_valid": submission_schema_valid,
            "hidden_labels_schema_valid": hidden_labels_schema_valid,
            "window_errors": window_errors,
            "submission_errors": submission_errors,
            "hidden_label_errors": hidden_label_errors,
            "window_id_match": window_id_match,
            "labels_window_id_match": labels_window_id_match,
        },
        "compliance": {
            "total_accounts": 0,
            "submitted_decisions": 0,
            "omitted_accounts_defaulted_to_wait": 0,
            "requested_action_counts": {},
            "effective_action_counts": {},
            "effective_human_touch_count": 0,
            "budgeted_human_touch_count": 0,
            "over_budget_downgraded_count": 0,
            "compliance_multiplier": 1.0,
        },
        "issues": [],
        "normalization": None,
        "terminology": build_sales_terminology(),
        "slice_diagnostics": None,
        "metrics": {
            "EnterpriseAllocationScore": None,
            "OfflineScore": None,
            "PolicyScore": None,
            "FitScore": None,
            "TimingScore": None,
            "ContactScore": None,
            "GroundingScore": None,
            "LiftScore": None,
        },
        "notes": [
            "Structure and compliance validation are always enforced before scoring.",
        ],
    }

    if not window_id_match:
        report["issues"].append(
            build_issue(
                "window_id_mismatch",
                "Submission window_id does not match the evaluation window.",
            )
        )

    if labels_data and not labels_window_id_match:
        report["issues"].append(
            build_issue(
                "labels_window_id_mismatch",
                "Hidden-label window_id does not match the evaluation window.",
            )
        )

    if not window_schema_valid:
        report["status"] = "invalid_window"
        return finalize_window_report(report)

    if not submission_schema_valid or not window_id_match:
        report["status"] = "invalid_submission"
        return finalize_window_report(report)

    if labels_data and (not hidden_labels_schema_valid or not labels_window_id_match):
        report["status"] = "invalid_hidden_labels"
        return finalize_window_report(report)

    final_decisions, issues, compliance = normalize_submission(window_data, submission_data)
    report["status"] = "ok_with_issues" if issues else "ok"
    report["scorable"] = True
    report["issues"] = issues
    report["compliance"] = compliance
    materialized_effective_decisions = materialize_effective_decisions(window_data, final_decisions)

    if labels_data:
        labels_by_account = {
            row["account_id"]
            for row in labels_data["account_outcomes"]
        }
        window_accounts = {
            account["account_id"]
            for account in window_data["accounts"]
        }
        missing_labels = sorted(window_accounts - labels_by_account)
        if missing_labels:
            report["status"] = "invalid_hidden_labels"
            report["issues"].append(
                build_issue(
                    "missing_account_labels",
                    f"Hidden labels are missing accounts: {missing_labels}",
                )
            )
            return finalize_window_report(report)

        touched_accounts = {
            decision["account_id"]
            for decision in materialized_effective_decisions
            if decision["chosen_action"] == "human_touch"
        }
        raw_trigger_rows = labels_data.get("trigger_outcomes", [])
        if raw_trigger_rows:
            unique_trigger_row_accounts = {
                row["account_id"]
                for row in raw_trigger_rows
            }
            if len(unique_trigger_row_accounts) != len(raw_trigger_rows):
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "duplicate_trigger_label_account",
                        "Trigger labels contain duplicate account rows.",
                    )
                )
                return finalize_window_report(report)

        trigger_rows = {
            row["account_id"]: row
            for row in raw_trigger_rows
        }
        missing_trigger_labels = sorted(
            account_id
            for account_id in touched_accounts
            if account_id not in trigger_rows
        )
        if missing_trigger_labels:
            report["status"] = "invalid_hidden_labels"
            report["issues"].append(
                build_issue(
                    "missing_trigger_labels",
                    f"Trigger labels are missing for touched accounts: {missing_trigger_labels}",
                )
            )
            return finalize_window_report(report)

        trigger_ids_in_window = {
            trigger["event_id"]
            for trigger in window_data["triggers"]
        }
        for account_id in touched_accounts:
            row = trigger_rows[account_id]
            trigger_ids = [trigger_row["event_id"] for trigger_row in row["triggers"]]
            unique_trigger_ids = set(trigger_ids)
            if len(unique_trigger_ids) != len(trigger_ids):
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "duplicate_trigger_label_id",
                        f"Trigger labels contain duplicate event IDs for account {account_id}.",
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

            invalid_window_trigger_ids = sorted(
                event_id
                for event_id in unique_trigger_ids
                if event_id not in trigger_ids_in_window
            )
            if invalid_window_trigger_ids:
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "invalid_trigger_label_id",
                        (
                            "Trigger labels reference events not present in the evaluation window: "
                            f"{invalid_window_trigger_ids}"
                        ),
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

            selected_trigger_id = next(
                decision["primary_trigger_event_id"]
                for decision in materialized_effective_decisions
                if decision["account_id"] == account_id and decision["chosen_action"] == "human_touch"
            )
            if selected_trigger_id not in unique_trigger_ids:
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "missing_selected_trigger_labels",
                        "Trigger labels are missing the selected trigger for a touched account.",
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

        raw_contact_rows = labels_data.get("contact_outcomes", [])
        if raw_contact_rows:
            unique_contact_row_accounts = {
                row["account_id"]
                for row in raw_contact_rows
            }
            if len(unique_contact_row_accounts) != len(raw_contact_rows):
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "duplicate_contact_label_account",
                        "Contact labels contain duplicate account rows.",
                    )
                )
                return finalize_window_report(report)

        contact_rows = {
            row["account_id"]: row
            for row in raw_contact_rows
        }
        missing_contact_labels = sorted(
            account_id
            for account_id in touched_accounts
            if account_id not in contact_rows
        )
        if missing_contact_labels:
            report["status"] = "invalid_hidden_labels"
            report["issues"].append(
                build_issue(
                    "missing_contact_labels",
                    f"Contact labels are missing for touched accounts: {missing_contact_labels}",
                )
            )
            return finalize_window_report(report)

        for account_id in touched_accounts:
            row = contact_rows[account_id]
            contact_ids = [contact_row["contact_id"] for contact_row in row["contacts"]]
            unique_contact_ids = set(contact_ids)

            if len(unique_contact_ids) != len(contact_ids):
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "duplicate_contact_label_id",
                        f"Contact labels contain duplicate contact IDs for account {account_id}.",
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

            if row["default_contact_id"] not in unique_contact_ids:
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "invalid_default_contact_label",
                        "Contact labels reference a default contact that is not present in the labeled contacts.",
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

            selected_contacts = next(
                decision["selected_contacts"]
                for decision in materialized_effective_decisions
                if decision["account_id"] == account_id and decision["chosen_action"] == "human_touch"
            )
            unlabeled_selected_contacts = sorted(
                contact_id
                for contact_id in selected_contacts
                if contact_id not in unique_contact_ids
            )
            if unlabeled_selected_contacts:
                report["status"] = "invalid_hidden_labels"
                report["issues"].append(
                    build_issue(
                        "missing_selected_contact_labels",
                        (
                            "Contact labels are missing selected contacts: "
                            f"{unlabeled_selected_contacts}"
                        ),
                        account_id=account_id,
                    )
                )
                return finalize_window_report(report)

    computed_metrics, metric_notes = compute_window_metrics(
        window_data,
        materialized_effective_decisions,
        labels_data=labels_data,
    )
    computed_metrics = apply_compliance_multiplier(computed_metrics, compliance)
    report["metrics"].update(computed_metrics)
    report["notes"].extend(metric_notes)
    if not math.isclose(compliance.get("compliance_multiplier", 1.0), 1.0):
        report["notes"].append(
            "Budget-sensitive score metrics include the compliance multiplier for over-budget submissions."
        )
    raw_slice_diagnostics = build_window_slice_diagnostics(
        window_data,
        materialized_effective_decisions,
        labels_data=labels_data,
    )
    report["slice_diagnostics"] = raw_slice_diagnostics

    if labels_data and include_normalization:
        random_baseline_submission = generate_window_submission(
            window_data,
            "random_within_icp",
            seed=normalization_seed,
        )
        random_baseline_report = evaluate_window(
            window_data,
            random_baseline_submission,
            labels_data=labels_data,
            include_effective_decisions=False,
            include_normalization=False,
            normalization_seed=normalization_seed,
        )
        report["normalization"] = build_score_normalization(
            report["metrics"],
            random_baseline_report["metrics"],
            baseline_name="random_within_icp",
        )
        report["slice_diagnostics"] = attach_slice_normalization(
            raw_slice_diagnostics,
            random_baseline_report.get("slice_diagnostics"),
            baseline_name="random_within_icp",
        )
        report["notes"].append(
            "Normalized metrics are reported relative to the random_within_icp baseline for this window."
        )

    if include_effective_decisions:
        report["effective_decisions"] = materialized_effective_decisions

    return finalize_window_report(report)


def evaluate_episode(
    episode_data: dict[str, Any],
    submission_data: dict[str, Any],
    labels_data: dict[str, Any] | None = None,
    *,
    include_window_reports: bool = False,
    include_normalization: bool = True,
    normalization_seed: int = 0,
) -> dict[str, Any]:
    episode_errors = validate_instance(episode_data, "policy_episode")
    submission_errors = validate_instance(submission_data, "policy_submission")
    labels_errors = validate_instance(labels_data, "policy_episode_labels") if labels_data else []

    episode_schema_valid = not episode_errors
    submission_schema_valid = not submission_errors
    labels_schema_valid = not labels_errors if labels_data else True
    episode_id_match = episode_data.get("episode_id") == submission_data.get("episode_id")
    labels_episode_id_match = (
        episode_data.get("episode_id") == labels_data.get("episode_id")
        if labels_data
        else True
    )

    report: dict[str, Any] = {
        "episode_id": episode_data.get("episode_id"),
        "status": "ok",
        "scorable": False,
        "validation": {
            "episode_schema_valid": episode_schema_valid,
            "submission_schema_valid": submission_schema_valid,
            "labels_schema_valid": labels_schema_valid,
            "episode_errors": episode_errors,
            "submission_errors": submission_errors,
            "labels_errors": labels_errors,
            "episode_id_match": episode_id_match,
            "labels_episode_id_match": labels_episode_id_match,
        },
        "issues": [],
        "normalization": None,
        "terminology": build_sales_terminology(),
        "slice_diagnostics": None,
        "metrics": {
            "EnterpriseAllocationScore": None,
            "OfflineScore": None,
            "PolicyScore": None,
            "FitScore": None,
            "TimingScore": None,
            "ContactScore": None,
            "GroundingScore": None,
            "LiftScore": None,
            "cumulative_incremental_pipeline": None,
            "cumulative_incremental_opps": None,
            "cumulative_incremental_meetings": None,
            "oracle_cumulative_incremental_pipeline": None,
            "oracle_cumulative_incremental_opps": None,
            "oracle_cumulative_incremental_meetings": None,
            "policy_pipeline_ratio": None,
            "policy_opp_ratio": None,
            "policy_meeting_ratio": None,
            "policy_regret": None,
            "policy_negative_regret_ratio": None,
        },
        "notes": [
            "Policy evaluation uses deterministic action transitions plus a greedy weekly oracle.",
            "Episode windows are evaluated in listed order.",
        ],
    }

    if not episode_id_match:
        report["issues"].append(
            build_issue(
                "episode_id_mismatch",
                "Submission episode_id does not match the policy episode.",
            )
        )

    if labels_data and not labels_episode_id_match:
        report["issues"].append(
            build_issue(
                "labels_episode_id_mismatch",
                "Policy label episode_id does not match the policy episode.",
            )
        )

    if not episode_schema_valid:
        report["status"] = "invalid_episode"
        return finalize_episode_report(report)

    if not submission_schema_valid or not episode_id_match:
        report["status"] = "invalid_submission"
        return finalize_episode_report(report)

    if labels_data and (not labels_schema_valid or not labels_episode_id_match):
        report["status"] = "invalid_policy_labels"
        return finalize_episode_report(report)

    windows = episode_data["windows"]
    submissions = submission_data["submissions"]

    window_ids = [window["window_id"] for window in windows]
    submission_window_ids = [submission["window_id"] for submission in submissions]
    if len(set(window_ids)) != len(window_ids):
        report["status"] = "invalid_episode"
        report["issues"].append(
            build_issue(
                "duplicate_episode_window_id",
                "Policy episode contains duplicate window IDs.",
            )
        )
        return finalize_episode_report(report)

    if len(set(submission_window_ids)) != len(submission_window_ids):
        report["status"] = "invalid_submission"
        report["issues"].append(
            build_issue(
                "duplicate_policy_submission_window_id",
                "Policy submission contains duplicate window IDs.",
            )
        )
        return finalize_episode_report(report)

    missing_submission_windows = sorted(set(window_ids) - set(submission_window_ids))
    extra_submission_windows = sorted(set(submission_window_ids) - set(window_ids))
    if missing_submission_windows or extra_submission_windows:
        report["status"] = "invalid_submission"
        if missing_submission_windows:
            report["issues"].append(
                build_issue(
                    "missing_policy_submission_windows",
                    f"Missing policy submission windows: {missing_submission_windows}",
                )
            )
        if extra_submission_windows:
            report["issues"].append(
                build_issue(
                    "extra_policy_submission_windows",
                    f"Unknown policy submission windows: {extra_submission_windows}",
                )
            )
        return finalize_episode_report(report)

    labels_by_window_id: dict[str, Any] = {}
    transition_rows_by_window_id: dict[str, Any] = {}
    if labels_data:
        label_windows = labels_data["windows"]
        label_window_ids = [window["window_id"] for window in label_windows]
        if len(set(label_window_ids)) != len(label_window_ids):
            report["status"] = "invalid_policy_labels"
            report["issues"].append(
                build_issue(
                    "duplicate_policy_label_window_id",
                    "Policy labels contain duplicate window IDs.",
                )
            )
            return finalize_episode_report(report)

        missing_label_windows = sorted(set(window_ids) - set(label_window_ids))
        extra_label_windows = sorted(set(label_window_ids) - set(window_ids))
        if missing_label_windows or extra_label_windows:
            report["status"] = "invalid_policy_labels"
            if missing_label_windows:
                report["issues"].append(
                    build_issue(
                        "missing_policy_label_windows",
                        f"Missing policy label windows: {missing_label_windows}",
                    )
                )
            if extra_label_windows:
                report["issues"].append(
                    build_issue(
                        "extra_policy_label_windows",
                        f"Unknown policy label windows: {extra_label_windows}",
                    )
                )
            return finalize_episode_report(report)

        for label_window in label_windows:
            if label_window["window_id"] != label_window["labels"]["window_id"]:
                report["status"] = "invalid_policy_labels"
                report["issues"].append(
                    build_issue(
                        "policy_label_window_id_mismatch",
                        "Nested hidden-label window_id does not match its parent policy label window.",
                        window_id=label_window["window_id"],
                    )
                )
                return finalize_episode_report(report)

            labels_by_window_id[label_window["window_id"]] = label_window["labels"]
            transition_rows_by_window_id[label_window["window_id"]] = label_window["policy_transitions"]

    window_map = {
        window["window_id"]: window
        for window in windows
    }
    submission_map = {
        submission["window_id"]: submission
        for submission in submissions
    }

    agent_state: dict[str, dict[str, Any]] = {}
    oracle_state: dict[str, dict[str, Any]] = {}
    cumulative_agent = {"pipeline": 0.0, "opps": 0.0, "meetings": 0.0}
    cumulative_oracle = {"pipeline": 0.0, "opps": 0.0, "meetings": 0.0}
    window_reports: list[dict[str, Any]] = []
    aggregated_issues: list[dict[str, Any]] = []
    score_accumulators: dict[str, list[float]] = {
        "OfflineScore": [],
        "FitScore": [],
        "TimingScore": [],
        "ContactScore": [],
        "GroundingScore": [],
        "LiftScore": [],
    }
    slice_aggregates = initialize_episode_slice_aggregates()

    for window_index, window_id in enumerate(window_ids):
        window_data = window_map[window_id]
        submission_window = submission_map[window_id]
        window_labels = labels_by_window_id.get(window_id) if labels_data else None

        window_report = evaluate_window(
            window_data,
            submission_window,
            labels_data=window_labels,
            include_effective_decisions=True,
        )
        if window_report["status"].startswith("invalid"):
            report["status"] = window_report["status"]
            for issue in window_report.get("issues", []):
                if "window_id" not in issue:
                    issue = {
                        **issue,
                        "window_id": window_id,
                    }
                aggregated_issues.append(issue)
            report["issues"] = aggregated_issues
            return finalize_episode_report(report)

        effective_decisions = window_report["effective_decisions"]
        adjusted_decisions, state_issues = apply_policy_state_constraints(
            window_id,
            window_index,
            effective_decisions,
            agent_state,
        )
        aggregated_issues.extend(window_report["issues"])
        aggregated_issues.extend(state_issues)
        eligible_account_ids = {
            account["account_id"]
            for account in window_data["accounts"]
            if is_account_eligible(agent_state, account["account_id"], window_index)
        }
        adjusted_window_metrics: dict[str, Any] = {}
        if eligible_account_ids:
            eligible_window_data = subset_window_to_accounts(window_data, eligible_account_ids)
            eligible_window_labels = subset_hidden_labels_to_accounts(window_labels, eligible_account_ids)
            eligible_adjusted_decisions = [
                decision
                for decision in adjusted_decisions
                if decision["account_id"] in eligible_account_ids
            ]
            adjusted_window_metrics, _ = compute_window_metrics(
                eligible_window_data,
                eligible_adjusted_decisions,
                labels_data=eligible_window_labels,
            )
            window_slice_diagnostics = build_window_slice_diagnostics(
                eligible_window_data,
                eligible_adjusted_decisions,
                labels_data=eligible_window_labels,
            )
        else:
            eligible_window_data = None
            eligible_window_labels = None
            eligible_adjusted_decisions = []
            window_slice_diagnostics = {
                "notes": build_slice_diagnostic_notes(),
                "dimensions": {
                    dimension: {}
                    for dimension in SLICE_DIMENSIONS
                },
            }
        for metric_name, metric_values in score_accumulators.items():
            value = adjusted_window_metrics.get(metric_name)
            if value is not None:
                metric_values.append(value)

        if labels_data:
            transition_rows = transition_rows_by_window_id[window_id]
            transition_map: dict[str, dict[str, Any]] = {}
            window_account_ids = {
                account["account_id"]
                for account in window_data["accounts"]
            }
            if len(transition_rows) != len(window_account_ids):
                report["status"] = "invalid_policy_labels"
                aggregated_issues.append(
                    build_issue(
                        "missing_policy_transitions",
                        "Policy transitions must contain exactly one row per account in the window.",
                        window_id=window_id,
                    )
                )
                report["issues"] = aggregated_issues
                return finalize_episode_report(report)

            for row in transition_rows:
                account_id = row["account_id"]
                if account_id in transition_map:
                    report["status"] = "invalid_policy_labels"
                    aggregated_issues.append(
                        build_issue(
                            "duplicate_policy_transition_account",
                            "Policy transitions contain duplicate account rows.",
                            account_id=account_id,
                            window_id=window_id,
                        )
                    )
                    report["issues"] = aggregated_issues
                    return finalize_episode_report(report)
                transition_map[account_id] = row["actions"]

            if set(transition_map) != window_account_ids:
                report["status"] = "invalid_policy_labels"
                aggregated_issues.append(
                    build_issue(
                        "incomplete_policy_transition_map",
                        "Policy transitions do not cover every account in the window.",
                        window_id=window_id,
                    )
                )
                report["issues"] = aggregated_issues
                return finalize_episode_report(report)

            agent_current_modifiers = extract_current_outcome_modifiers(agent_state)
            oracle_current_modifiers = extract_current_outcome_modifiers(oracle_state)
            adjusted_agent_labels = apply_outcome_modifiers_to_labels(window_labels, agent_current_modifiers)
            adjusted_oracle_labels = apply_outcome_modifiers_to_labels(window_labels, oracle_current_modifiers)

            oracle_decisions = compute_greedy_oracle_window_decisions(
                window_data,
                adjusted_oracle_labels,
                oracle_state,
                window_index,
            )

            agent_values = compute_window_incremental_values(adjusted_decisions, adjusted_agent_labels)
            oracle_values = compute_window_incremental_values(oracle_decisions, adjusted_oracle_labels)
            for key in cumulative_agent:
                cumulative_agent[key] += agent_values[key]
                cumulative_oracle[key] += oracle_values[key]

            agent_slice_incrementals: dict[tuple[str, str], dict[str, float]] = {}
            oracle_slice_incrementals: dict[tuple[str, str], dict[str, float]] = {}
            if eligible_window_data is not None:
                slice_values_by_account = derive_account_slice_values(eligible_window_data)
                for dimension in SLICE_DIMENSIONS:
                    grouped_account_ids: dict[str, set[str]] = {}
                    for account in eligible_window_data["accounts"]:
                        account_id = account["account_id"]
                        slice_value = slice_values_by_account[account_id][dimension]
                        grouped_account_ids.setdefault(slice_value, set()).add(account_id)

                    for slice_value, account_ids in grouped_account_ids.items():
                        slice_agent_labels = subset_hidden_labels_to_accounts(
                            adjusted_agent_labels,
                            account_ids,
                        )
                        slice_oracle_labels = subset_hidden_labels_to_accounts(
                            adjusted_oracle_labels,
                            account_ids,
                        )
                        agent_slice_incrementals[(dimension, slice_value)] = compute_window_incremental_values(
                            subset_effective_decisions_to_accounts(adjusted_decisions, account_ids),
                            slice_agent_labels,
                        )
                        oracle_slice_incrementals[(dimension, slice_value)] = compute_window_incremental_values(
                            subset_effective_decisions_to_accounts(oracle_decisions, account_ids),
                            slice_oracle_labels,
                        )

            update_episode_slice_aggregates(
                slice_aggregates,
                window_slice_diagnostics,
                agent_slice_incrementals=agent_slice_incrementals,
                oracle_slice_incrementals=oracle_slice_incrementals,
            )

            apply_policy_transitions(window_index, adjusted_decisions, transition_map, agent_state)
            apply_policy_transitions(window_index, oracle_decisions, transition_map, oracle_state)
        else:
            update_episode_slice_aggregates(
                slice_aggregates,
                window_slice_diagnostics,
            )

        window_summary = {
            "window_id": window_id,
            "status": "ok_with_issues" if (window_report["issues"] or state_issues) else "ok",
            "issue_count": len(window_report["issues"]) + len(state_issues),
            "compliance": window_report["compliance"],
            "slice_diagnostics": window_slice_diagnostics,
        }
        window_summary["metrics"] = {
            metric_name: adjusted_window_metrics.get(metric_name)
            for metric_name in (
                "OfflineScore",
                "FitScore",
                "TimingScore",
                "ContactScore",
                "GroundingScore",
                "LiftScore",
            )
            if adjusted_window_metrics.get(metric_name) is not None
        }
        if labels_data:
            window_summary["agent_incremental_pipeline"] = agent_values["pipeline"]
            window_summary["oracle_incremental_pipeline"] = oracle_values["pipeline"]

        window_reports.append(window_summary)

    report["status"] = "ok_with_issues" if aggregated_issues else "ok"
    report["scorable"] = True
    report["issues"] = aggregated_issues
    report["metrics"].update(
        {
            metric_name: average_or_none(values)
            for metric_name, values in score_accumulators.items()
        }
    )
    raw_episode_slice_diagnostics = finalize_episode_slice_diagnostics(
        slice_aggregates,
        include_policy_metrics=bool(labels_data),
    )
    report["slice_diagnostics"] = raw_episode_slice_diagnostics

    if labels_data:
        pipeline_ratio = ratio_against_oracle(cumulative_agent["pipeline"], cumulative_oracle["pipeline"])
        opp_ratio = ratio_against_oracle(cumulative_agent["opps"], cumulative_oracle["opps"])
        meeting_ratio = ratio_against_oracle(cumulative_agent["meetings"], cumulative_oracle["meetings"])
        policy_regret = max(cumulative_oracle["pipeline"] - cumulative_agent["pipeline"], 0.0)
        negative_regret_ratio = ratio_against_oracle(
            cumulative_oracle["pipeline"] - policy_regret,
            cumulative_oracle["pipeline"],
        )
        policy_score = 100.0 * (
            0.50 * pipeline_ratio
            + 0.25 * opp_ratio
            + 0.15 * meeting_ratio
            + 0.10 * negative_regret_ratio
        )
        enterprise_allocation_score = policy_score
        if report["metrics"]["OfflineScore"] is not None:
            enterprise_allocation_score = (
                0.60 * report["metrics"]["OfflineScore"]
                + 0.40 * policy_score
            )
        report["metrics"].update(
            {
                "EnterpriseAllocationScore": enterprise_allocation_score,
                "PolicyScore": policy_score,
                "cumulative_incremental_pipeline": cumulative_agent["pipeline"],
                "cumulative_incremental_opps": cumulative_agent["opps"],
                "cumulative_incremental_meetings": cumulative_agent["meetings"],
                "oracle_cumulative_incremental_pipeline": cumulative_oracle["pipeline"],
                "oracle_cumulative_incremental_opps": cumulative_oracle["opps"],
                "oracle_cumulative_incremental_meetings": cumulative_oracle["meetings"],
                "policy_pipeline_ratio": pipeline_ratio,
                "policy_opp_ratio": opp_ratio,
                "policy_meeting_ratio": meeting_ratio,
                "policy_regret": policy_regret,
                "policy_negative_regret_ratio": negative_regret_ratio,
            }
        )
        if include_normalization:
            random_baseline_submission = generate_episode_submission(
                episode_data,
                "random_within_icp",
                seed=normalization_seed,
            )
            random_baseline_report = evaluate_episode(
                episode_data,
                random_baseline_submission,
                labels_data,
                include_window_reports=False,
                include_normalization=False,
                normalization_seed=normalization_seed,
            )
            report["normalization"] = build_score_normalization(
                report["metrics"],
                random_baseline_report["metrics"],
                baseline_name="random_within_icp",
            )
            report["slice_diagnostics"] = attach_slice_normalization(
                raw_episode_slice_diagnostics,
                random_baseline_report.get("slice_diagnostics"),
                baseline_name="random_within_icp",
            )
            report["notes"].append(
                "Normalized metrics are reported relative to the random_within_icp baseline for this episode."
            )

    if include_window_reports:
        report["window_reports"] = window_reports

    return finalize_episode_report(report)


def summarize_metric_maps(
    metric_maps: list[dict[str, Any]],
    *,
    reducer: str,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ROBUSTNESS_SUMMARY_METRICS:
        values = [
            metric_map[metric_name]
            for metric_map in metric_maps
            if isinstance(metric_map.get(metric_name), (int, float))
            and not isinstance(metric_map.get(metric_name), bool)
        ]
        if not values:
            continue
        if reducer == "average":
            summary[metric_name] = sum(values) / len(values)
        elif reducer == "min":
            summary[metric_name] = min(values)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return summary


def build_robustness_case_summary(
    case: dict[str, Any],
    report: dict[str, Any],
    *,
    include_case_report: bool,
) -> dict[str, Any]:
    normalized_metrics = {}
    if report.get("normalization"):
        normalized_metrics = report["normalization"].get("normalized_metrics", {})

    summary = {
        "case_id": case["case_id"],
        "window_id": case["window"]["window_id"],
        "robustness_type": case["robustness_type"],
        "description": case.get("description"),
        "holdout_dimension": case.get("holdout_dimension"),
        "holdout_values": case.get("holdout_values", []),
        "shift_tags": case.get("shift_tags", []),
        "status": report["status"],
        "scorable": report["scorable"],
        "metrics": {
            metric_name: report["metrics"].get(metric_name)
            for metric_name in ROBUSTNESS_SUMMARY_METRICS
            if report["metrics"].get(metric_name) is not None
        },
        "normalized_metrics": normalized_metrics,
        "issue_count": len(report.get("issues", [])),
    }
    if include_case_report:
        summary["report"] = report
    return summary


def build_robustness_pack_summary(
    case_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_metric_maps = [case["metrics"] for case in case_summaries]
    normalized_metric_maps = [case["normalized_metrics"] for case in case_summaries]
    average_metrics = summarize_metric_maps(raw_metric_maps, reducer="average")
    worst_case_metrics = summarize_metric_maps(raw_metric_maps, reducer="min")
    average_normalized_metrics = summarize_metric_maps(normalized_metric_maps, reducer="average")
    worst_case_normalized_metrics = summarize_metric_maps(normalized_metric_maps, reducer="min")

    robustness_score = None
    score_basis = None
    if "EnterpriseAllocationScore" in average_normalized_metrics:
        score_basis = "normalized"
        robustness_score = (
            0.70 * average_normalized_metrics["EnterpriseAllocationScore"]
            + 0.30 * worst_case_normalized_metrics["EnterpriseAllocationScore"]
        )
    elif "EnterpriseAllocationScore" in average_metrics:
        score_basis = "raw"
        robustness_score = (
            0.70 * average_metrics["EnterpriseAllocationScore"]
            + 0.30 * worst_case_metrics["EnterpriseAllocationScore"]
        )

    underperforming_cases = []
    for case in case_summaries:
        normalized_enterprise_score = case["normalized_metrics"].get("EnterpriseAllocationScore")
        if normalized_enterprise_score is not None and normalized_enterprise_score <= 0.0:
            underperforming_cases.append(
                {
                    "case_id": case["case_id"],
                    "robustness_type": case["robustness_type"],
                    "normalized_enterprise_allocation_score": normalized_enterprise_score,
                }
            )

    robustness_type_breakdown: dict[str, Any] = {}
    for robustness_type in ROBUSTNESS_CASE_TYPES:
        type_cases = [
            case
            for case in case_summaries
            if case["robustness_type"] == robustness_type
        ]
        if not type_cases:
            continue
        robustness_type_breakdown[robustness_type] = {
            "case_count": len(type_cases),
            "average_metrics": summarize_metric_maps(
                [case["metrics"] for case in type_cases],
                reducer="average",
            ),
            "worst_case_metrics": summarize_metric_maps(
                [case["metrics"] for case in type_cases],
                reducer="min",
            ),
            "average_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in type_cases],
                reducer="average",
            ),
            "worst_case_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in type_cases],
                reducer="min",
            ),
        }

    holdout_dimension_breakdown: dict[str, Any] = {}
    holdout_dimensions = sorted(
        {
            case["holdout_dimension"]
            for case in case_summaries
            if case.get("holdout_dimension")
        }
    )
    for dimension in holdout_dimensions:
        dimension_cases = [
            case
            for case in case_summaries
            if case.get("holdout_dimension") == dimension
        ]
        holdout_dimension_breakdown[dimension] = {
            "case_count": len(dimension_cases),
            "average_metrics": summarize_metric_maps(
                [case["metrics"] for case in dimension_cases],
                reducer="average",
            ),
            "worst_case_metrics": summarize_metric_maps(
                [case["metrics"] for case in dimension_cases],
                reducer="min",
            ),
            "average_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in dimension_cases],
                reducer="average",
            ),
            "worst_case_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in dimension_cases],
                reducer="min",
            ),
        }

    shift_tag_breakdown: dict[str, Any] = {}
    shift_tags = sorted(
        {
            tag
            for case in case_summaries
            for tag in case.get("shift_tags", [])
        }
    )
    for tag in shift_tags:
        tag_cases = [
            case
            for case in case_summaries
            if tag in case.get("shift_tags", [])
        ]
        shift_tag_breakdown[tag] = {
            "case_count": len(tag_cases),
            "average_metrics": summarize_metric_maps(
                [case["metrics"] for case in tag_cases],
                reducer="average",
            ),
            "worst_case_metrics": summarize_metric_maps(
                [case["metrics"] for case in tag_cases],
                reducer="min",
            ),
            "average_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in tag_cases],
                reducer="average",
            ),
            "worst_case_normalized_metrics": summarize_metric_maps(
                [case["normalized_metrics"] for case in tag_cases],
                reducer="min",
            ),
        }

    return {
        "case_count": len(case_summaries),
        "status_counts": dict(Counter(case["status"] for case in case_summaries)),
        "average_metrics": average_metrics,
        "worst_case_metrics": worst_case_metrics,
        "average_normalized_metrics": average_normalized_metrics,
        "worst_case_normalized_metrics": worst_case_normalized_metrics,
        "robustness_score": robustness_score,
        "robustness_score_basis": score_basis,
        "underperforming_cases": underperforming_cases,
        "robustness_type_breakdown": robustness_type_breakdown,
        "holdout_dimension_breakdown": holdout_dimension_breakdown,
        "shift_tag_breakdown": shift_tag_breakdown,
    }


def evaluate_robustness_suite(
    suite_data: dict[str, Any],
    submission_data: dict[str, Any],
    *,
    include_case_reports: bool = False,
    include_normalization: bool = True,
    normalization_seed: int = 0,
) -> dict[str, Any]:
    suite_errors = validate_instance(suite_data, "robustness_suite")
    submission_errors = validate_instance(submission_data, "robustness_submission")

    suite_schema_valid = not suite_errors
    submission_schema_valid = not submission_errors
    suite_id_match = suite_data.get("suite_id") == submission_data.get("suite_id")

    report: dict[str, Any] = {
        "suite_id": suite_data.get("suite_id"),
        "status": "ok",
        "scorable": False,
        "validation": {
            "suite_schema_valid": suite_schema_valid,
            "submission_schema_valid": submission_schema_valid,
            "suite_errors": suite_errors,
            "submission_errors": submission_errors,
            "suite_id_match": suite_id_match,
        },
        "issues": [],
        "terminology": build_sales_terminology(),
        "summary": {
            "case_count": 0,
            "status_counts": {},
            "average_metrics": {},
            "worst_case_metrics": {},
            "average_normalized_metrics": {},
            "worst_case_normalized_metrics": {},
            "robustness_score": None,
            "robustness_score_basis": None,
            "underperforming_cases": [],
            "robustness_type_breakdown": {},
            "holdout_dimension_breakdown": {},
            "shift_tag_breakdown": {},
        },
        "notes": [
            "Robustness suites are diagnostic and do not change the public leaderboard score.",
            "Each robustness case is scored as an independent offline window.",
        ],
        "cases": [],
    }

    if not suite_id_match:
        report["issues"].append(
            build_issue(
                "suite_id_mismatch",
                "Robustness submission suite_id does not match the robustness suite.",
            )
        )

    if not suite_schema_valid:
        report["status"] = "invalid_suite"
        return finalize_robustness_report(report)

    if not submission_schema_valid or not suite_id_match:
        report["status"] = "invalid_submission"
        return finalize_robustness_report(report)

    cases = suite_data["cases"]
    submissions = submission_data["submissions"]
    case_ids = [case["case_id"] for case in cases]
    submission_case_ids = [item["case_id"] for item in submissions]

    if len(set(case_ids)) != len(case_ids):
        report["status"] = "invalid_suite"
        report["issues"].append(
            build_issue(
                "duplicate_robustness_case_id",
                "Robustness suite contains duplicate case IDs.",
            )
        )
        return finalize_robustness_report(report)

    if len(set(submission_case_ids)) != len(submission_case_ids):
        report["status"] = "invalid_submission"
        report["issues"].append(
            build_issue(
                "duplicate_robustness_submission_case_id",
                "Robustness submission contains duplicate case IDs.",
            )
        )
        return finalize_robustness_report(report)

    missing_submissions = sorted(set(case_ids) - set(submission_case_ids))
    extra_submissions = sorted(set(submission_case_ids) - set(case_ids))
    if missing_submissions or extra_submissions:
        report["status"] = "invalid_submission"
        if missing_submissions:
            report["issues"].append(
                build_issue(
                    "missing_robustness_case_submissions",
                    f"Missing robustness submissions for cases: {missing_submissions}",
                )
            )
        if extra_submissions:
            report["issues"].append(
                build_issue(
                    "extra_robustness_case_submissions",
                    f"Unknown robustness submissions supplied for cases: {extra_submissions}",
                )
            )
        return finalize_robustness_report(report)

    invalid_holdout_cases = [
        case["case_id"]
        for case in cases
        if case["robustness_type"] == "heldout_slice"
        and (
            not case.get("holdout_dimension")
            or not case.get("holdout_values")
        )
    ]
    if invalid_holdout_cases:
        report["status"] = "invalid_suite"
        report["issues"].append(
            build_issue(
                "invalid_holdout_case_metadata",
                (
                    "Held-out slice cases must declare holdout_dimension and holdout_values: "
                    f"{invalid_holdout_cases}"
                ),
            )
        )
        return finalize_robustness_report(report)

    submission_by_case_id = {
        item["case_id"]: item["submission"]
        for item in submissions
    }

    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        case_report = evaluate_window(
            case["window"],
            submission_by_case_id[case["case_id"]],
            labels_data=case.get("labels"),
            include_normalization=include_normalization,
            normalization_seed=normalization_seed,
        )
        if case_report["status"].startswith("invalid"):
            report["status"] = "invalid_submission"
            report["issues"].append(
                build_issue(
                    "invalid_robustness_case_submission",
                    (
                        f"Robustness case {case['case_id']} produced an invalid report status "
                        f"{case_report['status']}."
                    ),
                )
            )
            if include_case_reports:
                report["cases"].append(
                    build_robustness_case_summary(
                        case,
                        case_report,
                        include_case_report=True,
                    )
                )
            return finalize_robustness_report(report)

        if case_report["issues"]:
            report["issues"].extend(
                {
                    **issue,
                    "window_id": issue.get("window_id", case["window"]["window_id"]),
                }
                for issue in case_report["issues"]
            )

        case_summaries.append(
            build_robustness_case_summary(
                case,
                case_report,
                include_case_report=include_case_reports,
            )
        )

    report["status"] = "ok_with_issues" if report["issues"] else "ok"
    report["scorable"] = True
    report["cases"] = case_summaries
    report["summary"] = build_robustness_pack_summary(case_summaries)
    return finalize_robustness_report(report)


def generate_robustness_suite_submission(
    suite_data: dict[str, Any],
    baseline_name: str,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    submissions = []
    for index, case in enumerate(suite_data["cases"]):
        submissions.append(
            {
                "case_id": case["case_id"],
                "submission": generate_window_submission(
                    case["window"],
                    baseline_name,
                    seed=seed + index,
                ),
            }
        )
    return {
        "suite_id": suite_data["suite_id"],
        "submissions": submissions,
    }


def evaluate_robustness_suite_baseline(
    suite_data: dict[str, Any],
    baseline_name: str,
    *,
    include_case_reports: bool = False,
    include_submission: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    submission = generate_robustness_suite_submission(
        suite_data,
        baseline_name,
        seed=seed,
    )
    result = {
        "mode": "robustness_suite_baseline",
        "baseline_name": baseline_name,
        "report": evaluate_robustness_suite(
            suite_data,
            submission,
            include_case_reports=include_case_reports,
            include_normalization=include_normalization,
            normalization_seed=seed if normalization_seed is None else normalization_seed,
        ),
    }
    if include_submission:
        result["submission"] = submission
    return result


def evaluate_all_robustness_suite_baselines(
    suite_data: dict[str, Any],
    *,
    include_case_reports: bool = False,
    include_submissions: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    baselines: list[dict[str, Any]] = []
    for baseline_name in BASELINE_NAMES:
        submission = generate_robustness_suite_submission(
            suite_data,
            baseline_name,
            seed=seed,
        )
        entry = {
            "baseline_name": baseline_name,
            "report": evaluate_robustness_suite(
                suite_data,
                submission,
                include_case_reports=include_case_reports,
                include_normalization=include_normalization,
                normalization_seed=seed if normalization_seed is None else normalization_seed,
            ),
        }
        if include_submissions:
            entry["submission"] = submission
        baselines.append(entry)
    return {
        "mode": "robustness_suite_baseline_pack",
        "suite_id": suite_data["suite_id"],
        "baselines": baselines,
    }


def evaluate_window_baseline(
    window_data: dict[str, Any],
    baseline_name: str,
    *,
    labels_data: dict[str, Any] | None = None,
    include_effective_decisions: bool = False,
    include_submission: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    submission = generate_window_submission(window_data, baseline_name, seed=seed)
    result = {
        "mode": "window_baseline",
        "baseline_name": baseline_name,
        "report": evaluate_window(
            window_data,
            submission,
            labels_data=labels_data,
            include_effective_decisions=include_effective_decisions,
            include_normalization=include_normalization,
            normalization_seed=seed if normalization_seed is None else normalization_seed,
        ),
    }
    if include_submission:
        result["submission"] = submission
    return result


def evaluate_all_window_baselines(
    window_data: dict[str, Any],
    *,
    labels_data: dict[str, Any] | None = None,
    include_effective_decisions: bool = False,
    include_submissions: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    baseline_submissions = generate_all_window_submissions(window_data, seed=seed)
    baselines: list[dict[str, Any]] = []
    for baseline_name in BASELINE_NAMES:
        submission = baseline_submissions[baseline_name]
        entry = {
            "baseline_name": baseline_name,
            "report": evaluate_window(
                window_data,
                submission,
                labels_data=labels_data,
                include_effective_decisions=include_effective_decisions,
                include_normalization=include_normalization,
                normalization_seed=seed if normalization_seed is None else normalization_seed,
            ),
        }
        if include_submissions:
            entry["submission"] = submission
        baselines.append(entry)
    return {
        "mode": "window_baseline_pack",
        "window_id": window_data["window_id"],
        "baselines": baselines,
    }


def evaluate_episode_baseline(
    episode_data: dict[str, Any],
    baseline_name: str,
    *,
    labels_data: dict[str, Any] | None = None,
    include_window_reports: bool = False,
    include_submission: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    submission = generate_episode_submission(episode_data, baseline_name, seed=seed)
    result = {
        "mode": "episode_baseline",
        "baseline_name": baseline_name,
        "report": evaluate_episode(
            episode_data,
            submission,
            labels_data,
            include_window_reports=include_window_reports,
            include_normalization=include_normalization,
            normalization_seed=seed if normalization_seed is None else normalization_seed,
        ),
    }
    if include_submission:
        result["submission"] = submission
    return result


def evaluate_all_episode_baselines(
    episode_data: dict[str, Any],
    *,
    labels_data: dict[str, Any] | None = None,
    include_window_reports: bool = False,
    include_submissions: bool = False,
    seed: int = 0,
    include_normalization: bool = True,
    normalization_seed: int | None = None,
) -> dict[str, Any]:
    baseline_submissions = generate_all_episode_submissions(episode_data, seed=seed)
    baselines: list[dict[str, Any]] = []
    for baseline_name in BASELINE_NAMES:
        submission = baseline_submissions[baseline_name]
        entry = {
            "baseline_name": baseline_name,
            "report": evaluate_episode(
                episode_data,
                submission,
                labels_data,
                include_window_reports=include_window_reports,
                include_normalization=include_normalization,
                normalization_seed=seed if normalization_seed is None else normalization_seed,
            ),
        }
        if include_submissions:
            entry["submission"] = submission
        baselines.append(entry)
    return {
        "mode": "episode_baseline_pack",
        "episode_id": episode_data["episode_id"],
        "baselines": baselines,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SDR benchmark evaluator stub.")
    parser.add_argument("--robustness-suite", help="Path to a robustness suite JSON file.")
    parser.add_argument(
        "--robustness-submission",
        help="Path to a robustness suite submission JSON file.",
    )
    parser.add_argument("--window", help="Path to an evaluation window JSON file.")
    parser.add_argument("--submission", help="Path to a submission JSON file.")
    parser.add_argument(
        "--baseline",
        choices=BASELINE_NAMES,
        help="Generate and score a built-in baseline instead of loading a window submission.",
    )
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Generate and score all built-in baselines for the requested window or episode.",
    )
    parser.add_argument(
        "--labels",
        help="Optional path to a hidden-label JSON file for offline scoring.",
    )
    parser.add_argument("--episode", help="Path to a policy episode JSON file.")
    parser.add_argument("--episode-submission", help="Path to a policy episode submission JSON file.")
    parser.add_argument(
        "--episode-baseline",
        choices=BASELINE_NAMES,
        help="Generate and score a built-in baseline instead of loading an episode submission.",
    )
    parser.add_argument("--episode-labels", help="Optional path to a policy episode label JSON file.")
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List the available built-in baselines and exit.",
    )
    parser.add_argument(
        "--include-effective-decisions",
        action="store_true",
        help="Include the normalized decision set in the output report.",
    )
    parser.add_argument(
        "--include-generated-submissions",
        action="store_true",
        help="Include generated baseline submissions in the output for baseline runs.",
    )
    parser.add_argument(
        "--include-window-reports",
        action="store_true",
        help="Include per-window reports for policy episode evaluation.",
    )
    parser.add_argument(
        "--include-case-reports",
        action="store_true",
        help="Include per-case reports for robustness suite evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed used by stochastic reference baselines.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_baselines:
        report: Any = {
            "baselines": list(BASELINE_NAMES),
        }
    elif args.robustness_suite:
        suite_data = load_json(args.robustness_suite)
        if args.all_baselines:
            report = evaluate_all_robustness_suite_baselines(
                suite_data,
                include_case_reports=args.include_case_reports,
                include_submissions=args.include_generated_submissions,
                seed=args.seed,
            )
        elif args.baseline:
            report = evaluate_robustness_suite_baseline(
                suite_data,
                args.baseline,
                include_case_reports=args.include_case_reports,
                include_submission=args.include_generated_submissions,
                seed=args.seed,
            )
        else:
            if not args.robustness_submission:
                raise SystemExit("--robustness-submission is required when --robustness-suite is provided")
            robustness_submission = load_json(args.robustness_submission)
            report = evaluate_robustness_suite(
                suite_data,
                robustness_submission,
                include_case_reports=args.include_case_reports,
                normalization_seed=args.seed,
            )
    elif args.episode:
        episode_data = load_json(args.episode)
        episode_labels = load_json(args.episode_labels) if args.episode_labels else None
        if args.all_baselines:
            report = evaluate_all_episode_baselines(
                episode_data,
                labels_data=episode_labels,
                include_window_reports=args.include_window_reports,
                include_submissions=args.include_generated_submissions,
                seed=args.seed,
            )
        elif args.episode_baseline:
            report = evaluate_episode_baseline(
                episode_data,
                args.episode_baseline,
                labels_data=episode_labels,
                include_window_reports=args.include_window_reports,
                include_submission=args.include_generated_submissions,
                seed=args.seed,
            )
        else:
            if not args.episode_submission:
                raise SystemExit("--episode-submission is required when --episode is provided")
            episode_submission = load_json(args.episode_submission)
            report = evaluate_episode(
                episode_data,
                episode_submission,
                episode_labels,
                include_window_reports=args.include_window_reports,
                normalization_seed=args.seed,
            )
    else:
        if not args.window:
            raise SystemExit("--window is required for single-window evaluation")
        window_data = load_json(args.window)
        labels_data = load_json(args.labels) if args.labels else None
        if args.all_baselines:
            report = evaluate_all_window_baselines(
                window_data,
                labels_data=labels_data,
                include_effective_decisions=args.include_effective_decisions,
                include_submissions=args.include_generated_submissions,
                seed=args.seed,
            )
        elif args.baseline:
            report = evaluate_window_baseline(
                window_data,
                args.baseline,
                labels_data=labels_data,
                include_effective_decisions=args.include_effective_decisions,
                include_submission=args.include_generated_submissions,
                seed=args.seed,
            )
        else:
            if not args.submission:
                raise SystemExit("--window and --submission are required for single-window evaluation")
            submission_data = load_json(args.submission)
            report = evaluate_window(
                window_data,
                submission_data,
                labels_data=labels_data,
                include_effective_decisions=args.include_effective_decisions,
                normalization_seed=args.seed,
            )

    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
