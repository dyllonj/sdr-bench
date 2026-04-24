"""Canonical benchmark stages for the enterprise SDR motion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkStage:
    """A model-visible workflow slice with stable outputs and score families."""

    stage_id: str
    label: str
    description: str
    agent_jobs: tuple[str, ...]
    model_visible_tools: tuple[str, ...]
    outputs: tuple[str, ...]
    score_families: tuple[str, ...]
    inspirations: tuple[str, ...]


FULL_CYCLE_SDR_STAGE_IDS = (
    "account_discovery",
    "account_research",
    "buying_center_mapping",
    "qualification_discovery",
    "outreach_planning",
    "engagement_and_handoff",
    "weekly_allocation",
    "multi_week_book_management",
)

TOP_OF_FUNNEL_STAGE_IDS = (
    "account_discovery",
    "account_research",
    "buying_center_mapping",
    "qualification_discovery",
)

STAGE_MODES: dict[str, tuple[str, ...]] = {
    "top_of_funnel": TOP_OF_FUNNEL_STAGE_IDS,
    "full_cycle_sdr": FULL_CYCLE_SDR_STAGE_IDS,
}

BENCHMARK_STAGES: tuple[BenchmarkStage, ...] = (
    BenchmarkStage(
        stage_id="account_discovery",
        label="Account And Prospect Discovery",
        description=(
            "Find and rank ICP-fit enterprise accounts and candidate people from "
            "CRM, product-led signals, trigger feeds, and noisy contact rosters."
        ),
        agent_jobs=("prospect_search", "icp_filtering", "qualified_result_yield"),
        model_visible_tools=("list_accounts", "search_accounts", "search_people"),
        outputs=("ranked_accounts", "candidate_contacts", "criteria_evidence"),
        score_families=("relevance_precision", "effective_coverage", "fit_score"),
        inspirations=("people_search_bench",),
    ),
    BenchmarkStage(
        stage_id="account_research",
        label="Account Research",
        description=(
            "Build a grounded research brief that ties company context, strategic "
            "priorities, technographics, and recent events to the seller's value prop."
        ),
        agent_jobs=("company_research", "agentic_rag", "citation_grounding"),
        model_visible_tools=("get_account_context", "search_evidence", "get_seller_knowledge"),
        outputs=("account_research_brief", "why_account_codes", "citations"),
        score_families=("grounding_score", "research_relevance", "schema_accuracy"),
        inspirations=("microsoft_sales_qualification_bench", "microsoft_sales_research_bench"),
    ),
    BenchmarkStage(
        stage_id="buying_center_mapping",
        label="Buying-Center Mapping",
        description=(
            "Select the right people and roles for a first enterprise motion, with "
            "credit for complementary coverage instead of redundant titles."
        ),
        agent_jobs=("people_search", "persona_selection", "buying_role_coverage"),
        model_visible_tools=("get_account_context", "search_people", "compare_contacts"),
        outputs=("selected_contacts", "persona_rationale", "coverage_evidence"),
        score_families=("contact_score", "coverage_score", "information_utility"),
        inspirations=("people_search_bench",),
    ),
    BenchmarkStage(
        stage_id="qualification_discovery",
        label="Qualification Discovery",
        description=(
            "Infer what still needs to be learned about need, authority, timing, "
            "budget, pain, and urgency before seller handoff."
        ),
        agent_jobs=("discovery_planning", "qualification_gap_detection", "handoff_readiness"),
        model_visible_tools=("get_account_context", "get_engagement_history"),
        outputs=("qualification_plan", "missing_criteria", "handoff_readiness"),
        score_families=("qualification_accuracy", "handoff_accuracy", "timing_score"),
        inspirations=("microsoft_sales_qualification_bench",),
    ),
    BenchmarkStage(
        stage_id="outreach_planning",
        label="Personalized Outreach Planning",
        description=(
            "Choose the channel, message angle, and account-specific reason to engage "
            "without making copy quality the only objective."
        ),
        agent_jobs=("outreach_generation", "personalization", "channel_selection"),
        model_visible_tools=("get_account_context", "get_seller_knowledge", "draft_outreach"),
        outputs=("outreach_plan", "message_brief", "channel_plan"),
        score_families=("personalization_quality", "grounding_score", "policy_value"),
        inspirations=("microsoft_sales_qualification_bench",),
    ),
    BenchmarkStage(
        stage_id="engagement_and_handoff",
        label="Engagement And Handoff",
        description=(
            "Handle multi-turn prospect replies, answer product questions from seller "
            "knowledge, ask discovery questions, and hand off at the right moment."
        ),
        agent_jobs=("lead_engagement", "question_answering", "human_handoff"),
        model_visible_tools=("get_seller_knowledge", "get_engagement_history", "submit_handoff_decision"),
        outputs=("reply_plan", "discovery_questions", "handoff_decision"),
        score_families=("answer_quality", "discovery_coverage", "handoff_accuracy"),
        inspirations=("microsoft_sales_qualification_bench",),
    ),
    BenchmarkStage(
        stage_id="weekly_allocation",
        label="Weekly SDR Allocation",
        description=(
            "Allocate scarce personalized SDR touches across the book for the current "
            "week, optimizing incremental pipeline rather than raw conversion propensity."
        ),
        agent_jobs=("capacity_allocation", "next_best_action", "queue_prioritization"),
        model_visible_tools=("list_accounts", "get_account_context", "submit_weekly_decisions"),
        outputs=("weekly_decisions", "human_touch_queue", "next_best_actions"),
        score_families=("lift_score", "offline_score", "capacity_adjusted_value"),
        inspirations=("sdr_bench_original",),
    ),
    BenchmarkStage(
        stage_id="multi_week_book_management",
        label="Multi-Week Book Management",
        description=(
            "Manage a named-account SDR book over weeks as triggers decay, fatigue "
            "changes, meetings occur, and accounts move out of prospecting."
        ),
        agent_jobs=("policy_optimization", "state_tracking", "regret_minimization"),
        model_visible_tools=("list_accounts", "get_account_context", "submit_weekly_decisions"),
        outputs=("episode_policy", "weekly_trace", "book_state_decisions"),
        score_families=("policy_score", "incremental_pipeline", "negative_regret"),
        inspirations=("sdr_bench_original",),
    ),
)

_STAGES_BY_ID = {stage.stage_id: stage for stage in BENCHMARK_STAGES}


def get_benchmark_stage(stage_id: str) -> BenchmarkStage:
    """Return a benchmark stage by stable id."""

    try:
        return _STAGES_BY_ID[stage_id]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark stage: {stage_id}") from exc


def get_stage_ids_for_mode(mode: str) -> tuple[str, ...]:
    """Return ordered stage ids for a benchmark mode."""

    try:
        return STAGE_MODES[mode]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark mode: {mode}") from exc


def get_stages_for_mode(mode: str) -> tuple[BenchmarkStage, ...]:
    """Return ordered stage definitions for a benchmark mode."""

    return tuple(get_benchmark_stage(stage_id) for stage_id in get_stage_ids_for_mode(mode))
