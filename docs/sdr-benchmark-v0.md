# SDR Benchmark v0

## Core Question

Given limited SDR capacity, can a model allocate human attention to the accounts where it creates the most incremental pipeline?

This benchmark starts with prospecting allocation, then layers outreach and qualification on top. The design goal is to evaluate operating decisions, not just copy quality or propensity prediction.

## Design Principles

1. Optimize for incremental business impact under hard capacity, not generic conversion likelihood.
2. Separate account prioritization, action allocation, contact selection, and evidence quality.
3. Treat timing as first-class. Static lead scoring is not enough.
4. Reward grounded decisions with evidence packets tied to source records.
5. Support both offline evaluation and closed-loop policy evaluation.

## Capability Axes

The benchmark should report four capability axes separately, even if a single top-line score is also published.

| Axis | What it measures | Core tasks | Example failure mode |
|---|---|---|---|
| Fit | Is this account structurally worth working? | `T1`, `T3` | high-propensity but low-incremental accounts crowd out better targets |
| Timing | Is there a credible why-now reason to act this week? | `T7`, `P1` | stale or weak triggers drive wasted touches |
| Contact selection | Are the chosen people the right buying-center entry points? | `T5`, `T6` | redundant or irrelevant personas reduce meeting yield |
| Incremental lift | Does human attention change outcomes beyond baseline automation or wait? | `T2`, `T4`, `P1`, `P2` | model learns propensity instead of persuadability |

Recommended diagnostic reporting:

- `FitScore`: ranking quality on hidden structural-fit labels or long-run incremental value priors
- `TimingScore`: recency-weighted trigger relevance and reprioritization quality
- `ContactScore`: best-contact precision plus buying-center coverage
- `LiftScore`: uplift, `AUUC`, and policy value relative to baseline

## Sales-Facing Terminology

The benchmark keeps stable machine keys for schemas and evaluator outputs, but the intended SDR / sales-language reading is:

| Canonical key | SDR / sales-facing label |
|---|---|
| `human_touch` | Personalized SDR Outreach |
| `automated_outbound` | Sequence Enrollment |
| `nurture` | Marketing Nurture |
| `recycle` | Recycle / Snooze |
| `disqualify` | Disqualify |
| `wait` | Monitor / No Action |
| `FitScore` | ICP / Account Selection Score |
| `TimingScore` | Why-Now Score |
| `ContactScore` | Buying-Center Coverage Score |
| `GroundingScore` | Account Research Grounding Score |
| `LiftScore` | Incremental Pipeline Score |
| `OfflineScore` | Weekly Queue Prioritization Score |
| `PolicyScore` | Multi-Week Book Management Score |
| `EnterpriseAllocationScore` | Named-Account Book Score |

These are presentation labels, not replacement schema values. Submissions and labels should still use the canonical machine keys.

## V0 Benchmark Slice

v0 is explicitly scoped to an enterprise tech SDR motion.

- Seller archetype: enterprise SDR/BDR supporting AEs at B2B software, infrastructure, data, security, developer-tooling, or AI-platform vendors.
- Account archetype: named-account outbound into mid-market and enterprise companies, typically `500` to `20,000` employees.
- Motion: pre-opportunity prospecting and meeting creation under hard weekly human-capacity limits.
- Geography: start with North America and English-language outreach assumptions.
- Public leaderboard eligibility: net-new and product-led pre-opportunity accounts only.
- Out of scope for v0 leaderboard: renewals, active open opportunities, post-sales expansion plays, and full email-writing quality.

This keeps the benchmark close to the highest-value enterprise SDR bottleneck: deciding where scarce human research and personalization should go this week.

## V0 Decisions

The following choices are fixed for v0 so implementation can start.

| Area | v0 Decision |
|---|---|
| Ground truth | synthetic-first, enterprise-tech-specific simulator with hidden potential outcomes |
| Primary business objective | incremental weighted pipeline within `90` days |
| Secondary business objectives | incremental accepted opportunities within `45` days, meetings within `21` days |
| Task scope | reduced v0 cut: `T1`, `T2`, `T4`, `T5`, `T7`, `T8`, `P1` |
| Contact roster assumption | partial and noisy but person-deduped candidate rosters |
| Evidence grading | leaderboard scoring is deterministic on structured rationale codes and citations |
| Leaderboard shape | one public score plus offline/policy sub-scores and capability profile |

## Benchmark Layers

### Layer 1: Offline Ranking and Decision Tasks

Hidden-label tasks over large candidate sets with fixed weekly budgets.

- Candidate pool: `10,000` to `100,000` accounts per evaluation window.
- Capacity budget: examples `B in {25, 50, 200}` human SDR actions per week.
- Objective: maximize incremental meetings, opportunities, and pipeline value subject to budget.

### Layer 2: Policy and Simulator Tasks

A sequential environment where actions this week affect next week's state and downstream outcomes.

- Horizon: `4` to `12` weekly steps.
- Outcomes: reply, meeting, accepted opportunity, pipeline amount, conversion lag.
- Objective: maximize cumulative incremental value with budget and fatigue constraints.

## Canonical Data Model

All tasks use the same entity types so the benchmark can be extended without changing the interface.

### `account_snapshot`

```json
{
  "account_id": "acct_123",
  "snapshot_ts": "2026-03-02T00:00:00Z",
  "segment": "enterprise",
  "relationship_motion": "net_new",
  "account_tier": "tier_1_named",
  "industry": "financial_services",
  "employee_count": 4500,
  "revenue_band": "1b_5b",
  "hq_region": "NA",
  "sales_geo": "NA_enterprise",
  "technographics": {
    "cloud_provider": "aws",
    "identity_provider": "okta",
    "data_platform": "snowflake",
    "security_tools": ["crowdstrike"],
    "competitors_present": ["vendor_x"]
  },
  "crm_state": {
    "current_stage": "none",
    "owner_role": "SDR_pool",
    "past_meetings_365d": 0,
    "past_opps_365d": 0,
    "days_since_last_human_touch": 61,
    "days_since_last_outbound_sequence": 19,
    "has_open_opportunity": false,
    "open_support_escalation": false
  },
  "web_engagement": {
    "pricing_visits_30d": 4,
    "product_pages_30d": 12,
    "high_intent_content_downloads_30d": 1,
    "trial_signups_30d": 0
  },
  "product_or_usage_signals": {
    "seat_growth_30d": 0.18,
    "active_users_growth_30d": 0.09,
    "feature_adoption_flags": ["security_module"]
  },
  "trigger_events": ["evt_1", "evt_2"],
  "available_contacts": ["ct_1", "ct_2", "ct_3"],
  "label_window_id": "wk_2026_10"
}
```

### `contact_snapshot`

```json
{
  "contact_id": "ct_1",
  "account_id": "acct_123",
  "name_redacted": true,
  "title": "VP of IT",
  "function": "IT",
  "seniority": "VP",
  "department": "Infrastructure",
  "likely_buying_role": "technical_buyer",
  "is_decision_maker_proxy": 0.61,
  "historical_reply_rate_bucket": "medium",
  "channel_reachability": {
    "email_valid": true,
    "phone_valid": false,
    "linkedin_present": true
  },
  "recent_activity": {
    "job_change_90d": false,
    "content_engagement_30d": 2
  }
}
```

Contact roster assumptions for v0:

- Candidate rosters are partially observed, not complete org charts.
- Rosters are person-deduped, but near-duplicate titles may remain.
- Each account has `5` to `30` surfaced contacts.
- Some buying-center roles are missing; coverage metrics are normalized by the best achievable coverage within the provided roster.
- Models may only select surfaced contacts. They are not penalized for failing to pick an unavailable role.

### `trigger_event`

```json
{
  "event_id": "evt_1",
  "account_id": "acct_123",
  "event_ts": "2026-02-25T00:00:00Z",
  "event_type": "leadership_change",
  "source_type": "news",
  "confidence": 0.84,
  "recency_days": 5,
  "evidence_refs": ["doc_991", "doc_992"]
}
```

### `document_evidence`

```json
{
  "doc_id": "doc_991",
  "account_id": "acct_123",
  "source_type": "news",
  "published_ts": "2026-02-25T11:32:00Z",
  "excerpt": "Redacted excerpt backing the trigger.",
  "allowed_for_grounding": true,
  "grounding_support": {
    "why_now_codes": ["leadership_change_recent"],
    "related_event_ids": ["evt_1"]
  }
}
```

`grounding_support` is optional in the raw schema, but v0 scoring uses it when present to evaluate `T8` deterministically.

### `capacity_budget`

```json
{
  "window_id": "wk_2026_10",
  "human_sdr_actions": 200,
  "max_contacts_per_account": 3,
  "channel_costs": {
    "human_touch": 1.0,
    "automated_outbound": 0.15,
    "nurture": 0.05,
    "recycle": 0.02,
    "wait": 0.0,
    "disqualify": 0.0
  }
}
```

### `model_output`

```json
{
  "window_id": "wk_2026_10",
  "decisions": [
    {
      "account_id": "acct_123",
      "human_touch_rank": 1,
      "chosen_action": "human_touch",
      "action_score": 0.93,
      "selected_contacts": ["ct_1", "ct_2"],
      "primary_trigger_event_id": "evt_1",
      "evidence_brief": {
        "why_account_codes": ["enterprise_icp_fit", "intent_surge"],
        "why_account_text": "High-fit enterprise account with recent buying signals.",
        "why_now_code": "leadership_change_recent",
        "why_now_text": "Leadership change and pricing-page surge in last 7 days.",
        "why_persona_code": "technical_buyer_plus_security_champion",
        "why_persona_text": "VP IT plus Security Director match the buying center.",
        "why_channel_code": "email_valid_plus_recent_web_intent",
        "why_channel_text": "Email is valid and recent web engagement suggests reachable interest.",
        "citations": ["doc_991", "doc_992"]
      }
    }
  ]
}
```

Submission semantics:

- Omitted accounts default to `wait`.
- `human_touch_rank` is only required for accounts assigned `human_touch`.
- `selected_contacts`, `primary_trigger_event_id`, and `evidence_brief` are required for `human_touch`.
- Free-text rationale fields are optional for readability. Leaderboard scoring uses the structured codes and citations.

## Action Semantics

- `human_touch`: personalized SDR outreach this week, such as a custom email, call task, or LinkedIn touch backed by human research. Cost `1.0`.
- `automated_outbound`: sequence enrollment using approved templates and account-level targeting with limited rep time. Cost `0.15`.
- `nurture`: marketing nurture or lower-touch follow-up without direct SDR personalization this week. Cost `0.05`.
- `recycle`: recycle or snooze the account out of the active SDR queue for a fixed cooldown window, typically `4` to `8` weeks, because fit may exist but timing is currently poor. Cost `0.02`.
- `disqualify`: mark as out of scope for the benchmark horizon because the account is clearly out of ICP or operationally unreachable. Cost `0.0`.
- `wait`: monitor the account and keep it eligible next week without taking action now. Cost `0.0`.

`wait` and `recycle` are intentionally different:

- `wait` means "continue watching now."
- `recycle` means "stop spending queue attention for a defined cooldown period."

## Label Model

The benchmark should not use plain future conversion labels as the primary target.

### v0 Ground-Truth Decision

v0 uses a synthetic-first hidden-label setup for enterprise tech SDR allocation.

- Offline tasks use static weekly snapshots drawn from the simulator.
- Policy tasks use the same causal world model with temporal state transitions.
- If real enterprise-tech data is available, it should only be used to calibrate marginal distributions, trigger frequencies, title mixes, and base rates. It is not required for hidden labels.

Hidden counterfactual labels are produced from latent factors such as:

- structural fit
- timing and recency
- persona relevance
- channel reachability
- prior-touch fatigue
- market segment and account tier

The simulator must expose only observed CRM-like state and keep potential outcomes hidden.

Later versions may additionally support evaluation against historical randomized-assignment logs where treatment effect is identifiable.

Each `(account, week)` should have hidden potential outcomes for key actions:

```json
{
  "account_id": "acct_123",
  "window_id": "wk_2026_10",
  "potential_outcomes": {
    "human_touch": {
      "meeting_prob": 0.19,
      "opp_prob": 0.08,
      "pipeline_value": 14000
    },
    "automated_outbound": {
      "meeting_prob": 0.07,
      "opp_prob": 0.02,
      "pipeline_value": 3000
    },
    "wait": {
      "meeting_prob": 0.04,
      "opp_prob": 0.01,
      "pipeline_value": 1200
    }
  }
}
```

This allows evaluation of incremental lift:

- `incremental_meeting_lift = P(meeting | human_touch) - P(meeting | wait)`
- `incremental_opp_lift = P(opp | human_touch) - P(opp | wait)`
- `incremental_value_lift = value(human_touch) - value(next_best_nonhuman_action)`

For v0, `value(*)` means weighted pipeline generated within `90` days. Meeting and accepted-opportunity lift are secondary diagnostics.

## Task Suite

The v0 suite uses 10 tasks: 8 offline tasks and 2 policy tasks.

| ID | Layer | Task | Primary Output | Primary Score |
|---|---|---|---|---|
| T1 | Offline | Top-K account prioritization | ranked accounts | `uplift@K`, `precision@K`, `nDCG@K` |
| T2 | Offline | Persuadability ranking | ranked accounts by incremental value | `Qini@K`, `AUUC`, `uplift@K` |
| T3 | Offline | Capacity-aware portfolio allocation | top accounts under budget | `capacity_adjusted_value` |
| T4 | Offline | Next-best action assignment | action per account | policy value, macro F1 |
| T5 | Offline | In-account contact selection | top 1-3 contacts | contact `precision@M`, uplift delta |
| T6 | Offline | Buying-center coverage | contact set | coverage score, redundancy penalty |
| T7 | Offline | Trigger detection and timing | best why-now signal | trigger accuracy, recency-weighted gain |
| T8 | Offline | Evidence packet grounding | structured brief + citations | groundedness, schema accuracy |
| P1 | Policy | Weekly queue management simulator | rolling action policy | cumulative incremental meetings/opps |
| P2 | Policy | Quarter-scale pipeline simulator | rolling policy over 4-12 weeks | cumulative pipeline value, regret |

### T1. Top-K Account Prioritization Under Capacity

Input:
- `10k` to `100k` `account_snapshot` records for a week
- linked contacts, triggers, and evidence
- one `capacity_budget`

Output:
- a full ranking or top `B` accounts for `human_touch`

Scores:
- `precision@K` on hidden positive incremental accepted-opportunity labels
- `nDCG@K` using graded incremental pipeline-value labels
- `uplift@K` against baseline policy

Recommended definition:

`uplift@K = (1 / K) * sum_{i in topK} [Y_i(human_touch) - Y_i(baseline_policy)]`

Where `Y` is hidden incremental weighted pipeline value. Accepted-opportunity and meeting variants should also be reported as diagnostics.

### T2. Persuadability vs. Propensity

Purpose:
- punish models that prioritize accounts likely to convert without SDR help
- punish models that waste effort on low-fit, low-response accounts

Input:
- same as T1

Output:
- ranking by expected incremental value from `human_touch`

Scores:
- `AUUC`
- `Qini@K`
- `uplift@K`

Required baseline:
- compare against a propensity-only model trained on `P(outcome | no intervention distinction)`

Success condition:
- the model must beat propensity ranking at the same budget.

### T3. Capacity-Aware Portfolio Allocation

Purpose:
- test whether the model can make a good frontier decision, not just local rank choices

Additional input:
- action costs by account or segment
- optional rep-specialization constraints

Output:
- a set of accounts chosen under total action cost budget

Primary score:

`capacity_adjusted_value = sum incremental_value_i / sum action_cost_i`

Secondary scores:
- total incremental opportunities
- budget utilization
- segment fairness diagnostics

This task matters when some accounts consume more than one unit of SDR effort.

### T4. Next-Best Action Allocation

Action space:
- `human_touch`
- `automated_outbound`
- `nurture`
- `recycle`
- `disqualify`
- `wait`

Output:
- one action per account plus confidence

Primary score:
- expected policy value under hidden action-specific potential outcomes

Secondary scores:
- macro F1 on action labels if expert labels exist
- calibration of action confidence

Recommended evaluation:

`policy_value = sum_i V(action_i, account_i)`

Where `V` is hidden incremental value relative to `wait`.

### T5. In-Account Contact Selection

Purpose:
- separate account ranking from persona choice

Input:
- chosen accounts plus `5` to `30` contact candidates each

Output:
- top `1` to `3` contacts per account

Scores:
- contact `precision@M`
- `MRR` on best-contact identification
- downstream `meeting_lift_delta` relative to using a generic default persona

Recommended ground truth:
- hidden contact-level response and meeting uplift labels

### T6. Buying-Center Coverage

Purpose:
- reward complementary contact sets rather than three redundant titles

Output:
- a contact set per account

Primary score:

`coverage_score = role_coverage_gain - redundancy_penalty`

Example components:
- reward covering economic buyer, champion, technical buyer, and user
- penalize selecting multiple near-duplicate contacts in the same lane

Secondary score:
- opportunity uplift from the chosen contact set

### T7. Trigger Detection and Timing

Purpose:
- test whether the model can identify the best why-now reason from noisy evidence

Input:
- trigger events plus evidence documents, some relevant and some distractors

Output:
- one primary trigger, optional secondary trigger, and a timing recommendation

Scores:
- trigger classification accuracy
- evidence-supported relevance
- recency-weighted gain

Recommended recency weighting:

`recency_gain = relevance * exp(-lambda * age_in_days)`

This prevents stale triggers from scoring like fresh ones.

### T8. Evidence Packet Quality

Required brief fields:
- `why_account_codes`
- `why_now_code`
- `why_persona_code`
- `why_channel_code`
- `citations`

Primary scores:
- groundedness: each coded claim supported by cited evidence
- relevance: cited evidence matches chosen action and trigger
- schema accuracy: valid structured output with no missing required fields

Secondary scores:
- contradiction rate
- unsupported-claim rate
- brevity penalty beyond a token budget

v0 scoring is deterministic:

- every structured rationale code must map to an allowed evidence fact type
- citations must belong to the same account and time window
- a supported claim is a `(reason_code, cited_evidence)` pair whose annotated fact type, entity, and time window match
- optional free text is ignored for leaderboard scoring

### P1. Weekly Queue Management Simulator

Environment:
- weekly reprioritization over `4` steps
- new events arrive
- prior touches affect response probabilities and fatigue

Agent output each week:
- ranked accounts
- action per account
- contacts for human-touched accounts

Primary scores:
- cumulative incremental meetings
- cumulative incremental opportunities
- decision regret versus oracle policy

Key state transitions:
- touched accounts cool down or warm up based on response
- stale evidence decays
- opportunities remove accounts from prospecting pool

### P2. Quarter-Scale Pipeline Simulator

Environment:
- `8` to `12` week horizon
- meeting-to-opportunity lag
- capacity shifts by week
- account states evolve due to external triggers and internal actions

Primary scores:
- cumulative incremental pipeline value
- regret relative to oracle
- robustness across segment shifts

This is the benchmark layer closest to real SDR operations.

## Scoring Stack

The benchmark should report both task-level metrics and one top-line allocation score.

### Business Objective Stack

v0 uses a three-level objective stack:

1. Primary: incremental weighted pipeline within `90` days
2. Secondary: incremental accepted opportunities within `45` days
3. Tertiary: incremental meetings within `21` days

### Primary benchmark score

Recommended top-line score for the offline layer:

`AllocationScore = 0.45 * uplift_at_B + 0.20 * ndcg_at_B + 0.15 * policy_value_action + 0.10 * contact_uplift_delta + 0.10 * grounding_score`

Recommended top-line score for the policy layer:

`PolicyScore = 0.50 * cumulative_incremental_pipeline + 0.25 * cumulative_incremental_opps + 0.15 * meeting_gain + 0.10 * negative_regret`

These weights can be changed after pilot runs, but v0 should over-weight incremental value.

Recommended public leaderboard score:

`EnterpriseAllocationScore = 0.60 * OfflineScore + 0.40 * PolicyScore`

In addition to the public score, always publish:

- `OfflineScore` for weekly queue prioritization
- `PolicyScore` for multi-week book management
- `FitScore` for ICP and account selection
- `TimingScore` for why-now quality
- `ContactScore` for buying-center coverage
- `LiftScore` for incremental pipeline creation

### Capacity-aware metric

Use a metric that makes budget central:

`ValuePerAction@B = (sum_{i in selected(B)} incremental_value_i) / B`

If actions have heterogeneous cost:

`ValuePerUnitCost@B = (sum incremental_value_i) / (sum action_cost_i)`

### Normalization and aggregation

To aggregate metrics across windows and budgets, normalize each primary metric relative to fixed baselines:

`normalized_metric = 100 * (model_metric - random_icp_baseline) / (oracle_metric - random_icp_baseline)`

Aggregation rules:

- average equally across evaluation windows
- average equally across budget settings `B=25`, `B=50`, and `B=200`
- clip normalized metrics to `[-25, 125]` before weighted aggregation
- publish both raw and normalized metrics
- do not fold slice diagnostics into the public leaderboard

### Baselines

Every leaderboard should include:

1. random within ICP
2. rules-based trigger queue
3. propensity-only lead score
4. last-touch recency heuristic
5. business-as-usual action policy

Optional:
- oracle upper bound
- human expert ranking

Reference baseline implementations:

- `random_within_icp`: sample eligible enterprise-tech ICP accounts uniformly at random, then choose default persona by seniority heuristic.
- `rules_trigger_queue`: rank by a fixed weighted sum of recent trigger recency, web intent, and fit thresholds; assign `human_touch` to the top feasible prefix.
- `propensity_only_lead_score`: supervised model that predicts eventual accepted opportunity or pipeline without treatment distinction, then ranks by that prediction.
- `last_touch_recency`: prioritize accounts with the most recent high-intent or trigger evidence after cooldown checks, with no causal lift modeling.
- `bau_enterprise_sdr_policy`: deterministic enterprise SDR rule set:
  - `human_touch` if named-account fit is high, no open opportunity exists, cooldown has expired, and a strong trigger or intent threshold is met
  - `automated_outbound` if fit is high but timing is weaker
  - `nurture` if fit is medium and contactability is weak
  - `recycle` if fit is high but in cooldown or missing enough buying-center coverage
  - `disqualify` if clearly out of ICP
  - otherwise `wait`

## Data Construction

### Real-data path

Use historical data only if treatment assignment has enough randomness to support counterfactual estimation.

Preferred sources:
- randomized SDR queue experiments
- staggered capacity constraints
- natural experiments from rep absences or territory splits

Risk:
- observational CRM data will mostly measure propensity, not incremental effect

### Synthetic-data path

Use a synthetic org generator to create:
- realistic account graphs
- CRM histories
- contact rosters
- trigger streams
- hidden causal response surfaces

Synthetic data is acceptable if:
- the latent causal structure is explicit
- outputs are grounded in realistic CRM and go-to-market state
- benchmark splits prevent memorization of templates

For v0, this is the recommended path.

## Evaluation Protocol

### Splits

- Train/dev/test by time, not random row split
- Hold out industries and segments for robustness slices
- Include distribution-shift test windows with changing trigger prevalence
- Package held-out and shift cases into a separate robustness suite with worst-case and average reporting

### Budgets

Evaluate each model at several budgets:
- `B=25`
- `B=50`
- `B=200`

This prevents overfitting to one operating point.

### Submission format

Each evaluation window is submitted as one JSON object containing `window_id` and a `decisions` array.

Required fields by decision type:

- all submitted decisions: `account_id`, `chosen_action`, `action_score`
- `human_touch` decisions: `human_touch_rank`, `selected_contacts`, `primary_trigger_event_id`, `evidence_brief`
- `automated_outbound`, `nurture`, `recycle`, `disqualify`: `account_id`, `chosen_action`, `action_score`

Submission rules:

- omitted accounts are treated as `wait`
- account IDs must be unique within a window
- `human_touch_rank` values must be unique and contiguous from `1`
- selected contacts must belong to the chosen account
- citations must point to evidence objects from the same account and window

Invalid submission handling:

- schema-parse failure for a window: score `0` for that window
- invalid record fields: coerce that decision to `wait`
- invalid contacts or citations: drop them before scoring
- over-budget `human_touch` decisions: keep the best feasible prefix by `human_touch_rank`, coerce the rest to `wait`, and apply a `0.95` compliance multiplier to budget-sensitive metrics for that window

### Diagnostics

Report sliced performance by:
- segment
- industry
- region
- inbound intent presence
- pure net-new vs. product-led pre-opportunity
- sparse-data vs. dense-data accounts

Reference evaluator note:
- emit a `slice_diagnostics` block for offline windows and policy episodes
- include raw slice metrics always and baseline-normalized slice metrics when hidden labels are available
- keep slice diagnostics out of the public leaderboard aggregation

## v0 Deliverables

To make this benchmark real, the next implementation artifacts should be:

1. A canonical JSON schema for the five core entities.
2. A hidden-label offline dataset with `>= 10k` accounts per weekly window.
3. A policy simulator with action-specific transition logic.
4. A baseline pack with the five required baselines.
5. An evaluator that computes ranking, uplift, policy, and grounding metrics.
6. A rationale-code dictionary for `why_account`, `why_now`, `why_persona`, and `why_channel`.

## Recommended v0 Cuts

To keep the first version tractable:

- Start with `T1`, `T2`, `T4`, `T5`, `T7`, `T8`, and `P1`.
- Treat qualification as downstream outcome labels, not a separate generation task.
- Defer full email writing until allocation metrics are stable.
- Use one target market first: enterprise tech outbound in North America.

## Deferred v1 Questions

1. When should installed-base expansion and post-sales whitespace accounts be added as a separate slice?
2. When should optional real-log validation be added on top of the synthetic-first benchmark?
3. Should later versions score natural-language rationale quality beyond deterministic grounding?
