# Full-Cycle Enterprise SDR Direction

SDR Bench should become a full-cycle enterprise tech sales SDR benchmark, not only
a weekly account-allocation benchmark.

The target motion is a named-account B2B tech sales motion where an SDR agent must
find promising accounts and people, research why the account matters now,
personalize outreach, engage and qualify the prospect, decide whether to hand off
to a seller, and keep managing the book under capacity constraints over multiple
weeks.

## What We Borrow

### From Microsoft Sales Qualification Bench

Useful benchmark shape:

- Evaluate research, outreach, and engagement as separate jobs.
- Include seller knowledge such as value propositions, case studies, product docs,
  and approved answers to prospect questions.
- Score core quality, trustworthiness, grounding, and sales-specific success
  criteria separately before rolling into a composite.
- Treat handoff timing as a first-class outcome, not a hidden detail.

SDR Bench should use this structure but avoid being a vendor-product demo. The
agent should operate against a neutral enterprise tech seller profile and public
benchmark schemas.

Source: https://www.microsoft.com/en-us/dynamics-365/blog/business-leader/2025/12/11/dynamics-365-sets-the-bar-for-agentic-sales-qualification-on-new-benchmark/

### From PeopleSearchBench

Useful benchmark shape:

- Evaluate people discovery as retrieval, not only generation.
- Convert natural-language criteria into explicit, verifiable requirements.
- Score relevance precision, coverage, and information utility separately.
- Prefer criteria-grounded verification over broad subjective judging when the
  facts are checkable.

SDR Bench should adapt this to enterprise buying-center work: finding the right
roles, mapping coverage, and verifying that selected contacts satisfy the
account-specific SDR criteria.

Source: https://arxiv.org/abs/2603.27476

## Full-Cycle Stage Model

The canonical stage catalog lives in `src/sdr_bench/stages.py`.

| Stage | What the agent does | Primary evidence |
|---|---|---|
| `account_discovery` | Find and rank ICP-fit accounts and candidate people | CRM/account fields, product signals, public triggers |
| `account_research` | Build a grounded account brief tied to the seller's value prop | evidence docs, triggers, seller knowledge |
| `buying_center_mapping` | Select complementary personas and contacts | contact roster, titles, activity, role criteria |
| `qualification_discovery` | Identify need, authority, timing, budget, pain, and unknowns | account context, engagement history |
| `outreach_planning` | Choose channel, angle, and personalized message brief | account brief, contacts, approved seller messaging |
| `engagement_and_handoff` | Reply to prospect turns, ask discovery questions, and hand off at the right time | seller knowledge, prospect replies, qualification state |
| `weekly_allocation` | Allocate scarce SDR effort across the current book | account value, timing, fatigue, capacity |
| `multi_week_book_management` | Re-plan as account state changes over weeks | session trace, prior actions, state transitions |

## Evaluation Layers

Top-of-funnel mode should run first:

- account discovery
- account research
- buying-center mapping
- qualification discovery

Full-cycle mode adds:

- outreach planning
- engagement and handoff
- weekly allocation
- multi-week book management

This lets the repo support cheap pilot runs before running the full benchmark.
Top-of-funnel can prove whether models can retrieve and verify the right sales
objects. Full-cycle then tests whether that research turns into incremental
pipeline under realistic SDR capacity constraints.

## Scoring Families

The full-cycle benchmark should report these families separately:

- `FitScore`: account and ICP correctness.
- `ResearchGroundingScore`: account brief accuracy, citations, and schema
  validity.
- `PeopleSearchScore`: buying-center relevance precision, coverage, and profile
  utility.
- `QualificationScore`: need, authority, timing, budget, pain, and unknown-gap
  detection.
- `OutreachScore`: personalization, relevance, channel choice, and grounded
  message plan.
- `EngagementScore`: answer quality, discovery-question coverage, and handoff
  accuracy.
- `LiftScore`: incremental pipeline from scarce SDR effort.
- `PolicyScore`: multi-week book management value and regret.

The public top-line score should not let writing quality dominate. Full-cycle SDR
quality is only valid if the agent works the right accounts, the right people,
at the right time, and creates incremental pipeline that would not have happened
without SDR effort.

## Tooling Implications

Current tools cover the allocation core:

- `list_accounts`
- `get_account_context`
- `submit_weekly_decisions`

Next tool families should add the rest of the motion:

- `search_accounts` for prospect discovery and ICP filtering.
- `search_people` and `compare_contacts` for buying-center mapping.
- `get_seller_knowledge` for value props, case studies, product docs, and
  approved answers.
- `draft_outreach` for constrained outreach-plan generation.
- `get_engagement_history` for multi-turn lead qualification state.
- `submit_handoff_decision` for accepted meeting or seller handoff decisions.

These should stay behind the same narrow `execute(name, input) -> result`
interface so the harness, sandbox, and session trace can evolve independently.
