# Agent Sandbox TODOs

This roadmap moves SDR Bench from giant-context JSON completion toward a tool-mediated
agent environment while keeping the existing evaluator as the scoring authority.

## Phase 1: Public Sandbox Foundation

- [ ] Add `src/sdr_bench/agent/` as a separate package from the current prompt runner.
- [ ] Add a public window view that redacts scoring-only fields before any model-visible tool result.
- [ ] Redact `document_evidence.grounding_support` and `allowed_for_grounding` from tool-mode evidence.
- [ ] Add deterministic indexes for accounts, contacts, triggers, and evidence by account.
- [ ] Add leakage tests that fail if public payloads expose hidden-label or scoring-only keys.

## Phase 2: Minimal Tools

- [ ] Add `list_accounts(limit, cursor)` with bounded pagination and compact public summaries.
- [ ] Add `get_account_context(account_id)` for account-local public account, contacts, triggers, and evidence.
- [ ] Add `submit_weekly_decisions(decisions)` as the only mutating/finalizing tool.
- [ ] Return structured tool errors for invalid args, unknown IDs, and finalized runs.
- [ ] Log tool calls with stable result hashes, counts, latency, and public-only markers.
- [ ] Keep tools behind a narrow `execute(name, input)`-style boundary so the harness can swap hands without changing model-visible contracts.

## Phase 3: Agent Runner

- [ ] Add an opt-in tool-mode runner that loops over provider tool calls.
- [ ] Keep `run_window_model` and existing direct prompt mode backward compatible.
- [ ] Add an optional adapter protocol for provider tool turns.
- [ ] Implement mocked agent runner tests before provider-specific code.
- [ ] Add OpenAI and Anthropic tool-turn support behind the optional protocol.
- [ ] Treat the harness as stateless and recoverable from the durable trace/session artifact.
- [ ] Keep the session log outside the model context window; let future harnesses choose how much event history to rehydrate.

## Phase 4: Staged Benchmark Modes

- [ ] Add stage metadata for `top_of_funnel`, `why_now`, `buying_center`, `weekly_routing`, and `multi_week_policy`.
- [ ] Keep `full_cycle_sdr` deferred until allocation, timing, and contact selection are stable.
- [ ] Add `--modes` and `--interaction-mode prompt|tools` harness controls.
- [ ] Track model variants so prompt-mode and tool-mode results do not aggregate together.
- [ ] Derive lower-budget curves from a single ranked run where possible.

## Phase 5: Validity And Cost Controls

- [ ] Report deterministic baselines, prefilter baselines, and oracle ceilings for every pilot.
- [ ] Track tool-call count, viewed accounts, viewed docs, tokens, latency, and cost per score-point lift.
- [ ] Cache model/tool-turn results by model, prompt hash, dataset hash, schema hash, and seed.
- [ ] Use hosted batch/flex APIs for pilots; defer rented GPUs until the tool contract is stable.
- [ ] Run a small frontier canary only after local baseline/oracle audits pass.

## Phase 6: Managed-Agent Architecture Constraints

- [ ] Decouple the brain, hands, and session: model/harness, tool sandbox, and event log must be replaceable independently.
- [ ] Store append-only trace/session events durably enough that a failed harness can resume from the last event.
- [ ] Never place secrets, hidden labels, or scorer state inside model-visible tool results or sandbox state.
- [ ] Model context is a cache, not the source of truth; every irreversible compaction must be recoverable from trace/session events.
- [ ] Design for many brains and many hands: multiple model runs may share a public dataset, and one run may use multiple tool backends later.
- [ ] Avoid harness assumptions that only fit current models; keep context resets, search aids, and ranking helpers optional and measurable.
