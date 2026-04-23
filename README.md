# SDR Bench

Initial scoping docs for an SDR benchmark focused on allocation of limited human attention.

Start with the v0 spec in [docs/sdr-benchmark-v0.md](docs/sdr-benchmark-v0.md).

Machine-readable artifacts:

- Canonical schemas live in [`schemas/`](/home/deck/Work/sdr-bench/schemas)
- The evaluator stub lives in [`src/sdr_bench/evaluator.py`](/home/deck/Work/sdr-bench/src/sdr_bench/evaluator.py:1)
- Example inputs live in [`examples/`](/home/deck/Work/sdr-bench/examples)

Current evaluator coverage:

- Offline window scoring for fit, lift, trigger timing, contact selection, and deterministic evidence grounding
- Policy episode scoring with stateful transitions and a public score that blends average offline performance with policy performance
- Deterministic reference baseline pack for `random_within_icp`, `rules_trigger_queue`, `propensity_only_lead_score`, `last_touch_recency`, and `bau_enterprise_sdr_policy`
- Raw plus baseline-normalized score reporting relative to `random_within_icp`
- Slice diagnostics by `segment`, `industry`, `region`, `intent_presence`, `relationship_motion`, and `data_density`, with baseline-normalized slice scores in top-level window and episode reports
- Robustness-suite scoring for held-out cohort cases and distribution-shift windows, with pack-level worst-case and average summaries

Sales-facing terminology:

The evaluator keeps stable machine keys in schemas and JSON reports, but top-level reports now include both a `terminology` glossary and a `sales_view` block with SDR-friendly scorecards and routing summaries.

| Canonical key | SDR / sales-facing label |
|---|---|
| `human_touch` | Personalized SDR Outreach |
| `automated_outbound` | Sequence Enrollment |
| `nurture` | Marketing Nurture |
| `recycle` | Recycle / Snooze |
| `wait` | Monitor / No Action |
| `FitScore` | ICP / Account Selection Score |
| `TimingScore` | Why-Now Score |
| `ContactScore` | Buying-Center Coverage Score |
| `GroundingScore` | Account Research Grounding Score |
| `LiftScore` | Incremental Pipeline Score |
| `OfflineScore` | Weekly Queue Prioritization Score |
| `PolicyScore` | Multi-Week Book Management Score |
| `EnterpriseAllocationScore` | Named-Account Book Score |

Quick start:

```bash
PYTHONPATH=src python3 -m sdr_bench.evaluator \
  --window examples/sample_window.json \
  --submission examples/sample_submission.json \
  --labels examples/sample_hidden_labels.json \
  --seed 1 \
  --pretty
```

`--seed` controls deterministic baseline generation for stochastic reference runs and for the `random_within_icp` normalization anchor.
Top-level reports now include a `slice_diagnostics` block; episode runs with `--include-window-reports` also include raw per-window slice breakdowns on the eligible pool.

Robustness suite example:

```bash
PYTHONPATH=src python3 -m sdr_bench.evaluator \
  --robustness-suite examples/sample_robustness_suite.json \
  --robustness-submission examples/sample_robustness_submission.json \
  --include-case-reports \
  --seed 1 \
  --pretty
```

Policy episode example:

```bash
PYTHONPATH=src python3 -m sdr_bench.evaluator \
  --episode examples/sample_episode.json \
  --episode-submission examples/sample_policy_submission.json \
  --episode-labels examples/sample_policy_labels.json \
  --pretty
```

Single baseline example:

```bash
PYTHONPATH=src python3 -m sdr_bench.evaluator \
  --window examples/sample_window.json \
  --baseline bau_enterprise_sdr_policy \
  --labels examples/sample_hidden_labels.json \
  --include-generated-submissions \
  --pretty
```

All baselines example:

```bash
PYTHONPATH=src python3 -m sdr_bench.evaluator \
  --episode examples/sample_episode.json \
  --all-baselines \
  --episode-labels examples/sample_policy_labels.json \
  --pretty
```
