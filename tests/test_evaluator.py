from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.evaluator import evaluate_window
from sdr_bench.evaluator import evaluate_window_baseline
from sdr_bench.evaluator import evaluate_all_window_baselines
from sdr_bench.evaluator import evaluate_episode
from sdr_bench.evaluator import evaluate_episode_baseline
from sdr_bench.evaluator import evaluate_robustness_suite
from sdr_bench.evaluator import evaluate_robustness_suite_baseline
from sdr_bench.evaluator import load_json
from sdr_bench.baselines import BASELINE_NAMES
from sdr_bench.baselines import generate_window_submission


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = load_json(ROOT_DIR / "examples" / "sample_window.json")
        self.submission = load_json(ROOT_DIR / "examples" / "sample_submission.json")
        self.hidden_labels = load_json(ROOT_DIR / "examples" / "sample_hidden_labels.json")
        self.episode = load_json(ROOT_DIR / "examples" / "sample_episode.json")
        self.policy_submission = load_json(ROOT_DIR / "examples" / "sample_policy_submission.json")
        self.policy_labels = load_json(ROOT_DIR / "examples" / "sample_policy_labels.json")

    def build_subset_window_bundle(
        self,
        account_ids: set[str],
        *,
        window_id: str,
        submission_override: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        window = json.loads(json.dumps(self.window))
        labels = json.loads(json.dumps(self.hidden_labels))
        submission = json.loads(json.dumps(self.submission))

        window["window_id"] = window_id
        window["accounts"] = [
            account
            for account in window["accounts"]
            if account["account_id"] in account_ids
        ]
        for account in window["accounts"]:
            account["label_window_id"] = window_id
        window["contacts"] = [
            contact
            for contact in window["contacts"]
            if contact["account_id"] in account_ids
        ]
        window["triggers"] = [
            trigger
            for trigger in window["triggers"]
            if trigger["account_id"] in account_ids
        ]
        window["evidence"] = [
            document
            for document in window["evidence"]
            if document["account_id"] in account_ids
        ]
        window["capacity_budget"]["window_id"] = window_id
        window["capacity_budget"]["human_sdr_actions"] = min(
            window["capacity_budget"]["human_sdr_actions"],
            len(window["accounts"]),
        )

        labels["window_id"] = window_id
        labels["account_outcomes"] = [
            row
            for row in labels["account_outcomes"]
            if row["account_id"] in account_ids
        ]
        labels["contact_outcomes"] = [
            row
            for row in labels["contact_outcomes"]
            if row["account_id"] in account_ids
        ]
        labels["trigger_outcomes"] = [
            row
            for row in labels["trigger_outcomes"]
            if row["account_id"] in account_ids
        ]

        submission = {
            "window_id": window_id,
            "decisions": [
                decision
                for decision in submission["decisions"]
                if decision["account_id"] in account_ids
            ],
        }
        if submission_override is not None:
            submission = submission_override

        return window, labels, submission

    def build_sample_robustness_suite(
        self,
    ) -> tuple[dict[str, object], dict[str, object]]:
        holdout_window, holdout_labels, holdout_submission = self.build_subset_window_bundle(
            {"acct_123"},
            window_id="wk_holdout_finserv",
        )
        shift_window, shift_labels, shift_submission = self.build_subset_window_bundle(
            {"acct_123", "acct_456"},
            window_id="wk_shift_trigger_heavy",
        )
        holdout_window["capacity_budget"]["window_id"] = "wk_holdout_finserv"
        shift_window["capacity_budget"]["window_id"] = "wk_shift_trigger_heavy"

        for account in shift_window["accounts"]:
            if account["account_id"] == "acct_123":
                account["web_engagement"]["pricing_visits_30d"] = 6
                account["web_engagement"]["high_intent_content_downloads_30d"] = 2
            if account["account_id"] == "acct_456":
                account["web_engagement"]["product_pages_30d"] = 8
                account["web_engagement"]["trial_signups_30d"] = 2
        for trigger in shift_window["triggers"]:
            trigger["recency_days"] = max(1, trigger["recency_days"] - 2)

        suite = {
            "suite_id": "robust_v0_sample",
            "cases": [
                {
                    "case_id": "holdout_financial_services",
                    "robustness_type": "heldout_slice",
                    "description": "Held-out financial services named-account window.",
                    "holdout_dimension": "industry",
                    "holdout_values": ["financial_services"],
                    "window": holdout_window,
                    "labels": holdout_labels,
                },
                {
                    "case_id": "distribution_shift_trigger_heavy",
                    "robustness_type": "distribution_shift",
                    "description": "Trigger-heavy and higher-intent window with changed prevalence.",
                    "shift_tags": ["trigger_prevalence_up", "intent_density_up"],
                    "window": shift_window,
                    "labels": shift_labels,
                },
            ],
        }
        submission = {
            "suite_id": "robust_v0_sample",
            "submissions": [
                {
                    "case_id": "holdout_financial_services",
                    "submission": holdout_submission,
                },
                {
                    "case_id": "distribution_shift_trigger_heavy",
                    "submission": shift_submission,
                },
            ],
        }
        return suite, submission

    def test_valid_submission_is_scorable(self) -> None:
        report = evaluate_window(self.window, self.submission)

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["compliance"]["effective_human_touch_count"], 1)
        self.assertEqual(report["compliance"]["budgeted_human_touch_count"], 1)
        self.assertEqual(report["compliance"]["compliance_multiplier"], 1.0)
        self.assertAlmostEqual(report["metrics"]["GroundingScore"], 100.0)
        self.assertIsNone(report["metrics"]["OfflineScore"])

    def test_window_report_includes_sales_terminology_glossary(self) -> None:
        report = evaluate_window(self.window, self.submission)

        self.assertEqual(
            report["terminology"]["actions"]["human_touch"]["sales_label"],
            "Personalized SDR Outreach",
        )
        self.assertEqual(
            report["terminology"]["metrics"]["TimingScore"]["sales_label"],
            "Why-Now Score",
        )
        self.assertEqual(
            report["terminology"]["submission_fields"]["human_touch_rank"]["sales_label"],
            "Outreach Priority Rank",
        )

    def test_window_report_includes_sales_view(self) -> None:
        report = evaluate_window(
            self.window,
            self.submission,
            labels_data=self.hidden_labels,
            normalization_seed=1,
        )

        self.assertEqual(
            report["sales_view"]["benchmark_mode"],
            "Weekly Queue Prioritization Benchmark",
        )
        self.assertAlmostEqual(
            report["sales_view"]["scorecard"]["Named-Account Book Score"]["value"],
            100.0,
        )
        self.assertEqual(
            report["sales_view"]["routing_summary"]["effective_routing_mix"]["Personalized SDR Outreach"]["count"],
            1,
        )
        self.assertEqual(
            report["sales_view"]["benchmark_vs_anchor"]["anchor_baseline"]["sales_label"],
            "Random Named-Account Picklist",
        )

    def test_window_baselines_generate_scorable_submissions(self) -> None:
        for baseline_name in BASELINE_NAMES:
            with self.subTest(baseline=baseline_name):
                submission = generate_window_submission(self.window, baseline_name, seed=7)
                report = evaluate_window(
                    self.window,
                    submission,
                    labels_data=self.hidden_labels,
                )

                self.assertTrue(report["scorable"])
                self.assertIn(report["status"], {"ok", "ok_with_issues"})
                self.assertLessEqual(
                    report["compliance"]["effective_human_touch_count"],
                    self.window["capacity_budget"]["human_sdr_actions"],
                )

    def test_random_window_baseline_is_deterministic(self) -> None:
        first = generate_window_submission(self.window, "random_within_icp", seed=17)
        second = generate_window_submission(self.window, "random_within_icp", seed=17)

        self.assertEqual(first, second)

    def test_bau_window_baseline_matches_sample_optimum(self) -> None:
        result = evaluate_window_baseline(
            self.window,
            "bau_enterprise_sdr_policy",
            labels_data=self.hidden_labels,
        )

        self.assertEqual(result["report"]["status"], "ok")
        self.assertAlmostEqual(result["report"]["metrics"]["OfflineScore"], 100.0)

    def test_window_report_includes_baseline_normalization(self) -> None:
        report = evaluate_window(
            self.window,
            self.submission,
            labels_data=self.hidden_labels,
            normalization_seed=1,
        )

        self.assertEqual(report["normalization"]["baseline_name"], "random_within_icp")
        self.assertLess(report["normalization"]["baseline_metrics"]["OfflineScore"], 100.0)
        self.assertAlmostEqual(report["normalization"]["normalized_metrics"]["OfflineScore"], 100.0)
        self.assertAlmostEqual(
            report["normalization"]["normalized_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )

    def test_window_report_includes_slice_diagnostics(self) -> None:
        report = evaluate_window(
            self.window,
            self.submission,
            labels_data=self.hidden_labels,
            normalization_seed=1,
        )

        industry_slices = report["slice_diagnostics"]["dimensions"]["industry"]
        self.assertEqual(set(industry_slices), {"financial_services", "software"})
        self.assertEqual(industry_slices["financial_services"]["observation_count"], 1)
        self.assertEqual(industry_slices["software"]["human_touch_count"], 0)
        self.assertAlmostEqual(
            industry_slices["financial_services"]["metrics"]["OfflineScore"],
            100.0,
        )
        self.assertLess(
            industry_slices["software"]["metrics"]["OfflineScore"],
            industry_slices["financial_services"]["metrics"]["OfflineScore"],
        )
        self.assertEqual(
            industry_slices["software"]["normalization"]["baseline_name"],
            "random_within_icp",
        )
        self.assertLess(
            industry_slices["software"]["normalization"]["normalized_metrics"]["EnterpriseAllocationScore"],
            0.0,
        )

    def test_random_baseline_normalizes_to_zero(self) -> None:
        result = evaluate_window_baseline(
            self.window,
            "random_within_icp",
            labels_data=self.hidden_labels,
            seed=1,
        )

        self.assertAlmostEqual(
            result["report"]["normalization"]["normalized_metrics"]["OfflineScore"],
            0.0,
        )
        self.assertAlmostEqual(
            result["report"]["normalization"]["normalized_metrics"]["EnterpriseAllocationScore"],
            0.0,
        )

    def test_all_window_baselines_wrapper_returns_full_pack(self) -> None:
        result = evaluate_all_window_baselines(
            self.window,
            labels_data=self.hidden_labels,
        )

        self.assertEqual(result["mode"], "window_baseline_pack")
        self.assertEqual(len(result["baselines"]), len(BASELINE_NAMES))

    def test_over_budget_human_touch_is_coerced(self) -> None:
        over_budget_submission = json.loads(json.dumps(self.submission))
        over_budget_submission["decisions"][1] = {
            "account_id": "acct_456",
            "human_touch_rank": 2,
            "chosen_action": "human_touch",
            "action_score": 0.88,
            "selected_contacts": ["ct_3"],
            "primary_trigger_event_id": "evt_2",
            "evidence_brief": {
                "why_account_codes": ["enterprise_icp_fit"],
                "why_now_code": "usage_change_recent",
                "why_persona_code": "technical_buyer",
                "why_channel_code": "email_valid",
                "citations": ["doc_992"]
            }
        }

        report = evaluate_window(
            self.window,
            over_budget_submission,
            labels_data=self.hidden_labels,
            include_effective_decisions=True,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok_with_issues")
        self.assertEqual(report["compliance"]["effective_human_touch_count"], 1)
        self.assertEqual(report["compliance"]["over_budget_downgraded_count"], 1)
        self.assertEqual(report["compliance"]["compliance_multiplier"], 0.95)
        self.assertLess(report["metrics"]["OfflineScore"], 100.0)
        coerced = next(
            decision
            for decision in report["effective_decisions"]
            if decision["account_id"] == "acct_456"
        )
        self.assertEqual(coerced["chosen_action"], "wait")
        self.assertEqual(coerced["coerce_reason"], "over_budget_human_touch")

    def test_over_budget_compliance_multiplier_penalizes_budget_sensitive_scores(self) -> None:
        valid_wait_submission = json.loads(json.dumps(self.submission))
        valid_wait_submission["decisions"][1] = {
            "account_id": "acct_456",
            "chosen_action": "wait",
            "action_score": 0.05,
        }
        valid_wait_report = evaluate_window(
            self.window,
            valid_wait_submission,
            labels_data=self.hidden_labels,
        )

        over_budget_submission = json.loads(json.dumps(self.submission))
        over_budget_submission["decisions"][1] = {
            "account_id": "acct_456",
            "human_touch_rank": 2,
            "chosen_action": "human_touch",
            "action_score": 0.88,
            "selected_contacts": ["ct_3"],
            "primary_trigger_event_id": "evt_2",
            "evidence_brief": {
                "why_account_codes": ["enterprise_icp_fit"],
                "why_now_code": "usage_change_recent",
                "why_persona_code": "technical_buyer",
                "why_channel_code": "email_valid",
                "citations": ["doc_992"],
            },
        }
        over_budget_report = evaluate_window(
            self.window,
            over_budget_submission,
            labels_data=self.hidden_labels,
        )

        self.assertEqual(over_budget_report["compliance"]["compliance_multiplier"], 0.95)
        self.assertAlmostEqual(
            over_budget_report["metrics"]["OfflineScore"],
            valid_wait_report["metrics"]["OfflineScore"] * 0.95,
        )
        self.assertAlmostEqual(
            over_budget_report["metrics"]["GroundingScore"],
            valid_wait_report["metrics"]["GroundingScore"],
        )

    def test_invalid_contact_is_dropped_before_scoring(self) -> None:
        invalid_contact_submission = json.loads(json.dumps(self.submission))
        invalid_contact_submission["decisions"][0]["selected_contacts"].append("ct_3")

        report = evaluate_window(
            self.window,
            invalid_contact_submission,
            labels_data=self.hidden_labels,
            include_effective_decisions=True,
        )

        self.assertTrue(report["scorable"])
        decision = next(
            decision
            for decision in report["effective_decisions"]
            if decision["account_id"] == "acct_123"
        )
        self.assertEqual(decision["selected_contacts"], ["ct_1", "ct_2"])
        self.assertIn(
            "invalid_contact",
            {issue["code"] for issue in report["issues"]},
        )

    def test_hidden_labels_enable_offline_metrics(self) -> None:
        report = evaluate_window(
            self.window,
            self.submission,
            labels_data=self.hidden_labels,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["metrics"]["precision_at_capacity"], 1.0)
        self.assertAlmostEqual(report["metrics"]["ndcg_at_capacity"], 1.0)
        self.assertAlmostEqual(report["metrics"]["uplift_at_capacity"], 33000.0)
        self.assertAlmostEqual(report["metrics"]["action_policy_value"], 41000.0)
        self.assertAlmostEqual(report["metrics"]["trigger_accuracy_at_1"], 1.0)
        self.assertAlmostEqual(report["metrics"]["trigger_gain_ratio"], 1.0)
        self.assertAlmostEqual(report["metrics"]["TimingScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["contact_precision_at_selected"], 1.0)
        self.assertAlmostEqual(report["metrics"]["contact_mrr"], 1.0)
        self.assertAlmostEqual(report["metrics"]["contact_lift_ratio"], 1.0)
        self.assertAlmostEqual(report["metrics"]["ContactScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["GroundingScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["OfflineScore"], 100.0)

    def test_suboptimal_ranking_reduces_offline_score(self) -> None:
        bad_submission = {
            "window_id": "wk_2026_10",
            "decisions": [
                {
                    "account_id": "acct_456",
                    "human_touch_rank": 1,
                    "chosen_action": "human_touch",
                    "action_score": 0.88,
                    "selected_contacts": ["ct_3"],
                    "primary_trigger_event_id": "evt_2",
                    "evidence_brief": {
                        "why_account_codes": ["enterprise_icp_fit"],
                        "why_now_code": "usage_change_recent",
                        "why_persona_code": "technical_buyer",
                        "why_channel_code": "email_valid",
                        "citations": ["doc_992"]
                    }
                },
                {
                    "account_id": "acct_123",
                    "chosen_action": "automated_outbound",
                    "action_score": 0.77
                }
            ]
        }

        report = evaluate_window(
            self.window,
            bad_submission,
            labels_data=self.hidden_labels,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(report["metrics"]["OfflineScore"], 100.0)
        self.assertLess(report["metrics"]["FitScore"], 100.0)
        self.assertLess(report["metrics"]["LiftScore"], 100.0)

    def test_suboptimal_contact_selection_reduces_contact_score(self) -> None:
        bad_contact_submission = json.loads(json.dumps(self.submission))
        bad_contact_submission["decisions"][0]["selected_contacts"] = ["ct_4", "ct_2"]

        report = evaluate_window(
            self.window,
            bad_contact_submission,
            labels_data=self.hidden_labels,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(report["metrics"]["ContactScore"], 100.0)
        self.assertLess(report["metrics"]["contact_precision_at_selected"], 1.0)
        self.assertLess(report["metrics"]["contact_mrr"], 1.0)
        self.assertLess(report["metrics"]["OfflineScore"], 100.0)

    def test_suboptimal_trigger_selection_reduces_timing_score(self) -> None:
        bad_trigger_submission = json.loads(json.dumps(self.submission))
        bad_trigger_submission["decisions"][0]["primary_trigger_event_id"] = "evt_3"
        bad_trigger_submission["decisions"][0]["evidence_brief"]["citations"] = ["doc_993"]

        report = evaluate_window(
            self.window,
            bad_trigger_submission,
            labels_data=self.hidden_labels,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(report["metrics"]["TimingScore"], 100.0)
        self.assertLess(report["metrics"]["trigger_accuracy_at_1"], 1.0)
        self.assertLess(report["metrics"]["trigger_gain_ratio"], 1.0)
        self.assertLess(report["metrics"]["OfflineScore"], 100.0)

    def test_unsupported_evidence_packet_reduces_grounding_score(self) -> None:
        bad_grounding_submission = json.loads(json.dumps(self.submission))
        bad_grounding_submission["decisions"][0]["evidence_brief"]["citations"] = ["doc_991"]

        report = evaluate_window(
            self.window,
            bad_grounding_submission,
            labels_data=self.hidden_labels,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(report["metrics"]["GroundingScore"], 100.0)
        self.assertGreater(report["metrics"]["unsupported_claim_rate"], 0.0)
        self.assertLess(report["metrics"]["OfflineScore"], 100.0)

    def test_non_groundable_citation_is_dropped_before_scoring(self) -> None:
        window = json.loads(json.dumps(self.window))
        for document in window["evidence"]:
            if document["doc_id"] == "doc_998":
                document["allowed_for_grounding"] = False

        report = evaluate_window(
            window,
            self.submission,
            labels_data=self.hidden_labels,
            include_effective_decisions=True,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok_with_issues")
        decision = next(
            decision
            for decision in report["effective_decisions"]
            if decision["account_id"] == "acct_123"
        )
        self.assertNotIn("doc_998", decision["evidence_brief"]["citations"])
        self.assertIn(
            "disallowed_grounding_citation",
            {issue["code"] for issue in report["issues"]},
        )

    def test_missing_selected_contact_labels_invalidates_hidden_labels(self) -> None:
        bad_labels = json.loads(json.dumps(self.hidden_labels))
        bad_labels["contact_outcomes"][0]["contacts"] = [
            row
            for row in bad_labels["contact_outcomes"][0]["contacts"]
            if row["contact_id"] != "ct_2"
        ]

        report = evaluate_window(
            self.window,
            self.submission,
            labels_data=bad_labels,
        )

        self.assertEqual(report["status"], "invalid_hidden_labels")
        self.assertIn(
            "missing_selected_contact_labels",
            {issue["code"] for issue in report["issues"]},
        )

    def test_missing_selected_trigger_labels_invalidates_hidden_labels(self) -> None:
        bad_submission = json.loads(json.dumps(self.submission))
        bad_submission["decisions"][0]["primary_trigger_event_id"] = "evt_3"
        bad_submission["decisions"][0]["evidence_brief"]["citations"] = ["doc_993"]
        bad_labels = json.loads(json.dumps(self.hidden_labels))
        bad_labels["trigger_outcomes"][0]["triggers"] = [
            row
            for row in bad_labels["trigger_outcomes"][0]["triggers"]
            if row["event_id"] != "evt_3"
        ]

        report = evaluate_window(
            self.window,
            bad_submission,
            labels_data=bad_labels,
        )

        self.assertEqual(report["status"], "invalid_hidden_labels")
        self.assertIn(
            "missing_selected_trigger_labels",
            {issue["code"] for issue in report["issues"]},
        )

    def test_policy_episode_scores_successfully(self) -> None:
        report = evaluate_episode(
            self.episode,
            self.policy_submission,
            self.policy_labels,
            include_window_reports=True,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(report["metrics"]["cumulative_incremental_pipeline"], 79000.0)
        self.assertAlmostEqual(report["metrics"]["oracle_cumulative_incremental_pipeline"], 79000.0)
        self.assertAlmostEqual(report["metrics"]["policy_pipeline_ratio"], 1.0)
        self.assertAlmostEqual(report["metrics"]["PolicyScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["OfflineScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["GroundingScore"], 100.0)
        self.assertAlmostEqual(report["metrics"]["EnterpriseAllocationScore"], 100.0)
        self.assertEqual(len(report["window_reports"]), 2)

    def test_episode_report_includes_sales_terminology_glossary(self) -> None:
        report = evaluate_episode(
            self.episode,
            self.policy_submission,
            self.policy_labels,
        )

        self.assertEqual(
            report["terminology"]["metrics"]["PolicyScore"]["sales_label"],
            "Multi-Week Book Management Score",
        )
        self.assertEqual(
            report["terminology"]["baselines"]["bau_enterprise_sdr_policy"]["sales_label"],
            "Business-as-Usual SDR Playbook",
        )

    def test_episode_report_includes_sales_view(self) -> None:
        report = evaluate_episode(
            self.episode,
            self.policy_submission,
            self.policy_labels,
            normalization_seed=1,
        )

        self.assertEqual(
            report["sales_view"]["benchmark_mode"],
            "Multi-Week Book Management Benchmark",
        )
        self.assertAlmostEqual(
            report["sales_view"]["scorecard"]["Multi-Week Book Management Score"]["value"],
            100.0,
        )
        self.assertAlmostEqual(
            report["sales_view"]["supplemental_metrics"]["Incremental Pipeline Generated"]["value"],
            79000.0,
        )
        self.assertEqual(
            report["sales_view"]["benchmark_vs_anchor"]["anchor_baseline"]["sales_label"],
            "Random Named-Account Picklist",
        )

    def test_bau_episode_baseline_scores_cleanly(self) -> None:
        result = evaluate_episode_baseline(
            self.episode,
            "bau_enterprise_sdr_policy",
            labels_data=self.policy_labels,
            include_window_reports=True,
        )

        self.assertTrue(result["report"]["scorable"])
        self.assertEqual(result["report"]["status"], "ok")
        self.assertAlmostEqual(result["report"]["metrics"]["PolicyScore"], 100.0)

    def test_episode_report_includes_baseline_normalization(self) -> None:
        report = evaluate_episode(
            self.episode,
            self.policy_submission,
            self.policy_labels,
            normalization_seed=1,
        )

        self.assertEqual(report["normalization"]["baseline_name"], "random_within_icp")
        self.assertLess(report["normalization"]["baseline_metrics"]["PolicyScore"], 100.0)
        self.assertAlmostEqual(report["normalization"]["normalized_metrics"]["PolicyScore"], 100.0)
        self.assertAlmostEqual(
            report["normalization"]["normalized_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )

    def test_episode_report_includes_slice_diagnostics(self) -> None:
        report = evaluate_episode(
            self.episode,
            self.policy_submission,
            self.policy_labels,
            include_window_reports=True,
            normalization_seed=1,
        )

        industry_slices = report["slice_diagnostics"]["dimensions"]["industry"]
        self.assertEqual(industry_slices["financial_services"]["window_count"], 1)
        self.assertEqual(industry_slices["software"]["window_count"], 2)
        self.assertAlmostEqual(industry_slices["software"]["metrics"]["PolicyScore"], 100.0)
        self.assertLess(industry_slices["software"]["metrics"]["OfflineScore"], 100.0)
        self.assertEqual(
            industry_slices["software"]["normalization"]["baseline_name"],
            "random_within_icp",
        )
        self.assertAlmostEqual(
            industry_slices["software"]["normalization"]["normalized_metrics"]["PolicyScore"],
            100.0,
        )
        self.assertIsNone(
            report["window_reports"][0]["slice_diagnostics"]["dimensions"]["industry"]["software"]["normalization"]
        )

    def test_policy_episode_ineligible_account_is_coerced(self) -> None:
        bad_submission = json.loads(json.dumps(self.policy_submission))
        bad_submission["submissions"][1]["decisions"] = [
            {
                "account_id": "acct_123",
                "human_touch_rank": 1,
                "chosen_action": "human_touch",
                "action_score": 0.77,
                "selected_contacts": ["ct_1"],
                "primary_trigger_event_id": "evt_5",
                "evidence_brief": {
                    "why_account_codes": ["enterprise_icp_fit"],
                    "why_now_code": "compliance_deadline",
                    "why_persona_code": "technical_buyer",
                    "why_channel_code": "email_valid",
                    "citations": ["doc_995"]
                }
            }
        ]

        report = evaluate_episode(
            self.episode,
            bad_submission,
            self.policy_labels,
            include_window_reports=True,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok_with_issues")
        self.assertLess(report["metrics"]["PolicyScore"], 100.0)
        self.assertLess(report["metrics"]["GroundingScore"], 100.0)
        self.assertIn(
            "policy_ineligible_account",
            {issue["code"] for issue in report["issues"]},
        )

    def test_policy_episode_stateful_setup_action_matters(self) -> None:
        bad_submission = json.loads(json.dumps(self.policy_submission))
        bad_submission["submissions"][0]["decisions"][1] = {
            "account_id": "acct_456",
            "chosen_action": "wait",
            "action_score": 0.25
        }

        report = evaluate_episode(
            self.episode,
            bad_submission,
            self.policy_labels,
            include_window_reports=True,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(report["metrics"]["PolicyScore"], 100.0)
        self.assertLess(report["metrics"]["cumulative_incremental_pipeline"], 79000.0)
        self.assertGreater(report["metrics"]["policy_regret"], 0.0)

    def test_robustness_suite_scores_successfully(self) -> None:
        suite, submission = self.build_sample_robustness_suite()

        report = evaluate_robustness_suite(
            suite,
            submission,
            include_case_reports=True,
            normalization_seed=1,
        )

        self.assertTrue(report["scorable"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["summary"]["case_count"], 2)
        self.assertAlmostEqual(
            report["summary"]["average_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )
        self.assertAlmostEqual(
            report["summary"]["average_normalized_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )
        self.assertEqual(
            report["summary"]["robustness_type_breakdown"]["heldout_slice"]["case_count"],
            1,
        )
        self.assertEqual(
            report["summary"]["shift_tag_breakdown"]["trigger_prevalence_up"]["case_count"],
            1,
        )
        self.assertEqual(len(report["cases"]), 2)
        self.assertIn("report", report["cases"][0])

    def test_robustness_report_includes_sales_terminology_glossary(self) -> None:
        suite, submission = self.build_sample_robustness_suite()

        report = evaluate_robustness_suite(
            suite,
            submission,
        )

        self.assertEqual(
            report["terminology"]["actions"]["wait"]["sales_label"],
            "Monitor / No Action",
        )

    def test_robustness_report_includes_sales_view(self) -> None:
        suite, submission = self.build_sample_robustness_suite()

        report = evaluate_robustness_suite(
            suite,
            submission,
            normalization_seed=1,
        )

        self.assertEqual(
            report["sales_view"]["benchmark_mode"],
            "Robustness Diagnostics",
        )
        self.assertAlmostEqual(
            report["sales_view"]["summary"]["Robustness Score"]["value"],
            100.0,
        )
        self.assertEqual(
            report["sales_view"]["case_scoreboard"][0]["Named-Account Book Score"],
            100.0,
        )

    def test_robustness_suite_detects_underperforming_case(self) -> None:
        suite, submission = self.build_sample_robustness_suite()
        submission["submissions"][0]["submission"] = {
            "window_id": "wk_holdout_finserv",
            "decisions": [
                {
                    "account_id": "acct_123",
                    "chosen_action": "wait",
                    "action_score": 0.05,
                }
            ],
        }

        report = evaluate_robustness_suite(
            suite,
            submission,
            normalization_seed=1,
        )

        self.assertTrue(report["scorable"])
        self.assertLess(
            report["summary"]["worst_case_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )
        self.assertLess(
            report["summary"]["average_metrics"]["EnterpriseAllocationScore"],
            100.0,
        )
        self.assertGreaterEqual(len(report["summary"]["underperforming_cases"]), 1)

    def test_robustness_suite_baseline_scores_cleanly(self) -> None:
        suite, _ = self.build_sample_robustness_suite()

        result = evaluate_robustness_suite_baseline(
            suite,
            "bau_enterprise_sdr_policy",
            include_case_reports=True,
        )

        self.assertTrue(result["report"]["scorable"])
        self.assertIn(result["report"]["status"], {"ok", "ok_with_issues"})
        self.assertEqual(result["report"]["summary"]["case_count"], 2)


if __name__ == "__main__":
    unittest.main()
