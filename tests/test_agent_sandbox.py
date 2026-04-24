from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.agent import AgentSandbox
from sdr_bench.evaluator import load_json


SCORING_ONLY_KEYS = {
    "allowed_for_grounding",
    "grounding_support",
    "label_window_id",
}


def iter_key_paths(value: Any, *, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}"
            paths.append(child_path)
            paths.extend(iter_key_paths(child, path=child_path))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(iter_key_paths(child, path=f"{path}[{index}]"))
        return paths
    return []


class AgentSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = load_json(ROOT_DIR / "examples" / "sample_window.json")
        self.submission = load_json(ROOT_DIR / "examples" / "sample_submission.json")

    def assertToolOk(
        self,
        tool_result: dict[str, Any],
        *,
        tool_name: str,
    ) -> dict[str, Any]:
        self.assertIs(tool_result.get("ok"), True)
        self.assertEqual(tool_result.get("tool_name"), tool_name)
        self.assertNotIn("error", tool_result)
        self.assertIn("result", tool_result)
        self.assertRegex(tool_result.get("result_hash", ""), re.compile(r"^[a-f0-9]{64}$"))
        return tool_result["result"]

    def assertToolError(
        self,
        tool_result: dict[str, Any],
        *,
        tool_name: str,
        code: str,
    ) -> dict[str, Any]:
        self.assertIs(tool_result.get("ok"), False)
        self.assertEqual(tool_result.get("tool_name"), tool_name)
        self.assertRegex(tool_result.get("result_hash", ""), re.compile(r"^[a-f0-9]{64}$"))
        error = tool_result.get("error")
        self.assertIsInstance(error, dict)
        self.assertEqual(error.get("code"), code)
        self.assertIsInstance(error.get("message"), str)
        self.assertTrue(error["message"])
        return error

    def assertNoScoringOnlyKeys(self, payload: Any) -> None:
        leaked_paths = [
            path
            for path in iter_key_paths(payload)
            if path.rsplit(".", 1)[-1] in SCORING_ONLY_KEYS
        ]
        self.assertEqual([], leaked_paths)

    def test_public_tool_payloads_redact_scoring_only_fields(self) -> None:
        sandbox = AgentSandbox(self.window)

        account_page = sandbox.list_accounts(limit=2)
        account_page_payload = self.assertToolOk(account_page, tool_name="list_accounts")
        self.assertNoScoringOnlyKeys(account_page)

        context = sandbox.get_account_context("acct_123")
        context_payload = self.assertToolOk(context, tool_name="get_account_context")
        self.assertNoScoringOnlyKeys(context)

        self.assertTrue(account_page_payload["accounts"])
        self.assertTrue(context_payload["evidence"])

    def test_list_accounts_paginates_deterministically_and_enforces_limit_bounds(self) -> None:
        sandbox = AgentSandbox(self.window)

        first_page = self.assertToolOk(
            sandbox.list_accounts(limit=1),
            tool_name="list_accounts",
        )
        self.assertEqual(["acct_123"], [account["account_id"] for account in first_page["accounts"]])
        self.assertIsNotNone(first_page["next_cursor"])

        second_page = self.assertToolOk(
            sandbox.list_accounts(limit=1, cursor=first_page["next_cursor"]),
            tool_name="list_accounts",
        )
        self.assertEqual(["acct_456"], [account["account_id"] for account in second_page["accounts"]])
        self.assertIsNone(second_page["next_cursor"])

        self.assertToolError(
            sandbox.list_accounts(limit=0),
            tool_name="list_accounts",
            code="invalid_limit",
        )
        self.assertToolError(
            sandbox.list_accounts(limit=10_000),
            tool_name="list_accounts",
            code="invalid_limit",
        )

    def test_get_account_context_is_account_local(self) -> None:
        sandbox = AgentSandbox(self.window)

        context = self.assertToolOk(
            sandbox.get_account_context("acct_123"),
            tool_name="get_account_context",
        )

        self.assertEqual("acct_123", context["account"]["account_id"])
        self.assertEqual(
            {"acct_123"},
            {contact["account_id"] for contact in context["contacts"]},
        )
        self.assertEqual(
            {"acct_123"},
            {trigger["account_id"] for trigger in context["triggers"]},
        )
        self.assertEqual(
            {"acct_123"},
            {document["account_id"] for document in context["evidence"]},
        )

    def test_unknown_accounts_return_structured_errors(self) -> None:
        sandbox = AgentSandbox(self.window)

        context_error = self.assertToolError(
            sandbox.get_account_context("acct_missing"),
            tool_name="get_account_context",
            code="unknown_account",
        )
        self.assertEqual("acct_missing", context_error.get("account_id"))

        submit_error = self.assertToolError(
            sandbox.submit_weekly_decisions(
                [
                    {
                        "account_id": "acct_missing",
                        "chosen_action": "wait",
                        "action_score": 0.0,
                    }
                ]
            ),
            tool_name="submit_weekly_decisions",
            code="unknown_account",
        )
        self.assertEqual("acct_missing", submit_error.get("account_id"))
        self.assertFalse(sandbox.finalized)

    def test_submit_weekly_decisions_finalizes_and_records_trace_compliance_receipt(self) -> None:
        sandbox = AgentSandbox(self.window)

        submit_result = sandbox.submit_weekly_decisions(self.submission["decisions"])
        receipt = self.assertToolOk(submit_result, tool_name="submit_weekly_decisions")

        self.assertTrue(sandbox.finalized)
        self.assertEqual("finalized", receipt["status"])
        self.assertEqual(self.window["window_id"], receipt["window_id"])
        self.assertIn("compliance", receipt)
        self.assertEqual(1, receipt["compliance"]["effective_human_touch_count"])
        self.assertEqual(1, receipt["compliance"]["budgeted_human_touch_count"])

        trace_receipt = receipt["trace_receipt"]
        self.assertEqual("submit_weekly_decisions", trace_receipt["tool_name"])
        self.assertEqual(submit_result["result_hash"], trace_receipt["result_hash"])
        self.assertIs(trace_receipt["public_only"], True)
        self.assertIsInstance(trace_receipt["latency_ms"], (int, float))
        self.assertGreaterEqual(trace_receipt["latency_ms"], 0)
        self.assertGreaterEqual(trace_receipt["result_count"], 1)
        self.assertEqual(trace_receipt, sandbox.trace_events[-1])

        finalized_error = self.assertToolError(
            sandbox.submit_weekly_decisions(self.submission["decisions"]),
            tool_name="submit_weekly_decisions",
            code="finalized",
        )
        self.assertEqual(self.window["window_id"], finalized_error.get("window_id"))

    def test_result_hashes_are_stable_for_repeated_deterministic_calls(self) -> None:
        sandbox = AgentSandbox(self.window)

        first_page = sandbox.list_accounts(limit=1)
        second_page = sandbox.list_accounts(limit=1)
        self.assertToolOk(first_page, tool_name="list_accounts")
        self.assertToolOk(second_page, tool_name="list_accounts")
        self.assertEqual(first_page["result"], second_page["result"])
        self.assertEqual(first_page["result_hash"], second_page["result_hash"])

        first_context = sandbox.get_account_context("acct_123")
        second_context = sandbox.get_account_context("acct_123")
        self.assertToolOk(first_context, tool_name="get_account_context")
        self.assertToolOk(second_context, tool_name="get_account_context")
        self.assertEqual(first_context["result"], second_context["result"])
        self.assertEqual(first_context["result_hash"], second_context["result_hash"])


if __name__ == "__main__":
    unittest.main()
