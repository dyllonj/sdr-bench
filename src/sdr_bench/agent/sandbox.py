"""Read-only public tool sandbox for SDR Bench agent runs."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from sdr_bench.agent.public_views import build_public_window_view
from sdr_bench.agent.public_views import get_public_account_context
from sdr_bench.agent.trace import canonical_json_hash
from sdr_bench.evaluator import materialize_effective_decisions
from sdr_bench.evaluator import normalize_submission
from sdr_bench.evaluator import validate_instance


DEFAULT_MAX_PAGE_SIZE = 100


class AgentSandbox:
    """Deterministic public-data sandbox for one evaluation window.

    The sandbox owns the model-visible public view and a private copy of the
    original window used only for structural validation. It does not expose
    hidden labels, evaluator metrics, or scoring-only evidence annotations.
    """

    def __init__(
        self,
        window_data: dict[str, Any],
        *,
        max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
    ) -> None:
        if max_page_size < 1:
            raise ValueError("max_page_size must be positive")
        self._window_data = deepcopy(window_data)
        self.view = build_public_window_view(window_data)
        self.max_page_size = max_page_size
        self.finalized = False
        self.submission: dict[str, Any] | None = None
        self.effective_decisions: list[dict[str, Any]] | None = None
        self.trace_events: list[dict[str, Any]] = []

    @property
    def window_id(self) -> str:
        return self.view.window_id

    def list_accounts(
        self,
        *,
        limit: int = 25,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        tool_name = "list_accounts"
        if limit < 1 or limit > self.max_page_size:
            return self._error_result(
                tool_name,
                {
                    "code": "invalid_limit",
                    "message": f"limit must be between 1 and {self.max_page_size}",
                    "limit": limit,
                    "max_limit": self.max_page_size,
                },
                started=started,
            )

        account_ids = sorted(self.view.indexes.accounts_by_id)
        start_index = 0
        if cursor is not None:
            try:
                start_index = int(cursor)
            except ValueError:
                return self._error_result(
                    tool_name,
                    {
                        "code": "invalid_cursor",
                        "message": "cursor must be an integer offset encoded as a string",
                        "cursor": cursor,
                    },
                    started=started,
                )
            if start_index < 0 or start_index > len(account_ids):
                return self._error_result(
                    tool_name,
                    {
                        "code": "invalid_cursor",
                        "message": "cursor is outside the account list bounds",
                        "cursor": cursor,
                    },
                    started=started,
                )

        page_ids = account_ids[start_index : start_index + limit]
        next_index = start_index + limit
        result = {
            "window_id": self.window_id,
            "accounts": [
                deepcopy(self.view.indexes.accounts_by_id[account_id])
                for account_id in page_ids
            ],
            "next_cursor": str(next_index) if next_index < len(account_ids) else None,
            "total_accounts": len(account_ids),
        }
        return self._ok_result(tool_name, result, started=started)

    def get_account_context(self, account_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        tool_name = "get_account_context"
        try:
            context = get_public_account_context(self.view, account_id)
        except KeyError:
            return self._error_result(
                tool_name,
                {
                    "code": "unknown_account",
                    "message": f"account_id is not present in window {self.window_id}",
                    "account_id": account_id,
                },
                started=started,
            )
        result = {
            "window_id": self.window_id,
            **deepcopy(context),
        }
        return self._ok_result(tool_name, result, started=started)

    def submit_weekly_decisions(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        tool_name = "submit_weekly_decisions"
        if self.finalized:
            return self._error_result(
                tool_name,
                {
                    "code": "finalized",
                    "message": "weekly decisions have already been submitted",
                    "window_id": self.window_id,
                },
                started=started,
            )

        account_ids = set(self.view.indexes.accounts_by_id)
        for decision in decisions:
            account_id = decision.get("account_id")
            if account_id not in account_ids:
                return self._error_result(
                    tool_name,
                    {
                        "code": "unknown_account",
                        "message": f"account_id is not present in window {self.window_id}",
                        "account_id": account_id,
                    },
                    started=started,
                )

        submission = {
            "window_id": self.window_id,
            "decisions": deepcopy(decisions),
        }
        schema_errors = validate_instance(submission, "model_output")
        if schema_errors:
            return self._error_result(
                tool_name,
                {
                    "code": "validation_error",
                    "message": "submission does not satisfy model_output schema",
                    "validation_errors": schema_errors,
                },
                started=started,
            )

        final_decisions, issues, compliance = normalize_submission(
            self._window_data,
            submission,
        )
        materialized = materialize_effective_decisions(
            self._window_data,
            final_decisions,
        )
        self.finalized = True
        self.submission = submission
        self.effective_decisions = materialized

        result = {
            "status": "finalized",
            "window_id": self.window_id,
            "submitted_decision_count": len(decisions),
            "issue_count": len(issues),
            "issues": issues,
            "compliance": compliance,
        }
        tool_result = self._ok_result(tool_name, result, started=started)
        trace_receipt = self._trace_receipt(
            tool_name=tool_name,
            result_hash=tool_result["result_hash"],
            result=result,
            started=started,
        )
        self.trace_events.append(trace_receipt)
        tool_result["result"]["trace_receipt"] = trace_receipt
        return tool_result

    def execute(self, name: str, input: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a model-visible tool through a narrow dispatch boundary."""

        started = time.perf_counter()
        tool_input = {} if input is None else input
        if not isinstance(tool_input, dict):
            return self._error_result(
                name,
                {
                    "code": "invalid_args",
                    "message": "tool input must be an object",
                },
                started=started,
            )
        if name == "list_accounts":
            return self.list_accounts(**tool_input)
        if name == "get_account_context":
            return self.get_account_context(**tool_input)
        if name == "submit_weekly_decisions":
            return self.submit_weekly_decisions(**tool_input)
        return self._error_result(
            name,
            {
                "code": "unknown_tool",
                "message": f"unknown tool: {name}",
            },
            started=started,
        )

    def _ok_result(
        self,
        tool_name: str,
        result: dict[str, Any],
        *,
        started: float,
    ) -> dict[str, Any]:
        result_hash = canonical_json_hash(result)
        return {
            "ok": True,
            "tool_name": tool_name,
            "result": result,
            "result_hash": result_hash,
            "latency_ms": self._elapsed_ms(started),
            "public_only": True,
        }

    def _error_result(
        self,
        tool_name: str,
        error: dict[str, Any],
        *,
        started: float,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "tool_name": tool_name,
            "error": error,
            "result_hash": canonical_json_hash(error),
            "latency_ms": self._elapsed_ms(started),
            "public_only": True,
        }

    def _trace_receipt(
        self,
        *,
        tool_name: str,
        result_hash: str,
        result: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        return {
            "tool_name": tool_name,
            "result_hash": result_hash,
            "public_only": True,
            "latency_ms": self._elapsed_ms(started),
            "result_count": self._result_count(result),
        }

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _result_count(result: dict[str, Any]) -> int:
        if "accounts" in result and isinstance(result["accounts"], list):
            return len(result["accounts"])
        if "submitted_decision_count" in result:
            return int(result["submitted_decision_count"])
        return 1
