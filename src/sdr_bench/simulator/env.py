"""Stepwise environment for running policy episodes against a live model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sdr_bench.evaluator import apply_outcome_modifiers_to_labels
from sdr_bench.evaluator import apply_policy_state_constraints
from sdr_bench.evaluator import apply_policy_transitions
from sdr_bench.evaluator import compute_greedy_oracle_window_decisions
from sdr_bench.evaluator import compute_window_incremental_values
from sdr_bench.evaluator import compute_window_metrics
from sdr_bench.evaluator import extract_current_outcome_modifiers
from sdr_bench.evaluator import is_account_eligible
from sdr_bench.evaluator import materialize_effective_decisions
from sdr_bench.evaluator import normalize_submission
from sdr_bench.evaluator import subset_hidden_labels_to_accounts
from sdr_bench.evaluator import subset_window_to_accounts


class SDRBenchEnv:
    def __init__(
        self,
        episode_config: dict[str, Any],
        seed: int,
        labels_data: dict[str, Any],
    ) -> None:
        self.episode_config = deepcopy(episode_config)
        self.labels_data = deepcopy(labels_data)
        self.seed = seed
        self._window_index = 0
        self._agent_state: dict[str, dict[str, Any]] = {}

    def reset(self) -> dict[str, Any]:
        self._window_index = 0
        self._agent_state = {}
        return self._current_week_state()

    def _current_window(self) -> dict[str, Any]:
        return self.episode_config["windows"][self._window_index]

    def _current_labels_window(self) -> dict[str, Any]:
        return self.labels_data["windows"][self._window_index]

    def _current_week_state(self) -> dict[str, Any]:
        window = self._current_window()
        eligible_account_ids = {
            account["account_id"]
            for account in window["accounts"]
            if is_account_eligible(self._agent_state, account["account_id"], self._window_index)
        }
        public_window = subset_window_to_accounts(window, eligible_account_ids)
        return {
            "episode_id": self.episode_config["episode_id"],
            "week_index": self._window_index + 1,
            "total_weeks": len(self.episode_config["windows"]),
            "window_id": window["window_id"],
            "window": public_window,
            "eligible_account_ids": sorted(eligible_account_ids),
            "ineligible_account_ids": sorted(
                account["account_id"]
                for account in window["accounts"]
                if account["account_id"] not in eligible_account_ids
            ),
        }

    def step(self, submission: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
        window = self._current_window()
        labels_window = self._current_labels_window()
        current_state_snapshot = deepcopy(self._agent_state)

        current_outcome_modifiers = extract_current_outcome_modifiers(self._agent_state)
        adjusted_labels = apply_outcome_modifiers_to_labels(
            labels_window["labels"],
            current_outcome_modifiers,
        )
        final_decisions, issues, compliance = normalize_submission(window, submission)
        effective_decisions = materialize_effective_decisions(window, final_decisions)
        adjusted_decisions, state_issues = apply_policy_state_constraints(
            window["window_id"],
            self._window_index,
            effective_decisions,
            self._agent_state,
        )
        eligible_account_ids = {
            account["account_id"]
            for account in window["accounts"]
            if is_account_eligible(current_state_snapshot, account["account_id"], self._window_index)
        }
        eligible_window = subset_window_to_accounts(window, eligible_account_ids)
        eligible_labels = subset_hidden_labels_to_accounts(adjusted_labels, eligible_account_ids)
        eligible_decisions = [
            decision
            for decision in adjusted_decisions
            if decision["account_id"] in eligible_account_ids
        ]
        metrics, notes = compute_window_metrics(
            eligible_window,
            eligible_decisions,
            labels_data=eligible_labels,
        )
        oracle_decisions = compute_greedy_oracle_window_decisions(
            window,
            adjusted_labels,
            deepcopy(current_state_snapshot),
            self._window_index,
        )
        oracle_eligible_decisions = [
            decision
            for decision in oracle_decisions
            if decision["account_id"] in eligible_account_ids
        ]
        agent_values = compute_window_incremental_values(eligible_decisions, eligible_labels)
        oracle_values = compute_window_incremental_values(oracle_eligible_decisions, eligible_labels)

        transition_map = {
            row["account_id"]: row["actions"]
            for row in labels_window["policy_transitions"]
        }
        apply_policy_transitions(
            self._window_index,
            adjusted_decisions,
            transition_map,
            self._agent_state,
        )

        outcome = {
            "window_id": window["window_id"],
            "issues": issues + state_issues,
            "compliance": compliance,
            "metrics": metrics,
            "notes": notes,
            "effective_decisions": adjusted_decisions,
            "agent_incremental_values": agent_values,
            "oracle_incremental_values": oracle_values,
            "oracle_decisions": oracle_eligible_decisions,
        }

        self._window_index += 1
        done = self._window_index >= len(self.episode_config["windows"])
        next_state = None if done else self._current_week_state()
        return next_state, outcome, done
