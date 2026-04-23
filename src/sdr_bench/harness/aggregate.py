"""Aggregate sweep outputs into a normalized leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sdr_bench.io_utils import dump_json
from sdr_bench.io_utils import load_json

NORMALIZATION_CLIP_RANGE = (-25.0, 125.0)


def _clip(value: float) -> float:
    lower, upper = NORMALIZATION_CLIP_RANGE
    return max(lower, min(value, upper))


def _normalize(model_value: float | None, random_value: float | None, oracle_value: float | None) -> float | None:
    if model_value is None or random_value is None or oracle_value is None:
        return None
    denominator = oracle_value - random_value
    if denominator <= 0:
        return 100.0 if model_value >= oracle_value else 0.0
    return _clip(100.0 * (model_value - random_value) / denominator)


def _average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def _collect_model_rows(index: dict[str, Any], results_dir: Path) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for key in ("offline_runs", "policy_runs", "robustness_runs"):
        for row in index.get(key, []):
            by_model.setdefault(row["model_spec"], {"offline": [], "policy": [], "robustness": []})
            mode = "offline" if key == "offline_runs" else "policy" if key == "policy_runs" else "robustness"
            by_model[row["model_spec"]][mode].append(load_json(results_dir / row["path"]))

    leaderboard_rows: list[dict[str, Any]] = []
    for model_spec, payloads in by_model.items():
        offline_norm = [
            _normalize(
                record["evaluation"]["metrics"].get("OfflineScore"),
                record["random_baseline_report"]["metrics"].get("OfflineScore"),
                record["oracle_report"]["metrics"].get("OfflineScore"),
            )
            for record in payloads["offline"]
            if record.get("split") in {"dev", "test"}
        ]
        policy_norm = [
            _normalize(
                record["evaluation"]["metrics"].get("PolicyScore"),
                record["random_baseline_report"]["metrics"].get("PolicyScore"),
                record["oracle_report"]["metrics"].get("PolicyScore"),
            )
            for record in payloads["policy"]
        ]
        robustness_norm = [
            _normalize(
                record["evaluation"]["summary"]["average_metrics"].get("EnterpriseAllocationScore"),
                record["random_baseline_report"]["summary"]["average_metrics"].get("EnterpriseAllocationScore"),
                record["oracle_report"]["summary"]["average_metrics"].get("EnterpriseAllocationScore"),
            )
            for record in payloads["robustness"]
        ]
        fit_axis = [
            _normalize(
                record["evaluation"]["metrics"].get("FitScore"),
                record["random_baseline_report"]["metrics"].get("FitScore"),
                record["oracle_report"]["metrics"].get("FitScore"),
            )
            for record in payloads["offline"]
            if record.get("split") in {"dev", "test"}
        ]
        timing_axis = [
            _normalize(
                record["evaluation"]["metrics"].get("TimingScore"),
                record["random_baseline_report"]["metrics"].get("TimingScore"),
                record["oracle_report"]["metrics"].get("TimingScore"),
            )
            for record in payloads["offline"]
            if record.get("split") in {"dev", "test"}
        ]
        contact_axis = [
            _normalize(
                record["evaluation"]["metrics"].get("ContactScore"),
                record["random_baseline_report"]["metrics"].get("ContactScore"),
                record["oracle_report"]["metrics"].get("ContactScore"),
            )
            for record in payloads["offline"]
            if record.get("split") in {"dev", "test"}
        ]
        grounding_axis = [
            _normalize(
                record["evaluation"]["metrics"].get("GroundingScore"),
                record["random_baseline_report"]["metrics"].get("GroundingScore"),
                record["oracle_report"]["metrics"].get("GroundingScore"),
            )
            for record in payloads["offline"]
            if record.get("split") in {"dev", "test"}
        ]

        offline_score = _average(offline_norm)
        policy_score = _average(policy_norm)
        robustness_score = _average(robustness_norm)
        public_score = _average([offline_score, policy_score, robustness_score])
        total_input_tokens = sum(
            record["artifact"]["usage_log"].get("input_tokens", 0)
            for mode in payloads.values()
            for record in mode
        )
        total_output_tokens = sum(
            record["artifact"]["usage_log"].get("output_tokens", 0)
            for mode in payloads.values()
            for record in mode
        )
        total_latency_ms = sum(
            record["artifact"]["usage_log"].get("latency_ms", 0)
            for mode in payloads.values()
            for record in mode
        )
        total_cost_usd = round(
            sum(
                float(record.get("cost_usd", 0.0))
                for mode in payloads.values()
                for record in mode
            ),
            6,
        )
        leaderboard_rows.append(
            {
                "model_spec": model_spec,
                "offline_score": offline_score,
                "policy_score": policy_score,
                "robustness_score": robustness_score,
                "public_score": public_score,
                "fit_axis": _average(fit_axis),
                "timing_axis": _average(timing_axis),
                "contact_axis": _average(contact_axis),
                "grounding_axis": _average(grounding_axis),
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "latency_ms": total_latency_ms,
                "cost_usd": total_cost_usd,
            }
        )

    return sorted(
        leaderboard_rows,
        key=lambda row: (row["public_score"] is not None, row["public_score"] or -999.0),
        reverse=True,
    )


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| Model | Public | Offline | Policy | Robustness | Fit | Timing | Contact | Grounding | Tokens In | Tokens Out | Latency ms | Cost USD |\n"
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    lines = [header, divider]
    for row in rows:
        lines.append(
            "| {model} | {public} | {offline} | {policy} | {robustness} | {fit} | {timing} | {contact} | {grounding} | {inp} | {out} | {latency} | {cost} |\n".format(
                model=row["model_spec"],
                public=row["public_score"],
                offline=row["offline_score"],
                policy=row["policy_score"],
                robustness=row["robustness_score"],
                fit=row["fit_axis"],
                timing=row["timing_axis"],
                contact=row["contact_axis"],
                grounding=row["grounding_axis"],
                inp=row["input_tokens"],
                out=row["output_tokens"],
                latency=row["latency_ms"],
                cost=row["cost_usd"],
            )
        )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SDR Bench sweep outputs into a leaderboard.")
    parser.add_argument("--results", required=True, help="Results directory produced by sweep.py.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print leaderboard JSON.")
    args = parser.parse_args()

    results_dir = Path(args.results)
    index = load_json(results_dir / "index.json")
    rows = _collect_model_rows(index, results_dir)
    leaderboard = {
        "results_dir": str(results_dir),
        "models": rows,
    }
    dump_json(results_dir / "leaderboard.json", leaderboard, pretty=args.pretty)
    (results_dir / "leaderboard.md").write_text(_markdown_table(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
