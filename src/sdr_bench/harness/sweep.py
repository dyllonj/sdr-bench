"""Run model sweeps across offline windows, policy episodes, and robustness cases."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from sdr_bench.baselines import generate_oracle_episode_submission
from sdr_bench.baselines import generate_oracle_window_submission
from sdr_bench.evaluator import evaluate_episode
from sdr_bench.evaluator import evaluate_episode_baseline
from sdr_bench.evaluator import evaluate_robustness_suite
from sdr_bench.evaluator import evaluate_robustness_suite_baseline
from sdr_bench.evaluator import evaluate_window
from sdr_bench.evaluator import evaluate_window_baseline
from sdr_bench.io_utils import dump_json
from sdr_bench.io_utils import load_json
from sdr_bench.runner.episode_runner import run_policy_episode_model
from sdr_bench.runner.run import run_window_model


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def load_models_file(path: str | Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                models.append(current)
            current = {}
            remainder = stripped[2:]
            if remainder:
                key, value = remainder.split(":", 1)
                current[key.strip()] = _parse_scalar(value)
            continue
        if current is None:
            raise ValueError("models file must start with a list item")
        key, value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(value)

    if current:
        models.append(current)
    return models


def _safe_model_name(spec: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", spec)


def _estimate_cost(usage_log: dict[str, Any], model_entry: dict[str, Any]) -> float:
    input_rate = float(model_entry.get("input_cost_per_mtok", 0.0))
    output_rate = float(model_entry.get("output_cost_per_mtok", 0.0))
    return round(
        usage_log.get("input_tokens", 0) / 1_000_000 * input_rate
        + usage_log.get("output_tokens", 0) / 1_000_000 * output_rate,
        6,
    )


def _discover_window_rows(data_dir: Path) -> list[tuple[str, Path, Path]]:
    rows: list[tuple[str, Path, Path]] = []
    for split in ("train", "dev", "test"):
        windows_dir = data_dir / split / "windows"
        labels_dir = data_dir / split / "labels"
        if not windows_dir.exists():
            continue
        for window_path in sorted(windows_dir.glob("*.json")):
            label_path = labels_dir / window_path.name
            rows.append((split, window_path, label_path))
    return rows


def _discover_episode_rows(data_dir: Path) -> list[tuple[Path, Path]]:
    episodes_dir = data_dir / "episodes"
    labels_dir = data_dir / "episode_labels"
    if not episodes_dir.exists():
        return []
    return [
        (episode_path, labels_dir / episode_path.name)
        for episode_path in sorted(episodes_dir.glob("*.json"))
    ]


def _run_offline_sweep(
    results_dir: Path,
    models: list[dict[str, Any]],
    data_dir: Path,
    budgets: list[int],
    *,
    seed: int,
    pretty: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, window_path, label_path in _discover_window_rows(data_dir):
        window_data = load_json(window_path)
        labels_data = load_json(label_path)
        for model in models:
            for budget in budgets:
                artifact = run_window_model(
                    window_data,
                    model_spec=model["spec"],
                    budget=budget,
                    seed=seed,
                )
                evaluation = evaluate_window(
                    window_data,
                    artifact["submission"],
                    labels_data=labels_data,
                    normalization_seed=seed,
                )
                random_report = evaluate_window_baseline(
                    window_data,
                    "random_within_icp",
                    labels_data=labels_data,
                    seed=seed,
                )["report"]
                oracle_submission = generate_oracle_window_submission(window_data, labels_data)
                oracle_report = evaluate_window(
                    window_data,
                    oracle_submission,
                    labels_data=labels_data,
                    normalization_seed=seed,
                )
                record = {
                    "mode": "offline",
                    "split": split,
                    "window_id": window_data["window_id"],
                    "budget": budget,
                    "model": model,
                    "artifact": artifact,
                    "evaluation": evaluation,
                    "random_baseline_report": random_report,
                    "oracle_report": oracle_report,
                    "cost_usd": _estimate_cost(artifact["usage_log"], model),
                }
                output_path = (
                    results_dir
                    / "offline"
                    / _safe_model_name(model["spec"])
                    / f"{split}__{window_data['window_id']}__b{budget}.json"
                )
                dump_json(output_path, record, pretty=pretty)
                rows.append(
                    {
                        "model_spec": model["spec"],
                        "split": split,
                        "window_id": window_data["window_id"],
                        "budget": budget,
                        "path": output_path.relative_to(results_dir).as_posix(),
                    }
                )
    return rows


def _run_policy_sweep(
    results_dir: Path,
    models: list[dict[str, Any]],
    data_dir: Path,
    *,
    seed: int,
    pretty: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_path, labels_path in _discover_episode_rows(data_dir):
        episode_data = load_json(episode_path)
        labels_data = load_json(labels_path)
        for model in models:
            artifact = run_policy_episode_model(
                episode_data,
                labels_data,
                model_spec=model["spec"],
                seed=seed,
            )
            evaluation = evaluate_episode(
                episode_data,
                artifact["policy_submission"],
                labels_data,
                normalization_seed=seed,
            )
            random_report = evaluate_episode_baseline(
                episode_data,
                "random_within_icp",
                labels_data=labels_data,
                seed=seed,
            )["report"]
            oracle_submission = generate_oracle_episode_submission(episode_data, labels_data)
            oracle_report = evaluate_episode(
                episode_data,
                oracle_submission,
                labels_data,
                normalization_seed=seed,
            )
            record = {
                "mode": "policy",
                "episode_id": episode_data["episode_id"],
                "model": model,
                "artifact": artifact,
                "evaluation": evaluation,
                "random_baseline_report": random_report,
                "oracle_report": oracle_report,
                "cost_usd": _estimate_cost(artifact["usage_log"], model),
            }
            output_path = (
                results_dir
                / "policy"
                / _safe_model_name(model["spec"])
                / f"{episode_data['episode_id']}.json"
            )
            dump_json(output_path, record, pretty=pretty)
            rows.append(
                {
                    "model_spec": model["spec"],
                    "episode_id": episode_data["episode_id"],
                    "path": output_path.relative_to(results_dir).as_posix(),
                }
            )
    return rows


def _run_robustness_sweep(
    results_dir: Path,
    models: list[dict[str, Any]],
    data_dir: Path,
    *,
    seed: int,
    pretty: bool,
) -> list[dict[str, Any]]:
    suite_path = data_dir / "robustness" / "suite.json"
    if not suite_path.exists():
        return []

    suite = load_json(suite_path)
    rows: list[dict[str, Any]] = []
    for model in models:
        case_submissions = []
        case_usage_logs = []
        for case in suite["cases"]:
            artifact = run_window_model(
                case["window"],
                model_spec=model["spec"],
                seed=seed,
            )
            case_submissions.append(
                {
                    "case_id": case["case_id"],
                    "submission": artifact["submission"],
                }
            )
            case_usage_logs.append(
                {
                    "case_id": case["case_id"],
                    "usage_log": artifact["usage_log"],
                }
            )
        suite_submission = {
            "suite_id": suite["suite_id"],
            "submissions": case_submissions,
        }
        evaluation = evaluate_robustness_suite(
            suite,
            suite_submission,
            include_case_reports=True,
            normalization_seed=seed,
        )
        random_report = evaluate_robustness_suite_baseline(
            suite,
            "random_within_icp",
            seed=seed,
            include_case_reports=True,
        )["report"]
        oracle_submission = {
            "suite_id": suite["suite_id"],
            "submissions": [
                {
                    "case_id": case["case_id"],
                    "submission": generate_oracle_window_submission(case["window"], case["labels"]),
                }
                for case in suite["cases"]
            ],
        }
        oracle_report = evaluate_robustness_suite(
            suite,
            oracle_submission,
            include_case_reports=True,
            normalization_seed=seed,
        )
        total_usage = {
            "input_tokens": sum(item["usage_log"]["input_tokens"] for item in case_usage_logs),
            "output_tokens": sum(item["usage_log"]["output_tokens"] for item in case_usage_logs),
            "latency_ms": sum(item["usage_log"]["latency_ms"] for item in case_usage_logs),
            "cases": case_usage_logs,
        }
        record = {
            "mode": "robustness",
            "model": model,
            "artifact": {
                "submission": suite_submission,
                "usage_log": total_usage,
                "model_spec": model["spec"],
                "seed": seed,
            },
            "evaluation": evaluation,
            "random_baseline_report": random_report,
            "oracle_report": oracle_report,
            "cost_usd": _estimate_cost(total_usage, model),
        }
        output_path = (
            results_dir
            / "robustness"
            / _safe_model_name(model["spec"])
            / "suite.json"
        )
        dump_json(output_path, record, pretty=pretty)
        rows.append(
            {
                "model_spec": model["spec"],
                "path": output_path.relative_to(results_dir).as_posix(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full SDR Bench sweep.")
    parser.add_argument("--models-file", required=True, help="Path to the models YAML file.")
    parser.add_argument("--data", required=True, help="Path to the generated dataset root.")
    parser.add_argument("--budgets", default="25,50,200", help="Comma-separated offline budgets.")
    parser.add_argument("--out", required=True, help="Output directory for results.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic run seed.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print result JSON.")
    args = parser.parse_args()

    data_dir = Path(args.data)
    results_dir = Path(args.out)
    models = load_models_file(args.models_file)
    budgets = [int(value.strip()) for value in args.budgets.split(",") if value.strip()]

    offline_runs = _run_offline_sweep(
        results_dir,
        models,
        data_dir,
        budgets,
        seed=args.seed,
        pretty=args.pretty,
    )
    policy_runs = _run_policy_sweep(
        results_dir,
        models,
        data_dir,
        seed=args.seed,
        pretty=args.pretty,
    )
    robustness_runs = _run_robustness_sweep(
        results_dir,
        models,
        data_dir,
        seed=args.seed,
        pretty=args.pretty,
    )
    index = {
        "seed": args.seed,
        "models": models,
        "offline_runs": offline_runs,
        "policy_runs": policy_runs,
        "robustness_runs": robustness_runs,
    }
    dump_json(results_dir / "index.json", index, pretty=True)


if __name__ == "__main__":
    main()
