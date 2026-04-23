"""CLI and orchestration for stepwise policy episode runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sdr_bench.io_utils import dump_json
from sdr_bench.io_utils import load_json
from sdr_bench.runner.adapters import load_adapter
from sdr_bench.runner.prompts import build_episode_prompt
from sdr_bench.runner.repair import generate_with_repair
from sdr_bench.simulator.env import SDRBenchEnv


def _prompt_hash(system: str, user: str, schema: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "system": system,
            "user": user,
            "schema": schema,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def infer_episode_labels_path(episode_path: str | Path) -> Path:
    episode_path = Path(episode_path)
    candidate = episode_path.parent.parent / "episode_labels" / episode_path.name
    if candidate.exists():
        return candidate
    alt_candidate = episode_path.parent.parent / "episode_labels" / f"{episode_path.stem}_labels.json"
    if alt_candidate.exists():
        return alt_candidate
    raise FileNotFoundError(f"Could not infer episode labels for {episode_path}")


def run_policy_episode_model(
    episode_data: dict[str, Any],
    labels_data: dict[str, Any],
    *,
    model_spec: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    adapter = load_adapter(model_spec)
    env = SDRBenchEnv(episode_data, seed=seed, labels_data=labels_data)
    state = env.reset()
    history: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    window_logs: list[dict[str, Any]] = []
    prompt_hashes: list[str] = []

    while state is not None:
        system, user, schema = build_episode_prompt(state, history)
        prompt_hashes.append(_prompt_hash(system, user, schema))
        result = generate_with_repair(
            adapter,
            window_data=state["window"],
            system=system,
            user=user,
            json_schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        next_state, outcome, done = env.step(result.submission)
        submissions.append(result.submission)
        window_logs.append(
            {
                "window_id": state["window_id"],
                "usage_log": result.usage_log,
                "outcome": {
                    "metrics": outcome["metrics"],
                    "issue_count": len(outcome["issues"]),
                    "compliance": outcome["compliance"],
                },
            }
        )
        history.append(
            {
                "window_id": state["window_id"],
                "metrics": outcome["metrics"],
                "issue_count": len(outcome["issues"]),
                "effective_human_touch_count": outcome["compliance"]["effective_human_touch_count"],
            }
        )
        state = None if done else next_state

    total_input_tokens = sum(item["usage_log"]["input_tokens"] for item in window_logs)
    total_output_tokens = sum(item["usage_log"]["output_tokens"] for item in window_logs)
    total_latency_ms = sum(item["usage_log"]["latency_ms"] for item in window_logs)
    return {
        "policy_submission": {
            "episode_id": episode_data["episode_id"],
            "submissions": submissions,
        },
        "usage_log": {
            "windows": window_logs,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_ms": total_latency_ms,
        },
        "prompt_hashes": prompt_hashes,
        "model_spec": model_spec,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model adapter through an SDR Bench policy episode.")
    parser.add_argument("--episode", required=True, help="Path to a public policy episode JSON file.")
    parser.add_argument("--episode-labels", help="Optional path to the hidden policy episode labels.")
    parser.add_argument("--model", required=True, help="Model adapter spec.")
    parser.add_argument("--out", required=True, help="Output artifact path.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum completion tokens per week.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic run seed recorded in output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    episode_data = load_json(args.episode)
    labels_path = Path(args.episode_labels) if args.episode_labels else infer_episode_labels_path(args.episode)
    labels_data = load_json(labels_path)
    artifact = run_policy_episode_model(
        episode_data,
        labels_data,
        model_spec=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    dump_json(args.out, artifact, pretty=args.pretty)


if __name__ == "__main__":
    main()
