"""CLI for running a model over one offline evaluation window."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from sdr_bench.agent.runner import run_window_agent_model as run_window_tool_agent
from sdr_bench.io_utils import dump_json
from sdr_bench.io_utils import load_json
from sdr_bench.runner.adapters import load_adapter
from sdr_bench.runner.prompts import build_window_prompt
from sdr_bench.runner.repair import generate_with_repair


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


def run_window_model(
    window_data: dict[str, Any],
    *,
    model_spec: str,
    budget: int | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    adapter = load_adapter(model_spec)
    system, user, schema = build_window_prompt(window_data, budget=budget)
    result = generate_with_repair(
        adapter,
        window_data=window_data,
        system=system,
        user=user,
        json_schema=schema,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "submission": result.submission,
        "usage_log": result.usage_log,
        "prompt_hash": _prompt_hash(system, user, schema),
        "model_spec": model_spec,
        "seed": seed,
    }


def run_window_model_with_tools(
    window_data: dict[str, Any],
    *,
    model_spec: str,
    budget: int | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    adapter = load_adapter(model_spec)
    if not hasattr(adapter, "create_turn"):
        raise RuntimeError(f"Adapter {model_spec} does not support tool-mode turns.")
    return run_window_tool_agent(
        window_data,
        adapter=adapter,
        model_spec=model_spec,
        budget=budget,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SDR Bench model adapter on one offline window.")
    parser.add_argument("--window", required=True, help="Path to an evaluation window JSON file.")
    parser.add_argument("--model", required=True, help="Model adapter spec.")
    parser.add_argument("--out", required=True, help="Path to write the run artifact JSON.")
    parser.add_argument("--budget", type=int, help="Optional override for human SDR action capacity.")
    parser.add_argument(
        "--interaction-mode",
        choices=("direct", "tools"),
        default="direct",
        help="Model interaction mode. direct preserves the original full-prompt runner; tools uses the public agent sandbox.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic run seed recorded in output.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    window_data = load_json(args.window)
    if args.interaction_mode == "tools":
        artifact = run_window_model_with_tools(
            window_data,
            model_spec=args.model,
            budget=args.budget,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    else:
        artifact = run_window_model(
            window_data,
            model_spec=args.model,
            budget=args.budget,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
    dump_json(args.out, artifact, pretty=args.pretty)


if __name__ == "__main__":
    main()
