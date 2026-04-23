"""CLI entry point for building the synthetic SDR Bench dataset."""

from __future__ import annotations

import argparse

from sdr_bench.evaluator import validate_instance
from sdr_bench.simulator.corpus import build_corpus
from sdr_bench.simulator.corpus import corpus_files
from sdr_bench.simulator.corpus import write_corpus


def _validate_payloads(files: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for path, payload in files.items():
        if path.endswith("/windows/" + path.split("/")[-1]):
            schema_name = "evaluation_window"
        elif "/labels/" in path and not path.startswith("episode_labels/"):
            schema_name = "hidden_labels"
        elif path.startswith("episodes/"):
            schema_name = "policy_episode"
        elif path.startswith("episode_labels/"):
            schema_name = "policy_episode_labels"
        elif path == "robustness/suite.json":
            schema_name = "robustness_suite"
        elif path.endswith("submission_template.json") and path.startswith("robustness/"):
            schema_name = "robustness_submission"
        elif path.endswith("submission_template.json"):
            schema_name = "model_output"
        else:
            continue

        payload_errors = validate_instance(payload, schema_name)
        for error in payload_errors:
            errors.append(f"{path}: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic SDR Bench corpus.")
    parser.add_argument("--out", required=True, help="Output directory for the corpus.")
    parser.add_argument("--n-windows", type=int, default=12, help="Number of offline windows to generate.")
    parser.add_argument("--accounts-per-window", type=int, default=10000, help="Accounts per offline window.")
    parser.add_argument("--seed", type=int, default=1, help="Deterministic generator seed.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print emitted JSON files.")
    args = parser.parse_args()

    corpus = build_corpus(
        seed=args.seed,
        n_windows=args.n_windows,
        accounts_per_window=args.accounts_per_window,
    )
    files = corpus_files(corpus)
    validation_errors = _validate_payloads(files)
    if validation_errors:
        raise SystemExit("Corpus validation failed:\n- " + "\n- ".join(validation_errors))
    write_corpus(args.out, corpus, pretty=args.pretty)


if __name__ == "__main__":
    main()
