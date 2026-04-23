"""Corpus builder for SDR Bench synthetic datasets."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from sdr_bench.io_utils import dump_json
from sdr_bench.simulator.causal import HOLDOUT_INDUSTRIES
from sdr_bench.simulator.causal import HOLDOUT_REGION
from sdr_bench.simulator.generator import evolve_account_blueprint
from sdr_bench.simulator.generator import generate_window_bundle_from_blueprints
from sdr_bench.simulator.generator import sample_account_blueprint


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _window_id(index: int) -> str:
    return f"wk_2026_{index + 1:02d}"


def _select_split(index: int, n_windows: int) -> str:
    train_cutoff = max(1, int(n_windows * 0.60))
    dev_cutoff = max(train_cutoff + 1, int(n_windows * 0.80))
    if index < train_cutoff:
        return "train"
    if index < dev_cutoff:
        return "dev"
    return "test"


def _score_blueprint(blueprint: dict[str, Any], rng: random.Random) -> float:
    latents = blueprint["latents"]
    return (
        0.48 * latents["structural_fit"]
        + 0.42 * latents["timing_heat"]
        + 0.10 * latents["channel_reach"]
        + rng.uniform(-0.05, 0.05)
    )


def _wait_submission_template(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "window_id": window["window_id"],
        "decisions": [
            {
                "account_id": account["account_id"],
                "chosen_action": "wait",
                "action_score": 0.0,
            }
            for account in window["accounts"]
        ],
    }


def _robustness_submission_template(suite: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite_id": suite["suite_id"],
        "submissions": [
            {
                "case_id": case["case_id"],
                "submission": _wait_submission_template(case["window"]),
            }
            for case in suite["cases"]
        ],
    }


def _materialize_offline_windows(
    rng: random.Random,
    *,
    n_windows: int,
    accounts_per_window: int,
) -> dict[str, list[dict[str, Any]]]:
    pool_size = max(accounts_per_window * 2, accounts_per_window + 32)
    master_pool = [
        sample_account_blueprint(rng, account_index=index + 1, allow_holdouts=False)
        for index in range(pool_size)
    ]
    split_rows = {"train": [], "dev": [], "test": []}

    for window_index in range(n_windows):
        scenario = {
            "human_budget_ratio": 0.05 + 0.015 * ((window_index % 3) + 1),
        }
        if window_index:
            for blueprint in master_pool:
                evolve_account_blueprint(blueprint, rng, scenario=scenario)

        ranked = sorted(
            master_pool,
            key=lambda blueprint: (_score_blueprint(blueprint, rng), blueprint["account_id"]),
            reverse=True,
        )
        active_blueprints = ranked[:accounts_per_window]
        bundle = generate_window_bundle_from_blueprints(
            active_blueprints,
            window_id=_window_id(window_index),
            week_index=window_index,
            scenario=scenario,
            seed=rng.randint(0, 10_000_000),
        )
        split_rows[_select_split(window_index, n_windows)].append(bundle)

    return split_rows


def _build_episode_artifacts(
    rng: random.Random,
    *,
    accounts_per_window: int,
    episode_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    episode_labels: list[dict[str, Any]] = []
    episode_length = 4

    for episode_index in range(episode_count):
        pool = [
            sample_account_blueprint(
                rng,
                account_index=100_000 + episode_index * accounts_per_window + index + 1,
                allow_holdouts=False,
            )
            for index in range(accounts_per_window)
        ]
        episode_windows: list[dict[str, Any]] = []
        episode_label_windows: list[dict[str, Any]] = []
        for week_index in range(episode_length):
            if week_index:
                for blueprint in pool:
                    evolve_account_blueprint(
                        blueprint,
                        rng,
                        scenario={"trigger_shift": 0.03 if week_index % 2 else 0.0},
                    )
            bundle = generate_window_bundle_from_blueprints(
                pool,
                window_id=f"ep_{episode_index + 1:02d}_wk_{week_index + 1:02d}",
                week_index=week_index,
                scenario={"human_budget_ratio": 0.08},
                seed=rng.randint(0, 10_000_000),
            )
            episode_windows.append(bundle["window"])
            episode_label_windows.append(
                {
                    "window_id": bundle["window"]["window_id"],
                    "labels": bundle["labels"],
                    "policy_transitions": bundle["policy_transitions"],
                }
            )

        episode_id = f"ep_{episode_index + 1:02d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "windows": episode_windows,
            }
        )
        episode_labels.append(
            {
                "episode_id": episode_id,
                "windows": episode_label_windows,
            }
        )

    return episodes, episode_labels


def _build_holdout_pool(
    rng: random.Random,
    *,
    count: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    account_index = 500_000
    while len(pool) < count:
        blueprint = sample_account_blueprint(
            rng,
            account_index=account_index,
            allow_holdouts=True,
        )
        account_index += 1
        if (
            blueprint["industry"] in HOLDOUT_INDUSTRIES
            or blueprint["hq_region"] == HOLDOUT_REGION
        ):
            pool.append(blueprint)
    return pool


def _build_robustness_suite(
    rng: random.Random,
    *,
    accounts_per_window: int,
) -> dict[str, Any]:
    holdout_pool = _build_holdout_pool(rng, count=accounts_per_window)
    heldout_bundle = generate_window_bundle_from_blueprints(
        holdout_pool,
        window_id="wk_holdout_reserved",
        week_index=0,
        scenario={"human_budget_ratio": 0.07},
        seed=rng.randint(0, 10_000_000),
    )

    shift_pool = [
        sample_account_blueprint(
            rng,
            account_index=700_000 + index,
            allow_holdouts=False,
        )
        for index in range(accounts_per_window)
    ]
    for blueprint in shift_pool:
        evolve_account_blueprint(
            blueprint,
            rng,
            scenario={"trigger_shift": 0.12, "fatigue_shift": 0.06},
        )
    shifted_bundle = generate_window_bundle_from_blueprints(
        shift_pool,
        window_id="wk_shift_trigger_heavy",
        week_index=0,
        scenario={
            "trigger_multiplier": 1.35,
            "trigger_shift": 0.12,
            "fatigue_shift": 0.06,
            "enterprise_share": 0.82,
            "human_budget_ratio": 0.06,
        },
        seed=rng.randint(0, 10_000_000),
    )

    return {
        "suite_id": "robust_v0_synth",
        "cases": [
            {
                "case_id": "heldout_reserved_slice",
                "robustness_type": "heldout_slice",
                "description": "Held-out industries and region reserved from the standard training windows.",
                "holdout_dimension": "industry",
                "holdout_values": list(HOLDOUT_INDUSTRIES),
                "window": heldout_bundle["window"],
                "labels": heldout_bundle["labels"],
            },
            {
                "case_id": "distribution_shift_trigger_heavy",
                "robustness_type": "distribution_shift",
                "description": "Higher trigger prevalence, higher fatigue, and a more enterprise-heavy account mix.",
                "shift_tags": ["trigger_prevalence_up", "fatigue_up", "segment_mix_enterprise_up"],
                "window": shifted_bundle["window"],
                "labels": shifted_bundle["labels"],
            },
        ],
    }


def build_corpus(
    *,
    seed: int,
    n_windows: int,
    accounts_per_window: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    split_bundles = _materialize_offline_windows(
        rng,
        n_windows=n_windows,
        accounts_per_window=accounts_per_window,
    )
    episodes, episode_labels = _build_episode_artifacts(
        rng,
        accounts_per_window=max(8, min(accounts_per_window, 128)),
        episode_count=2 if n_windows >= 8 else 1,
    )
    robustness_suite = _build_robustness_suite(
        rng,
        accounts_per_window=max(8, min(accounts_per_window, 128)),
    )
    first_test_window = (
        split_bundles["test"][0]["window"]
        if split_bundles["test"]
        else split_bundles["dev"][0]["window"]
    )

    return {
        "seed": seed,
        "splits": split_bundles,
        "episodes": episodes,
        "episode_labels": episode_labels,
        "robustness_suite": robustness_suite,
        "submission_template": _wait_submission_template(first_test_window),
        "robustness_submission_template": _robustness_submission_template(robustness_suite),
    }


def corpus_files(corpus: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for split, bundles in corpus["splits"].items():
        for bundle in bundles:
            window_id = bundle["window"]["window_id"]
            files[f"{split}/windows/{window_id}.json"] = bundle["window"]
            files[f"{split}/labels/{window_id}.json"] = bundle["labels"]

    for episode in corpus["episodes"]:
        files[f"episodes/{episode['episode_id']}.json"] = episode
    for labels in corpus["episode_labels"]:
        files[f"episode_labels/{labels['episode_id']}.json"] = labels

    files["robustness/suite.json"] = corpus["robustness_suite"]
    files["submission_template.json"] = corpus["submission_template"]
    files["robustness/submission_template.json"] = corpus["robustness_submission_template"]

    manifest = {
        "seed": corpus["seed"],
        "counts": {
            "train_windows": len(corpus["splits"]["train"]),
            "dev_windows": len(corpus["splits"]["dev"]),
            "test_windows": len(corpus["splits"]["test"]),
            "episodes": len(corpus["episodes"]),
            "robustness_cases": len(corpus["robustness_suite"]["cases"]),
        },
        "hashes": {
            path: _hash_payload(payload)
            for path, payload in sorted(files.items())
        },
    }
    files["manifest.json"] = manifest
    return files


def write_corpus(out_dir: str | Path, corpus: dict[str, Any], *, pretty: bool = True) -> list[Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for relative_path, payload in corpus_files(corpus).items():
        path = output_dir / relative_path
        dump_json(path, payload, pretty=pretty)
        written.append(path)
    return written
