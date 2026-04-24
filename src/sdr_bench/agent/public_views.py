"""Public model-visible views over SDR Bench evaluation windows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from sdr_bench.agent.types import PublicWindowView
from sdr_bench.agent.types import WindowIndexes


SCORING_ONLY_KEYS = frozenset(
    {
        "account_outcomes",
        "agent_incremental_values",
        "allowed_for_grounding",
        "contact_outcomes",
        "cooldown_weeks",
        "deal_size",
        "grounding_support",
        "hidden_labels",
        "label_window_id",
        "labels",
        "latents",
        "meeting_gain",
        "meeting_prob",
        "next_window_outcome_modifiers",
        "opp_gain",
        "opp_prob",
        "oracle_decisions",
        "oracle_incremental_values",
        "pipeline_gain",
        "pipeline_value",
        "policy_transitions",
        "potential_outcomes",
        "relevance_gain",
        "remove_from_episode",
        "trigger_outcomes",
    }
)

_ENTITY_ID_KEYS = {
    "accounts": "account_id",
    "contacts": "contact_id",
    "triggers": "event_id",
    "evidence": "doc_id",
}


def redact_scoring_fields(payload: Any) -> Any:
    """Return a deep copy with hidden/scoring-only fields removed."""

    if isinstance(payload, dict):
        return {
            key: redact_scoring_fields(value)
            for key, value in payload.items()
            if key not in SCORING_ONLY_KEYS
        }
    if isinstance(payload, list):
        return [redact_scoring_fields(value) for value in payload]
    return deepcopy(payload)


def publicize_window(window_data: dict[str, Any]) -> dict[str, Any]:
    """Build the public, model-visible form of an evaluation window."""

    return redact_scoring_fields(window_data)


def _entity_id(entity: dict[str, Any], id_key: str, collection_name: str) -> str:
    raw_id = entity.get(id_key)
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError(f"{collection_name} entity is missing non-empty {id_key}")
    return raw_id


def _sorted_entities(
    entities: list[dict[str, Any]],
    *,
    collection_name: str,
    id_key: str,
) -> list[dict[str, Any]]:
    return sorted(
        entities,
        key=lambda entity: _entity_id(entity, id_key, collection_name),
    )


def _by_id(
    entities: list[dict[str, Any]],
    *,
    collection_name: str,
    id_key: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for entity in _sorted_entities(
        entities,
        collection_name=collection_name,
        id_key=id_key,
    ):
        entity_id = _entity_id(entity, id_key, collection_name)
        if entity_id in indexed:
            raise ValueError(f"duplicate {id_key} in {collection_name}: {entity_id}")
        indexed[entity_id] = entity
    return indexed


def _group_by_account(
    entities: list[dict[str, Any]],
    *,
    account_ids: list[str],
    collection_name: str,
    id_key: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped = {account_id: [] for account_id in account_ids}
    for entity in sorted(
        entities,
        key=lambda item: (
            str(item.get("account_id", "")),
            _entity_id(item, id_key, collection_name),
        ),
    ):
        account_id = entity.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError(f"{collection_name} entity is missing non-empty account_id")
        grouped.setdefault(account_id, []).append(entity)
    return {
        account_id: grouped[account_id]
        for account_id in sorted(grouped)
    }


def _public_entities(
    public_window: dict[str, Any],
    collection_name: str,
) -> list[dict[str, Any]]:
    raw_entities = public_window.get(collection_name, [])
    if not isinstance(raw_entities, list):
        raise ValueError(f"window {collection_name} must be a list")
    for entity in raw_entities:
        if not isinstance(entity, dict):
            raise ValueError(f"window {collection_name} entries must be objects")
    return raw_entities


def _build_indexes_from_public_window(public_window: dict[str, Any]) -> WindowIndexes:
    accounts = _public_entities(public_window, "accounts")
    contacts = _public_entities(public_window, "contacts")
    triggers = _public_entities(public_window, "triggers")
    evidence = _public_entities(public_window, "evidence")

    accounts_by_id = _by_id(
        accounts,
        collection_name="accounts",
        id_key=_ENTITY_ID_KEYS["accounts"],
    )
    account_ids = sorted(accounts_by_id)

    return WindowIndexes(
        accounts_by_id=accounts_by_id,
        contacts_by_id=_by_id(
            contacts,
            collection_name="contacts",
            id_key=_ENTITY_ID_KEYS["contacts"],
        ),
        triggers_by_id=_by_id(
            triggers,
            collection_name="triggers",
            id_key=_ENTITY_ID_KEYS["triggers"],
        ),
        evidence_by_id=_by_id(
            evidence,
            collection_name="evidence",
            id_key=_ENTITY_ID_KEYS["evidence"],
        ),
        contacts_by_account=_group_by_account(
            contacts,
            account_ids=account_ids,
            collection_name="contacts",
            id_key=_ENTITY_ID_KEYS["contacts"],
        ),
        triggers_by_account=_group_by_account(
            triggers,
            account_ids=account_ids,
            collection_name="triggers",
            id_key=_ENTITY_ID_KEYS["triggers"],
        ),
        evidence_by_account=_group_by_account(
            evidence,
            account_ids=account_ids,
            collection_name="evidence",
            id_key=_ENTITY_ID_KEYS["evidence"],
        ),
    )


def build_window_indexes(window_data: dict[str, Any]) -> WindowIndexes:
    """Build deterministic indexes over a publicized copy of a window."""

    return _build_indexes_from_public_window(publicize_window(window_data))


def build_public_window_view(window_data: dict[str, Any]) -> PublicWindowView:
    """Return the redacted window plus deterministic public indexes."""

    public_window = publicize_window(window_data)
    window_id = public_window.get("window_id")
    if not isinstance(window_id, str) or not window_id:
        raise ValueError("window is missing non-empty window_id")
    return PublicWindowView(
        window_id=window_id,
        window=public_window,
        indexes=_build_indexes_from_public_window(public_window),
    )


def public_window_view_to_dict(view: PublicWindowView) -> dict[str, Any]:
    """Convert a public view dataclass to a plain JSON-compatible dict."""

    return asdict(view)


def find_scoring_fields(payload: Any, *, prefix: str = "") -> list[str]:
    """Return dotted paths for any known hidden/scoring-only keys."""

    if isinstance(payload, dict):
        paths: list[str] = []
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in SCORING_ONLY_KEYS:
                paths.append(path)
                continue
            paths.extend(find_scoring_fields(value, prefix=path))
        return paths
    if isinstance(payload, list):
        paths = []
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(find_scoring_fields(value, prefix=path))
        return paths
    return []


def get_public_account_context(
    view: PublicWindowView,
    account_id: str,
) -> dict[str, Any]:
    """Return account-local public context from a prepared public view."""

    account = view.indexes.accounts_by_id.get(account_id)
    if account is None:
        raise KeyError(account_id)
    return {
        "account": account,
        "contacts": view.indexes.contacts_by_account.get(account_id, []),
        "triggers": view.indexes.triggers_by_account.get(account_id, []),
        "evidence": view.indexes.evidence_by_account.get(account_id, []),
    }
