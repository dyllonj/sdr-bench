from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.evaluator import load_json
from sdr_bench.evaluator import validate_instance
from sdr_bench.rationale_codes import GROUNDING_CODE_FIELDS
from sdr_bench.rationale_codes import build_rationale_catalog
from sdr_bench.rationale_codes import supports_grounding_claim


class RationaleCodeTests(unittest.TestCase):
    def test_catalog_matches_schema(self) -> None:
        errors = validate_instance(build_rationale_catalog(), "rationale_codes")
        self.assertEqual(errors, [])

    def test_examples_only_use_known_codes(self) -> None:
        catalog = build_rationale_catalog()
        known_codes = {
            category: set(entries)
            for category, entries in catalog.items()
        }

        for path in sorted((ROOT_DIR / "examples").glob("*.json")):
            payload = load_json(path)
            for brief_key, support_key, is_multi in GROUNDING_CODE_FIELDS:
                found = self._extract_codes(payload, brief_key)
                for code in found:
                    self.assertIn(code, known_codes[support_key], msg=f"{path.name}: unknown code {code}")

    def test_sample_submission_claims_are_supported_by_dictionary(self) -> None:
        window = load_json(ROOT_DIR / "examples" / "sample_window.json")
        submission = load_json(ROOT_DIR / "examples" / "sample_submission.json")
        trigger_by_id = {
            trigger["event_id"]: trigger
            for trigger in window["triggers"]
        }
        evidence_by_id = {
            document["doc_id"]: document
            for document in window["evidence"]
        }

        decision = submission["decisions"][0]
        brief = decision["evidence_brief"]
        cited_docs = [evidence_by_id[doc_id] for doc_id in brief["citations"]]

        for brief_key, support_key, is_multi in GROUNDING_CODE_FIELDS:
            codes = brief[brief_key] if is_multi else [brief[brief_key]]
            for code in codes:
                self.assertTrue(
                    any(
                        supports_grounding_claim(
                            support_key,
                            code,
                            document,
                            trigger_by_id=trigger_by_id,
                        )
                        for document in cited_docs
                    ),
                    msg=f"unsupported sample claim: {support_key}:{code}",
                )

    def _extract_codes(self, payload: object, key: str) -> list[str]:
        found: list[str] = []
        if isinstance(payload, dict):
            for current_key, value in payload.items():
                if current_key == key:
                    if isinstance(value, list):
                        found.extend(item for item in value if isinstance(item, str))
                    elif isinstance(value, str):
                        found.append(value)
                found.extend(self._extract_codes(value, key))
        elif isinstance(payload, list):
            for item in payload:
                found.extend(self._extract_codes(item, key))
        return found


if __name__ == "__main__":
    unittest.main()
