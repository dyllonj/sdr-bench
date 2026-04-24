from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.agent import SellerKnowledgeError
from sdr_bench.agent import query_seller_knowledge
from sdr_bench.evaluator import load_json


class SellerKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_json(ROOT_DIR / "examples" / "sample_seller_profile.json")

    def test_query_filters_sample_seller_profile_by_section_and_query(self) -> None:
        result = query_seller_knowledge(
            self.profile,
            section="case_studies",
            query="financial",
            limit=2,
        )

        self.assertEqual("neutral_enterprise_tech_v1", result["seller_profile_id"])
        self.assertEqual("case_studies", result["section"])
        self.assertEqual(1, result["total_matches"])
        self.assertEqual("case_studies", result["items"][0]["section"])
        self.assertIn("financial", result["items"][0]["knowledge_id"])

    def test_query_rejects_invalid_section(self) -> None:
        with self.assertRaises(SellerKnowledgeError) as context:
            query_seller_knowledge(self.profile, section="hidden")

        self.assertEqual("unknown_section", context.exception.code)
        self.assertIn("allowed_sections", context.exception.details)

    def test_query_rejects_invalid_limit(self) -> None:
        with self.assertRaises(SellerKnowledgeError) as context:
            query_seller_knowledge(self.profile, limit=0)

        self.assertEqual("invalid_limit", context.exception.code)
        self.assertEqual(20, context.exception.details["max_limit"])


if __name__ == "__main__":
    unittest.main()
