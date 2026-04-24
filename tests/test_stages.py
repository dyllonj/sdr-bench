from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sdr_bench.stages import BENCHMARK_STAGES
from sdr_bench.stages import FULL_CYCLE_SDR_STAGE_IDS
from sdr_bench.stages import TOP_OF_FUNNEL_STAGE_IDS
from sdr_bench.stages import get_benchmark_stage
from sdr_bench.stages import get_stage_ids_for_mode
from sdr_bench.stages import get_stages_for_mode


class BenchmarkStageTests(unittest.TestCase):
    def test_stage_ids_are_unique_and_lookup_is_stable(self) -> None:
        stage_ids = [stage.stage_id for stage in BENCHMARK_STAGES]

        self.assertEqual(len(stage_ids), len(set(stage_ids)))
        for stage_id in stage_ids:
            self.assertEqual(stage_id, get_benchmark_stage(stage_id).stage_id)

    def test_top_of_funnel_is_prefix_of_full_cycle_motion(self) -> None:
        self.assertEqual(
            TOP_OF_FUNNEL_STAGE_IDS,
            FULL_CYCLE_SDR_STAGE_IDS[: len(TOP_OF_FUNNEL_STAGE_IDS)],
        )
        self.assertEqual(
            TOP_OF_FUNNEL_STAGE_IDS,
            get_stage_ids_for_mode("top_of_funnel"),
        )
        self.assertEqual(
            FULL_CYCLE_SDR_STAGE_IDS,
            get_stage_ids_for_mode("full_cycle_sdr"),
        )

    def test_full_cycle_covers_sales_qualification_and_people_search_jobs(self) -> None:
        stages = get_stages_for_mode("full_cycle_sdr")
        all_jobs = {job for stage in stages for job in stage.agent_jobs}
        all_inspirations = {
            inspiration
            for stage in stages
            for inspiration in stage.inspirations
        }

        self.assertIn("prospect_search", all_jobs)
        self.assertIn("company_research", all_jobs)
        self.assertIn("outreach_generation", all_jobs)
        self.assertIn("lead_engagement", all_jobs)
        self.assertIn("capacity_allocation", all_jobs)
        self.assertIn("people_search_bench", all_inspirations)
        self.assertIn("microsoft_sales_qualification_bench", all_inspirations)

    def test_every_stage_has_outputs_scores_and_tools(self) -> None:
        for stage in BENCHMARK_STAGES:
            self.assertTrue(stage.label)
            self.assertTrue(stage.description)
            self.assertTrue(stage.outputs, stage.stage_id)
            self.assertTrue(stage.score_families, stage.stage_id)
            self.assertTrue(stage.model_visible_tools, stage.stage_id)


if __name__ == "__main__":
    unittest.main()
