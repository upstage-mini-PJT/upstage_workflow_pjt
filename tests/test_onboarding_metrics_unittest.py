import unittest

from evaluations.onboarding.metrics import (
    confidence_match_rate,
    decision_exact_match_rate,
    required_docs_exact_match_rate,
    retrieval_topk_jaccard_mean,
    sufficiency_flip_rate,
)


class OnboardingMetricsTests(unittest.TestCase):
    def test_metrics_basic(self):
        rows = [
            {
                "scenario_id": "S1",
                "decision_signature": "A",
                "required_document_ids": ["x", "y"],
                "evidence_sufficient": True,
                "final_plan_confidence": "high",
                "retrieval_top_source_ids": ["a", "b"],
            },
            {
                "scenario_id": "S1",
                "decision_signature": "A",
                "required_document_ids": ["x", "y"],
                "evidence_sufficient": False,
                "final_plan_confidence": "high",
                "retrieval_top_source_ids": ["a", "c"],
            },
            {
                "scenario_id": "S2",
                "decision_signature": "B",
                "required_document_ids": ["z"],
                "evidence_sufficient": True,
                "final_plan_confidence": "medium",
                "retrieval_top_source_ids": ["m"],
            },
            {
                "scenario_id": "S2",
                "decision_signature": "C",
                "required_document_ids": ["z"],
                "evidence_sufficient": True,
                "final_plan_confidence": "low",
                "retrieval_top_source_ids": ["m"],
            },
        ]

        self.assertAlmostEqual(decision_exact_match_rate(rows), 0.75, places=6)
        self.assertAlmostEqual(required_docs_exact_match_rate(rows), 1.0, places=6)
        self.assertAlmostEqual(sufficiency_flip_rate(rows), 0.5, places=6)
        self.assertAlmostEqual(confidence_match_rate(rows), 0.75, places=6)
        self.assertGreaterEqual(retrieval_topk_jaccard_mean(rows), 0.0)


if __name__ == "__main__":
    unittest.main()

