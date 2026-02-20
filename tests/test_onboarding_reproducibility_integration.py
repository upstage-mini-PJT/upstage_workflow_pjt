import os
import unittest

from evaluations.onboarding.common import load_dataset, prepare_runtime, run_onboarding_once


@unittest.skipUnless(
    str(os.getenv("RUN_ONBOARDING_INTEGRATION_TESTS", "false")).strip().lower() == "true",
    "integration test disabled (set RUN_ONBOARDING_INTEGRATION_TESTS=true)",
)
class OnboardingReproducibilityIntegrationTests(unittest.TestCase):
    def test_same_input_keeps_core_decision_fields(self):
        dataset = load_dataset("evaluations/onboarding/dataset.jsonl")
        self.assertTrue(dataset)
        case = dataset[0]

        runtime = prepare_runtime()
        first = run_onboarding_once(case, runtime, repeat_index=0)
        second = run_onboarding_once(case, runtime, repeat_index=1)

        self.assertTrue(str(first.get("decision_signature", "")).strip())
        self.assertTrue(str(second.get("decision_signature", "")).strip())
        self.assertEqual(
            sorted(first.get("required_document_ids") or []),
            sorted(second.get("required_document_ids") or []),
        )
        self.assertEqual(first.get("evidence_sufficient"), second.get("evidence_sufficient"))
        self.assertEqual(
            str(first.get("final_plan_confidence", "")).strip().lower(),
            str(second.get("final_plan_confidence", "")).strip().lower(),
        )


if __name__ == "__main__":
    unittest.main()

