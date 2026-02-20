import unittest

from agents.onboarding_agent.decision_diagnostics import (
    append_trace,
    build_trace_entry,
    canonical_json,
    compute_decision_signature,
    fingerprint,
)


class DecisionDiagnosticsTests(unittest.TestCase):
    def test_canonical_json_is_order_independent(self):
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(fingerprint(left), fingerprint(right))

    def test_append_trace_updates_latency_and_fallback_flag(self):
        state = {"decision_trace": [], "step_latency_ms": {}, "quality_flags": []}
        entry = build_trace_entry(
            step_id="issue_planning",
            input_obj={"a": 1},
            output_obj={"b": 2},
            rationale_summary="test",
            fallback_used=True,
            latency_ms=123,
        )
        trace, step_latency_ms, quality_flags = append_trace(state, entry)
        self.assertEqual(len(trace), 1)
        self.assertEqual(step_latency_ms.get("issue_planning"), 123)
        self.assertIn("fallback:issue_planning", quality_flags)

    def test_signature_ignores_explanation_text_changes(self):
        base_state = {
            "required_document_ids": ["billing_statement", "nhis_statement"],
            "evidence_sufficient": True,
            "final_plan_confidence": "high",
            "retrieval_candidates": [
                {"source_id": "A"},
                {"source_id": "B"},
            ],
            "decision_explanation": "첫 번째 설명",
        }
        changed_text_state = {
            **base_state,
            "decision_explanation": "완전히 다른 설명 문구",
        }
        self.assertEqual(
            compute_decision_signature(base_state),
            compute_decision_signature(changed_text_state),
        )

    def test_signature_changes_when_required_ids_change(self):
        state_a = {
            "required_document_ids": ["billing_statement", "nhis_statement"],
            "evidence_sufficient": True,
            "final_plan_confidence": "high",
            "retrieval_candidates": [{"source_id": "A"}],
        }
        state_b = {
            **state_a,
            "required_document_ids": ["billing_statement", "medical_certificate"],
        }
        self.assertNotEqual(
            compute_decision_signature(state_a),
            compute_decision_signature(state_b),
        )


if __name__ == "__main__":
    unittest.main()

