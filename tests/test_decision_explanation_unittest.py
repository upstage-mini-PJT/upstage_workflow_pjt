import unittest

from agents.onboarding_agent.nodes import explain_decision_node


class DecisionExplanationNodeTests(unittest.TestCase):
    def test_returns_empty_when_sufficiency_is_false(self):
        result = explain_decision_node({"evidence_sufficient": False}, config={})
        self.assertEqual(result, {})

    def test_builds_fallback_explanation_without_chat_client(self):
        state = {
            "evidence_sufficient": True,
            "denial_statement_text": (
                "수신: 양찬우 고객님\n"
                "계약번호: 2024-NS-556677\n"
                "연락처: 010-1234-5678\n"
                "약관상 면책 사유에 해당되어 보험금 지급이 어렵습니다."
            ),
            "relevant_terms": (
                "[Section: 보장하지 않는 손해 | Article: 제3조(면책)]\n"
                "고의 또는 약관상 면책 사유에 해당하는 경우 보장하지 않습니다."
            ),
            "plan": "현재 자료 기준 보험사 측 면책 주장 근거를 확인한 상태입니다.",
            "extracted_document_infos": [
                {
                    "key_data": "진단명: 특정 상해",
                    "evidence_or_grounds": "상해 발생 경위가 약관 면책 요건과 일부 일치",
                    "helpful_notes": "",
                }
            ],
        }

        result = explain_decision_node(state, config={})
        self.assertIn("decision_summary", result)
        self.assertIn("decision_explanation", result)
        self.assertTrue(result["decision_explanation"])
        self.assertIn("보험사", result["decision_summary"]["insurer_claim"])
        self.assertGreaterEqual(len(result["decision_summary"]["policy_clauses"]), 1)
        self.assertIn("약관 근거", result["decision_explanation"])
        self.assertIn("[Section:", result["decision_explanation"])
        self.assertIn("AI가 만든 참고용", result["decision_explanation"])
        self.assertIn("최종 판단과 결정의 책임", result["decision_explanation"])
        self.assertNotIn("양찬우", result["decision_explanation"])
        self.assertNotIn("2024-NS-556677", result["decision_explanation"])
        self.assertNotIn("010-1234-5678", result["decision_explanation"])


if __name__ == "__main__":
    unittest.main()
