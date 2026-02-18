import unittest
from unittest.mock import patch

from agents.onboarding_agent.nodes import (
    final_planning_node,
    issue_planning_node,
    request_additional_documents_node,
    retrieve_terms_node,
)


class PlanningRetrievalPipelineTests(unittest.TestCase):
    def test_issue_planning_fallback_without_chat_client(self):
        state = {
            "denial_statement_text": "보험사는 약관상 면책 및 고지의무 위반을 근거로 보험금 지급을 거절했습니다."
        }

        result = issue_planning_node(state, config={})

        self.assertTrue(result["issue_hypotheses"])
        self.assertTrue(result["query_plan"])
        self.assertIn("query_seed", result["query_plan"][0])
        self.assertIn("priority", result["query_plan"][0])

    @patch("tools.retrieve_terms.ensure_vectordb_ready")
    @patch("tools.retrieve_terms.retrieve_terms_candidates")
    @patch("tools.retrieve_terms.merge_candidates_rrf")
    @patch("tools.retrieve_terms.format_candidates_to_terms")
    @patch("agents.onboarding_agent.nodes._generate_hyde_queries")
    def test_retrieve_terms_uses_issue_plan_and_multi_queries(
        self,
        mock_generate_hyde_queries,
        mock_format_terms,
        mock_merge_rrf,
        mock_retrieve_candidates,
        mock_ensure_vectordb,
    ):
        mock_ensure_vectordb.return_value = object()
        mock_generate_hyde_queries.side_effect = (
            lambda _chat, _denial, intent: [f"hyde::{intent.get('query_seed', '')}"]
        )
        mock_retrieve_candidates.side_effect = lambda **kwargs: [
            {
                "source_id": f"src::{kwargs['query'][:20]}",
                "title": "[Section: 보장 | Article: 제1조 | Policy: 20200101]",
                "snippet": "보장 요건 관련 요약",
                "full_text": "약관 본문",
                "query": kwargs["query"],
                "rank": 1,
                "score": 0.0,
                "matched_keywords": [],
            }
        ]
        mock_merge_rrf.return_value = [
            {
                "source_id": "src::final",
                "title": "[Section: 보장 | Article: 제1조 | Policy: 20200101]",
                "snippet": "최종 선택 조항",
                "full_text": "최종 약관 본문",
                "score": 1.7,
                "rank": 1,
                "matched_keywords": ["면책"],
            }
        ]
        mock_format_terms.return_value = "FORMATTED_TERMS"

        state = {
            "denial_statement_text": "보험사는 상해 발생 경위가 약관상 면책 조항에 해당한다고 주장합니다.",
            "query_plan": [
                {
                    "intent": "면책 조항 확인",
                    "query_seed": "상해 면책 조항",
                    "must_keywords": ["면책", "상해"],
                    "priority": 1,
                },
                {
                    "intent": "보장 요건 확인",
                    "query_seed": "상해 보장 요건",
                    "must_keywords": ["보장", "요건"],
                    "priority": 2,
                },
            ],
            "policy_date": "20200101",
        }
        config = {"configurable": {"policy_vectordb": object()}}

        result = retrieve_terms_node(state, config)

        self.assertEqual(result["relevant_terms"], "FORMATTED_TERMS")
        self.assertGreaterEqual(len(result["retrieval_queries"]), 3)
        self.assertTrue(
            any("지급거절 약관 근거 보장 요건 면책" in q for q in result["retrieval_queries"])
        )
        self.assertEqual(mock_retrieve_candidates.call_count, len(result["retrieval_queries"]))
        self.assertEqual(result["retrieval_candidates"][0]["source_id"], "src::final")
        self.assertEqual(result["retrieval_candidates"][0]["matched_keywords"], ["면책"])

    def test_final_planning_fallback_without_chat_client(self):
        state = {
            "denial_statement_text": "보험사는 고지의무 위반을 근거로 지급거절 결정을 통보했습니다.",
            "relevant_terms": "",
            "issue_hypotheses": ["고지의무 위반 해당 여부"],
            "retrieval_candidates": [],
        }

        result = final_planning_node(state, config={})

        self.assertTrue(result["plan"])
        self.assertTrue(result["required_documents"])
        self.assertTrue(result["required_document_ids"])
        self.assertIn("denial_notice", result["required_document_ids"])
        self.assertEqual(result["final_plan_confidence"], "low")

    @patch("agents.onboarding_agent.nodes.interrupt")
    def test_request_additional_documents_uses_catalog_selection(self, mock_interrupt):
        mock_interrupt.return_value = ["/tmp/mock1.pdf", "/tmp/mock2.pdf"]
        state = {
            "required_document_ids": ["invalid_id", "denial_notice", "medical_certificate", "denial_notice"],
        }

        result = request_additional_documents_node(state, config={})

        payload = mock_interrupt.call_args.args[0]
        self.assertEqual(payload["required_document_ids"], ["denial_notice", "medical_certificate"])
        self.assertEqual(
            payload["required_documents"],
            ["보험금 지급거절 통지서 원문", "진단서/의사 소견서"],
        )
        self.assertEqual(len(payload["required_document_items"]), 2)
        self.assertEqual(result["additional_document_paths"], ["/tmp/mock1.pdf", "/tmp/mock2.pdf"])


if __name__ == "__main__":
    unittest.main()
