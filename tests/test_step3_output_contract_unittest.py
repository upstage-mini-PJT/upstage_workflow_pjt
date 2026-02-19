import unittest

from agents.data_analysis_agent.pipeline import (
    _apply_domain_filter,
    _build_next_steps,
    _build_user_guidance_node,
    _dedupe_actions,
    _finalize_node,
)
from core.schemas.case_context import normalize_structured_case
from main import _mask_payload_recursive, _render_user_guidance_4sections


class Step3OutputContractTests(unittest.TestCase):
    def _sample_case(self) -> dict:
        return normalize_structured_case(
            {
                "denial_summary": "실손 의료비 청구 부지급",
                "denial_reasons": ["실손", "입원 의료비 부지급"],
                "policy_clauses": ["실손의료비 약관 제1조"],
                "timeline": [],
                "evidence_summary": [],
                "open_questions": [],
            }
        )

    def test_domain_filter_removes_unrelated_items_when_valid_pool_exists(self):
        items = [
            {
                "source_type": "CASELAW",
                "doc_id": f"CASE-{idx}",
                "title": "실손의료비 입원 치료비 부지급 판례",
                "snippet": "약관 해석 및 보험금 청구 관련 판단",
                "score": 0.9 - idx * 0.01,
                "rerank_score": 0.95 - idx * 0.01,
            }
            for idx in range(1, 7)
        ]
        items.append(
            {
                "source_type": "DISPUTE",
                "doc_id": "AUTO-1",
                "title": "자동차 교통사고 대물 분쟁",
                "snippet": "차량 수리비와 대인 배상 분쟁 사례",
                "score": 0.99,
                "rerank_score": 0.99,
            }
        )

        filtered, trace = _apply_domain_filter(items, self._sample_case())
        filtered_ids = {str(item.get("doc_id", "")) for item in filtered}

        self.assertIn(trace.get("mode"), {"strict", "relaxed", "fallback_soft_deny", "fallback_hard"})
        self.assertGreaterEqual(len(filtered), 5)
        self.assertNotIn("AUTO-1", filtered_ids)

    def test_domain_filter_fallback_preserves_minimum_items(self):
        items = [
            {
                "source_type": "CASELAW",
                "doc_id": "AUTO-2",
                "title": "자동차 대물 사고 분쟁",
                "snippet": "교통사고 차량 손해 위주",
                "score": 0.8,
                "rerank_score": 0.8,
            },
            {
                "source_type": "DISPUTE",
                "doc_id": "AUTO-3",
                "title": "운전자 보험 대인 분쟁",
                "snippet": "차량 사고 손해배상",
                "score": 0.7,
                "rerank_score": 0.7,
            },
        ]

        filtered, trace = _apply_domain_filter(items, self._sample_case())
        self.assertTrue(filtered)
        self.assertIn(trace.get("mode"), {"relaxed", "fallback_soft_deny", "fallback_hard"})

    def test_domain_filter_soft_deny_cap_is_applied(self):
        items = [
            {
                "source_type": "CASELAW",
                "doc_id": f"CASE-{idx}",
                "title": "실손 의료비 약관 분쟁 판례",
                "snippet": "입원 치료비 보험금 청구 관련 판단",
                "score": 0.8,
                "rerank_score": 0.8,
            }
            for idx in range(1, 5)
        ]
        items.extend(
            [
                {
                    "source_type": "DISPUTE",
                    "doc_id": "SOFT-1",
                    "title": "실손 자동차 혼합 분쟁",
                    "snippet": "보험금 청구와 교통사고 쟁점이 함께 있음",
                    "score": 0.79,
                    "rerank_score": 0.79,
                },
                {
                    "source_type": "DISPUTE",
                    "doc_id": "SOFT-2",
                    "title": "의료비 차량 사고 혼합 분쟁",
                    "snippet": "입원 치료비와 자동차 사고가 함께 언급",
                    "score": 0.78,
                    "rerank_score": 0.78,
                },
            ]
        )

        filtered, trace = _apply_domain_filter(items, self._sample_case())
        self.assertEqual(trace.get("mode"), "fallback_soft_deny")
        self.assertEqual(len(filtered), 5)
        self.assertLessEqual(int(trace.get("soft_deny_included_count", 0)), 1)

    def test_actions_and_next_steps_count_are_aligned(self):
        raw_actions = [
            {
                "action_id": "ACTION-1",
                "title": "쟁점별 반박서 + 추가 증빙 병행",
                "detail": "핵심 쟁점별 반박서를 작성하세요.",
                "priority": "high",
                "linked_issue_id": "ISSUE-1",
            },
            {
                "action_id": "ACTION-2",
                "title": "반박 논리 보강",
                "detail": "쟁점별 반박서 항목을 보강하세요.",
                "priority": "medium",
                "linked_issue_id": "ISSUE-1",
            },
            {
                "action_id": "ACTION-3",
                "title": "증빙 보강",
                "detail": "건강보험 급여/비급여 확인서를 준비하세요.",
                "priority": "high",
                "linked_issue_id": "ISSUE-2",
            },
            {
                "action_id": "ACTION-4",
                "title": "제출 체크",
                "detail": "제출 순서를 최종 점검하세요.",
                "priority": "medium",
                "linked_issue_id": "ISSUE-2",
            },
        ]
        deduped = _dedupe_actions(raw_actions)
        steps = _build_next_steps(
            actions=deduped,
            required_documents=["진단서/의사 소견서", "진료비 세부산정내역서/영수증"],
            conclusion="추가 증빙 보강이 필요합니다.",
            next_step_hint="쟁점별로 자료를 묶어 제출하세요.",
        )

        self.assertEqual(len(deduped), len(steps))
        self.assertTrue(deduped)
        for idx, action in enumerate(deduped, start=1):
            self.assertEqual(action.get("action_id"), f"ACTION-{idx}")

    def test_minimum_actions_are_enforced(self):
        deduped = _dedupe_actions(
            [
                {
                    "action_id": "ACTION-1",
                    "title": "쟁점별 반박서 작성",
                    "detail": "반박서를 작성하세요.",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-3",
                }
            ],
            default_issue_id="ISSUE-3",
            min_actions=2,
        )
        self.assertGreaterEqual(len(deduped), 2)
        for action in deduped:
            self.assertEqual(action.get("linked_issue_id"), "ISSUE-3")

    def test_finalize_contract_hides_score_and_internal_trace(self):
        state = {
            "issue_tree": {"root_title": "테스트", "nodes": [{"issue_id": "ISSUE-1", "title": "쟁점"}]},
            "gap_analysis": [{"issue_id": "ISSUE-1", "missing_evidence": "진단서", "impact": "high"}],
            "recommended_actions": [
                {
                    "action_id": "ACTION-1",
                    "title": "증빙 보강",
                    "detail": "진단서를 보강 제출하세요.",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-1",
                }
            ],
            "success_probability_public": {
                "band": "MEDIUM",
                "positive_drivers": ["근거 확보"],
                "negative_drivers": ["추가 확인 필요"],
                "assumptions": ["추가 소견서 제출 가정"],
            },
            "evidence_pack": [
                {
                    "evidence_id": "EV-1",
                    "issue_id": "ISSUE-1",
                    "evidence_title": "테스트 판례",
                    "summary": "요약",
                    "relevance_score": 0.9,
                    "provenance": [{"source_type": "caselaw", "source_id": "1"}],
                }
            ],
            "scoring_trace": {"total_score": 82},
            "internal_scoring": {"total_score": 82},
        }
        result = _finalize_node(state)["analysis_result"]

        self.assertTrue({"issue_tree", "gap_analysis", "recommended_actions", "success_probability", "evidence_pack"}.issubset(result.keys()))
        self.assertNotIn("score", result.get("success_probability", {}))
        self.assertNotIn("scoring_trace", result)
        self.assertNotIn("internal_scoring", result)

    def test_finalize_masks_pii_in_actions(self):
        state = {
            "issue_tree": {"root_title": "테스트", "nodes": [{"issue_id": "ISSUE-1", "title": "쟁점"}]},
            "gap_analysis": [],
            "recommended_actions": [
                {
                    "action_id": "ACTION-1",
                    "title": "양찬우 고객님 연락 및 계약번호 확인",
                    "detail": "연락처: 010-1234-5678, 계약번호: 2024-NS-556677",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-1",
                }
            ],
            "success_probability_public": {"band": "MEDIUM", "positive_drivers": [], "negative_drivers": [], "assumptions": []},
            "evidence_pack": [],
        }
        result = _finalize_node(state)["analysis_result"]
        action = result.get("recommended_actions", [{}])[0]
        self.assertNotIn("양찬우", str(action.get("title", "")))
        self.assertNotIn("010-1234-5678", str(action.get("detail", "")))
        self.assertNotIn("2024-NS-556677", str(action.get("detail", "")))

    def test_user_guidance_step_count_matches_recommended_actions(self):
        actions = _dedupe_actions(
            [
                {
                    "action_id": "ACTION-1",
                    "title": "쟁점별 반박서 작성",
                    "detail": "쟁점별 반박서를 정리하세요.",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-1",
                },
                {
                    "action_id": "ACTION-2",
                    "title": "증빙 보강",
                    "detail": "입퇴원 확인서를 준비하세요.",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-2",
                },
            ]
        )
        state = {
            "success_probability_public": {"band": "MEDIUM"},
            "recommended_actions": actions,
            "evidence_pack": [
                {
                    "evidence_id": "CASELAW:10",
                    "issue_id": "ISSUE-1",
                    "evidence_title": "판례 A",
                    "summary": "요약 A",
                    "provenance": [{"source_type": "caselaw", "source_id": "10"}],
                },
                {
                    "evidence_id": "DISPUTE:20",
                    "issue_id": "ISSUE-2",
                    "evidence_title": "분쟁사례 B",
                    "summary": "요약 B",
                    "provenance": [{"source_type": "dispute_case", "source_id": "20"}],
                },
            ],
            "gap_analysis": [
                {"issue_id": "ISSUE-1", "missing_evidence": "진단서", "impact": "high", "rationale": "치료 필요성 확인"},
                {"issue_id": "ISSUE-2", "missing_evidence": "입퇴원 기록", "impact": "medium", "rationale": "입원 사실 확인"},
            ],
            "issue_tree": {
                "nodes": [
                    {"issue_id": "ISSUE-1", "title": "약관 해석 쟁점", "description": "약관 문구 충돌"},
                    {"issue_id": "ISSUE-2", "title": "증빙 충족 쟁점", "description": "증빙 부족 여부"},
                ]
            },
            "structured_case": {
                "user_info": {
                    "required_documents": ["진단서/의사 소견서", "입퇴원 확인서"],
                }
            },
        }
        guidance = _build_user_guidance_node(state)["user_guidance"]
        self.assertEqual(len(guidance.get("next_steps", [])), len(actions))
        self.assertTrue(guidance.get("issue_brief"))

    def test_user_guidance_render_uses_issue_point_label(self):
        text = _render_user_guidance_4sections(
            {
                "plain_summary": "- 현재 판단: MEDIUM",
                "issue_brief": [
                    {
                        "issue_id": "ISSUE-1",
                        "title": "약관 해석 쟁점",
                        "why_it_matters": "핵심 판단에 직접 영향",
                        "needed_evidence": "진단서",
                    }
                ],
                "next_steps": ["1) 무엇: 증빙 보강\n   방법: 준비\n   이유: 필요\n   준비물: 진단서\n   완료기준: 완료"],
                "evidence_guide": [],
                "disclaimer": "참고용",
            }
        )
        self.assertIn("쟁점 포인트:", text)
        self.assertNotIn("왜 중요?:", text)

    def test_recursive_mask_masks_nested_payload(self):
        payload = {
            "decision_summary": {"user_situation": "양찬우 고객님 연락처: 010-1234-5678"},
            "analysis_result": {"recommended_actions": [{"detail": "계약번호: 2024-NS-556677"}]},
        }
        masked = _mask_payload_recursive(payload)
        serialized = str(masked)
        self.assertNotIn("양찬우", serialized)
        self.assertNotIn("010-1234-5678", serialized)
        self.assertNotIn("2024-NS-556677", serialized)


if __name__ == "__main__":
    unittest.main()
