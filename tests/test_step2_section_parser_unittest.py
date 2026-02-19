import unittest

from agents.data_analysis_agent.pipeline import (
    _build_next_steps,
    _build_user_guidance_node,
    _parse_numbered_sections,
    _simplify_korean_terms,
)


class Step2SectionParserTests(unittest.TestCase):
    def test_parse_parenthesis_format(self):
        text = """1) 지금 상황\n- A\n2) 보험사 판단\n- B\n5) 한 줄 결론\n- C\n7) 쉬운 설명\n- D"""
        sections = _parse_numbered_sections(text)
        self.assertEqual(sections.get("situation"), "- A")
        self.assertEqual(sections.get("insurer_claim"), "- B")
        self.assertEqual(sections.get("conclusion"), "- C")
        self.assertEqual(sections.get("plain_explanation"), "- D")

    def test_parse_dot_format(self):
        text = """1. 지금 상황\n- A\n2. 보험사 주장\n- B\n5. 결론\n- C"""
        sections = _parse_numbered_sections(text)
        self.assertEqual(sections.get("situation"), "- A")
        self.assertEqual(sections.get("insurer_claim"), "- B")
        self.assertEqual(sections.get("conclusion"), "- C")

    def test_parse_number_only_header_then_title_line(self):
        text = """1 )\n지금 상황\n- A\n2 )\n보험사 판단\n- B"""
        sections = _parse_numbered_sections(text)
        self.assertEqual(sections.get("situation"), "- A")
        self.assertEqual(sections.get("insurer_claim"), "- B")

    def test_alias_mapping(self):
        text = """6. 후속 조치\n- X\n8. 유의사항\n- Y"""
        sections = _parse_numbered_sections(text)
        self.assertEqual(sections.get("next_step_hint"), "- X")
        self.assertEqual(sections.get("notice"), "- Y")

    def test_parse_hyphen_delimiter_format(self):
        text = """1 - 지금 상황\n- A\n2 - 보험사 주장\n- B"""
        sections = _parse_numbered_sections(text)
        self.assertEqual(sections.get("situation"), "- A")
        self.assertEqual(sections.get("insurer_claim"), "- B")

    def test_parse_failure_returns_empty(self):
        sections = _parse_numbered_sections("헤더 없이 일반 본문만 있는 텍스트")
        self.assertEqual(sections, {})

    def test_simplify_terms_is_idempotent(self):
        text = "건강보험 미적용 항목(비급여)와 본인 부담금(자기부담금)을 확인"
        once = _simplify_korean_terms(text)
        twice = _simplify_korean_terms(once)
        self.assertEqual(once, twice)
        self.assertNotIn("건강보험 미적용 항목(건강보험 미적용 항목(비급여))", twice)

    def test_next_steps_structure_contains_four_labels(self):
        actions = [
            {
                "action_id": "ACTION-1",
                "title": "증빙 보강",
                "detail": "건강보험 급여/비급여 확인서를 준비하세요.",
                "priority": "high",
                "linked_issue_id": "ISSUE-1",
            }
        ]
        steps = _build_next_steps(
            actions=actions,
            required_documents=["건강보험 급여/비급여 확인서"],
            conclusion="추가 자료가 필요합니다.",
            next_step_hint="약관 조항과 제출 서류를 함께 정리하세요.",
        )
        self.assertEqual(len(steps), len(actions))
        first = steps[0]
        self.assertIn("무엇:", first)
        self.assertIn("방법:", first)
        self.assertIn("이유:", first)
        self.assertIn("준비물:", first)
        self.assertIn("완료기준:", first)

    def test_user_guidance_summary_is_step3_focused(self):
        state = {
            "success_probability_public": {"band": "MEDIUM"},
            "recommended_actions": [
                {
                    "action_id": "ACTION-1",
                    "title": "증빙 보강",
                    "detail": "핵심 증빙을 준비해 제출하세요.",
                    "priority": "high",
                    "linked_issue_id": "ISSUE-1",
                }
            ],
            "evidence_pack": [
                {
                    "evidence_id": "EV-1",
                    "issue_id": "ISSUE-1",
                    "evidence_title": "테스트 판례",
                    "summary": "요약",
                    "provenance": [{"source_type": "caselaw", "source_id": "1"}],
                }
            ],
            "gap_analysis": [{"issue_id": "ISSUE-1", "impact": "high"}],
            "issue_tree": {"nodes": [{"issue_id": "ISSUE-1"}]},
            "structured_case": {
                "user_info": {
                    "decision_summary_user_situation": "사용자 상황",
                    "decision_summary_insurer_claim": "보험사 주장",
                    "decision_summary_conclusion_reason": "결론",
                }
            },
        }
        result = _build_user_guidance_node(state)
        summary = str(result["user_guidance"]["plain_summary"])
        self.assertNotIn("고객님 상황:", summary)
        self.assertNotIn("보험사 판단:", summary)
        self.assertIn("현재 판단:", summary)
        issue_brief = result["user_guidance"].get("issue_brief", [])
        self.assertTrue(issue_brief)
        self.assertIn("why_it_matters", issue_brief[0])


if __name__ == "__main__":
    unittest.main()
