"""
Step3 Data Analysis Pipeline

보험금 부지급 대응 분석의 전체 흐름을 제어합니다.
(Retrieval -> Issue Analysis -> Gap Analysis -> Scoring -> Strategy Planning -> Packaging)
"""

import sys
import os
import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.schemas.case_context import Step3Input, StructuredCase
from core.schemas.analysis import (
    Step3Output, Meta, AnalysisReport, StrategyPlan, ResearchPack,
    Issue, GapAnalysis, SuccessProbability, CaseLawResult, ProvenanceRef,
    MissingEvidence, Inconsistency, RecommendedAction,
    PrimaryClaim, SupportingPoint, CounterArgument, StepTask
)
from tools.data_analysis_tools.caselaw.mock_client import MockCaseLawClient
from tools.data_analysis_tools.caselaw.caselaw_types import SearchQuery, CaseLawDoc
from core.scoring.features import extract_features
from core.scoring.rubric import calculate_probability


def _extract_json_payload(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    inline = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if inline:
        return json.loads(inline.group(1))

    raise ValueError("LLM response does not contain valid JSON")


def build_gemini_case_adjustment_llm(
    model: str = "gemini-2.0-flash",
    api_key_env: str = "GEMINI_API_KEY",
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Gemini REST API 기반 사례 보정치 판단 함수 생성기"""

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    instruction = (
        "너는 보험 분쟁 사례 기반 점수 보정기다. "
        "반드시 JSON만 반환한다. "
        "스키마: {\"adjustment\": int, \"rationale\": str, \"cited_case_ids\": list[str]}. "
        "adjustment는 -15~15 정수 범위를 지켜라. "
        "cited_case_ids는 payload.top_cases의 case_id만 사용하라."
    )

    def _call(payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {"text": json.dumps(payload, ensure_ascii=False)},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API HTTPError: {e.code} {detail}")

        candidates = raw.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_out = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        data = _extract_json_payload(text_out or "")

        return {
            "adjustment": data.get("adjustment", 0),
            "rationale": data.get("rationale", ""),
            "cited_case_ids": data.get("cited_case_ids", []),
        }

    return _call


class AnalysisPipeline:

    def __init__(
        self,
        mock_db_path: str = None,
        case_adjustment_llm: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        파이프라인 초기화

        Args:
            mock_db_path: mock 판례 DB 경로
            case_adjustment_llm: 사례 기반 보정치 판단 LLM 함수(선택)
                입력: payload(dict)
                출력 예시: {"adjustment": 6, "rationale": "...", "cited_case_ids": ["mock_2022_015"]}
        """
        self.caselaw_client = MockCaseLawClient(mock_db_path)
        self.case_adjustment_llm = case_adjustment_llm

    def run(self, input_data: Step3Input) -> Step3Output:
        """
        Step3 분석 파이프라인 실행
        """
        print(f"[*] Starting Step3 Analysis for Case: {input_data.meta.case_id}")

        # 1. Retrieval (판례 검색)
        print("[1/6] Retrieving similar caselaws...")
        search_query = SearchQuery(
            keywords=input_data.research_inputs.caselaw_query.keywords,
            filters=input_data.research_inputs.caselaw_query.filters,
            top_k=int(input_data.research_inputs.caselaw_query.filters.get("top_k", 5)),
        )
        caselaws = self.caselaw_client.search(search_query)

        # 2. Issue Analysis (쟁점 분석)
        print("[2/6] Analyzing issues...")
        issue_tree = self._analyze_issues(input_data.structured_case, caselaws)

        # 3. Gap Analysis (증빙 갭 분석)
        print("[3/6] Identifying evidence gaps...")
        gap_analysis = self._perform_gap_analysis(input_data.structured_case)

        # 4. Success Probability Scoring (2단계: 판례 1차 + 사례/LLM 보정)
        print("[4/6] Calculating success probability...")
        features = extract_features(input_data.structured_case, caselaws)

        # 1차: 판례 중심 점수
        precedent_features = self._build_precedent_only_features(features)
        precedent_probability = calculate_probability(precedent_features)

        # 2차: 사례 기반 보정(LLM 또는 휴리스틱)
        case_adjustment, adjustment_notes = self._compute_case_adjustment(
            case=input_data.structured_case,
            caselaws=caselaws,
            precedent_score=precedent_probability.score_0_to_100,
        )

        total_score = max(0, min(100, precedent_probability.score_0_to_100 + case_adjustment))
        total_band = self._score_to_band(total_score)

        drivers_positive = list(precedent_probability.drivers_positive)
        drivers_negative = list(precedent_probability.drivers_negative)
        if case_adjustment > 0:
            drivers_positive.append(f"사례 기반 보정 +{case_adjustment}점")
        elif case_adjustment < 0:
            drivers_negative.append(f"사례 기반 보정 {case_adjustment}점")

        probability = SuccessProbability(
            score_0_to_100=total_score,
            band=total_band,
            drivers_positive=drivers_positive,
            drivers_negative=drivers_negative,
            assumptions=list(precedent_probability.assumptions)
            + adjustment_notes
            + [
                f"precedent_score={precedent_probability.score_0_to_100}",
                f"case_adjustment={case_adjustment}",
                f"total_score={total_score}",
            ],
        )

        # 5. Re-review Strategy Planning (재심의 전략 수립)
        print("[5/6] Drafting re-review strategy...")
        strategy = self._plan_strategy(
            input_data.structured_case,
            issue_tree,
            probability,
            caselaws,
            input_data.analysis_options.risk_level,
        )

        # 6. Final Packaging (최종 결과 취합)
        print("[6/6] Packaging results...")
        
        # 분석 리포트 생성
        recommended_actions = self._build_recommended_actions(
            gap_analysis,
            probability,
            caselaws,
        )

        analysis_report = AnalysisReport(
            issue_tree=issue_tree,
            gap_analysis=gap_analysis,
            recommended_actions=recommended_actions,
            success_probability=probability
        )

        # 리서치 팩 생성
        research_pack = self._package_research(caselaws)

        # 최종 출력 생성
        return Step3Output(
            meta=Meta(
                case_id=input_data.meta.case_id,
                timestamp=datetime.now().isoformat()
            ),
            analysis_report=analysis_report,
            strategy_plan=strategy,
            research_pack=research_pack
        )

    def _analyze_issues(self, case: StructuredCase, caselaws: List[CaseLawDoc]) -> List[Issue]:
        """
        쟁점 트리 생성 (규칙 기반)
        """
        issues = []
        top_refs = [
            ProvenanceRef(ref_type="CASELAW", ref_id=c.case_id)
            for c in caselaws[:2]
        ]

        reason_to_sub_issues = {
            "LACK_OF_NECESSITY": ["의학적 필요성 입증 여부", "치료 대체 가능성 검토"],
            "PREEXISTING": ["기왕증과 현증상 인과관계", "발병 시점 입증"],
            "DOCUMENT_INSUFFICIENT": ["서류 누락 여부", "증빙 신뢰도 보강 필요성"],
        }

        for reason in case.denial_reasons:
            sub_issues = reason_to_sub_issues.get(
                reason.reason_code,
                ["약관 해석의 적절성", "유사 판례와의 비교"],
            )
            if reason.related_clause:
                sub_issues.append(f"관련 조항({reason.related_clause}) 해석의 타당성")
            issues.append(Issue(
                issue=f"보험사의 '{reason.reason_text}' 사유에 대한 타당성 검토",
                sub_issues=sub_issues,
                provenance=top_refs,
            ))

        if not issues:
            issues.append(Issue(
                issue="보험금 부지급의 약관상 근거 및 사실관계 정합성 검토",
                sub_issues=["부지급 통지서 사유의 구체성", "유사 판례와의 정합성"],
                provenance=top_refs,
            ))

        return issues

    def _perform_gap_analysis(self, case: StructuredCase) -> GapAnalysis:
        """
        부족 증빙 분석 (규칙 기반)
        """
        sources = [(fact.source or "") for fact in case.facts]
        source_blob = " ".join(sources)

        missing_evidence: List[MissingEvidence] = []
        inconsistencies: List[Inconsistency] = []

        has_medical_record = ("진료기록" in source_blob) or ("의무기록" in source_blob)
        has_doctor_note = ("소견서" in source_blob) or ("진단서" in source_blob)
        has_receipt = ("영수증" in source_blob) or ("진료비" in source_blob) or ("세부내역" in source_blob)

        if not has_medical_record:
            missing_evidence.append(MissingEvidence(
                evidence_type="MEDICAL_RECORD",
                why_needed="치료 경과 및 의학적 필요성을 객관적으로 입증해야 합니다.",
                how_to_get="병원 원무과에서 진료기록/의무기록 사본을 발급받습니다.",
                priority="HIGH",
            ))

        if not has_doctor_note:
            missing_evidence.append(MissingEvidence(
                evidence_type="DOCTOR_NOTE",
                why_needed="보험사의 필요성 부인 사유에 직접 반박 근거가 필요합니다.",
                how_to_get="주치의에게 치료 필요성 및 대체불가성을 포함한 소견서를 요청합니다.",
                priority="HIGH",
            ))

        if not has_receipt:
            missing_evidence.append(MissingEvidence(
                evidence_type="RECEIPT",
                why_needed="실제 발생 비용 및 청구 범위를 명확히 해야 합니다.",
                how_to_get="진료비 영수증과 진료비 세부내역서를 재발급받습니다.",
                priority="MED",
            ))

        fact_dates = [f.date for f in case.facts if f.date]
        if fact_dates and fact_dates != sorted(fact_dates):
            inconsistencies.append(Inconsistency(
                field="timeline",
                detail="사실 타임라인의 날짜 순서가 뒤섞여 있습니다. 사건 순서를 재정렬해야 합니다.",
                priority="MED",
            ))

        if len(case.open_questions) >= 2:
            inconsistencies.append(Inconsistency(
                field="open_questions",
                detail="미해결 쟁점이 다수 존재하여 현재 자료만으로는 주장 완결성이 낮습니다.",
                priority="MED",
            ))

        return GapAnalysis(
            missing_evidence=missing_evidence,
            inconsistencies=inconsistencies,
        )

    def _build_precedent_only_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """판례 점수만 반영되도록 비판례 feature를 중립화"""
        precedent_only = dict(features)
        neutral_defaults = {
            "evidence_completeness": 0.4,
            "has_medical_records": None,
            "has_doctor_note": None,
            "procedural_compliance": None,
            "timeline_consistency": 0.6,
            "policy_clarity": 0.7,
            "clause_interpretation_favorable": None,
            "claim_amount_reasonableness": 0.5,
            "denial_reason_strength": 0.5,
            "has_emergency_situation": None,
            "fact_count": 2,
        }
        precedent_only.update(neutral_defaults)
        return precedent_only

    def _compute_case_adjustment(
        self,
        case: StructuredCase,
        caselaws: List[CaseLawDoc],
        precedent_score: int,
    ) -> Tuple[int, List[str]]:
        """
        사례 기반 보정치 계산

        1) LLM 함수가 주입된 경우 해당 결과 사용
        2) 없으면 휴리스틱 fallback
        3) 가드레일 적용 후 최종 보정치 확정
        """
        if not caselaws:
            return 0, ["사례 기반 보정 미적용: 검색된 사례 없음"]

        payload = {
            "case_id": case.case_id,
            "insurance_type": case.insurance_type,
            "denial_reasons": [d.reason_code for d in case.denial_reasons],
            "top_cases": [
                {
                    "case_id": c.case_id,
                    "title": c.title,
                    "relevance_score": c.relevance_score,
                    "result": c.result,
                }
                for c in caselaws[:3]
            ],
            "precedent_score": precedent_score,
        }

        if self.case_adjustment_llm is not None:
            try:
                raw = self.case_adjustment_llm(payload)
            except Exception as exc:
                fallback = self._heuristic_case_adjustment(payload)
                adj, notes = self._apply_adjustment_guardrails(
                    raw_adjustment=fallback.get("adjustment", 0),
                    rationale=f"LLM 실패 fallback: {exc}",
                    cited_case_ids=fallback.get("cited_case_ids", []),
                    caselaws=caselaws,
                    precedent_score=precedent_score,
                )
                return adj, notes
        else:
            raw = self._heuristic_case_adjustment(payload)

        adjustment, notes = self._apply_adjustment_guardrails(
            raw_adjustment=raw.get("adjustment", 0),
            rationale=raw.get("rationale", ""),
            cited_case_ids=raw.get("cited_case_ids", []),
            caselaws=caselaws,
            precedent_score=precedent_score,
        )
        return adjustment, notes

    def _heuristic_case_adjustment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 미연동 시 사용하는 보정치 휴리스틱"""
        top_cases = payload.get("top_cases", [])
        if not top_cases:
            return {"adjustment": 0, "rationale": "사례 없음", "cited_case_ids": []}

        adjustment = 0
        top_rel = float(top_cases[0].get("relevance_score", 0.0) or 0.0)

        if top_rel >= 0.8:
            adjustment += 6
        elif top_rel >= 0.6:
            adjustment += 3
        elif top_rel < 0.3:
            adjustment -= 3

        win_like = 0
        lose_like = 0
        for c in top_cases:
            result = str(c.get("result", "") or "")
            if any(k in result for k in ["승소", "인용"]):
                win_like += 1
            if any(k in result for k in ["패소", "기각", "각하"]):
                lose_like += 1

        if win_like >= 2:
            adjustment += 4
        elif lose_like >= 2:
            adjustment -= 4

        cited = [c.get("case_id") for c in top_cases[:2] if c.get("case_id")]
        return {
            "adjustment": adjustment,
            "rationale": "top 사례 관련도/판정 경향 기반 보정",
            "cited_case_ids": cited,
        }

    def _apply_adjustment_guardrails(
        self,
        raw_adjustment: Any,
        rationale: str,
        cited_case_ids: List[str],
        caselaws: List[CaseLawDoc],
        precedent_score: int,
    ) -> Tuple[int, List[str]]:
        """사례 보정치 안정화 가드레일"""
        notes: List[str] = []

        try:
            adjustment = int(round(float(raw_adjustment)))
        except Exception:
            adjustment = 0
            notes.append("사례 보정치 파싱 실패: 0으로 대체")

        # Guardrail 1: 근거 없는 보정 금지
        valid_ids = {c.case_id for c in caselaws}
        if not rationale or len(rationale.strip()) < 8:
            adjustment = 0
            notes.append("사례 보정치 무효화: rationale 부족")

        if not cited_case_ids or any(cid not in valid_ids for cid in cited_case_ids):
            adjustment = 0
            notes.append("사례 보정치 무효화: 유효 provenance 부재")

        # Guardrail 2: 보정 폭 제한
        if adjustment > 15:
            adjustment = 15
            notes.append("사례 보정치 상한 적용(+15)")
        if adjustment < -15:
            adjustment = -15
            notes.append("사례 보정치 하한 적용(-15)")

        # Guardrail 3: 1차 점수와 과도한 충돌 방지
        if precedent_score < 20 and adjustment > 8:
            adjustment = 8
            notes.append("저신뢰 1차 점수 보호: +8로 제한")
        if precedent_score > 85 and adjustment < -8:
            adjustment = -8
            notes.append("고신뢰 1차 점수 보호: -8로 제한")

        return adjustment, notes

    def _score_to_band(self, score: int) -> str:
        if score < 40:
            return "LOW"
        if score < 70:
            return "MEDIUM"
        return "HIGH"

    def _build_recommended_actions(
        self,
        gap_analysis: GapAnalysis,
        probability: SuccessProbability,
        caselaws: List[CaseLawDoc],
    ) -> List[RecommendedAction]:
        """갭/확률 기반 권장 액션 생성"""
        actions: List[RecommendedAction] = []
        top_ref = [ProvenanceRef(ref_type="CASELAW", ref_id=caselaws[0].case_id)] if caselaws else []
        idx = 1

        for evidence in gap_analysis.missing_evidence:
            actions.append(RecommendedAction(
                action_id=f"A{idx}",
                action=f"{evidence.evidence_type} 자료 보강",
                rationale=evidence.why_needed,
                effort="MED",
                expected_impact="HIGH" if evidence.priority == "HIGH" else "MED",
                provenance=top_ref,
            ))
            idx += 1

        if probability.band != "HIGH":
            actions.append(RecommendedAction(
                action_id=f"A{idx}",
                action="부지급 사유별 반박서(쟁점별 1페이지) 작성",
                rationale="핵심 쟁점을 문서화하면 재심의 단계에서 심사자의 판단 부담을 낮출 수 있습니다.",
                effort="MED",
                expected_impact="HIGH",
                provenance=top_ref,
            ))

        return actions

    def _plan_strategy(
        self,
        case: StructuredCase,
        issues: List[Issue],
        probability: SuccessProbability,
        caselaws: List[CaseLawDoc],
        risk_level: str,
    ) -> StrategyPlan:
        """
        재심의 전략 수립 (규칙 기반)
        """
        top_refs = [
            ProvenanceRef(ref_type="CASELAW", ref_id=c.case_id)
            for c in caselaws[:2]
        ]
        strongest_issue = issues[0].issue if issues else "부지급 사유의 합리성"
        overview = (
            f"핵심 쟁점은 '{strongest_issue}'입니다. "
            f"현재 성공 확률은 {probability.score_0_to_100}점({probability.band})으로 평가되며, "
            "의학적 필요성 및 약관 해석 근거를 중심으로 재심의를 진행합니다."
        )

        claim = PrimaryClaim(
            claim="약관상 보험금 지급요건 충족",
            supporting_points=[
                SupportingPoint(
                    point="유사 분쟁 사례에서 치료 목적성이 인정된 판례가 확인됩니다.",
                    provenance=top_refs,
                )
            ],
            counterarguments_expected=[
                CounterArgument(
                    counter="치료의 필요성이 낮거나 선택적 치료라는 반론",
                    rebuttal_outline="주치의 소견서와 진료기록으로 치료 필요성, 대체불가성을 제시합니다.",
                )
            ],
        )

        recommended_channel = "INSURER_REVIEW"
        if probability.band == "LOW":
            recommended_channel = "DISPUTE_MEDIATION"
        if risk_level == "AGGRESSIVE" and probability.band != "HIGH":
            recommended_channel = "INSURANCE_ASSOCIATION"

        plan = [
            StepTask(step=1, task="부족 증빙(진료기록/소견서/영수증) 확보", owner="USER", due_days=3),
            StepTask(step=2, task="부지급 사유별 반박서 및 재심의서 초안 생성", owner="SYSTEM", due_days=0),
            StepTask(step=3, task="제출 채널 및 첨부 양식 점검 후 제출", owner="SYSTEM", due_days=0),
        ]

        return StrategyPlan(
            strategy_overview=overview,
            primary_claims=[claim],
            step_by_step_plan=plan,
            recommended_channel=recommended_channel,
            provenance=top_refs,
        )

    def _package_research(self, caselaws: List[CaseLawDoc]) -> ResearchPack:
        """
        검색된 판례를 최종 포맷으로 변환
        """
        results = []
        for c in caselaws:
            results.append(CaseLawResult(
                case_id=c.case_id,
                title=c.title,
                court=c.court,
                date=c.date,
                holding_summary=c.summary,
                relevance_score=c.relevance_score,
                key_quotes=[],
                provenance=[ProvenanceRef(ref_type="CASELAW", ref_id=c.case_id)]
            ))
        return ResearchPack(caselaw_results=results, web_sources=[])


# ============================================================================
# 간단 테스트 실행부
# ============================================================================
if __name__ == "__main__":
    from core.schemas.case_context import (
        Step3Input,
        Meta as InputMeta,
        ResearchInputs,
        CaseLawQuery,
        WebSearchQuery,
        Fact,
        DenialReason,
        PolicyClause,
        AnalysisOptions,
    )

    def make_input(
        case_id: str,
        facts,
        reason_code: str,
        reason_text: str,
        clause_id: str,
        clause_text: str,
        keywords,
        insurance_type: str,
        claim_amount: int,
        risk_level: str = "BALANCED",
    ) -> Step3Input:
        return Step3Input(
            meta=InputMeta(case_id=case_id),
            structured_case=StructuredCase(
                case_id=case_id,
                facts=[Fact(date=d, event=e, source=s) for d, e, s in facts],
                denial_reasons=[
                    DenialReason(
                        reason_code=reason_code,
                        reason_text=reason_text,
                        related_clause=clause_id,
                    )
                ],
                policy_clauses=[PolicyClause(clause_id=clause_id, clause_text=clause_text)],
                insurance_type=insurance_type,
                claim_amount=claim_amount,
            ),
            research_inputs=ResearchInputs(
                caselaw_query=CaseLawQuery(keywords=keywords, filters={"top_k": 3}),
                web_search_query=WebSearchQuery(keywords=keywords[:3], max_results=5),
            ),
            analysis_options=AnalysisOptions(risk_level=risk_level),
        )

    sample_inputs = [
        # HIGH x4
        make_input(
            "USER-HIGH-001",
            [
                ("2025-11-03", "하지정맥류 진단 및 역류 소견 확인", "진료기록"),
                ("2025-11-10", "고주파 수술 시행", "수술기록지"),
                ("2025-11-25", "주치의 소견서 발급", "의사소견서"),
            ],
            "LACK_OF_NECESSITY",
            "의학적 필요성 부족",
            "제5조 2항",
            "의학적으로 필요하다고 인정되는 수술 및 입원치료를 보장",
            ["하지정맥류", "실손보험", "수술", "치료목적", "의학적필요성", "입원"],
            "실손의료보험",
            2800000,
        ),
        make_input(
            "USER-HIGH-002",
            [
                ("2025-08-14", "유방 양성종양 통증 악화", "진료기록"),
                ("2025-08-20", "맘모톰 시술로 종양 제거", "수술기록"),
                ("2025-08-26", "치료 목적 소견서 제출", "소견서"),
            ],
            "LACK_OF_NECESSITY",
            "조직검사 목적 시술로 판단",
            "제7조 1항",
            "치료 목적 시술은 보장 대상",
            ["맘모톰", "유방", "양성종양", "치료목적", "실손보험"],
            "실손의료보험",
            1900000,
        ),
        make_input(
            "USER-HIGH-003",
            [
                ("2025-07-03", "자궁근종 크기 증가 및 통증", "진료기록"),
                ("2025-07-10", "하이푸 시술 후 입원 관찰", "입퇴원확인서"),
                ("2025-07-14", "합병증 위험 관련 의사소견", "의사소견서"),
            ],
            "LACK_OF_NECESSITY",
            "통원으로도 가능했다는 사유",
            "제5조 2항",
            "의학적으로 필요하다고 인정되는 입원은 보장",
            ["자궁근종", "하이푸", "입원", "통증관리", "신의료기술"],
            "실손의료보험",
            3200000,
        ),
        make_input(
            "USER-HIGH-004",
            [
                ("2025-12-01", "급성 복통으로 응급실 내원", "응급기록"),
                ("2025-12-01", "응급 수술 시행", "수술기록"),
                ("2025-12-05", "입원 필요성 소견서", "진단서"),
            ],
            "LACK_OF_NECESSITY",
            "응급성 부족 주장",
            "제9조 3항",
            "응급 상황에서의 치료는 보장",
            ["응급", "수술", "입원", "실손보험", "의학적필요성"],
            "실손의료보험",
            4100000,
        ),

        # MEDIUM x4
        make_input(
            "USER-MED-001",
            [
                ("2025-09-02", "허리 통증으로 도수치료 시작", "진료기록"),
                ("2025-09-20", "통원 치료 반복", "통원기록"),
            ],
            "LACK_OF_NECESSITY",
            "과잉 진료 의심",
            "제6조 1항",
            "의학적 필요성이 있는 치료는 보장",
            ["도수치료", "통증", "입원", "실손보험"],
            "실손의료보험",
            1300000,
        ),
        make_input(
            "USER-MED-002",
            [
                ("2025-06-10", "백내장 진단", "진료기록"),
                ("2025-06-12", "수술 후 단기 입원", "입퇴원확인서"),
            ],
            "LACK_OF_NECESSITY",
            "입원 적정성 부족",
            "제5조 2항",
            "입원 필요성이 인정되는 경우 보장",
            ["백내장", "입원치료", "수술", "실손보험"],
            "실손의료보험",
            900000,
        ),
        make_input(
            "USER-MED-003",
            [
                ("2025-05-21", "갑상선 결절 확인", "검사결과"),
                ("2025-05-30", "고주파절제술 시행", "수술기록"),
            ],
            "LACK_OF_NECESSITY",
            "예방적 시술이라는 사유",
            "제8조 1항",
            "치료 목적 수술은 보장",
            ["갑상선", "고주파절제술", "수술필요성", "실손보험"],
            "실손의료보험",
            1700000,
        ),
        make_input(
            "USER-MED-004",
            [
                ("2025-04-09", "전립선비대증 약물치료 반응 저조", "진료기록"),
                ("2025-04-15", "유로리프트 시술", "수술기록"),
            ],
            "LACK_OF_NECESSITY",
            "고가 치료 선택 사유",
            "제6조 3항",
            "의학적으로 필요한 치료비를 보장",
            ["전립선비대증", "유로리프트", "과잉진료", "실손보험"],
            "실손의료보험",
            2600000,
        ),

        # LOW x4
        make_input(
            "USER-LOW-001",
            [
                ("2025-03-03", "피로 회복 목적 영양주사", "외래기록"),
                ("2025-03-10", "수액 치료 반복", "영수증"),
            ],
            "EXCLUSION",
            "치료 목적 불명확",
            "면책조항 2항",
            "건강증진 목적 치료는 보장하지 아니함",
            ["영양주사", "수액", "비급여", "치료목적"],
            "실손의료보험",
            450000,
        ),
        make_input(
            "USER-LOW-002",
            [
                ("2025-02-12", "단순 두통으로 MRI 촬영", "검사기록"),
                ("2025-02-14", "보험금 청구", "청구서"),
            ],
            "EXCLUSION",
            "과잉검사 판단",
            "면책조항 1항",
            "의학적 필요성 없는 검사는 보장 제외",
            ["뇌MRI", "두통", "과잉검사", "비급여"],
            "실손의료보험",
            1200000,
        ),
        make_input(
            "USER-LOW-003",
            [
                ("2025-01-08", "항암 종료 후 요양병원 입원", "입원확인서"),
                ("2025-01-20", "면역주사 치료", "치료기록"),
            ],
            "EXCLUSION",
            "암 직접치료 아님",
            "암특약 4항",
            "암 직접치료 목적 입원만 보장",
            ["요양병원", "암보험", "직접치료", "면역치료"],
            "암보험",
            3500000,
        ),
        make_input(
            "USER-LOW-004",
            [
                ("2025-10-01", "리스 차량 사고로 전손 처리", "사고접수기록"),
                ("2025-10-05", "리스계약 위약금 보상 요청", "리스계약서"),
                ("2025-10-12", "약관상 담보 제외로 지급거절", "부지급통지서"),
            ],
            "EXCLUSION",
            "약관상 담보 제외",
            "면책조항 3항",
            "리스 위약금 및 간접손해는 보상하지 아니함",
            ["자동차", "리스계약", "위약금", "대물배상", "전손"],
            "자동차보험",
            7000000,
        ),
    ]

    pipeline = AnalysisPipeline()

    print("\n" + "=" * 60)
    print("STEP3 ANALYSIS SAMPLE RUN (12 CASES)")
    print("=" * 60)

    band_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    for sample in sample_inputs:
        result = pipeline.run(sample)
        score = result.analysis_report.success_probability.score_0_to_100
        band = result.analysis_report.success_probability.band
        band_counts[band] = band_counts.get(band, 0) + 1

        print(f"Case ID: {result.meta.case_id}")
        print(f"  Score/Band: {score} / {band}")
        if result.research_pack.caselaw_results:
            top = result.research_pack.caselaw_results[0]
            print(f"  Top Case: {top.case_id} ({top.relevance_score:.2f})")
        else:
            print("  Top Case: None")
        print("-" * 60)

    print("Band Summary:", band_counts)
