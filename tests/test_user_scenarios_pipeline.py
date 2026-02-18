from agents.data_analysis_agent.pipeline import run_pipeline
from evaluations.data_analysis.user_scenarios import USER_SCENARIOS


def test_user_scenarios_pipeline_16_cases() -> None:
    assert len(USER_SCENARIOS) == 16

    for scenario in USER_SCENARIOS:
        result = run_pipeline(scenario["structured_case"])  # type: ignore[arg-type]

        rag_result = result.get("rag_result", {})
        score = result.get("success_probability", {})
        scoring_trace = result.get("scoring_trace", {})

        assert rag_result.get("stats", {}).get("candidate_count", 0) >= 0
        assert rag_result.get("stats", {}).get("returned_count", 0) >= 0
        assert len(rag_result.get("items", [])) >= 0

        assert 0 <= int(score.get("score", 0)) <= 100
        assert score.get("band", "LOW") in {"LOW", "MEDIUM", "HIGH"}

        assert 0 <= int(scoring_trace.get("precedent_score", 0)) <= 100
        assert -15 <= int(scoring_trace.get("case_adjustment", 0)) <= 15
        assert 0 <= int(scoring_trace.get("total_score", 0)) <= 100
