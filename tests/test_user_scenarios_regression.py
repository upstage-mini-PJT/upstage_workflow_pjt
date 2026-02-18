from agents.data_analysis_agent.pipeline import run_pipeline
from evaluations.data_analysis.user_scenarios import USER_SCENARIOS
from evaluations.data_analysis.user_scenarios_expectations import EXPECTED_SCENARIO_CONSTRAINTS


def test_user_scenarios_regression_constraints() -> None:
    assert len(USER_SCENARIOS) == 16
    assert len(EXPECTED_SCENARIO_CONSTRAINTS) == 16

    for scenario in USER_SCENARIOS:
        scenario_id = str(scenario["scenario_id"])
        expected = EXPECTED_SCENARIO_CONSTRAINTS[scenario_id]

        result = run_pipeline(scenario["structured_case"])  # type: ignore[arg-type]
        rag_result = result.get("rag_result", {})
        success_probability = result.get("success_probability", {})
        scoring_trace = result.get("scoring_trace", {})

        total_score = int(scoring_trace.get("total_score", 0))
        success_band = str(success_probability.get("band", "LOW"))
        rag_items = len(rag_result.get("items", []))

        assert success_band == expected["expected_band"]
        assert int(expected["min_total_score"]) <= total_score <= int(expected["max_total_score"])
        assert rag_items >= int(expected["min_rag_items"])
