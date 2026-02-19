import os
import unittest
from unittest.mock import patch

from agents.data_analysis_agent.pipeline import _build_step3_trace_config, run_pipeline
from core.schemas.case_context import normalize_structured_case
from main import _analysis_options_from_env


def _sample_case() -> dict:
    return normalize_structured_case(
        {
            "denial_summary": "실손 의료비 청구 부지급",
            "denial_reasons": ["약관상 비급여 제외"],
            "policy_clauses": ["실손의료비 약관 제1조"],
            "timeline": [],
            "evidence_summary": [],
            "open_questions": [],
        }
    )


def _analysis_result_stub() -> dict:
    return {
        "analysis_result": {
            "issue_tree": {"root_title": "테스트", "nodes": []},
            "gap_analysis": [],
            "recommended_actions": [],
            "success_probability": {
                "band": "LOW",
                "positive_drivers": [],
                "negative_drivers": [],
                "assumptions": [],
            },
            "evidence_pack": [],
        }
    }


class _DummyApp:
    def __init__(self) -> None:
        self.captured_input = None
        self.captured_config = None

    def invoke(self, input_state, config=None, **kwargs):  # noqa: ANN001
        self.captured_input = input_state
        self.captured_config = config
        return _analysis_result_stub()


class Step3TracingTests(unittest.TestCase):
    def test_run_pipeline_invoke_receives_config(self):
        dummy_app = _DummyApp()
        with patch("agents.data_analysis_agent.pipeline.build_graph", return_value=dummy_app):
            run_pipeline(_sample_case(), analysis_options={"thread_id": "thread-main-1", "entrypoint": "main"})

        self.assertIsNotNone(dummy_app.captured_config)
        self.assertEqual(dummy_app.captured_config.get("run_name"), "step3_data_analysis")
        self.assertEqual(dummy_app.captured_config.get("configurable", {}).get("thread_id"), "thread-main-1")

    def test_default_trace_config_contains_required_fields(self):
        state = {
            "analysis_options": {"risk_level": "BALANCED", "output_style": "USER_READABLE"},
            "retrieval_mode": "plain",
            "rag_result": {},
        }
        config = _build_step3_trace_config(state)
        metadata = config.get("metadata", {})

        self.assertEqual(config.get("run_name"), "step3_data_analysis")
        self.assertIn("step3", config.get("tags", []))
        self.assertIn("data_analysis", config.get("tags", []))
        self.assertIn("risk:balanced", config.get("tags", []))
        self.assertIn("retrieval:plain", config.get("tags", []))
        self.assertEqual(metadata.get("component"), "step3")
        self.assertEqual(metadata.get("entrypoint"), "unknown")
        self.assertTrue(str(config.get("configurable", {}).get("thread_id", "")).startswith("step3-"))

    def test_trace_overrides_are_applied(self):
        state = {
            "analysis_options": {
                "risk_level": "AGGRESSIVE",
                "output_style": "EXPERT_LIKE",
                "thread_id": "thread-eval-1",
                "entrypoint": "eval",
                "trace_run_name": "custom_step3_run",
                "trace_tags": ["custom-a", "custom-b"],
                "trace_metadata": {"dataset_path": "evaluations/data_analysis/dataset.jsonl"},
                "llm_enrichment_enabled": False,
            },
            "retrieval_mode": "hyde",
            "rag_result": {"items": [{"doc_id": "D1"}]},
        }
        config = _build_step3_trace_config(state)
        metadata = config.get("metadata", {})

        self.assertEqual(config.get("run_name"), "custom_step3_run")
        self.assertIn("custom-a", config.get("tags", []))
        self.assertIn("custom-b", config.get("tags", []))
        self.assertEqual(config.get("configurable", {}).get("thread_id"), "thread-eval-1")
        self.assertEqual(metadata.get("entrypoint"), "eval")
        self.assertEqual(metadata.get("dataset_path"), "evaluations/data_analysis/dataset.jsonl")
        self.assertTrue(metadata.get("has_external_rag"))
        self.assertFalse(metadata.get("llm_enrichment_enabled"))

    def test_run_pipeline_works_when_tracing_env_off(self):
        dummy_app = _DummyApp()
        with (
            patch.dict(
                os.environ,
                {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING_V2": "false", "LANGSMITH_TRACING": "false"},
                clear=False,
            ),
            patch("agents.data_analysis_agent.pipeline.build_graph", return_value=dummy_app),
        ):
            result = run_pipeline(_sample_case())

        self.assertTrue({"issue_tree", "gap_analysis", "recommended_actions", "success_probability", "evidence_pack"}.issubset(result.keys()))
        self.assertFalse(dummy_app.captured_config.get("metadata", {}).get("tracing_enabled_env"))

    def test_main_analysis_options_include_thread_id(self):
        with patch.dict(os.environ, {"STEP3_TRACE_TAGS": "main,cli"}, clear=False):
            options = _analysis_options_from_env("thread-main-2")

        self.assertEqual(options.get("thread_id"), "thread-main-2")
        self.assertEqual(options.get("entrypoint"), "main")
        self.assertEqual(options.get("trace_tags"), ["main", "cli"])


if __name__ == "__main__":
    unittest.main()
