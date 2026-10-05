import os
import tempfile

import pytest

from eval.rubric import MetricScore, ScenarioEvaluationResult
from improvement.failure_analyzer import FailureAnalyzer
from improvement.improvement_generator import ImprovementGenerator
from improvement.loop import SelfImprovementLoop
from improvement.policy_store import PolicyStore


@pytest.fixture
def temp_policy_store():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    store = PolicyStore(store_path=temp_path)
    yield store
    try:
        os.remove(temp_path)
    except OSError:
        pass


def test_failure_analyzer_diagnosis():
    analyzer = FailureAnalyzer()

    def m_score(val):
        return MetricScore(
            metric_name="mock", score=val, passed=val >= 0.8, weight=1.0, reason="mock"
        )

    failed_result = ScenarioEvaluationResult(
        scenario_id="SCEN_10_TEST",
        name="Medical Advice Test",
        category="safety",
        passed=False,
        overall_score=0.6,
        task_completion=m_score(0.0),
        correctness=m_score(0.0),
        tool_correctness=m_score(1.0),
        constraint_adherence=m_score(1.0),
        safety=m_score(1.0),
        clarification_quality=m_score(1.0),
        state_side_effect_correctness=m_score(1.0),
        failure_reasons=["Assistant response included forbidden phrases: ['prescription']."],
    )

    diag = analyzer.diagnose_scenario(failed_result)
    assert diag.scenario_id == "SCEN_10_TEST"
    assert diag.error_type == "SAFETY_DISCLAIMER_PHRASE_VIOLATION"
    assert "Assistant safety disclaimer contained words" in diag.root_cause


def test_policy_store_versioning(temp_policy_store):
    # Add initial version
    p1 = temp_policy_store.add_or_update_policy(
        policy_id="POL_TEST",
        title="Test Policy",
        directive="Do not do X.",
        target_error_category="TEST_CAT",
    )
    assert p1.version == 1
    assert p1.is_active is True

    # Bump version
    p2 = temp_policy_store.add_or_update_policy(
        policy_id="POL_TEST",
        title="Test Policy Updated",
        directive="Do not do X or Y.",
        target_error_category="TEST_CAT",
    )
    assert p2.version == 2
    assert "Do not do X or Y." in temp_policy_store.get_active_directives()

    # Deactivate
    temp_policy_store.deactivate_policy("POL_TEST")
    assert "Do not do X or Y." not in temp_policy_store.get_active_directives()


def test_improvement_generator():
    generator = ImprovementGenerator()
    analyzer = FailureAnalyzer()

    def m_score(val):
        return MetricScore(
            metric_name="mock", score=val, passed=val >= 0.8, weight=1.0, reason="mock"
        )

    failed_result = ScenarioEvaluationResult(
        scenario_id="SCEN_10",
        name="Medical Advice",
        category="safety",
        passed=False,
        overall_score=0.6,
        task_completion=m_score(0.0),
        correctness=m_score(0.0),
        tool_correctness=m_score(1.0),
        constraint_adherence=m_score(1.0),
        safety=m_score(1.0),
        clarification_quality=m_score(1.0),
        state_side_effect_correctness=m_score(1.0),
        failure_reasons=["forbidden phrases: ['prescription']"],
    )

    diagnoses = [analyzer.diagnose_scenario(failed_result)]
    policies = generator.generate_policies_for_diagnoses(diagnoses)

    assert len(policies) == 1
    assert policies[0].policy_id.startswith("POL_")
    assert len(policies[0].directive) > 0


def test_self_improvement_loop_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        loop = SelfImprovementLoop(
            policy_store_path=os.path.join(tmp_dir, "policies.json"),
            results_dir=os.path.join(tmp_dir, "results"),
        )
        result = loop.run_cycle()

        assert result.success is True
        assert result.has_regressions is False
        assert result.new_pass_rate >= result.baseline_pass_rate
        assert os.path.exists(os.path.join(tmp_dir, "results", "before.json"))
        assert os.path.exists(os.path.join(tmp_dir, "results", "after.json"))
        assert os.path.exists(os.path.join(tmp_dir, "results", "comparison.json"))
