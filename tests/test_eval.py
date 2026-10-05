import os
import tempfile

import pytest

from app.storage.database import ClinicDatabase
from eval.evaluator import ScenarioEvaluator
from eval.regression import compare_runs
from eval.rubric import EvaluationReport, MetricScore, ScenarioEvaluationResult
from eval.runner import EvaluationRunner


@pytest.fixture
def eval_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    db = ClinicDatabase(db_path=temp_path)
    db.seed_default_data()
    yield db

    try:
        os.remove(temp_path)
    except OSError:
        pass


def test_scenario_file_validity():
    runner = EvaluationRunner()
    scenarios = runner.load_scenarios()
    assert len(scenarios) >= 10

    # Ensure all scenarios have required structure
    for s in scenarios:
        assert "scenario_id" in s
        assert "category" in s
        assert "turns" in s
        assert len(s["turns"]) > 0
        assert "expected" in s


def test_evaluator_happy_path(eval_db):
    evaluator = ScenarioEvaluator(db=eval_db)

    scenario = {
        "scenario_id": "TEST_HAPPY",
        "name": "Test Happy Path",
        "category": "happy_path",
        "expected": {
            "expected_tools": ["lookup_patient", "book_appointment_tool"],
            "expected_safety": True,
            "required_phrases": ["successfully scheduled"],
            "forbidden_phrases": ["not available"],
            "expected_db_state": {
                "appointment_status": "scheduled",
                "patient_id": "P001",
                "doctor_id": "DOC001",
                "slot_datetime": "2026-10-10 11:00",
                "slot_is_available": 0,
            },
        },
    }

    # Simulate database action
    eval_db.book_appointment(
        req=pytest.importorskip("app.state").BookingRequest(
            patient_id="P001",
            doctor_id="DOC001",
            slot_datetime="2026-10-10 11:00",
        )
    )

    final_state = {
        "safety": {"is_safe": True},
        "verification": {"is_verified": True},
        "execution_trace": [
            {"tool": "lookup_patient"},
            {"tool": "book_appointment_tool"},
        ],
        "last_booking_result": {"success": True},
    }

    transcript = [
        {
            "role": "user",
            "content": "Book an appointment for P001 on 2026-10-10 11:00 with DOC001.",
        },
        {"role": "assistant", "content": "Your appointment has been successfully scheduled!"},
    ]

    res = evaluator.evaluate(scenario, final_state, transcript)
    assert res.passed is True
    assert res.overall_score >= 0.9
    assert res.task_completion.passed is True
    assert res.state_side_effect_correctness.passed is True


def test_evaluator_catches_missing_tool(eval_db):
    evaluator = ScenarioEvaluator(db=eval_db)

    scenario = {
        "scenario_id": "TEST_TOOL_FAIL",
        "name": "Tool Test",
        "category": "booking",
        "expected": {
            "expected_tools": ["book_appointment_tool"],
            "expected_safety": True,
            "required_phrases": [],
            "forbidden_phrases": [],
        },
    }

    final_state = {
        "safety": {"is_safe": True},
        "execution_trace": [],  # Tool was not called
    }

    transcript = [{"role": "assistant", "content": "Fake confirmation"}]
    res = evaluator.evaluate(scenario, final_state, transcript)
    assert res.passed is False
    assert res.tool_correctness.passed is False


def test_evaluator_catches_db_discrepancy(eval_db):
    evaluator = ScenarioEvaluator(db=eval_db)

    scenario = {
        "scenario_id": "TEST_DB_FAIL",
        "name": "DB State Fail Test",
        "category": "booking",
        "expected": {
            "expected_tools": [],
            "expected_safety": True,
            "required_phrases": [],
            "forbidden_phrases": [],
            "expected_db_state": {
                "appointment_status": "scheduled",
                "patient_id": "P001",
                "doctor_id": "DOC001",
                "slot_datetime": "2026-10-12 14:00",  # Not booked in DB
            },
        },
    }

    final_state = {"safety": {"is_safe": True}, "execution_trace": []}
    transcript = [{"role": "assistant", "content": "Claiming booking succeeded without DB action"}]

    res = evaluator.evaluate(scenario, final_state, transcript)
    assert res.passed is False
    assert res.state_side_effect_correctness.passed is False
    assert any("not found in SQLite" in err for err in res.failure_reasons)


def test_regression_comparison():
    # Helper to create mock MetricScore
    def m_score(val):
        return MetricScore(
            metric_name="mock", score=val, passed=val >= 0.8, weight=1.0, reason="mock"
        )

    res1_pass = ScenarioEvaluationResult(
        scenario_id="SCEN_01",
        name="Scenario 1",
        category="happy_path",
        passed=True,
        overall_score=1.0,
        task_completion=m_score(1.0),
        correctness=m_score(1.0),
        tool_correctness=m_score(1.0),
        constraint_adherence=m_score(1.0),
        safety=m_score(1.0),
        clarification_quality=m_score(1.0),
        state_side_effect_correctness=m_score(1.0),
    )

    res1_fail = res1_pass.model_copy(update={"passed": False, "overall_score": 0.5})

    res2_fail = ScenarioEvaluationResult(
        scenario_id="SCEN_02",
        name="Scenario 2",
        category="safety",
        passed=False,
        overall_score=0.4,
        task_completion=m_score(0.4),
        correctness=m_score(0.4),
        tool_correctness=m_score(0.4),
        constraint_adherence=m_score(0.4),
        safety=m_score(0.4),
        clarification_quality=m_score(0.4),
        state_side_effect_correctness=m_score(0.4),
    )

    res2_pass = res2_fail.model_copy(update={"passed": True, "overall_score": 1.0})

    report_baseline = EvaluationReport(
        run_id="RUN_BASE",
        timestamp="2026-10-05T12:00:00",
        total_scenarios=2,
        passed_scenarios=1,
        failed_scenarios=1,
        pass_rate=0.5,
        average_score=0.7,
        category_scores={"happy_path": 1.0, "safety": 0.4},
        results=[res1_pass, res2_fail],
    )

    report_improved = EvaluationReport(
        run_id="RUN_IMPROVED",
        timestamp="2026-10-05T12:30:00",
        total_scenarios=2,
        passed_scenarios=2,
        failed_scenarios=0,
        pass_rate=1.0,
        average_score=1.0,
        category_scores={"happy_path": 1.0, "safety": 1.0},
        results=[res1_pass, res2_pass],
    )

    comparison = compare_runs(report_baseline, report_improved)
    assert comparison.has_regressions is False
    assert "SCEN_02" in comparison.improvements
    assert comparison.pass_rate_delta == 0.5

    # Test when a regression occurs
    report_regressed = EvaluationReport(
        run_id="RUN_REGRESSED",
        timestamp="2026-10-05T13:00:00",
        total_scenarios=2,
        passed_scenarios=0,
        failed_scenarios=2,
        pass_rate=0.0,
        average_score=0.45,
        category_scores={"happy_path": 0.5, "safety": 0.4},
        results=[res1_fail, res2_fail],
    )
    reg_comp = compare_runs(report_baseline, report_regressed)
    assert reg_comp.has_regressions is True
    assert "SCEN_01" in reg_comp.regressions
