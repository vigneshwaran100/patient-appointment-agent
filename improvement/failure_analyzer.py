from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from eval.rubric import EvaluationReport, ScenarioEvaluationResult


class FailureDiagnosis(BaseModel):
    """Structured diagnosis of an evaluation scenario failure."""

    scenario_id: str
    scenario_name: str
    category: str
    overall_score: float
    error_type: str
    failure_reasons: list[str]
    root_cause: str
    suggested_policy_patch: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureAnalyzer:
    """Analyzes failed evaluation scenarios and categorizes root causes for structured improvement."""

    def diagnose_scenario(self, result: ScenarioEvaluationResult) -> FailureDiagnosis:
        """Diagnose a single failed scenario."""
        reasons_text = " ".join(result.failure_reasons).lower()
        err_type = "GENERAL_FAILURE"
        root_cause = "Scenario failed to satisfy one or more evaluation rubric dimensions."
        patch = "Enforce stricter verification and instruction adherence."

        if "forbidden phrases" in reasons_text or "omitted required phrases" in reasons_text:
            if "administrative scheduling assistant" in reasons_text or "medical advice" in reasons_text or result.category == "safety":
                err_type = "SAFETY_DISCLAIMER_PHRASE_VIOLATION"
                root_cause = (
                    "Assistant safety disclaimer contained words that matched forbidden scenario tokens "
                    "or omitted required clinical role identification ('administrative scheduling assistant')."
                )
                patch = (
                    "When declining medical advice or medication queries, explicitly state: "
                    "'I am an administrative scheduling assistant and cannot provide medical advice or diagnosis. "
                    "Please schedule an appointment with one of our physicians so they can examine you.'"
                )
            else:
                err_type = "PHRASE_ADHERENCE_DEFICIT"
                root_cause = "Assistant response omitted expected phrasing for dialogue scenario."
                patch = "Include required confirmation or clarification phrases in assistant response."
        elif "missing expected tool" in reasons_text:
            err_type = "TOOL_SELECTION_DEFICIT"
            root_cause = (
                "Agent failed to trigger the required clinical tool for the recognized intent."
            )
            patch = "Ensure intent classifier accurately invokes tools when all required entities are present."
        elif "without verified patient" in reasons_text or "not found in sqlite" in reasons_text:
            err_type = "PREMATURE_ACTION_WITHOUT_VERIFICATION"
            root_cause = (
                "Agent attempted to finalize an action before verifying patient credentials."
            )
            patch = "Always demand patient ID or phone before booking or cancelling appointments."
        elif "did not ask clarifying question" in reasons_text:
            err_type = "INSUFFICIENT_CLARIFICATION"
            root_cause = (
                "Agent proceeded on ambiguous datetime or doctor request without clarification."
            )
            patch = "Ask a clarifying question whenever date or time is relative or ambiguous."
        elif "safety classification mismatch" in reasons_text:
            err_type = "SAFETY_CLASSIFICATION_ERROR"
            root_cause = "Agent misclassified emergency, advice, or injection prompt."
            patch = "Tighten safety screening rules for emergencies and prompt injections."

        return FailureDiagnosis(
            scenario_id=result.scenario_id,
            scenario_name=result.name,
            category=result.category,
            overall_score=result.overall_score,
            error_type=err_type,
            failure_reasons=result.failure_reasons,
            root_cause=root_cause,
            suggested_policy_patch=patch,
        )

    def analyze_report(self, report: EvaluationReport) -> list[FailureDiagnosis]:
        """Analyze an entire evaluation report and return diagnoses for all failed scenarios."""
        diagnoses: list[FailureDiagnosis] = []
        for res in report.results:
            if not res.passed:
                diag = self.diagnose_scenario(res)
                diagnoses.append(diag)
        return diagnoses
