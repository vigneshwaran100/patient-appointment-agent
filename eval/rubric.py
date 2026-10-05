from __future__ import annotations

from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    """Evaluation score for a single metric criterion."""

    metric_name: str
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    passed: bool
    weight: float = Field(default=1.0, ge=0.0)
    reason: str


class ScenarioEvaluationResult(BaseModel):
    """Comprehensive evaluation record for one benchmark scenario."""

    scenario_id: str
    name: str
    category: str
    passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)

    # 7 Required Evaluation Dimensions
    task_completion: MetricScore
    correctness: MetricScore
    tool_correctness: MetricScore
    constraint_adherence: MetricScore
    safety: MetricScore
    clarification_quality: MetricScore
    state_side_effect_correctness: MetricScore

    failure_reasons: list[str] = Field(default_factory=list)
    dialogue_transcript: list[dict[str, str]] = Field(default_factory=list)
    tool_calls_executed: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Aggregated benchmark report across all scenarios."""

    run_id: str
    timestamp: str
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    pass_rate: float
    average_score: float
    category_scores: dict[str, float] = Field(default_factory=dict)
    results: list[ScenarioEvaluationResult] = Field(default_factory=list)


# --- Weights for Metric Computation ---

METRIC_WEIGHTS = {
    "task_completion": 0.20,
    "correctness": 0.15,
    "tool_correctness": 0.15,
    "constraint_adherence": 0.15,
    "safety": 0.15,
    "clarification_quality": 0.10,
    "state_side_effect_correctness": 0.10,
}
