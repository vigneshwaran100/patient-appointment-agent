from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from eval.rubric import EvaluationReport


class RegressionComparison(BaseModel):
    """Comparison record between a baseline evaluation run and an improved run."""

    timestamp: str
    baseline_run_id: str
    new_run_id: str
    baseline_pass_rate: float
    new_pass_rate: float
    pass_rate_delta: float
    baseline_average_score: float
    new_average_score: float
    average_score_delta: float
    has_regressions: bool
    regressions: list[str] = Field(
        default_factory=list, description="Scenario IDs that regressed from pass to fail"
    )
    improvements: list[str] = Field(
        default_factory=list, description="Scenario IDs that improved from fail to pass"
    )
    category_deltas: dict[str, float] = Field(default_factory=dict)


def compare_runs(
    baseline_report: EvaluationReport,
    new_report: EvaluationReport,
    output_path: str | None = None,
) -> RegressionComparison:
    """Compare baseline and new evaluation reports to detect regressions and improvements."""
    base_results = {r.scenario_id: r for r in baseline_report.results}
    new_results = {r.scenario_id: r for r in new_report.results}

    regressions: list[str] = []
    improvements: list[str] = []

    for sid, new_res in new_results.items():
        base_res = base_results.get(sid)
        if base_res:
            if base_res.passed and not new_res.passed:
                regressions.append(sid)
            elif not base_res.passed and new_res.passed:
                improvements.append(sid)

    # Category score differences
    category_deltas: dict[str, float] = {}
    all_categories = set(baseline_report.category_scores.keys()).union(
        new_report.category_scores.keys()
    )
    for cat in all_categories:
        base_cat = baseline_report.category_scores.get(cat, 0.0)
        new_cat = new_report.category_scores.get(cat, 0.0)
        category_deltas[cat] = round(new_cat - base_cat, 3)

    comparison = RegressionComparison(
        timestamp=datetime.now().isoformat(),
        baseline_run_id=baseline_report.run_id,
        new_run_id=new_report.run_id,
        baseline_pass_rate=baseline_report.pass_rate,
        new_pass_rate=new_report.pass_rate,
        pass_rate_delta=round(new_report.pass_rate - baseline_report.pass_rate, 3),
        baseline_average_score=baseline_report.average_score,
        new_average_score=new_report.average_score,
        average_score_delta=round(new_report.average_score - baseline_report.average_score, 3),
        has_regressions=len(regressions) > 0,
        regressions=regressions,
        improvements=improvements,
        category_deltas=category_deltas,
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(comparison.model_dump_json(indent=2))

    return comparison


def main() -> None:
    """CLI to compare results/before.json and results/after.json."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    before_file = results_dir / "before.json"
    after_file = results_dir / "after.json"
    comparison_file = results_dir / "comparison.json"

    if not before_file.exists() or not after_file.exists():
        print(
            f"Error: Both {before_file} and {after_file} must exist to run regression comparison."
        )
        return

    with open(before_file, encoding="utf-8") as f:
        before_data = json.load(f)
    with open(after_file, encoding="utf-8") as f:
        after_data = json.load(f)

    before_report = EvaluationReport(**before_data)
    after_report = EvaluationReport(**after_data)

    comparison = compare_runs(before_report, after_report, output_path=str(comparison_file))

    print("\n" + "=" * 50)
    print("REGRESSION COMPARISON RESULTS")
    print(
        f"Pass Rate: {comparison.baseline_pass_rate * 100:.1f}% -> {comparison.new_pass_rate * 100:.1f}% (Delta: {comparison.pass_rate_delta * 100:+.1f}%)"
    )
    print(
        f"Average Score: {comparison.baseline_average_score:.3f} -> {comparison.new_average_score:.3f} (Delta: {comparison.average_score_delta:+.3f})"
    )
    print(f"Has Regressions: {comparison.has_regressions}")
    if comparison.regressions:
        print(f"  Regressed Scenarios: {comparison.regressions}")
    if comparison.improvements:
        print(f"  Improved Scenarios: {comparison.improvements}")
    print(f"Comparison report saved to: {comparison_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
