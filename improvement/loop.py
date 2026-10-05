from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval.regression import compare_runs
from eval.runner import EvaluationRunner
from improvement.failure_analyzer import FailureAnalyzer, FailureDiagnosis
from improvement.improvement_generator import ImprovementGenerator
from improvement.policy_store import PolicyItem, PolicyStore


class ImprovementLoopResult(BaseModel):
    """Result of an autonomous self-improvement execution cycle."""

    success: bool
    baseline_pass_rate: float
    new_pass_rate: float
    pass_rate_delta: float
    diagnoses: list[FailureDiagnosis] = Field(default_factory=list)
    proposed_policies: list[PolicyItem] = Field(default_factory=list)
    has_regressions: bool = False
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    comparison_summary: dict[str, Any] = Field(default_factory=dict)


class SelfImprovementLoop:
    """Orchestrates: Run -> Evaluate -> Failure -> Diagnosis -> Improvement -> Version -> Re-run -> Regression."""

    def __init__(
        self,
        scenarios_path: str | None = None,
        policy_store_path: str | None = None,
        results_dir: str | None = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.scenarios_path = scenarios_path or str(base_dir / "eval" / "scenarios.json")
        self.results_dir = Path(results_dir or str(base_dir / "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.policy_store = PolicyStore(store_path=policy_store_path)
        self.analyzer = FailureAnalyzer()
        self.generator = ImprovementGenerator()

    def run_cycle(self) -> ImprovementLoopResult:
        """Execute the full end-to-end self-improvement loop."""
        before_file = str(self.results_dir / "before.json")
        after_file = str(self.results_dir / "after.json")
        comp_file = str(self.results_dir / "comparison.json")

        # ----------------------------------------------------
        # Step 1: Run Baseline Evaluation
        # ----------------------------------------------------
        print("\n[Step 1/6] Running Baseline Benchmark Evaluation...")
        baseline_runner = EvaluationRunner(
            scenarios_path=self.scenarios_path,
            custom_policies=[],
        )
        baseline_report = baseline_runner.run_all(output_path=before_file)
        print(
            f"  -> Baseline Pass Rate: {baseline_report.pass_rate * 100:.1f}% ({baseline_report.passed_scenarios}/{baseline_report.total_scenarios})"
        )

        # ----------------------------------------------------
        # Step 2: Failure Analysis & Root Cause Diagnosis
        # ----------------------------------------------------
        print("\n[Step 2/6] Performing Structured Failure Analysis...")
        diagnoses = self.analyzer.analyze_report(baseline_report)
        print(f"  -> Identified {len(diagnoses)} scenario failure(s).")
        for diag in diagnoses:
            print(f"     • [{diag.scenario_id}] {diag.error_type}: {diag.root_cause}")

        if not diagnoses:
            print("\nAll scenarios already passed with 100% accuracy. System is optimal.")
            return ImprovementLoopResult(
                success=True,
                baseline_pass_rate=baseline_report.pass_rate,
                new_pass_rate=baseline_report.pass_rate,
                pass_rate_delta=0.0,
                diagnoses=[],
                proposed_policies=[],
                has_regressions=False,
            )

        # ----------------------------------------------------
        # Step 3: Generate Targeted Structured Improvements
        # ----------------------------------------------------
        print("\n[Step 3/6] Generating Targeted Policy Improvements...")
        proposed_policies = self.generator.generate_policies_for_diagnoses(diagnoses)
        for pol in proposed_policies:
            print(f"  -> Formulated policy [{pol.policy_id}]: {pol.directive[:80]}...")

        # ----------------------------------------------------
        # Step 4: Stage & Version Policies into Policy Store
        # ----------------------------------------------------
        print("\n[Step 4/6] Versioning and Staging Policies...")
        staged_items: list[PolicyItem] = []
        for pol in proposed_policies:
            staged = self.policy_store.add_or_update_policy(
                policy_id=pol.policy_id,
                title=pol.title,
                directive=pol.directive,
                target_error_category=pol.target_error_category,
            )
            staged_items.append(staged)
            print(f"  -> Committed [{staged.policy_id}] version v{staged.version}")

        # Active policies to test in candidate run
        candidate_directives = self.policy_store.get_active_directives()

        # ----------------------------------------------------
        # Step 5: Re-evaluate on Same Scenarios with Candidate Policies
        # ----------------------------------------------------
        print("\n[Step 5/6] Re-evaluating Candidate Agent on Same Benchmark Scenarios...")
        improved_runner = EvaluationRunner(
            scenarios_path=self.scenarios_path,
            custom_policies=candidate_directives,
        )
        after_report = improved_runner.run_all(output_path=after_file)
        print(
            f"  -> Candidate Pass Rate: {after_report.pass_rate * 100:.1f}% ({after_report.passed_scenarios}/{after_report.total_scenarios})"
        )

        # ----------------------------------------------------
        # Step 6: Regression Check & Comparison Report
        # ----------------------------------------------------
        print("\n[Step 6/6] Executing Regression Check...")
        comparison = compare_runs(
            baseline_report=baseline_report,
            new_report=after_report,
            output_path=comp_file,
        )

        if comparison.has_regressions:
            print(f"  [!] REGRESSION DETECTED in scenarios: {comparison.regressions}")
            print("  Rolling back candidate policies...")
            for pol in staged_items:
                self.policy_store.deactivate_policy(pol.policy_id)
            cycle_success = False
        else:
            print(f"  [+] Zero regressions. Improvements validated in: {comparison.improvements}")
            print(f"  [+] Pass rate delta: {comparison.pass_rate_delta * 100:+.1f}%")
            cycle_success = True

        return ImprovementLoopResult(
            success=cycle_success,
            baseline_pass_rate=baseline_report.pass_rate,
            new_pass_rate=after_report.pass_rate,
            pass_rate_delta=comparison.pass_rate_delta,
            diagnoses=diagnoses,
            proposed_policies=staged_items,
            has_regressions=comparison.has_regressions,
            regressions=comparison.regressions,
            improvements=comparison.improvements,
            comparison_summary=comparison.model_dump(),
        )


def main() -> None:
    """CLI to execute the full self-improvement loop."""
    loop = SelfImprovementLoop()
    result = loop.run_cycle()

    print("\n" + "=" * 60)
    print("SELF-IMPROVEMENT LOOP SUMMARY")
    print(f"Cycle Success: {result.success}")
    print(f"Baseline Pass Rate: {result.baseline_pass_rate * 100:.1f}%")
    print(f"New Pass Rate:      {result.new_pass_rate * 100:.1f}%")
    print(f"Pass Rate Delta:    {result.pass_rate_delta * 100:+.1f}%")
    print(f"Has Regressions:    {result.has_regressions}")
    print(f"Improvements:       {result.improvements}")
    print("=" * 60)


if __name__ == "__main__":
    main()
