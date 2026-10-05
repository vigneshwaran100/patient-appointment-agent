from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agent import SchedulingAgent
from app.prompts import clear_learned_policies
from app.storage.database import ClinicDatabase, get_db, reset_db_singleton
from eval.evaluator import ScenarioEvaluator
from eval.rubric import EvaluationReport, ScenarioEvaluationResult


class EvaluationRunner:
    """Benchmark harness executing scenarios and producing verified evaluation reports."""

    def __init__(
        self,
        scenarios_path: str | None = None,
        custom_policies: list[str] | None = None,
    ):
        base_dir = Path(__file__).resolve().parent
        self.scenarios_path = scenarios_path or str(base_dir / "scenarios.json")
        self.custom_policies = custom_policies or []

    def load_scenarios(self) -> list[dict[str, Any]]:
        """Load benchmark scenarios from JSON file."""
        with open(self.scenarios_path, encoding="utf-8") as f:
            return json.load(f)

    def run_scenario(self, scenario: dict[str, Any]) -> ScenarioEvaluationResult:
        """Execute one scenario against an isolated SQLite test database."""
        # 1. Create isolated temporary database for test determinism
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db_path = f.name

        db = ClinicDatabase(db_path=temp_db_path)
        db.seed_default_data()
        get_db(db_path=temp_db_path)

        # 2. Initialize Agent
        clear_learned_policies()
        agent = SchedulingAgent(custom_policies=self.custom_policies)

        transcript: list[dict[str, str]] = []

        # 3. Execute conversational turns
        try:
            for user_input in scenario.get("turns", []):
                transcript.append({"role": "user", "content": user_input})
                resp = agent.process_turn(user_input)
                transcript.append({"role": "assistant", "content": resp.response})

            # 4. Evaluate run and database state
            evaluator = ScenarioEvaluator(db=db)
            result = evaluator.evaluate(
                scenario=scenario,
                final_state=agent.conversation_state,
                transcript=transcript,
            )
            return result

        finally:
            reset_db_singleton()
            try:
                os.remove(temp_db_path)
            except OSError:
                pass

    def run_all(self, output_path: str | None = None) -> EvaluationReport:
        """Run all loaded benchmark scenarios and compile comprehensive report."""
        scenarios = self.load_scenarios()
        results: list[ScenarioEvaluationResult] = []

        category_totals: dict[str, list[float]] = {}

        for scen in scenarios:
            result = self.run_scenario(scen)
            results.append(result)
            category_totals.setdefault(result.category, []).append(result.overall_score)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = round(passed / total, 3) if total > 0 else 0.0
        avg_score = round(sum(r.overall_score for r in results) / total, 3) if total > 0 else 0.0

        category_scores = {
            cat: round(sum(scores) / len(scores), 3) for cat, scores in category_totals.items()
        }

        report = EvaluationReport(
            run_id=f"RUN_{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now().isoformat(),
            total_scenarios=total,
            passed_scenarios=passed,
            failed_scenarios=failed,
            pass_rate=pass_rate,
            average_score=avg_score,
            category_scores=category_scores,
            results=results,
        )

        # Persist report if destination specified
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

        return report


def main() -> None:
    """CLI runner executing benchmark suite and saving results."""
    runner = EvaluationRunner()
    results_dir = Path(__file__).resolve().parent.parent / "results"
    output_file = str(results_dir / "before.json")

    print(f"Running evaluation benchmark on scenarios from {runner.scenarios_path}...")
    report = runner.run_all(output_path=output_file)

    print("\n" + "=" * 50)
    print(f"EVALUATION COMPLETE - Run ID: {report.run_id}")
    print(
        f"Pass Rate: {report.pass_rate * 100:.1f}% ({report.passed_scenarios}/{report.total_scenarios} passed)"
    )
    print(f"Average Score: {report.average_score:.3f}")
    print("\nCategory Performance:")
    for cat, score in report.category_scores.items():
        print(f"  • {cat}: {score:.3f}")
    print(f"\nReport written to: {output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
