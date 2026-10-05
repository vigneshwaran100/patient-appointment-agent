from __future__ import annotations

import re
from datetime import datetime

from improvement.failure_analyzer import FailureDiagnosis
from improvement.policy_store import PolicyItem


class ImprovementGenerator:
    """Synthesizes structured, versioned policy improvements to remedy diagnosed failures."""

    def generate_policies_for_diagnoses(
        self, diagnoses: list[FailureDiagnosis]
    ) -> list[PolicyItem]:
        """Convert failure diagnoses into concrete versioned policy proposals."""
        policies: list[PolicyItem] = []
        now = datetime.now().isoformat()

        for diag in diagnoses:
            # Generate deterministic policy ID from category and error type
            clean_type = re.sub(r"[^A-Za-z0-9_]+", "", diag.error_type.upper())
            policy_id = f"POL_{clean_type}"

            title = f"Remedy for {diag.scenario_name}"
            directive = diag.suggested_policy_patch.strip()

            policy_item = PolicyItem(
                policy_id=policy_id,
                version=1,
                title=title,
                directive=directive,
                target_error_category=diag.error_type,
                is_active=True,
                created_at=now,
            )
            policies.append(policy_item)

        return policies
