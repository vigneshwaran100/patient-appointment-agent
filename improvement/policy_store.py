from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class PolicyItem(BaseModel):
    """Versioned clinical policy directive."""

    policy_id: str
    version: int = 1
    title: str
    directive: str
    target_error_category: str
    is_active: bool = True
    created_at: str


class PolicyStore:
    """Manages versioned, structured policy rules that are dynamically compiled into the agent."""

    def __init__(self, store_path: str | None = None):
        if not store_path:
            base_dir = Path(__file__).resolve().parent
            store_path = str(base_dir / "policies" / "policies.json")
        self.store_path = Path(store_path)
        self.policies: dict[str, PolicyItem] = {}
        self.load()

    def _ensure_dir(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load versioned policies from storage file."""
        if self.store_path.exists():
            with open(self.store_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    self.policies = {}
                    return
                data = json.loads(content)
                self.policies = {
                    item["policy_id"]: PolicyItem(**item) for item in data.get("policies", [])
                }
        else:
            self.policies = {}

    def save(self) -> None:
        """Persist current policy collection to storage."""
        self._ensure_dir()
        serialized = [item.model_dump() for item in self.policies.values()]
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump({"policies": serialized}, f, indent=2)

    def add_or_update_policy(
        self,
        policy_id: str,
        title: str,
        directive: str,
        target_error_category: str,
    ) -> PolicyItem:
        """Add a new policy or bump its version if directive changed."""
        now = datetime.now().isoformat()
        if policy_id in self.policies:
            existing = self.policies[policy_id]
            new_version = existing.version + 1
            item = PolicyItem(
                policy_id=policy_id,
                version=new_version,
                title=title,
                directive=directive,
                target_error_category=target_error_category,
                is_active=True,
                created_at=now,
            )
        else:
            item = PolicyItem(
                policy_id=policy_id,
                version=1,
                title=title,
                directive=directive,
                target_error_category=target_error_category,
                is_active=True,
                created_at=now,
            )

        self.policies[policy_id] = item
        self.save()
        return item

    def deactivate_policy(self, policy_id: str) -> bool:
        """Deactivate a policy directive."""
        if policy_id in self.policies:
            self.policies[policy_id].is_active = False
            self.save()
            return True
        return False

    def get_active_directives(self) -> list[str]:
        """Return list of string directives for all active policies."""
        return [item.directive for item in self.policies.values() if item.is_active]

    def list_policies(self) -> list[PolicyItem]:
        """Return list of all registered policies."""
        return list(self.policies.values())
