from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.graph import build_agent_graph
from app.prompts import (
    get_active_learned_policies,
    register_learned_policy,
)
from app.state import AgentState


class AgentResponse(BaseModel):
    """Structured response from the scheduling agent."""

    response: str
    intent: str | None = None
    is_safe: bool = True
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class SchedulingAgent:
    """Production-grade conversational agent for patient appointment scheduling."""

    def __init__(self, custom_policies: list[str] | None = None):
        self.workflow = build_agent_graph()
        self.app = self.workflow.compile()
        policies = list(custom_policies or [])
        for p in policies:
            register_learned_policy(p)

        self.conversation_state: AgentState = {
            "messages": [],
            "verification": {"is_verified": False},
            "patient": None,
            "intent": None,
            "safety": {"is_safe": True},
            "requested_specialty": None,
            "requested_doctor_id": None,
            "requested_datetime": None,
            "requested_patient_id": None,
            "requested_patient_phone": None,
            "requested_patient_name": None,
            "requested_patient_dob": None,
            "appointment_id": None,
            "is_ambiguous": False,
            "available_slots": [],
            "selected_slot": None,
            "last_booking_result": None,
            "last_cancellation_result": None,
            "lookup_failed": False,
            "active_policies": policies,
            "turn_count": 0,
            "execution_trace": [],
            "errors": [],
        }

    def add_policy(self, policy_text: str) -> None:
        """Register a runtime policy directive."""
        register_learned_policy(policy_text)
        if policy_text not in self.conversation_state.setdefault("active_policies", []):
            self.conversation_state["active_policies"].append(policy_text)

    def reset(self) -> None:
        """Reset conversation state for a new session."""
        self.conversation_state = {
            "messages": [],
            "verification": {"is_verified": False},
            "patient": None,
            "intent": None,
            "safety": {"is_safe": True},
            "requested_specialty": None,
            "requested_doctor_id": None,
            "requested_datetime": None,
            "requested_patient_id": None,
            "requested_patient_phone": None,
            "requested_patient_name": None,
            "requested_patient_dob": None,
            "appointment_id": None,
            "is_ambiguous": False,
            "available_slots": [],
            "selected_slot": None,
            "last_booking_result": None,
            "last_cancellation_result": None,
            "last_registration_result": None,
            "lookup_failed": False,
            "active_policies": get_active_learned_policies(),
            "turn_count": 0,
            "execution_trace": [],
            "errors": [],
        }

    def process_turn(self, user_message: str) -> AgentResponse:
        """Process one conversational turn through the LangGraph workflow."""
        # 1. Update state with incoming user message
        self.conversation_state.setdefault("messages", []).append(
            {
                "role": "user",
                "content": user_message,
            }
        )
        self.conversation_state["turn_count"] = self.conversation_state.get("turn_count", 0) + 1
        self.conversation_state["internal_steps"] = 0

        # Clear per-turn transient tool results
        self.conversation_state["last_booking_result"] = None
        self.conversation_state["last_cancellation_result"] = None

        # 2. Invoke compiled LangGraph graph
        output_state = self.app.invoke(self.conversation_state)
        self.conversation_state = output_state

        # 3. Extract assistant's latest response
        assistant_messages = [
            m.get("content", "").strip()
            for m in self.conversation_state.get("messages", [])
            if m.get("role") == "assistant" and m.get("content", "").strip()
        ]
        last_response = (
            assistant_messages[-1]
            if assistant_messages
            else "I have processed your request. How else may I assist you with your appointment today?"
        )

        # Extract tool calls from execution trace for this turn
        trace = self.conversation_state.get("execution_trace", [])
        tool_calls = [item for item in trace if "tool" in item]

        return AgentResponse(
            response=last_response,
            intent=self.conversation_state.get("intent"),
            is_safe=self.conversation_state.get("safety", {}).get("is_safe", True),
            tool_calls=tool_calls,
            state=dict(self.conversation_state),
        )


def create_agent(custom_policies: list[str] | None = None) -> SchedulingAgent:
    """Factory helper to construct a configured SchedulingAgent."""
    return SchedulingAgent(custom_policies=custom_policies)
