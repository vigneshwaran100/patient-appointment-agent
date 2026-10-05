from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

# --- Enums ---


class DoctorSpecialty(StrEnum):
    CARDIOLOGY = "Cardiology"
    DERMATOLOGY = "Dermatology"
    GENERAL_MEDICINE = "General Medicine"
    PEDIATRICS = "Pediatrics"
    ORTHOPEDICS = "Orthopedics"


class AppointmentStatus(StrEnum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


class IntentType(StrEnum):
    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    CHECK_AVAILABILITY = "check_availability"
    REGISTER = "register"
    GENERAL_INQUIRY = "general_inquiry"
    EMERGENCY = "emergency"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# --- Domain Models ---


class Patient(BaseModel):
    """Patient entity registered in the clinic."""

    patient_id: str = Field(description="Unique patient identifier, e.g., P001")
    name: str = Field(description="Full name of patient")
    phone: str = Field(description="Contact phone number")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")
    created_at: str | None = None

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as err:
            raise ValueError(f"Invalid DOB format '{v}'. Expected YYYY-MM-DD.") from err
        return v


class PatientVerificationStatus(BaseModel):
    """State tracking whether patient identity is confirmed for booking/cancelling."""

    is_verified: bool = False
    patient: Patient | None = None
    failed_attempts: int = 0
    verification_method: str | None = None


class Doctor(BaseModel):
    """Physician entity in the clinic."""

    doctor_id: str = Field(description="Unique doctor identifier, e.g., DOC001")
    name: str = Field(description="Full title and name, e.g., Dr. Alice Smith")
    specialty: str = Field(description="Medical specialty")
    room: str | None = Field(default=None, description="Clinic room number")


class TimeSlot(BaseModel):
    """A discrete schedule slot for a physician."""

    slot_id: int | None = None
    doctor_id: str
    doctor_name: str
    specialty: str
    slot_datetime: str = Field(description="ISO format string: YYYY-MM-DD HH:MM")
    is_available: bool = True

    @field_validator("slot_datetime")
    @classmethod
    def validate_slot_datetime(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M")
        except ValueError as err:
            raise ValueError(f"Invalid slot format '{v}'. Expected YYYY-MM-DD HH:MM.") from err
        return v


class Appointment(BaseModel):
    """Confirmed appointment record."""

    appointment_id: str
    patient_id: str
    doctor_id: str
    doctor_name: str | None = None
    slot_datetime: str
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    reason: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


# --- Operation Requests & Results ---


class BookingRequest(BaseModel):
    patient_id: str
    doctor_id: str
    slot_datetime: str
    reason: str | None = "Routine consultation"


class BookingResult(BaseModel):
    success: bool
    appointment: Appointment | None = None
    error_message: str | None = None
    alternative_slots: list[TimeSlot] = Field(default_factory=list)


class CancellationRequest(BaseModel):
    appointment_id: str
    patient_id: str | None = None
    reason: str | None = None


class CancellationResult(BaseModel):
    success: bool
    appointment_id: str | None = None
    error_message: str | None = None


class RescheduleRequest(BaseModel):
    appointment_id: str
    new_slot_datetime: str
    new_doctor_id: str | None = None
    reason: str | None = None


class RescheduleResult(BaseModel):
    success: bool
    old_appointment_id: str | None = None
    new_appointment: Appointment | None = None
    error_message: str | None = None
    alternative_slots: list[TimeSlot] = Field(default_factory=list)


class PatientRegistrationRequest(BaseModel):
    name: str
    phone: str
    dob: str


class PatientRegistrationResult(BaseModel):
    success: bool
    patient: Patient | None = None
    error_code: str | None = None
    error_message: str | None = None


class SafetyClassification(BaseModel):
    """Structured output from guardrail evaluation on user input."""

    category: Literal[
        "safe",
        "medical_advice",
        "emergency",
        "prompt_injection",
        "out_of_scope",
    ] = "safe"
    severity: Literal["low", "medium", "high", "critical"] = "low"
    reason: str = ""
    allowed: bool = True
    suggested_action: str | None = None

    # Backwards compatibility properties
    @property
    def is_safe(self) -> bool:
        return self.allowed

    @property
    def is_medical_emergency(self) -> bool:
        return self.category == "emergency"

    @property
    def is_prompt_injection(self) -> bool:
        return self.category == "prompt_injection"

    @property
    def is_out_of_scope(self) -> bool:
        return self.category in {"medical_advice", "out_of_scope"}

    @property
    def explanation(self) -> str:
        return self.reason


# --- LangGraph State Schema ---


class AgentState(TypedDict, total=False):
    """Orchestration state passed between LangGraph nodes."""

    # Conversation thread
    messages: list[dict[str, Any]]

    # Patient authentication & context
    verification: dict[str, Any]
    patient: dict[str, Any] | None

    # Routing & triage
    intent: str | None
    safety: dict[str, Any]

    # Active clinical context
    requested_specialty: str | None
    requested_doctor_id: str | None
    requested_datetime: str | None
    requested_patient_id: str | None
    requested_patient_phone: str | None
    requested_patient_name: str | None
    requested_patient_dob: str | None
    appointment_id: str | None
    is_ambiguous: bool

    # Slots & tool outputs
    available_slots: list[dict[str, Any]]
    selected_slot: dict[str, Any] | None
    last_booking_result: dict[str, Any] | None
    last_cancellation_result: dict[str, Any] | None
    last_registration_result: dict[str, Any] | None

    # Lookup failure tracking (prevents re-lookup loop after failure)
    lookup_failed: bool

    # Dynamic policies & tracking
    active_policies: list[str]
    turn_count: int
    execution_trace: list[dict[str, Any]]
    errors: list[str]
