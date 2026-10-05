"""LangChain tool definitions for the Patient Appointment Scheduling Agent.

All tools here call existing deterministic SQLite tools.
The LLM NEVER executes SQL directly.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.tools.availability import check_availability as _check_availability
from app.tools.availability import list_doctors_tool as _list_doctors_tool
from app.tools.booking import book_appointment_tool as _book_appointment_tool
from app.tools.cancellation import cancel_appointment_tool as _cancel_appointment_tool
from app.tools.cancellation import reschedule_appointment_tool as _reschedule_appointment_tool
from app.tools.patient import lookup_patient as _lookup_patient
from app.tools.patient import register_patient as _register_patient


@tool
def lookup_patient(
    patient_id: str | None = None,
    phone: str | None = None,
    name: str | None = None,
    dob: str | None = None,
) -> dict[str, Any]:
    """Look up an existing patient record in the clinic database.

    Provide patient_id (e.g. 'P001'), phone (e.g. '555-0199'), or name + date of birth (YYYY-MM-DD).
    Returns verified patient details or an error code.
    """
    return _lookup_patient(patient_id=patient_id, phone=phone, name=name, dob=dob)


@tool
def register_patient(
    name: str,
    phone: str,
    dob: str,
) -> dict[str, Any]:
    """Register a new patient into the clinic database.

    Requires full name, contact phone number, and date of birth (YYYY-MM-DD or DD/MM/YYYY).
    Prevents duplicates and returns the newly assigned unique Patient ID (e.g. 'P006').
    """
    return _register_patient(name=name, phone=phone, dob=dob)


@tool
def list_doctors_tool(specialty: str | None = None) -> dict[str, Any]:
    """List physicians and their specialties in the clinic.

    Optionally filter by specialty (e.g., 'Cardiology', 'Dermatology', 'General Medicine', 'Pediatrics', 'Orthopedics').
    """
    return _list_doctors_tool(specialty=specialty)


@tool
def check_availability(
    doctor_id: str | None = None,
    specialty: str | None = None,
    date: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Query available open appointment slots from the clinic schedule in SQLite.

    Optionally filter by doctor_id (e.g. 'DOC001'), specialty (e.g. 'Cardiology'), and/or specific date (YYYY-MM-DD).
    Returns only verified open slots from the database.
    """
    return _check_availability(doctor_id=doctor_id, specialty=specialty, date=date, limit=limit)


@tool
def book_appointment_tool(
    patient_id: str,
    doctor_id: str,
    slot_datetime: str,
    reason: str = "Routine consultation",
) -> dict[str, Any]:
    """Book an appointment slot atomically in the clinic database.

    Requires verified patient_id (e.g. 'P001'), doctor_id (e.g. 'DOC001'), and slot_datetime (format: 'YYYY-MM-DD HH:MM').
    Returns confirmed appointment details with unique Appointment ID (e.g. 'APT1002') or alternative available slots if unavailable.
    """
    return _book_appointment_tool(
        patient_id=patient_id,
        doctor_id=doctor_id,
        slot_datetime=slot_datetime,
        reason=reason,
    )


@tool
def cancel_appointment_tool(
    appointment_id: str,
    patient_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Cancel a scheduled appointment and release the slot back to available in SQLite.

    Requires appointment_id (e.g. 'APT1001'). Optionally provide patient_id for authorization check.
    """
    return _cancel_appointment_tool(
        appointment_id=appointment_id,
        patient_id=patient_id,
        reason=reason,
    )


@tool
def reschedule_appointment_tool(
    appointment_id: str,
    new_slot_datetime: str,
    new_doctor_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Reschedule an existing appointment to a new slot datetime and/or doctor atomically.

    Requires existing appointment_id and target new_slot_datetime ('YYYY-MM-DD HH:MM').
    """
    return _reschedule_appointment_tool(
        appointment_id=appointment_id,
        new_slot_datetime=new_slot_datetime,
        new_doctor_id=new_doctor_id,
        reason=reason,
    )


CLINIC_TOOLS = [
    lookup_patient,
    register_patient,
    list_doctors_tool,
    check_availability,
    book_appointment_tool,
    cancel_appointment_tool,
    reschedule_appointment_tool,
]

CLINIC_TOOLS_BY_NAME = {t.name: t for t in CLINIC_TOOLS}
