from __future__ import annotations

from datetime import datetime
from typing import Any

from app.state import BookingRequest
from app.storage.database import ClinicDatabase, get_db


def book_appointment_tool(
    patient_id: str,
    doctor_id: str,
    slot_datetime: str,
    reason: str | None = "Routine consultation",
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Deterministically book an appointment in SQLite.

    Guarantees:
    - Input validation (date-time format, non-empty IDs).
    - Prevention of double-booking doctor slots.
    - Prevention of duplicate simultaneous bookings for the patient.
    - Atomic transaction and state update.
    - Alternative slots provided on conflict.
    """
    database = db or get_db()

    # Input sanitization and validation
    clean_patient_id = patient_id.strip() if patient_id else ""
    clean_doctor_id = doctor_id.strip() if doctor_id else ""
    clean_slot = slot_datetime.strip() if slot_datetime else ""
    clean_reason = reason.strip() if reason else "Routine consultation"

    if not clean_patient_id:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: patient_id is mandatory.",
        }

    if not clean_doctor_id:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: doctor_id is mandatory.",
        }

    if not clean_slot:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: slot_datetime is mandatory.",
        }

    try:
        datetime.strptime(clean_slot, "%Y-%m-%d %H:%M")
    except ValueError:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": f"Invalid slot datetime format '{clean_slot}'. Expected YYYY-MM-DD HH:MM.",
        }

    # Verify patient exists
    patient = database.find_patient(patient_id=clean_patient_id)
    if not patient:
        return {
            "success": False,
            "appointment": None,
            "error_code": "PATIENT_NOT_FOUND",
            "error_message": f"Patient with ID '{clean_patient_id}' is not registered.",
        }

    # Verify doctor exists
    doctors = database.list_doctors()
    doc_match = next((d for d in doctors if d.doctor_id == clean_doctor_id), None)
    if not doc_match:
        return {
            "success": False,
            "appointment": None,
            "error_code": "DOCTOR_NOT_FOUND",
            "error_message": f"Doctor with ID '{clean_doctor_id}' does not exist.",
        }

    # Execute booking through transactional repository
    req = BookingRequest(
        patient_id=clean_patient_id,
        doctor_id=clean_doctor_id,
        slot_datetime=clean_slot,
        reason=clean_reason,
    )
    result = database.book_appointment(req)

    if not result.success:
        # Determine specific error code
        err_msg = result.error_message or "Booking failed"
        if "already has an active appointment" in err_msg:
            code = "DUPLICATE_BOOKING"
        elif "not available" in err_msg:
            code = "SLOT_UNAVAILABLE"
        elif "just been booked" in err_msg:
            code = "DOUBLE_BOOKING"
        else:
            code = "BOOKING_FAILED"

        return {
            "success": False,
            "appointment": None,
            "error_code": code,
            "error_message": err_msg,
            "alternative_slots": [s.model_dump() for s in result.alternative_slots],
        }

    return {
        "success": True,
        "appointment": result.appointment.model_dump() if result.appointment else None,
        "error_code": None,
        "error_message": None,
    }
