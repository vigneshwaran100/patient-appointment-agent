from __future__ import annotations

from datetime import datetime
from typing import Any

from app.state import CancellationRequest, RescheduleRequest
from app.storage.database import ClinicDatabase, get_db


def cancel_appointment_tool(
    appointment_id: str,
    patient_id: str | None = None,
    reason: str | None = None,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Deterministically cancel an existing appointment and release its slot back to the clinic.

    Guarantees:
    - Input validation.
    - Identification of non-existent or previously cancelled appointments.
    - Patient authorization check if patient_id is supplied.
    - Immediate release of slot in doctor_availability.
    """
    database = db or get_db()

    clean_appt_id = appointment_id.strip() if appointment_id else ""
    clean_patient_id = patient_id.strip() if patient_id else None
    clean_reason = reason.strip() if reason else None

    if not clean_appt_id:
        return {
            "success": False,
            "appointment_id": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: appointment_id is mandatory.",
        }

    # Execute cancellation
    req = CancellationRequest(
        appointment_id=clean_appt_id,
        patient_id=clean_patient_id,
        reason=clean_reason,
    )
    result = database.cancel_appointment(req)

    if not result.success:
        err_msg = result.error_message or "Cancellation failed"
        if "not found" in err_msg.lower():
            code = "APPOINTMENT_NOT_FOUND"
        elif "already cancelled" in err_msg.lower():
            code = "ALREADY_CANCELLED"
        elif "unauthorized" in err_msg.lower():
            code = "UNAUTHORIZED_CANCELLATION"
        else:
            code = "CANCELLATION_FAILED"

        return {
            "success": False,
            "appointment_id": clean_appt_id,
            "error_code": code,
            "error_message": err_msg,
        }

    return {
        "success": True,
        "appointment_id": clean_appt_id,
        "error_code": None,
        "error_message": None,
        "message": f"Appointment {clean_appt_id} successfully cancelled and slot released.",
    }


def reschedule_appointment_tool(
    appointment_id: str,
    new_slot_datetime: str,
    new_doctor_id: str | None = None,
    reason: str | None = None,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Deterministically reschedule an existing appointment to a new available slot.

    Atomically releases the old slot and reserves the new slot.
    """
    database = db or get_db()

    clean_appt_id = appointment_id.strip() if appointment_id else ""
    clean_slot = new_slot_datetime.strip() if new_slot_datetime else ""
    clean_doctor_id = new_doctor_id.strip() if new_doctor_id else None
    clean_reason = reason.strip() if reason else None

    if not clean_appt_id:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: appointment_id is mandatory.",
        }

    if not clean_slot:
        return {
            "success": False,
            "appointment": None,
            "error_code": "INVALID_INPUT",
            "error_message": "Missing required field: new_slot_datetime is mandatory.",
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

    req = RescheduleRequest(
        appointment_id=clean_appt_id,
        new_slot_datetime=clean_slot,
        new_doctor_id=clean_doctor_id,
        reason=clean_reason,
    )
    result = database.reschedule_appointment(req)

    if not result.success:
        err_msg = result.error_message or "Reschedule failed"
        if "not found" in err_msg.lower():
            code = "APPOINTMENT_NOT_FOUND"
        elif "not available" in err_msg.lower():
            code = "SLOT_UNAVAILABLE"
        else:
            code = "RESCHEDULE_FAILED"

        return {
            "success": False,
            "appointment": None,
            "error_code": code,
            "error_message": err_msg,
            "alternative_slots": [s.model_dump() for s in result.alternative_slots],
        }

    return {
        "success": True,
        "old_appointment_id": result.old_appointment_id,
        "appointment": result.new_appointment.model_dump() if result.new_appointment else None,
        "error_code": None,
        "error_message": None,
    }
