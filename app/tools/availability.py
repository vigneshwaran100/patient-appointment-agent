from __future__ import annotations

from datetime import datetime
from typing import Any

from app.storage.database import ClinicDatabase, get_db


def check_availability(
    doctor_id: str | None = None,
    specialty: str | None = None,
    date: str | None = None,
    limit: int = 10,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Look up available appointment slots from the deterministic database.

    Filters by doctor_id, specialty, or date prefix (YYYY-MM-DD).
    Never fabricates or hallucinates availability.
    """
    database = db or get_db()

    # Input validation
    clean_doctor_id = doctor_id.strip() if doctor_id else None
    clean_specialty = specialty.strip() if specialty else None
    clean_date = date.strip() if date else None

    if clean_date:
        # Validate format (YYYY-MM-DD)
        try:
            datetime.strptime(clean_date, "%Y-%m-%d")
        except ValueError:
            return {
                "success": False,
                "total_slots": 0,
                "slots": [],
                "error_code": "INVALID_INPUT",
                "error_message": f"Invalid date format '{clean_date}'. Expected YYYY-MM-DD.",
            }

    if limit < 1 or limit > 50:
        return {
            "success": False,
            "total_slots": 0,
            "slots": [],
            "error_code": "INVALID_INPUT",
            "error_message": "Query limit must be an integer between 1 and 50.",
        }

    slots = database.get_available_slots(
        doctor_id=clean_doctor_id,
        specialty=clean_specialty,
        date_prefix=clean_date,
        limit=limit,
    )

    return {
        "success": True,
        "total_slots": len(slots),
        "slots": [slot.model_dump() for slot in slots],
        "error_code": None,
        "error_message": None if slots else "No available slots found for the requested criteria.",
    }


def list_doctors_tool(
    specialty: str | None = None,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Retrieve list of clinic physicians, optionally filtered by medical specialty."""
    database = db or get_db()
    clean_specialty = specialty.strip() if specialty else None
    doctors = database.list_doctors(specialty=clean_specialty)

    return {
        "success": True,
        "total_doctors": len(doctors),
        "doctors": [doc.model_dump() for doc in doctors],
        "error_code": None,
        "error_message": None
        if doctors
        else f"No doctors found for specialty '{clean_specialty}'.",
    }
