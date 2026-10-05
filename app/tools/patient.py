from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.storage.database import ClinicDatabase, get_db


def normalize_dob(dob_str: str) -> str | None:
    """Normalize various date of birth formats into standard YYYY-MM-DD."""
    if not dob_str:
        return None
    cleaned = dob_str.strip()
    formats = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d")
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def lookup_patient(
    patient_id: str | None = None,
    phone: str | None = None,
    name: str | None = None,
    dob: str | None = None,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Look up a patient by patient_id, phone, or name + date of birth.

    Returns structured output with patient details or clear failure code.
    Never returns free-form unverified claims.
    """
    database = db or get_db()

    # Input sanitization
    clean_id = patient_id.strip() if patient_id else None
    clean_phone = phone.strip() if phone else None
    clean_name = name.strip() if name else None
    clean_dob = dob.strip() if dob else None

    # Validation: at least one identifying parameter must be given
    if not any([clean_id, clean_phone, clean_name]):
        return {
            "success": False,
            "patient": None,
            "error_code": "INVALID_INPUT",
            "error_message": "At least one lookup criteria (patient_id, phone, or name) must be provided.",
        }

    # Validate DOB format if supplied
    if clean_dob:
        try:
            datetime.strptime(clean_dob, "%Y-%m-%d")
        except ValueError:
            return {
                "success": False,
                "patient": None,
                "error_code": "INVALID_INPUT",
                "error_message": f"Invalid date of birth format '{clean_dob}'. Expected YYYY-MM-DD.",
            }

    # Execute deterministic lookup
    patient = database.find_patient(
        patient_id=clean_id,
        phone=clean_phone,
        name=clean_name,
        dob=clean_dob,
    )

    if not patient:
        return {
            "success": False,
            "patient": None,
            "error_code": "PATIENT_NOT_FOUND",
            "error_message": "No patient record found matching the provided identification details.",
        }

    return {
        "success": True,
        "patient": patient.model_dump(),
        "error_code": None,
        "error_message": None,
    }


def register_patient(
    name: str,
    phone: str,
    dob: str,
    db: ClinicDatabase | None = None,
) -> dict[str, Any]:
    """Register a new patient into the clinic database with validation and duplicate prevention.

    Returns structured result with the assigned unique patient ID.
    Never allows the LLM to access SQLite directly.
    """
    database = db or get_db()

    clean_name = name.strip() if name else ""
    clean_phone = phone.strip() if phone else ""
    clean_dob = dob.strip() if dob else ""

    # 1. Validation
    if not clean_name or len(clean_name) < 2:
        return {
            "success": False,
            "patient": None,
            "error_code": "INVALID_NAME",
            "error_message": "A valid full name (at least 2 characters) is required.",
        }

    # Extract digits or basic length check for phone
    digits_phone = re.sub(r"[^\d]", "", clean_phone)
    if not digits_phone or len(digits_phone) < 7:
        return {
            "success": False,
            "patient": None,
            "error_code": "INVALID_PHONE",
            "error_message": "A valid contact phone number with at least 7 digits is required.",
        }

    norm_dob = normalize_dob(clean_dob)
    if not norm_dob:
        return {
            "success": False,
            "patient": None,
            "error_code": "INVALID_DOB",
            "error_message": f"Invalid date of birth '{clean_dob}'. Please use YYYY-MM-DD or DD/MM/YYYY.",
        }

    # 2. Duplicate Detection
    existing_by_phone = database.find_patient(phone=clean_phone)
    if existing_by_phone:
        return {
            "success": False,
            "patient": None,
            "error_code": "DUPLICATE_PATIENT",
            "error_message": f"A patient with phone {clean_phone} is already registered (Patient ID: {existing_by_phone.patient_id}).",
        }

    existing_by_identity = database.find_patient(name=clean_name, dob=norm_dob)
    if existing_by_identity:
        return {
            "success": False,
            "patient": None,
            "error_code": "DUPLICATE_PATIENT",
            "error_message": f"Patient '{clean_name}' with date of birth {norm_dob} is already registered (Patient ID: {existing_by_identity.patient_id}).",
        }

    # 3. Deterministic Insertion in SQLite
    try:
        new_patient = database.register_patient(
            name=clean_name,
            phone=clean_phone,
            dob=norm_dob,
        )
        return {
            "success": True,
            "patient": new_patient.model_dump(),
            "error_code": None,
            "error_message": None,
        }
    except Exception as e:
        return {
            "success": False,
            "patient": None,
            "error_code": "DATABASE_ERROR",
            "error_message": f"Failed to register patient: {e}",
        }


# Alias for consistency with other tools
register_patient_tool = register_patient
