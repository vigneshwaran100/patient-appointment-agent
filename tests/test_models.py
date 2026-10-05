import pytest
from pydantic import ValidationError

from app.state import (
    BookingRequest,
    Patient,
    SafetyClassification,
    TimeSlot,
)


def test_patient_model_valid():
    patient = Patient(
        patient_id="P001",
        name="Sarah Connor",
        phone="555-0199",
        dob="1984-05-12",
    )
    assert patient.patient_id == "P001"
    assert patient.name == "Sarah Connor"


def test_patient_model_invalid_dob():
    with pytest.raises(ValidationError):
        Patient(
            patient_id="P001",
            name="Sarah Connor",
            phone="555-0199",
            dob="invalid-dob",
        )


def test_timeslot_model_valid():
    slot = TimeSlot(
        doctor_id="DOC001",
        doctor_name="Dr. Alice Smith",
        specialty="Cardiology",
        slot_datetime="2026-10-10 09:00",
        is_available=True,
    )
    assert slot.doctor_id == "DOC001"
    assert slot.is_available is True


def test_timeslot_model_invalid_datetime():
    with pytest.raises(ValidationError):
        TimeSlot(
            doctor_id="DOC001",
            doctor_name="Dr. Alice Smith",
            specialty="Cardiology",
            slot_datetime="2026/10/10 9am",
        )


def test_booking_request_defaults():
    req = BookingRequest(
        patient_id="P001",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 11:00",
    )
    assert req.reason == "Routine consultation"


def test_safety_classification_defaults():
    safety = SafetyClassification()
    assert safety.is_safe is True
    assert safety.is_medical_emergency is False
    assert safety.is_prompt_injection is False
