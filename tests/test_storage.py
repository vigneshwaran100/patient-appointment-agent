import os
import tempfile

import pytest

from app.state import (
    AppointmentStatus,
    BookingRequest,
    CancellationRequest,
    RescheduleRequest,
)
from app.storage.database import ClinicDatabase


@pytest.fixture
def test_db():
    """Create a temporary SQLite database seeded with default test fixtures."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    db = ClinicDatabase(db_path=temp_path)
    db.seed_default_data()
    yield db

    # Cleanup temporary database
    try:
        os.remove(temp_path)
    except OSError:
        pass


def test_seed_data_loaded(test_db):
    doctors = test_db.list_doctors()
    assert len(doctors) >= 5

    patient = test_db.find_patient(patient_id="P001")
    assert patient is not None
    assert patient.name == "Sarah Connor"


def test_find_patient_variations(test_db):
    # Lookup by phone
    p1 = test_db.find_patient(phone="555-0144")
    assert p1 is not None
    assert p1.patient_id == "P002"

    # Lookup by name
    p2 = test_db.find_patient(name="Emily Blunt")
    assert p2 is not None
    assert p2.patient_id == "P003"


def test_get_available_slots(test_db):
    # DOC001 has 2026-10-10 09:00 booked in seed data
    slots = test_db.get_available_slots(doctor_id="DOC001")
    slot_times = [s.slot_datetime for s in slots]
    assert "2026-10-10 09:00" not in slot_times
    assert "2026-10-10 11:00" in slot_times


def test_successful_booking(test_db):
    req = BookingRequest(
        patient_id="P002",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 11:00",
        reason="Consultation",
    )
    res = test_db.book_appointment(req)
    assert res.success is True
    assert res.appointment is not None
    assert res.appointment.status == AppointmentStatus.SCHEDULED

    # Verify slot is no longer returned in available slots
    slots = test_db.get_available_slots(doctor_id="DOC001")
    slot_times = [s.slot_datetime for s in slots]
    assert "2026-10-10 11:00" not in slot_times


def test_booking_conflict_prevention(test_db):
    # Try booking already booked slot
    req = BookingRequest(
        patient_id="P002",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 09:00",  # already booked by P001 in seed data
    )
    res = test_db.book_appointment(req)
    assert res.success is False
    assert "not available" in res.error_message
    assert len(res.alternative_slots) > 0


def test_duplicate_patient_booking_same_time(test_db):
    # P001 is already booked at 2026-10-10 09:00 with DOC001
    # Try booking P001 at the same time with another doctor if slot exists
    test_db.add_availability_slot("DOC003", "2026-10-10 09:00")
    req = BookingRequest(
        patient_id="P001",
        doctor_id="DOC003",
        slot_datetime="2026-10-10 09:00",
    )
    res = test_db.book_appointment(req)
    assert res.success is False
    assert "already has an active appointment" in res.error_message


def test_cancellation_and_slot_release(test_db):
    # APT1001 is P001's appointment with DOC001 at 2026-10-10 09:00
    res = test_db.cancel_appointment(
        CancellationRequest(appointment_id="APT1001", patient_id="P001")
    )
    assert res.success is True

    # Slot should now be freed and available
    slots = test_db.get_available_slots(doctor_id="DOC001")
    slot_times = [s.slot_datetime for s in slots]
    assert "2026-10-10 09:00" in slot_times


def test_reschedule_appointment(test_db):
    # Reschedule APT1001 from 2026-10-10 09:00 to 2026-10-10 11:00
    req = RescheduleRequest(
        appointment_id="APT1001",
        new_slot_datetime="2026-10-10 11:00",
    )
    res = test_db.reschedule_appointment(req)
    assert res.success is True
    assert res.new_appointment is not None

    # Check old slot is freed and new slot is taken
    slots = test_db.get_available_slots(doctor_id="DOC001")
    slot_times = [s.slot_datetime for s in slots]
    assert "2026-10-10 09:00" in slot_times
    assert "2026-10-10 11:00" not in slot_times
