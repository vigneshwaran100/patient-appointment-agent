import os
import tempfile

import pytest

from app.storage.database import ClinicDatabase
from app.tools.availability import check_availability, list_doctors_tool
from app.tools.booking import book_appointment_tool
from app.tools.cancellation import cancel_appointment_tool, reschedule_appointment_tool
from app.tools.patient import lookup_patient, register_patient_tool


@pytest.fixture
def isolated_db():
    """Create a temporary seeded database for deterministic tool testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    db = ClinicDatabase(db_path=temp_path)
    db.seed_default_data()
    yield db

    try:
        os.remove(temp_path)
    except OSError:
        pass


# ==========================================
# 1. Patient Found / Not Found Tests
# ==========================================


def test_patient_lookup_found_by_id(isolated_db):
    res = lookup_patient(patient_id="P001", db=isolated_db)
    assert res["success"] is True
    assert res["patient"]["patient_id"] == "P001"
    assert res["patient"]["name"] == "Sarah Connor"
    assert res["error_code"] is None


def test_patient_lookup_found_by_phone(isolated_db):
    res = lookup_patient(phone="555-0144", db=isolated_db)
    assert res["success"] is True
    assert res["patient"]["name"] == "John Doe"


def test_patient_lookup_found_by_name_and_dob(isolated_db):
    res = lookup_patient(name="Emily Blunt", dob="1983-02-23", db=isolated_db)
    assert res["success"] is True
    assert res["patient"]["patient_id"] == "P003"


def test_patient_lookup_not_found(isolated_db):
    res = lookup_patient(patient_id="P999", db=isolated_db)
    assert res["success"] is False
    assert res["patient"] is None
    assert res["error_code"] == "PATIENT_NOT_FOUND"


def test_patient_lookup_invalid_inputs(isolated_db):
    # Empty inputs
    res1 = lookup_patient(db=isolated_db)
    assert res1["success"] is False
    assert res1["error_code"] == "INVALID_INPUT"

    # Malformed DOB
    res2 = lookup_patient(name="Sarah Connor", dob="1984/05/12", db=isolated_db)
    assert res2["success"] is False
    assert res2["error_code"] == "INVALID_INPUT"


# ==========================================
# 2. Available / No-Availability Tests
# ==========================================


def test_availability_found(isolated_db):
    res = check_availability(specialty="Cardiology", db=isolated_db)
    assert res["success"] is True
    assert res["total_slots"] > 0
    assert all(s["specialty"] == "Cardiology" for s in res["slots"])


def test_availability_no_slots_found(isolated_db):
    # Search date far in the future with no slots declared
    res = check_availability(date="2099-01-01", db=isolated_db)
    assert res["success"] is True
    assert res["total_slots"] == 0
    assert len(res["slots"]) == 0
    assert "No available slots" in res["error_message"]


def test_availability_invalid_date_format(isolated_db):
    res = check_availability(date="tomorrow", db=isolated_db)
    assert res["success"] is False
    assert res["error_code"] == "INVALID_INPUT"


def test_list_doctors_by_specialty(isolated_db):
    res = list_doctors_tool(specialty="Dermatology", db=isolated_db)
    assert res["success"] is True
    assert res["total_doctors"] >= 1
    assert res["doctors"][0]["name"] == "Dr. Robert Chen"


# ==========================================
# 3. Successful Booking Tests
# ==========================================


def test_successful_booking(isolated_db):
    # 2026-10-10 11:00 is an open slot for Dr. Alice Smith (DOC001)
    res = book_appointment_tool(
        patient_id="P002",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 11:00",
        reason="Blood pressure follow up",
        db=isolated_db,
    )
    assert res["success"] is True
    assert res["appointment"] is not None
    assert res["appointment"]["doctor_id"] == "DOC001"
    assert res["appointment"]["slot_datetime"] == "2026-10-10 11:00"
    assert res["appointment"]["status"] == "scheduled"

    # Verify slot is no longer reported in availability
    avail = check_availability(doctor_id="DOC001", date="2026-10-10", db=isolated_db)
    slot_times = [s["slot_datetime"] for s in avail["slots"]]
    assert "2026-10-10 11:00" not in slot_times


# ==========================================
# 4. Double Booking Prevention Tests
# ==========================================


def test_double_booking_prevention(isolated_db):
    # DOC001 has slot 2026-10-10 09:00 already booked by P001 in seed data
    res = book_appointment_tool(
        patient_id="P002",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 09:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["appointment"] is None
    assert res["error_code"] in ["SLOT_UNAVAILABLE", "DOUBLE_BOOKING"]
    assert len(res["alternative_slots"]) > 0


# ==========================================
# 5. Duplicate Booking Prevention Tests
# ==========================================


def test_duplicate_booking_prevention(isolated_db):
    # P001 already has appointment at 2026-10-10 09:00
    # Try booking P001 at the same datetime with another doctor
    isolated_db.add_availability_slot("DOC003", "2026-10-10 09:00")
    res = book_appointment_tool(
        patient_id="P001",
        doctor_id="DOC003",
        slot_datetime="2026-10-10 09:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "DUPLICATE_BOOKING"
    assert "already has an active appointment" in res["error_message"]


# ==========================================
# 6. Invalid Booking Tests
# ==========================================


def test_invalid_booking_missing_fields(isolated_db):
    res = book_appointment_tool(
        patient_id="",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 11:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_INPUT"


def test_invalid_booking_bad_datetime_format(isolated_db):
    res = book_appointment_tool(
        patient_id="P001",
        doctor_id="DOC001",
        slot_datetime="2026-10-10T11:00:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_INPUT"


def test_invalid_booking_unknown_patient(isolated_db):
    res = book_appointment_tool(
        patient_id="UNKNOWN_PATIENT",
        doctor_id="DOC001",
        slot_datetime="2026-10-10 11:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "PATIENT_NOT_FOUND"


def test_invalid_booking_unknown_doctor(isolated_db):
    res = book_appointment_tool(
        patient_id="P001",
        doctor_id="DOC_NONEXISTENT",
        slot_datetime="2026-10-10 11:00",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "DOCTOR_NOT_FOUND"


# ==========================================
# 7. Successful Cancellation Tests
# ==========================================


def test_successful_cancellation(isolated_db):
    # APT1001 is Sarah Connor's (P001) appointment with DOC001
    res = cancel_appointment_tool(
        appointment_id="APT1001",
        patient_id="P001",
        reason="Schedule conflict",
        db=isolated_db,
    )
    assert res["success"] is True
    assert res["appointment_id"] == "APT1001"
    assert res["error_code"] is None


# ==========================================
# 8. Invalid / Unknown Cancellation Tests
# ==========================================


def test_cancel_unknown_appointment(isolated_db):
    res = cancel_appointment_tool(appointment_id="APT_DOES_NOT_EXIST", db=isolated_db)
    assert res["success"] is False
    assert res["error_code"] == "APPOINTMENT_NOT_FOUND"


def test_cancel_already_cancelled_appointment(isolated_db):
    # First cancellation succeeds
    res1 = cancel_appointment_tool(appointment_id="APT1001", db=isolated_db)
    assert res1["success"] is True

    # Second cancellation fails with ALREADY_CANCELLED
    res2 = cancel_appointment_tool(appointment_id="APT1001", db=isolated_db)
    assert res2["success"] is False
    assert res2["error_code"] == "ALREADY_CANCELLED"


def test_cancel_unauthorized_patient_mismatch(isolated_db):
    # APT1001 belongs to P001, try cancelling as P002
    res = cancel_appointment_tool(
        appointment_id="APT1001",
        patient_id="P002",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "UNAUTHORIZED_CANCELLATION"


# ==========================================
# 9. Slot Release After Cancellation & Reschedule Tests
# ==========================================


def test_slot_released_after_cancellation(isolated_db):
    # Before cancellation: 2026-10-10 09:00 is unavailable
    avail_before = check_availability(doctor_id="DOC001", date="2026-10-10", db=isolated_db)
    slots_before = [s["slot_datetime"] for s in avail_before["slots"]]
    assert "2026-10-10 09:00" not in slots_before

    # Cancel appointment
    cancel_res = cancel_appointment_tool(appointment_id="APT1001", db=isolated_db)
    assert cancel_res["success"] is True

    # After cancellation: 2026-10-10 09:00 is now released and available
    avail_after = check_availability(doctor_id="DOC001", date="2026-10-10", db=isolated_db)
    slots_after = [s["slot_datetime"] for s in avail_after["slots"]]
    assert "2026-10-10 09:00" in slots_after


def test_reschedule_tool_releases_and_claims_slots(isolated_db):
    # Reschedule APT1001 from 2026-10-10 09:00 to 2026-10-10 11:00
    res = reschedule_appointment_tool(
        appointment_id="APT1001",
        new_slot_datetime="2026-10-10 11:00",
        db=isolated_db,
    )
    assert res["success"] is True
    assert res["appointment"]["slot_datetime"] == "2026-10-10 11:00"

    # Old slot should be available, new slot should now be unavailable
    avail = check_availability(doctor_id="DOC001", date="2026-10-10", db=isolated_db)
    slot_times = [s["slot_datetime"] for s in avail["slots"]]
    assert "2026-10-10 09:00" in slot_times
    assert "2026-10-10 11:00" not in slot_times


# ==========================================
# 5. Patient Registration Tool Tests
# ==========================================


def test_patient_registration_success(isolated_db):
    res = register_patient_tool(
        name="Bruce Wayne",
        phone="555-8888",
        dob="1980-04-17",
        db=isolated_db,
    )
    assert res["success"] is True
    assert res["patient"]["name"] == "Bruce Wayne"
    assert res["patient"]["patient_id"].startswith("P")

    # Verify patient is immediately queryable in SQLite
    found = isolated_db.find_patient(patient_id=res["patient"]["patient_id"])
    assert found is not None
    assert found.name == "Bruce Wayne"


def test_patient_registration_duplicate_phone(isolated_db):
    # P001 has phone 555-0199 in seed data
    res = register_patient_tool(
        name="Another Person",
        phone="555-0199",
        dob="1990-01-01",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "DUPLICATE_PATIENT"


def test_patient_registration_duplicate_name_and_dob(isolated_db):
    # P001 is Sarah Connor, 1984-05-12
    res = register_patient_tool(
        name="Sarah Connor",
        phone="555-9999",
        dob="1984-05-12",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "DUPLICATE_PATIENT"


def test_patient_registration_invalid_dob(isolated_db):
    res = register_patient_tool(
        name="Peter Parker",
        phone="555-7777",
        dob="invalid-dob",
        db=isolated_db,
    )
    assert res["success"] is False
    assert res["error_code"] == "INVALID_DOB"

