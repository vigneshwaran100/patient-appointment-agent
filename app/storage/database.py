from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.state import (
    Appointment,
    AppointmentStatus,
    BookingRequest,
    BookingResult,
    CancellationRequest,
    CancellationResult,
    Doctor,
    Patient,
    RescheduleRequest,
    RescheduleResult,
    TimeSlot,
)


class ClinicDatabase:
    """Production-grade SQLite database manager for clinic appointments and state."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_settings().database_path
        self._ensure_directory()
        self.initialize_schema()
        if not self.list_doctors():
            self.seed_default_data()

    def _ensure_directory(self) -> None:
        """Create parent directory for database file if missing."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a contextual connection with WAL mode and foreign key enforcement."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create tables, indexes, and constraints if they do not exist."""
        with self.get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL UNIQUE,
                    dob TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS doctors (
                    doctor_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    room TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS doctor_availability (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
                    slot_datetime TEXT NOT NULL,
                    is_available INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(doctor_id, slot_datetime)
                );

                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id),
                    slot_datetime TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(doctor_id, slot_datetime, status)
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_id TEXT,
                    details TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_avail_doc_slot 
                    ON doctor_availability(doctor_id, slot_datetime, is_available);
                CREATE INDEX IF NOT EXISTS idx_appt_patient 
                    ON appointments(patient_id, status);
                CREATE INDEX IF NOT EXISTS idx_appt_doc_slot 
                    ON appointments(doctor_id, slot_datetime);
                """
            )

    def reset_database(self) -> None:
        """Drop all tables and reinitialize schema (used for testing and eval resets)."""
        with self.get_connection() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS audit_logs;
                DROP TABLE IF EXISTS appointments;
                DROP TABLE IF EXISTS doctor_availability;
                DROP TABLE IF EXISTS doctors;
                DROP TABLE IF EXISTS patients;
                """
            )
        self.initialize_schema()

    def log_audit(self, event_type: str, details: str, entity_id: str | None = None) -> None:
        """Record audit event into SQLite."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (timestamp, event_type, entity_id, details) VALUES (?, ?, ?, ?)",
                (now, event_type, entity_id, details),
            )

    # --- Patient Operations ---

    def generate_next_patient_id(self) -> str:
        """Generate next sequential patient ID (e.g., P006, P007)."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT patient_id FROM patients WHERE patient_id LIKE 'P%'").fetchall()
            max_num = 0
            for r in rows:
                pid = r["patient_id"]
                match = re.match(r"^P0*(\d+)$", pid)
                if match:
                    try:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
                    except ValueError:
                        pass
            if max_num > 0:
                return f"P{max_num + 1:03d}"
            return f"P{uuid.uuid4().hex[:6].upper()}"

    def register_patient(
        self, name: str, phone: str, dob: str, patient_id: str | None = None
    ) -> Patient:
        """Register a new patient into the clinic database."""
        pid = patient_id or self.generate_next_patient_id()
        now = datetime.now().isoformat()
        patient = Patient(patient_id=pid, name=name, phone=phone, dob=dob, created_at=now)
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO patients (patient_id, name, phone, dob, created_at) VALUES (?, ?, ?, ?, ?)",
                (patient.patient_id, patient.name, patient.phone, patient.dob, patient.created_at),
            )
        self.log_audit("PATIENT_REGISTERED", f"Registered patient {pid}: {name}", entity_id=pid)
        return patient

    def find_patient(
        self,
        patient_id: str | None = None,
        phone: str | None = None,
        name: str | None = None,
        dob: str | None = None,
    ) -> Patient | None:
        """Find patient by ID, Phone, or Name + DOB."""
        with self.get_connection() as conn:
            if patient_id:
                row = conn.execute(
                    "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
                ).fetchone()
            elif phone:
                # Normalize phone lookup by digits or raw format
                clean_phone = phone.strip()
                row = conn.execute(
                    "SELECT * FROM patients WHERE phone = ?", (clean_phone,)
                ).fetchone()
            elif name and dob:
                row = conn.execute(
                    "SELECT * FROM patients WHERE LOWER(name) = LOWER(?) AND dob = ?",
                    (name.strip(), dob.strip()),
                ).fetchone()
            elif name:
                row = conn.execute(
                    "SELECT * FROM patients WHERE LOWER(name) = LOWER(?)",
                    (name.strip(),),
                ).fetchone()
            else:
                return None

            if row:
                return Patient(
                    patient_id=row["patient_id"],
                    name=row["name"],
                    phone=row["phone"],
                    dob=row["dob"],
                    created_at=row["created_at"],
                )
        return None

    # --- Doctor & Availability Operations ---

    def add_doctor(
        self, doctor_id: str, name: str, specialty: str, room: str | None = None
    ) -> Doctor:
        """Add a physician into the clinic directory."""
        now = datetime.now().isoformat()
        doc = Doctor(doctor_id=doctor_id, name=name, specialty=specialty, room=room)
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO doctors (doctor_id, name, specialty, room, created_at) VALUES (?, ?, ?, ?, ?)",
                (doc.doctor_id, doc.name, doc.specialty, doc.room, now),
            )
        return doc

    def list_doctors(self, specialty: str | None = None) -> list[Doctor]:
        """List doctors, optionally filtered by medical specialty."""
        with self.get_connection() as conn:
            if specialty:
                rows = conn.execute(
                    "SELECT * FROM doctors WHERE LOWER(specialty) = LOWER(?) ORDER BY name ASC",
                    (specialty.strip(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM doctors ORDER BY specialty, name ASC").fetchall()

            return [
                Doctor(
                    doctor_id=r["doctor_id"],
                    name=r["name"],
                    specialty=r["specialty"],
                    room=r["room"],
                )
                for r in rows
            ]

    def add_availability_slot(self, doctor_id: str, slot_datetime: str) -> None:
        """Seed or declare an available slot for a doctor."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO doctor_availability (doctor_id, slot_datetime, is_available)
                VALUES (?, ?, 1)
                """,
                (doctor_id, slot_datetime),
            )

    def get_available_slots(
        self,
        doctor_id: str | None = None,
        specialty: str | None = None,
        date_prefix: str | None = None,
        limit: int = 10,
    ) -> list[TimeSlot]:
        """Query open slots joined with doctor info, ensuring no confirmed appointment conflicts."""
        query = """
            SELECT da.id, da.doctor_id, da.slot_datetime, da.is_available, d.name AS doctor_name, d.specialty
            FROM doctor_availability da
            JOIN doctors d ON da.doctor_id = d.doctor_id
            LEFT JOIN appointments a ON da.doctor_id = a.doctor_id 
                AND da.slot_datetime = a.slot_datetime 
                AND a.status = 'scheduled'
            WHERE da.is_available = 1
              AND a.appointment_id IS NULL
        """
        params: list[Any] = []

        if doctor_id:
            query += " AND da.doctor_id = ?"
            params.append(doctor_id)
        if specialty:
            query += " AND LOWER(d.specialty) = LOWER(?)"
            params.append(specialty.strip())
        if date_prefix:
            query += " AND da.slot_datetime LIKE ?"
            params.append(f"{date_prefix}%")

        query += " ORDER BY da.slot_datetime ASC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                TimeSlot(
                    slot_id=r["id"],
                    doctor_id=r["doctor_id"],
                    doctor_name=r["doctor_name"],
                    specialty=r["specialty"],
                    slot_datetime=r["slot_datetime"],
                    is_available=bool(r["is_available"]),
                )
                for r in rows
            ]

    # --- Appointment Operations ---

    def get_appointment(self, appointment_id: str) -> Appointment | None:
        """Fetch appointment by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT a.*, d.name as doctor_name
                FROM appointments a
                JOIN doctors d ON a.doctor_id = d.doctor_id
                WHERE a.appointment_id = ?
                """,
                (appointment_id,),
            ).fetchone()
            if row:
                return Appointment(
                    appointment_id=row["appointment_id"],
                    patient_id=row["patient_id"],
                    doctor_id=row["doctor_id"],
                    doctor_name=row["doctor_name"],
                    slot_datetime=row["slot_datetime"],
                    status=AppointmentStatus(row["status"]),
                    reason=row["reason"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None

    def get_patient_appointments(
        self, patient_id: str, status: AppointmentStatus | None = None
    ) -> list[Appointment]:
        """Retrieve appointments for a given patient."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT a.*, d.name as doctor_name
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.doctor_id
                    WHERE a.patient_id = ? AND a.status = ?
                    ORDER BY a.slot_datetime ASC
                    """,
                    (patient_id, status.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT a.*, d.name as doctor_name
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.doctor_id
                    WHERE a.patient_id = ?
                    ORDER BY a.slot_datetime ASC
                    """,
                    (patient_id,),
                ).fetchall()

            return [
                Appointment(
                    appointment_id=r["appointment_id"],
                    patient_id=r["patient_id"],
                    doctor_id=r["doctor_id"],
                    doctor_name=r["doctor_name"],
                    slot_datetime=r["slot_datetime"],
                    status=AppointmentStatus(r["status"]),
                    reason=r["reason"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def book_appointment(self, req: BookingRequest) -> BookingResult:
        """Atomically verify availability and book appointment."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            # 1. Verify patient exists
            p = conn.execute(
                "SELECT patient_id FROM patients WHERE patient_id = ?", (req.patient_id,)
            ).fetchone()
            if not p:
                return BookingResult(
                    success=False, error_message=f"Patient ID '{req.patient_id}' not found."
                )

            # 2. Check doctor exists
            doc = conn.execute(
                "SELECT name, specialty FROM doctors WHERE doctor_id = ?", (req.doctor_id,)
            ).fetchone()
            if not doc:
                return BookingResult(
                    success=False, error_message=f"Doctor ID '{req.doctor_id}' not found."
                )

            # 3. Check for patient duplicate booking conflict at that slot
            patient_conflict = conn.execute(
                "SELECT appointment_id FROM appointments WHERE patient_id = ? AND slot_datetime = ? AND status = 'scheduled'",
                (req.patient_id, req.slot_datetime),
            ).fetchone()
            if patient_conflict:
                return BookingResult(
                    success=False,
                    error_message=f"Patient already has an active appointment at {req.slot_datetime}.",
                )

            # 4. Check doctor availability slot
            avail_row = conn.execute(
                "SELECT id, is_available FROM doctor_availability WHERE doctor_id = ? AND slot_datetime = ?",
                (req.doctor_id, req.slot_datetime),
            ).fetchone()
            if not avail_row or avail_row["is_available"] == 0:
                alternatives = self.get_available_slots(doctor_id=req.doctor_id, limit=3)
                return BookingResult(
                    success=False,
                    error_message=f"Doctor {doc['name']} is not available at {req.slot_datetime}.",
                    alternative_slots=alternatives,
                )

            # 5. Check if doctor already has an active appointment at that time
            doc_conflict = conn.execute(
                "SELECT appointment_id FROM appointments WHERE doctor_id = ? AND slot_datetime = ? AND status = 'scheduled'",
                (req.doctor_id, req.slot_datetime),
            ).fetchone()
            if doc_conflict:
                alternatives = self.get_available_slots(doctor_id=req.doctor_id, limit=3)
                return BookingResult(
                    success=False,
                    error_message=f"Slot {req.slot_datetime} has just been booked by another patient.",
                    alternative_slots=alternatives,
                )

            # 6. Commit booking and mark slot unavailable
            appt_id = f"APT{uuid.uuid4().hex[:6].upper()}"
            conn.execute(
                """
                INSERT INTO appointments (appointment_id, patient_id, doctor_id, slot_datetime, status, reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'scheduled', ?, ?, ?)
                """,
                (appt_id, req.patient_id, req.doctor_id, req.slot_datetime, req.reason, now, now),
            )
            conn.execute(
                "UPDATE doctor_availability SET is_available = 0 WHERE doctor_id = ? AND slot_datetime = ?",
                (req.doctor_id, req.slot_datetime),
            )

        self.log_audit(
            "APPOINTMENT_BOOKED",
            f"Booked {appt_id} for patient {req.patient_id}",
            entity_id=appt_id,
        )

        appt = Appointment(
            appointment_id=appt_id,
            patient_id=req.patient_id,
            doctor_id=req.doctor_id,
            doctor_name=doc["name"],
            slot_datetime=req.slot_datetime,
            status=AppointmentStatus.SCHEDULED,
            reason=req.reason,
            created_at=now,
            updated_at=now,
        )
        return BookingResult(success=True, appointment=appt)

    def cancel_appointment(self, req: CancellationRequest) -> CancellationResult:
        """Cancel an appointment and free the doctor slot."""
        now = datetime.now().isoformat()
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?",
                (req.appointment_id,),
            ).fetchone()
            if not row:
                return CancellationResult(
                    success=False, error_message=f"Appointment '{req.appointment_id}' not found."
                )

            if row["status"] == "cancelled":
                return CancellationResult(
                    success=False, error_message="Appointment is already cancelled."
                )

            if req.patient_id and row["patient_id"] != req.patient_id:
                return CancellationResult(
                    success=False,
                    error_message="Unauthorized: Patient ID does not match appointment record.",
                )

            # Update status
            conn.execute(
                "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE appointment_id = ?",
                (now, req.appointment_id),
            )
            # Free up slot in doctor availability
            conn.execute(
                "UPDATE doctor_availability SET is_available = 1 WHERE doctor_id = ? AND slot_datetime = ?",
                (row["doctor_id"], row["slot_datetime"]),
            )

        self.log_audit(
            "APPOINTMENT_CANCELLED", f"Cancelled {req.appointment_id}", entity_id=req.appointment_id
        )
        return CancellationResult(success=True, appointment_id=req.appointment_id)

    def reschedule_appointment(self, req: RescheduleRequest) -> RescheduleResult:
        """Atomically cancel old appointment and book new slot."""
        with self.get_connection() as conn:
            old_row = conn.execute(
                "SELECT * FROM appointments WHERE appointment_id = ?",
                (req.appointment_id,),
            ).fetchone()
            if not old_row:
                return RescheduleResult(
                    success=False, error_message=f"Appointment '{req.appointment_id}' not found."
                )
            if old_row["status"] != "scheduled":
                return RescheduleResult(
                    success=False,
                    error_message=f"Cannot reschedule an appointment with status '{old_row['status']}'.",
                )

            target_doctor_id = req.new_doctor_id or old_row["doctor_id"]

            # Check new slot availability
            avail_row = conn.execute(
                "SELECT id, is_available FROM doctor_availability WHERE doctor_id = ? AND slot_datetime = ?",
                (target_doctor_id, req.new_slot_datetime),
            ).fetchone()
            if not avail_row or avail_row["is_available"] == 0:
                alternatives = self.get_available_slots(doctor_id=target_doctor_id, limit=3)
                return RescheduleResult(
                    success=False,
                    error_message=f"Requested slot {req.new_slot_datetime} is not available.",
                    alternative_slots=alternatives,
                )

            # Cancel old
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE appointments SET status = 'rescheduled', updated_at = ? WHERE appointment_id = ?",
                (now, req.appointment_id),
            )
            conn.execute(
                "UPDATE doctor_availability SET is_available = 1 WHERE doctor_id = ? AND slot_datetime = ?",
                (old_row["doctor_id"], old_row["slot_datetime"]),
            )

            # Create new
            new_appt_id = f"APT{uuid.uuid4().hex[:6].upper()}"
            reason = req.reason or old_row["reason"]
            conn.execute(
                """
                INSERT INTO appointments (appointment_id, patient_id, doctor_id, slot_datetime, status, reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'scheduled', ?, ?, ?)
                """,
                (
                    new_appt_id,
                    old_row["patient_id"],
                    target_doctor_id,
                    req.new_slot_datetime,
                    reason,
                    now,
                    now,
                ),
            )
            conn.execute(
                "UPDATE doctor_availability SET is_available = 0 WHERE doctor_id = ? AND slot_datetime = ?",
                (target_doctor_id, req.new_slot_datetime),
            )

            doc_row = conn.execute(
                "SELECT name FROM doctors WHERE doctor_id = ?", (target_doctor_id,)
            ).fetchone()

        self.log_audit(
            "APPOINTMENT_RESCHEDULED",
            f"Rescheduled {req.appointment_id} to new appointment {new_appt_id}",
            entity_id=new_appt_id,
        )

        new_appt = Appointment(
            appointment_id=new_appt_id,
            patient_id=old_row["patient_id"],
            doctor_id=target_doctor_id,
            doctor_name=doc_row["name"] if doc_row else None,
            slot_datetime=req.new_slot_datetime,
            status=AppointmentStatus.SCHEDULED,
            reason=reason,
            created_at=now,
            updated_at=now,
        )
        return RescheduleResult(
            success=True,
            old_appointment_id=req.appointment_id,
            new_appointment=new_appt,
        )

    # --- Seeding for Evaluation and Tests ---

    def seed_default_data(self) -> None:
        """Seed known patients, doctors, and slots for reproducible tests and evaluation scenarios."""
        # 1. Add doctors
        doctors = [
            ("DOC001", "Dr. Alice Smith", "Cardiology", "Room 301"),
            ("DOC002", "Dr. Robert Chen", "Dermatology", "Room 205"),
            ("DOC003", "Dr. Maria Garcia", "General Medicine", "Room 102"),
            ("DOC004", "Dr. James Wilson", "Pediatrics", "Room 110"),
            ("DOC005", "Dr. Linda Taylor", "Orthopedics", "Room 404"),
        ]
        for doc_id, name, specialty, room in doctors:
            self.add_doctor(doc_id, name, specialty, room)

        # 2. Add patients
        patients = [
            ("Sarah Connor", "555-0199", "1984-05-12", "P001"),
            ("John Doe", "555-0144", "1990-11-23", "P002"),
            ("Emily Blunt", "555-0182", "1983-02-23", "P003"),
            ("Michael Scott", "555-0177", "1965-03-15", "P004"),
        ]
        for name, phone, dob, pid in patients:
            if not self.find_patient(patient_id=pid):
                self.register_patient(name=name, phone=phone, dob=dob, patient_id=pid)

        # 3. Add slots
        slots = [
            # Dr. Alice Smith (Cardiology)
            ("DOC001", "2026-10-10 09:00"),
            ("DOC001", "2026-10-10 11:00"),
            ("DOC001", "2026-10-12 14:00"),
            ("DOC001", "2026-10-12 16:00"),
            # Dr. Robert Chen (Dermatology)
            ("DOC002", "2026-10-11 10:00"),
            ("DOC002", "2026-10-11 15:30"),
            ("DOC002", "2026-10-13 09:30"),
            # Dr. Maria Garcia (General Medicine)
            ("DOC003", "2026-10-08 08:30"),
            ("DOC003", "2026-10-08 13:00"),
            ("DOC003", "2026-10-09 10:00"),
            ("DOC003", "2026-10-09 14:00"),
            # Dr. James Wilson (Pediatrics)
            ("DOC004", "2026-10-14 09:00"),
            ("DOC004", "2026-10-14 11:30"),
            # Dr. Linda Taylor (Orthopedics)
            ("DOC005", "2026-10-15 10:00"),
            ("DOC005", "2026-10-15 13:30"),
        ]
        for doc_id, slot_dt in slots:
            self.add_availability_slot(doc_id, slot_dt)

        # 4. Add one existing appointment for Sarah Connor (P001) with Dr. Alice Smith
        # to test cancellation / reschedule / duplicate detection scenarios
        existing_appt = self.get_appointment("APT1001")
        if not existing_appt:
            with self.get_connection() as conn:
                now = datetime.now().isoformat()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO appointments (appointment_id, patient_id, doctor_id, slot_datetime, status, reason, created_at, updated_at)
                    VALUES ('APT1001', 'P001', 'DOC001', '2026-10-10 09:00', 'scheduled', 'Annual cardiovascular checkup', ?, ?)
                    """,
                    (now, now),
                )
                conn.execute(
                    "UPDATE doctor_availability SET is_available = 0 WHERE doctor_id = 'DOC001' AND slot_datetime = '2026-10-10 09:00'"
                )


_db_singleton: ClinicDatabase | None = None


def get_db(db_path: str | None = None) -> ClinicDatabase:
    """Return database singleton instance."""
    global _db_singleton
    if _db_singleton is None or (db_path and _db_singleton.db_path != db_path):
        _db_singleton = ClinicDatabase(db_path)
    return _db_singleton


def reset_db_singleton() -> None:
    """Reset the database singleton instance (used between tests)."""
    global _db_singleton
    _db_singleton = None
