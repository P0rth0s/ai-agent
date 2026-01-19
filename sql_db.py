from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
import psycopg2
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "ai_agent_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )

def check_appointment_overlap(cur, start_dt: datetime, end_dt: datetime, exclude_id: Optional[int] = None) -> Optional[dict]:
    """Check if an appointment overlaps with existing appointments.
    
    Args:
        cur: Database cursor
        start_dt: Start datetime of appointment
        end_dt: End datetime of appointment
        exclude_id: Optional appointment ID to exclude from overlap check (for updates)
    
    Returns:
        Dictionary with overlapping appointment details if overlap exists, None otherwise
    """
    if exclude_id:
        cur.execute("""
            SELECT id, appointment_title, start_time, estimated_end_time
            FROM appointments
            WHERE (start_time, estimated_end_time) OVERLAPS (%s::timestamp, %s::timestamp)
            AND id != %s
        """, (start_dt, end_dt, exclude_id))
    else:
        cur.execute("""
            SELECT id, appointment_title, start_time, estimated_end_time
            FROM appointments
            WHERE (start_time, estimated_end_time) OVERLAPS (%s::timestamp, %s::timestamp)
        """, (start_dt, end_dt))
    
    return cur.fetchone()

def get_previous_appointment(cur, reference_time: datetime, exclude_id: Optional[int] = None) -> Optional[dict]:
    """Get the appointment scheduled immediately before a given time.
    
    Args:
        cur: Database cursor
        reference_time: Reference datetime
        exclude_id: Optional appointment ID to exclude
    
    Returns:
        Dictionary with appointment details if found, None otherwise
    """
    if exclude_id:
        cur.execute("""
            SELECT id, appointment_title, address, estimated_end_time
            FROM appointments
            WHERE estimated_end_time <= %s AND id != %s
            ORDER BY estimated_end_time DESC
            LIMIT 1
        """, (reference_time, exclude_id))
    else:
        cur.execute("""
            SELECT id, appointment_title, address, estimated_end_time
            FROM appointments
            WHERE estimated_end_time <= %s
            ORDER BY estimated_end_time DESC
            LIMIT 1
        """, (reference_time,))
    return cur.fetchone()

def get_next_appointment(cur, reference_time: datetime, exclude_id: Optional[int] = None) -> Optional[dict]:
    """Get the appointment scheduled immediately after a given time.
    
    Args:
        cur: Database cursor
        reference_time: Reference datetime
        exclude_id: Optional appointment ID to exclude
    
    Returns:
        Dictionary with appointment details if found, None otherwise
    """
    if exclude_id:
        cur.execute("""
            SELECT id, appointment_title, address, start_time
            FROM appointments
            WHERE start_time >= %s AND id != %s
            ORDER BY start_time ASC
            LIMIT 1
        """, (reference_time, exclude_id))
    else:
        cur.execute("""
            SELECT id, appointment_title, address, start_time
            FROM appointments
            WHERE start_time >= %s
            ORDER BY start_time ASC
            LIMIT 1
        """, (reference_time,))
    return cur.fetchone()

def get_previous_appointment_full(cur, reference_time: datetime) -> Optional[dict]:
    """Get full details of the appointment scheduled immediately before a given time.
    
    Args:
        cur: Database cursor
        reference_time: Reference datetime
    
    Returns:
        Dictionary with full appointment details if found, None otherwise
    """
    cur.execute("""
        SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
        FROM appointments
        WHERE estimated_end_time <= %s
        ORDER BY estimated_end_time DESC
        LIMIT 1
    """, (reference_time,))
    return cur.fetchone()

def get_next_appointment_full(cur, reference_time: datetime) -> Optional[dict]:
    """Get full details of the appointment scheduled immediately after a given time.
    
    Args:
        cur: Database cursor
        reference_time: Reference datetime
    
    Returns:
        Dictionary with full appointment details if found, None otherwise
    """
    cur.execute("""
        SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
        FROM appointments
        WHERE start_time >= %s
        ORDER BY start_time ASC
        LIMIT 1
    """, (reference_time,))
    return cur.fetchone()

def get_all_appointments(cur, date_filter: Optional[str] = None) -> list:
    """Get all appointments, optionally filtered by date.
    
    Args:
        cur: Database cursor
        date_filter: Optional date string in YYYY-MM-DD format
    
    Returns:
        List of appointment dictionaries
    """
    if date_filter:
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
            FROM appointments
            WHERE DATE(start_time) = %s
            ORDER BY start_time
        """, (date_filter,))
    else:
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
            FROM appointments
            ORDER BY start_time
        """)
    return cur.fetchall()

def get_appointment_by_id(cur, appointment_id: int) -> Optional[dict]:
    """Get an appointment by its ID.
    
    Args:
        cur: Database cursor
        appointment_id: Appointment ID
    
    Returns:
        Dictionary with appointment details if found, None otherwise
    """
    cur.execute("SELECT * FROM appointments WHERE id = %s", (appointment_id,))
    return cur.fetchone()

def insert_appointment(cur, customer_name: str, address: str, title: str, 
                      description: str, start_time: datetime, end_time: datetime) -> int:
    """Insert a new appointment into the database.
    
    Args:
        cur: Database cursor
        customer_name: Customer full name
        address: Full address
        title: Appointment title
        description: Appointment notes/description
        start_time: Start datetime
        end_time: End datetime
    
    Returns:
        ID of the newly created appointment
    """
    cur.execute("""
        INSERT INTO appointments (customer_name, address, appointment_title, notes, start_time, estimated_end_time)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (customer_name, address, title, description, start_time, end_time))
    return cur.fetchone()['id']

def delete_appointment_by_id(cur, appointment_id: int) -> Optional[str]:
    """Delete an appointment by its ID.
    
    Args:
        cur: Database cursor
        appointment_id: Appointment ID
    
    Returns:
        Appointment title if found and deleted, None otherwise
    """
    cur.execute("SELECT appointment_title FROM appointments WHERE id = %s", (appointment_id,))
    appointment = cur.fetchone()
    
    if not appointment:
        return None
    
    cur.execute("DELETE FROM appointments WHERE id = %s", (appointment_id,))
    return appointment['appointment_title']

def update_appointment_fields(cur, appointment_id: int, updates: list, params: list) -> Optional[str]:
    """Update appointment fields.
    
    Args:
        cur: Database cursor
        appointment_id: Appointment ID
        updates: List of SQL SET clauses (e.g., ["appointment_title = %s", "address = %s"])
        params: List of parameter values corresponding to updates
    
    Returns:
        Updated appointment title if successful, None otherwise
    """
    if not updates:
        return None
    
    params.append(appointment_id)
    query = f"UPDATE appointments SET {', '.join(updates)} WHERE id = %s RETURNING appointment_title"
    cur.execute(query, params)
    result = cur.fetchone()
    return result['appointment_title'] if result else None
