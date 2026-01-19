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
