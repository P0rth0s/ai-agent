from datetime import datetime
from typing import Dict, Any
import logging
from psycopg2.extras import RealDictCursor

from langchain_core.tools import tool

from calendar_agent.sql_db import get_db_connection, get_previous_appointment_full, get_next_appointment_full

logger = logging.getLogger(__name__)

@tool
def get_previous_task(reference_time: str) -> Dict[str, Any]:
    """Get the appointment scheduled immediately before a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointment (dict): Appointment data with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes (only if found)
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: get_previous_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment = get_previous_appointment_full(cur, ref_dt)
        cur.close()
        conn.close()
        
        if not appointment:
            return {"success": True, "appointment": None}
        
        return {
            "success": True,
            "appointment": {
                "id": appointment['id'],
                "appointment_title": appointment['appointment_title'],
                "customer_name": appointment['customer_name'],
                "start_time": appointment['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": appointment['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": appointment['address'],
                "notes": appointment['notes']
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in get_previous_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to get previous task - {str(e)}"}

@tool
def get_next_task(reference_time: str) -> Dict[str, Any]:
    """Get the appointment scheduled immediately after a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointment (dict): Appointment data with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes (only if found)
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: get_next_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment = get_next_appointment_full(cur, ref_dt)
        cur.close()
        conn.close()
        
        if not appointment:
            return {"success": True, "appointment": None}
        
        return {
            "success": True,
            "appointment": {
                "id": appointment['id'],
                "appointment_title": appointment['appointment_title'],
                "customer_name": appointment['customer_name'],
                "start_time": appointment['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": appointment['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": appointment['address'],
                "notes": appointment['notes']
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in get_next_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to get next task - {str(e)}"}
