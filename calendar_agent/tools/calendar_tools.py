from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from psycopg2.extras import RealDictCursor

from langchain_core.tools import tool

from calendar_agent.sql_db import (
    get_db_connection,
    check_appointment_overlap,
    get_previous_appointment,
    get_next_appointment,
    get_all_appointments,
    get_appointment_by_id,
    insert_appointment,
    delete_appointment_by_id,
    update_appointment_fields
)
from calendar_agent.maps import validate_travel_time_from_previous, validate_travel_time_to_next
from calendar_agent.weaviate_db import find_related_appointments, store_appointment_in_vector_db
from calendar_agent.validation import validate_weekend, validate_business_hours, validate_service_area

logger = logging.getLogger(__name__)

@tool
def add_task(title: str, date: str, start_time: str, description: str, customer_name: str, address: str, estimated_end_time: Optional[str] = None) -> Dict[str, Any]:
    """Add an appointment to the calendar. All fields are required except estimated_end_time.
    
    Args:
        title: Appointment title (required)
        date: Date in YYYY-MM-DD format (required)
        start_time: Time in HH:MM format (required)
        description: Appointment notes/description (optional)
        customer_name: Customer full name (required)
        address: Full address (required)
        estimated_end_time: Estimated end time in HH:MM format (optional, defaults to 1 hour after start).
    
    Returns:
        Dictionary with:
        - success (bool): True if appointment created, False if error
        - appointment_id (int): ID of created appointment (only if success=True)
        - title (str): Title of the appointment (only if success=True)
        - date (str): Date of appointment (only if success=True)
        - time (str): Start time of appointment (only if success=True)
        - error (str): Error message (only if success=False)
        - related_appointments (list): List of related past appointments with fields: appointment_id, title, address, description (only if success=True and related found)
    """
    logger.info(f"🔧 TOOL CALLED: add_task(title='{title}', date='{date}', start_time='{start_time}', customer='{customer_name}', estimated_end='{estimated_end_time}')")
    try:
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        
        # Only calculate estimated_end_time if not provided
        if estimated_end_time:
            end_dt = datetime.strptime(f"{date} {estimated_end_time}", "%Y-%m-%d %H:%M")
        else:
            # Default to 1 hour duration
            end_dt = start_dt + timedelta(hours=1)
        
        # Validate business rules
        weekend_error = validate_weekend(start_dt)
        if weekend_error:
            return {"success": False, **weekend_error}
        
        hours_error = validate_business_hours(start_dt, end_dt)
        if hours_error:
            return {"success": False, **hours_error}
        
        area_error = validate_service_area(address)
        if area_error:
            return {"success": False, **area_error}
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check for related past appointments using vector search
        related_appointments = find_related_appointments(customer_name, title, description, address)
        
        # Check for overlapping appointments
        overlapping = check_appointment_overlap(cur, start_dt, end_dt)
        if overlapping:
            cur.close()
            conn.close()
            return {
                "success": False,
                "error": f"This appointment overlaps with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
            }
        
        # Check travel time from previous appointment
        previous_apt = get_previous_appointment(cur, start_dt)
        travel_error = validate_travel_time_from_previous(previous_apt, start_dt, address)
        if travel_error:
            cur.close()
            conn.close()
            return {"success": False, **travel_error}
        
        # Check travel time to next appointment
        next_apt = get_next_appointment(cur, end_dt)
        travel_error = validate_travel_time_to_next(next_apt, end_dt, address)
        if travel_error:
            cur.close()
            conn.close()
            return {"success": False, **travel_error}
        
        appointment_id = insert_appointment(cur, customer_name, address, title, description, start_dt, end_dt)
        conn.commit()
        cur.close()
        conn.close()
        
        # Store appointment in vector database for future similarity searches
        store_appointment_in_vector_db(appointment_id, customer_name, title, description, address, start_dt)
        
        result = {
            "success": True,
            "appointment_id": appointment_id,
            "title": title,
            "date": date,
            "time": start_time
        }
        
        if related_appointments:
            result["related_appointments"] = related_appointments
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in add_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to create appointment - {str(e)}"}

@tool
def list_tasks(date: Optional[str] = None) -> Dict[str, Any]:
    """List all appointments with complete details including ID, title, customer, time, address, and notes.
    
    Args:
        date: Optional date filter in YYYY-MM-DD format. If not provided, returns ALL appointments.
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointments (list): List of appointment dictionaries, each with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes
        - date_filter (str): The date filter used, if any
        - count (int): Number of appointments returned
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: list_tasks(date={date})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointments = get_all_appointments(cur, date)
        cur.close()
        conn.close()
        
        # Convert datetime objects to strings for JSON serialization
        appointments_list = []
        for apt in appointments:
            appointments_list.append({
                "id": apt['id'],
                "appointment_title": apt['appointment_title'],
                "customer_name": apt['customer_name'],
                "start_time": apt['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": apt['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": apt['address'],
                "notes": apt['notes']
            })
        
        result = {
            "success": True,
            "appointments": appointments_list,
            "count": len(appointments_list)
        }
        
        if date:
            result["date_filter"] = date
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in list_tasks: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to list appointments - {str(e)}"}

@tool
def delete_task(task_id: int) -> Dict[str, Any]:
    """Delete an appointment from the calendar by its ID.
    
    Args:
        task_id: The ID of the appointment to delete
    
    Returns:
        Dictionary with:
        - success (bool): True if deleted, False if error
        - task_id (int): ID of the deleted appointment (only if success=True)
        - title (str): Title of deleted appointment (only if success=True)
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: delete_task(task_id={task_id})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment_title = delete_appointment_by_id(cur, task_id)
        
        if not appointment_title:
            cur.close()
            conn.close()
            return {"success": False, "error": f"Appointment with ID {task_id} not found."}
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"success": True, "task_id": task_id, "title": appointment_title}
    except Exception as e:
        logger.error(f"❌ Error in delete_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to delete appointment - {str(e)}"}

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None,
                customer_name: Optional[str] = None, address: Optional[str] = None) -> Dict[str, Any]:
    """Update an appointment's details. Provide task_id and fields to update.
    
    Args:
        task_id: ID of the appointment to update
        title: New appointment title (optional)
        date: New date in YYYY-MM-DD format (optional)
        time: New start time in HH:MM format (optional)
        description: New description/notes (optional)
        customer_name: New customer name (optional)
        address: New address (optional)
    
    Returns:
        Dictionary with:
        - success (bool): True if updated, False if error
        - task_id (int): ID of the updated appointment (only if success=True)
        - title (str): Title of updated appointment (only if success=True)
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: update_task(task_id={task_id}, updates={{title={title}, date={date}, time={time}}})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current appointment
        appointment = get_appointment_by_id(cur, task_id)
        
        if not appointment:
            cur.close()
            conn.close()
            return {"success": False, "error": f"Appointment with ID {task_id} not found."}
        
        # Build update query dynamically
        updates = []
        params = []
        
        if title:
            updates.append("appointment_title = %s")
            params.append(title)
        if customer_name:
            updates.append("customer_name = %s")
            params.append(customer_name)
        if address:
            updates.append("address = %s")
            params.append(address)
        if description:
            updates.append("notes = %s")
            params.append(description)
        if date or time:
            current_date = appointment['start_time'].strftime('%Y-%m-%d') if not date else date
            current_time = appointment['start_time'].strftime('%H:%M') if not time else time
            new_start = datetime.strptime(f"{current_date} {current_time}", "%Y-%m-%d %H:%M")
            # Keep same duration
            duration = appointment['estimated_end_time'] - appointment['start_time']
            new_end = new_start + duration
            
            # Validate business rules
            weekend_error = validate_weekend(new_start)
            if weekend_error:
                cur.close()
                conn.close()
                return {"success": False, **weekend_error}
            
            hours_error = validate_business_hours(new_start, new_end)
            if hours_error:
                cur.close()
                conn.close()
                return {"success": False, **hours_error}
            
            # Check for overlapping appointments (excluding current appointment)
            overlapping = check_appointment_overlap(cur, new_start, new_end, exclude_id=task_id)
            if overlapping:
                cur.close()
                conn.close()
                return {
                    "success": False,
                    "error": f"This time change would overlap with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
                }
            
            # Check travel time constraints when time is changing
            appointment_address = address if address else appointment['address']
            
            # Validate service area if address is changing
            if address:
                area_error = validate_service_area(address)
                if area_error:
                    cur.close()
                    conn.close()
                    return {"success": False, **area_error}
            
            # Check travel time from previous appointment
            previous_apt = get_previous_appointment(cur, new_start, exclude_id=task_id)
            travel_error = validate_travel_time_from_previous(previous_apt, new_start, appointment_address)
            if travel_error:
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, new_end, exclude_id=task_id)
            travel_error = validate_travel_time_to_next(next_apt, new_end, appointment_address)
            if travel_error:
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            updates.append("start_time = %s")
            params.append(new_start)
            updates.append("estimated_end_time = %s")
            params.append(new_end)
        elif address:
            # If only address is changing (not time), check travel time constraints with current times
            # Validate service area
            area_error = validate_service_area(address)
            if area_error:
                cur.close()
                conn.close()
                return {"success": False, **area_error}
            
            # Check travel time from previous appointment
            previous_apt = get_previous_appointment(cur, appointment['start_time'], exclude_id=task_id)
            travel_error = validate_travel_time_from_previous(previous_apt, appointment['start_time'], address)
            if travel_error:
                # Adjust error message for address change context
                travel_error['error'] = travel_error['error'].replace('Not enough travel time', 'Changing address would require')
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, appointment['estimated_end_time'], exclude_id=task_id)
            travel_error = validate_travel_time_to_next(next_apt, appointment['estimated_end_time'], address)
            if travel_error:
                # Adjust error message for address change context
                travel_error['error'] = travel_error['error'].replace('Not enough travel time', 'Changing address would require')
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
        
        if not updates:
            cur.close()
            conn.close()
            return {"success": False, "error": "No updates provided."}
        
        result_title = update_appointment_fields(cur, task_id, updates, params)
        
        if not result_title:
            cur.close()
            conn.close()
            return {"success": False, "error": f"Failed to update appointment {task_id}"}
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"success": True, "task_id": task_id, "title": result_title}
    except Exception as e:
        logger.error(f"❌ Error in update_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to update appointment - {str(e)}"}
