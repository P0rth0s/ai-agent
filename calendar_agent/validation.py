from datetime import datetime
from typing import Optional, Dict, Any
from calendar_agent.maps import check_travel_time

def validate_weekend(appointment_date: datetime) -> Optional[Dict[str, str]]:
    """Validate that appointment is not on a weekend.
    
    Args:
        appointment_date: Datetime of the appointment
    
    Returns:
        Error dictionary if invalid, None if valid
    """
    if appointment_date.weekday() in [5, 6]:  # Saturday=5, Sunday=6
        day_name = appointment_date.strftime('%A')
        return {
            "error": f"Cannot schedule appointments on weekends. {day_name} is not available. Please choose a weekday (Monday-Friday)."
        }
    return None

def validate_business_hours(start_time: datetime, end_time: datetime) -> Optional[Dict[str, str]]:
    """Validate that appointment is within business hours (08:00-18:00).
    
    Args:
        start_time: Start datetime of the appointment
        end_time: End datetime of the appointment
    
    Returns:
        Error dictionary if invalid, None if valid
    """
    start_hour = start_time.hour + start_time.minute / 60
    end_hour = end_time.hour + end_time.minute / 60
    
    if start_hour < 8 or end_hour > 18:
        return {
            "error": f"Appointments must be scheduled between 08:00 and 18:00. Requested time: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} is outside business hours."
        }
    return None

def validate_service_area(address: str, base_location: str = "Bozeman, MT 59715", max_minutes: int = 120) -> Optional[Dict[str, str]]:
    """Validate that appointment address is within service area.
    
    Args:
        address: Appointment address to validate
        base_location: Base location to measure from
        max_minutes: Maximum travel time in minutes
    
    Returns:
        Error dictionary if invalid, None if valid
    """
    travel_minutes = check_travel_time(base_location, address)
    if travel_minutes is not None and travel_minutes > max_minutes:
        return {
            "error": f"Appointment location is {travel_minutes} minutes from Bozeman, MT. Service area is limited to locations within {max_minutes} minutes of Bozeman."
        }
    return None
