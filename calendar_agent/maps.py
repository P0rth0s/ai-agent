from dotenv import load_dotenv
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
import logging
import googlemaps

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_travel_time(origin: str, destination: str) -> Optional[int]:
    """Check driving time between two addresses.
    
    Args:
        origin: Starting address
        destination: Ending address
    
    Returns:
        Travel time in minutes, or None if calculation fails
    """
    try:
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
        result = gmaps.distance_matrix(origin, destination, mode="driving")
        
        if result['rows'][0]['elements'][0]['status'] == 'OK':
            duration_seconds = result['rows'][0]['elements'][0]['duration']['value']
            duration_minutes = duration_seconds / 60
            return int(duration_minutes)
        else:
            return None
    except Exception as e:
        logger.error(f"❌ Error in check_travel_time: {type(e).__name__}: {str(e)}")
        return None

def validate_travel_time_from_previous(previous_apt: Dict[str, Any], new_start_time: datetime, new_address: str) -> Optional[Dict[str, Any]]:
    """Validate if there's enough travel time from a previous appointment.
    
    Args:
        previous_apt: Dictionary with previous appointment data (id, appointment_title, address, estimated_end_time)
        new_start_time: Start time of the new/updated appointment
        new_address: Address of the new/updated appointment
    
    Returns:
        Error dictionary with 'error' field if insufficient travel time, None if valid or no check needed
    """
    if not previous_apt or not previous_apt.get('address') or not new_address:
        return None
    
    travel_minutes = check_travel_time(previous_apt['address'], new_address)
    if travel_minutes is None:
        return None
    
    time_gap = (new_start_time - previous_apt['estimated_end_time']).total_seconds() / 60
    if time_gap < travel_minutes:
        suggested_time = (previous_apt['estimated_end_time'] + timedelta(minutes=travel_minutes)).strftime('%H:%M')
        return {
            "error": f"Not enough travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling at {suggested_time} or later."
        }
    
    return None

def validate_travel_time_to_next(next_apt: Dict[str, Any], new_end_time: datetime, new_address: str) -> Optional[Dict[str, Any]]:
    """Validate if there's enough travel time to the next appointment.
    
    Args:
        next_apt: Dictionary with next appointment data (id, appointment_title, address, start_time)
        new_end_time: End time of the new/updated appointment
        new_address: Address of the new/updated appointment
    
    Returns:
        Error dictionary with 'error' field if insufficient travel time, None if valid or no check needed
    """
    if not next_apt or not next_apt.get('address') or not new_address:
        return None
    
    travel_minutes = check_travel_time(new_address, next_apt['address'])
    if travel_minutes is None:
        return None
    
    time_gap = (next_apt['start_time'] - new_end_time).total_seconds() / 60
    if time_gap < travel_minutes:
        return {
            "error": f"Not enough travel time to next appointment '{next_apt['appointment_title']}' (ID: {next_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling earlier or adjusting the end time."
        }
    
    return None
