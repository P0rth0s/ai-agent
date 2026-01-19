from dotenv import load_dotenv
from typing import Optional
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
