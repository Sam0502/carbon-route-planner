"""
Test script to verify Google Maps API connection.
"""
import os
from dotenv import load_dotenv
import googlemaps
from datetime import datetime

# Load environment variables
load_dotenv()

def test_maps_api():
    """Test Google Maps API with a simple directions request."""
    # Print environment info
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("ERROR: No API key found in environment variables.")
        return False
    
    print(f"Found API key: {api_key[:5]}...{api_key[-4:]}")
    
    try:
        # Initialize client
        gmaps = googlemaps.Client(key=api_key)
        
        # Test a simple directions request
        now = datetime.now()
        directions = gmaps.directions("New York, NY", 
                                      "Boston, MA",
                                      mode="driving",
                                      departure_time=now)
        
        if directions:
            print(f"SUCCESS: Received {len(directions)} routes from API.")
            print(f"First route has {len(directions[0].get('legs', []))} legs.")
            return True
        else:
            print("ERROR: No directions returned.")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_maps_api()
