"""
Enhanced test script to troubleshoot Google Maps API access.

This script will:
1. Verify the API key is correctly loaded
2. Test different API services (Directions, Geocoding, Distance Matrix)
3. Provide detailed error information
"""
import os
import json
import googlemaps
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def test_maps_api_comprehensive():
    """Run comprehensive tests on all Google Maps API services we use."""
    # Get API key
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("ERROR: No API key found in environment variables.")
        print("Make sure GOOGLE_MAPS_API_KEY is set in your .env file or environment.")
        return False
    
    print(f"Found API key: {api_key[:5]}...{api_key[-4:]}")
    
    # Initialize client
    try:
        gmaps = googlemaps.Client(key=api_key)
        print("✅ Successfully initialized Google Maps client")
    except Exception as e:
        print(f"❌ Failed to initialize Google Maps client: {str(e)}")
        return False
    
    # Test 1: Geocoding API
    print("\n--- Testing Geocoding API ---")
    try:
        geocode_result = gmaps.geocode("New York, NY")
        if geocode_result:
            print(f"✅ Geocoding API returned {len(geocode_result)} results")
            print(f"   First result: {geocode_result[0]['formatted_address']}")
        else:
            print("❌ Geocoding API returned no results")
    except Exception as e:
        print(f"❌ Geocoding API error: {str(e)}")
    
    # Test 2: Directions API
    print("\n--- Testing Directions API ---")
    try:
        directions = gmaps.directions(
            "New York, NY", 
            "Boston, MA",
            mode="driving",
            departure_time=datetime.now()
        )
        if directions:
            print(f"✅ Directions API returned {len(directions)} routes")
            leg = directions[0].get('legs', [{}])[0]
            distance = leg.get('distance', {}).get('text', 'unknown')
            duration = leg.get('duration', {}).get('text', 'unknown')
            print(f"   Route distance: {distance}, duration: {duration}")
        else:
            print("❌ Directions API returned no results")
    except Exception as e:
        print(f"❌ Directions API error: {str(e)}")
    
    # Test 3: Distance Matrix API
    print("\n--- Testing Distance Matrix API ---")
    try:
        matrix = gmaps.distance_matrix(
            ["New York, NY"],
            ["Boston, MA"],
            mode="driving"
        )
        if matrix and matrix.get('status') == 'OK':
            print(f"✅ Distance Matrix API returned successfully")
            element = matrix['rows'][0]['elements'][0]
            if element.get('status') == 'OK':
                distance = element.get('distance', {}).get('text', 'unknown')
                duration = element.get('duration', {}).get('text', 'unknown')
                print(f"   Distance: {distance}, duration: {duration}")
            else:
                print(f"❌ Distance Matrix element status: {element.get('status')}")
        else:
            print(f"❌ Distance Matrix API status: {matrix.get('status')}")
    except Exception as e:
        print(f"❌ Distance Matrix API error: {str(e)}")
    
    # Test 4: Directions API with waypoints
    print("\n--- Testing Directions API with waypoints ---")
    try:
        directions = gmaps.directions(
            "New York, NY",
            "Boston, MA",
            waypoints=["Philadelphia, PA"],
            mode="driving",
            departure_time=datetime.now()
        )
        if directions:
            print(f"✅ Directions API with waypoints returned {len(directions)} routes")
            print(f"   Route has {len(directions[0].get('legs', []))} legs")
        else:
            print("❌ Directions API with waypoints returned no results")
    except Exception as e:
        print(f"❌ Directions API with waypoints error: {str(e)}")
    
    print("\n--- Google Maps API Service Requirements ---")
    print("To use this application, make sure the following services are enabled for your API key:")
    print("1. Directions API")
    print("2. Distance Matrix API")
    print("3. Geocoding API")
    print("4. Places API (optional)")
    print("\nYou can enable these services in the Google Cloud Console:")
    print("https://console.cloud.google.com/apis/library")

if __name__ == "__main__":
    test_maps_api_comprehensive()
