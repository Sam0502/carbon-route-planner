"""
Script to update the application to use the enhanced maps client.
"""
import os
import time
from enhanced_maps_client import EnhancedMapsClient

def update_app():
    """Update the app to use the enhanced maps client."""
    print("Updating application to use enhanced Maps client...")
    
    # Get the API key from environment
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    
    # Create an instance of the enhanced maps client
    maps_client = EnhancedMapsClient(api_key)
    
    # Test the client
    print("\nTesting enhanced Maps client with known city pairs...")
    test_routes = [
        ("New York", "Boston"),
        ("New York", "clayton"),
        ("clayton", "Boston"),
    ]
    
    for origin, destination in test_routes:
        print(f"\nTesting route from '{origin}' to '{destination}':")
        route = maps_client.get_route(origin, destination)
        total_distance, total_duration, segments = maps_client.extract_route_data(route)
        print(f"Total distance: {total_distance:.1f} km, duration: {total_duration:.2f} hours")
        print(f"Route has {len(route.get('legs', []))} legs")
        if maps_client.last_route_warnings:
            print(f"Warnings: {maps_client.last_route_warnings}")
        
        # Pause between requests to avoid rate limits
        time.sleep(1)
    
    print("\nTest complete. To use the enhanced maps client in your app:")
    print("1. Add 'from enhanced_maps_client import EnhancedMapsClient' to app.py")
    print("2. Change '@st.cache_resource\\ndef get_maps_client():' to use EnhancedMapsClient")
    print("\nThank you for using the enhanced Maps client!")

if __name__ == "__main__":
    update_app()
