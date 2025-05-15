"""
Test script to validate aviation duration calculations.
"""
import os
from vehicle_data import get_vehicle_by_id, get_all_vehicles
from enhanced_maps_client import EnhancedMapsClient
from optimizer import RouteOptimizer
from carbon_calculator import calculate_route_footprint
from models import ShipmentRequest

def test_aviation_duration():
    """Test aviation transport duration calculations."""
    print("Testing aviation transport duration calculations...")
    
    # Initialize the maps client with a bad API key to force demo mode
    maps_client = EnhancedMapsClient(api_key="invalid_key_to_force_demo_mode")
    print(f"Using demo mode: {maps_client.demo_mode}")
    
    # Initialize the route optimizer
    optimizer = RouteOptimizer(maps_client)
    
    # Define a long-distance route that would typically use air transportation
    origin = "New York, NY, USA"
    destination = "Los Angeles, CA, USA"
    
    # Get route data
    route_data = maps_client.get_route(origin, destination)
      # Extract route data
    total_distance, total_duration, segments_data = maps_client.extract_route_data(route_data)
    
    print(f"\nRoute: {origin} → {destination}")
    print(f"Total distance: {total_distance:.1f} km")
    print(f"Standard duration: {total_duration:.2f} hours")
    
    # Check if air_duration is available
    if segments_data and 'air_duration' in segments_data[0]:
        print(f"Aviation duration: {segments_data[0]['air_duration']:.2f} hours")
    else:
        print("Aviation duration not available in segments data")
    
    # Print raw route data to debug
    print("\nDEBUG - Route data structure:")
    if 'legs' in route_data:
        for i, leg in enumerate(route_data['legs']):
            print(f"Leg {i+1}:")
            print(f"  Distance: {leg.get('distance', {})}")
            print(f"  Duration: {leg.get('duration', {})}")
            print(f"  Has air_duration: {'air_duration' in leg}")
            if 'air_duration' in leg:
                print(f"  Air duration: {leg.get('air_duration', {})}")
    else:
        print("No legs found in route data")

    # Test with different vehicle types
    vehicles = {
        "standard_diesel": "Standard Diesel Truck (Ground)",
        "cargo_plane_medium": "Medium Cargo Plane (Air)",
        "cargo_plane_small": "Small Cargo Plane (Air)",
        "sustainable_aviation": "Sustainable Aviation (Air)"
    }
    
    print("\nTesting different vehicle types:")
    print("-" * 80)
    print(f"{'Vehicle Type':<30} {'Distance (km)':<15} {'Duration (h)':<15} {'Speed (km/h)':<15}")
    print("-" * 80)
    
    for vehicle_id, vehicle_name in vehicles.items():
        # Get vehicle speed
        vehicle = get_vehicle_by_id(vehicle_id)
        
        # Create a route with this vehicle
        route = calculate_route_footprint(
            waypoints=[origin, destination],
            segments_data=segments_data,
            vehicle_type_id=vehicle_id,
            use_ai_prediction=True
        )
        
        # Calculate the effective speed
        effective_speed = total_distance / route.total_duration if route.total_duration > 0 else 0
        
        print(f"{vehicle_name:<30} {total_distance:<15.1f} {route.total_duration:<15.2f} {effective_speed:<15.1f}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_aviation_duration()
