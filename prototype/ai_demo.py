"""
Demonstration of AI-based carbon footprint prediction capabilities.

This script demonstrates how to use the AI prediction features for more 
accurate carbon footprint estimation based on dynamic factors.
"""
import os
import sys
import json
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from models import ShipmentRequest, Route
from maps_client import MapsClient
from optimizer import RouteOptimizer
from ai_predictor import get_predictor
from carbon_calculator import calculate_segment_footprint, calculate_route_footprint
from vehicle_data import get_all_vehicles, get_vehicle_by_id

def display_route_comparison(standard_route: Route, ai_route: Route) -> None:
    """Display a comparison between standard and AI-enhanced route calculations."""
    print("\n" + "=" * 80)
    print(f"ROUTE COMPARISON: Standard vs. AI-enhanced prediction")
    print("=" * 80)
    
    print(f"\nTotal Distance: {standard_route.total_distance:.2f} km")
    print(f"Total Duration: {standard_route.total_duration:.2f} hours")
    
    print("\nCO2e Estimates:")
    print(f"  Standard calculation: {standard_route.total_co2e:.2f} kg CO2e")
    print(f"  AI-enhanced calculation: {ai_route.total_co2e:.2f} kg CO2e")
    print(f"  Difference: {ai_route.total_co2e - standard_route.total_co2e:.2f} kg CO2e " +
          f"({((ai_route.total_co2e / standard_route.total_co2e) - 1) * 100:.1f}%)")
    
    print("\nSegment breakdown:")
    print("-" * 80)
    print(f"{'Origin':30} {'Destination':30} {'Standard CO2e':15} {'AI CO2e':15} {'Diff %':10}")
    print("-" * 80)
    
    for i in range(len(standard_route.segments)):
        std_segment = standard_route.segments[i]
        ai_segment = ai_route.segments[i]
        
        diff_pct = ((ai_segment.co2e / std_segment.co2e) - 1) * 100 if std_segment.co2e > 0 else 0
        
        print(f"{std_segment.origin[:30]:30} {std_segment.destination[:30]:30} " +
              f"{std_segment.co2e:15.2f} {ai_segment.co2e:15.2f} {diff_pct:+10.1f}%")
    
    # Display AI prediction factors if available
    if hasattr(ai_route.segments[0], 'prediction_metadata') and ai_route.segments[0].prediction_metadata:
        print("\nAI Prediction Factors:")
        for key, value in ai_route.segments[0].prediction_metadata.get("adjustments", {}).items():
            print(f"  {key}: {value:.3f}")
    
    print("=" * 80)

def run_demo_comparison() -> None:
    """Run a demonstration comparing standard and AI-enhanced predictions."""
    try:
        # Define a sample route
        test_route = {
            "origin": "Berlin, Germany",
            "destination": "Munich, Germany",
            "intermediate_stops": ["Leipzig, Germany", "Nuremberg, Germany"],
            "weight_tons": 15.0
        }
        
        print(f"Running AI prediction demo for route: {test_route['origin']} → "
              f"{' → '.join(test_route['intermediate_stops'])} → {test_route['destination']}")
        
        # Set up dummy maps client for demo (won't make actual API calls)
        class DummyMapsClient:
            def __init__(self):
                # Store the route waypoints for extract_route_data to use
                self.origin = test_route["origin"]
                self.destination = test_route["destination"]
                self.waypoints = test_route["intermediate_stops"]
                
            def get_route(self, origin, destination, waypoints=None):
                # Store these for extract_route_data to use later
                self.origin = origin
                self.destination = destination
                self.waypoints = waypoints
                return {"dummy": "route_data"}
                
            def extract_route_data(self, route):
                # Create realistic but dummy data
                segments = []
                waypoints_list = [self.origin]
                if self.waypoints:
                    waypoints_list.extend(self.waypoints)
                waypoints_list.append(self.destination)
                
                total_distance = 0
                total_duration = 0
                
                # Generate segment data
                for i in range(len(waypoints_list) - 1):
                    # Realistic distances between German cities
                    distances = {
                        "Berlin-Leipzig": 190,
                        "Leipzig-Nuremberg": 285,
                        "Nuremberg-Munich": 170,
                        # Add fallbacks for any combination
                        "Berlin-Nuremberg": 440,
                        "Berlin-Munich": 585,
                        "Leipzig-Munich": 430
                    }
                    
                    origin = waypoints_list[i]
                    destination = waypoints_list[i + 1]
                    
                    # Try to find the specific segment distance, otherwise use a default
                    key = f"{origin.split(',')[0]}-{destination.split(',')[0]}"
                    rev_key = f"{destination.split(',')[0]}-{origin.split(',')[0]}"
                    
                    distance = distances.get(key, distances.get(rev_key, 200))  # Default 200km
                    duration = distance / 80  # Assume 80 km/h average speed
                    
                    segments.append({
                        "distance": distance,
                        "duration": duration
                    })
                    
                    total_distance += distance
                    total_duration += duration
                
                return total_distance, total_duration, segments
        
        # Simulate Maps API results
        maps_client = DummyMapsClient()
        
        # Create a request for standard calculation (no AI)
        standard_request = ShipmentRequest(
            origin=test_route["origin"],
            destination=test_route["destination"],
            intermediate_stops=test_route["intermediate_stops"],
            weight_tons=test_route["weight_tons"],
            use_ai_prediction=False
        )
        
        # Create a request for AI-enhanced calculation
        # Include environmental factors for more realistic prediction
        ai_request = ShipmentRequest(
            origin=test_route["origin"],
            destination=test_route["destination"],
            intermediate_stops=test_route["intermediate_stops"],
            weight_tons=test_route["weight_tons"],
            use_ai_prediction=True,
            # Add specific environmental factors for each segment
            terrain_factors=[1.0, 1.2, 1.1],  # Relatively flat to slightly hilly
            temperatures=[22.0, 18.0, 20.0],  # Temperature in Celsius
            traffic_levels=[0.6, 0.4, 0.7]     # Moderate to high traffic
        )
        
        # Process route with standard and AI-enhanced calculations
        optimizer = RouteOptimizer(maps_client)
        
        # Extract the route data once
        all_waypoints = [ai_request.origin]
        all_waypoints.extend(ai_request.intermediate_stops)
        all_waypoints.append(ai_request.destination)
        
        total_distance, total_duration, segments_data = maps_client.extract_route_data({"dummy": "route"})
        
        # Calculate routes with standard and AI prediction methods
        standard_route = calculate_route_footprint(
            waypoints=all_waypoints,
            segments_data=segments_data,
            vehicle_type_id="standard_diesel",
            weight_tons=standard_request.weight_tons,
            use_ai_prediction=False
        )
        
        ai_route = calculate_route_footprint(
            waypoints=all_waypoints,
            segments_data=segments_data,
            vehicle_type_id="standard_diesel",
            weight_tons=ai_request.weight_tons,
            use_ai_prediction=True,
            terrain_factors=ai_request.terrain_factors,
            temperatures=ai_request.temperatures,
            traffic_levels=ai_request.traffic_levels
        )
        
        # Add vehicle attributes to both routes
        vehicle = get_vehicle_by_id("standard_diesel")
        standard_route.set_vehicle_attributes(vehicle)
        ai_route.set_vehicle_attributes(vehicle)
        
        # Display comparison between the two calculations
        display_route_comparison(standard_route, ai_route)
        
        # Run a single segment prediction with detailed output to demonstrate 
        # the AI-based prediction capabilities
        print("\nSINGLE SEGMENT AI PREDICTION EXAMPLE:")
        print("-" * 80)
        
        # Get the AI predictor and make a prediction
        predictor = get_predictor()
        
        # Define parameters for a single segment
        prediction_params = {
            "vehicle_type_id": "standard_diesel",
            "distance": 200.0,
            "avg_speed": 75.0,
            "weight_tons": 15.0,
            "terrain_factor": 1.2,  # Slightly hilly
            "temperature": 22.0,    # Warm day
            "traffic_level": 0.7    # Heavy traffic
        }
        
        print(f"Requesting prediction with parameters:")
        for k, v in prediction_params.items():
            print(f"  {k}: {v}")
        
        # Get prediction
        co2e, metadata = predictor.predict_co2e(**prediction_params)
        
        # Show the prediction result
        print(f"\nAI Prediction result: {co2e:.2f} kg CO2e")
        
        if "adjustments" in metadata:
            print("\nPrediction factors:")
            for factor, value in metadata["adjustments"].items():
                print(f"  {factor}: {value:.3f}")
        
        print("\nDemo batch prediction capability with multiple segments:")
        print("-" * 80)
        
        # Set up batch prediction request
        batch_request = [
            {
                "vehicle_type_id": "standard_diesel",
                "distance": 190.0,
                "avg_speed": 80.0,
                "weight_tons": 15.0,
                "terrain_factor": 1.0,
                "temperature": 22.0,
                "traffic_level": 0.6
            },
            {
                "vehicle_type_id": "standard_diesel",
                "distance": 285.0,
                "avg_speed": 75.0,
                "weight_tons": 15.0, 
                "terrain_factor": 1.2,
                "temperature": 18.0,
                "traffic_level": 0.4
            },
            {
                "vehicle_type_id": "standard_diesel",
                "distance": 170.0,
                "avg_speed": 70.0,
                "weight_tons": 15.0,
                "terrain_factor": 1.1,
                "temperature": 20.0,
                "traffic_level": 0.7
            }
        ]
        
        batch_results = predictor.batch_predict(batch_request)
        
        print("Batch prediction results:")
        for i, (co2e, metadata) in enumerate(batch_results):
            print(f"  Segment {i+1}: {co2e:.2f} kg CO2e")
            
        print("\nCompare different vehicle types with AI prediction:")
        print("-" * 80)
        
        # Compare different vehicle types
        vehicle_types = ["standard_diesel", "eco_diesel", "electric", "hybrid", "cng"]
        
        for vehicle_type in vehicle_types:
            ai_route = calculate_route_footprint(
                waypoints=all_waypoints,
                segments_data=segments_data,
                vehicle_type_id=vehicle_type,
                weight_tons=test_route["weight_tons"],
                use_ai_prediction=True,
                terrain_factors=ai_request.terrain_factors,
                temperatures=ai_request.temperatures,
                traffic_levels=ai_request.traffic_levels
            )
            
            # Get vehicle info
            vehicle = get_vehicle_by_id(vehicle_type)
            print(f"{vehicle.name:25}: {ai_route.total_co2e:.2f} kg CO2e")
        
        print("\nAI prediction demo complete!")
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Demo of AI-enhanced Carbon Footprint Prediction")
    print("=" * 50)
    run_demo_comparison()