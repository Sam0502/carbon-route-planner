"""
Google Maps API integration for route planning.
"""
import os
import googlemaps
from typing import List, Dict, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MapsClient:
    """Client for interacting with Google Maps APIs."""
    
    def __init__(self, api_key: str = None):
        """Initialize Google Maps client with API key."""
        self.demo_mode = False
        
        if api_key is None:
            api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if api_key is None:
                print("Warning: No API key found. Using demo mode.")
                self.demo_mode = True
                self.client = None
                return
        
        # Initialize the API client without skipping to demo mode
        self.client = googlemaps.Client(key=api_key)
        print(f"Initialized Maps client with API key: {api_key[:5]}...{api_key[-4:]}")
            
    def get_route(self, origin: str, destination: str, waypoints: List[str] = None) -> Dict[str, Any]:
        """Get route information between origin and destination with optional waypoints."""
        if self.demo_mode:
            return self._get_demo_route(origin, destination, waypoints)
        
        try:    
            now = datetime.now()
            print(f"Requesting directions from {origin} to {destination}")
            
            if waypoints:
                directions = self.client.directions(
                    origin=origin,
                    destination=destination,
                    waypoints=waypoints,
                    mode="driving",
                    departure_time=now
                )
            else:
                directions = self.client.directions(
                    origin=origin,
                    destination=destination,
                    mode="driving",
                    departure_time=now
                )
                
            print(f"Received {len(directions)} routes from API")
            
            if not directions:
                print(f"Warning: No route found between {origin} and {destination}. Using demo route.")
                return self._get_demo_route(origin, destination, waypoints)
                
            return directions[0]
            
        except googlemaps.exceptions.ApiError as e:
            print(f"API ERROR: Google Maps API error: {str(e)}. Using demo route.")
            return self._get_demo_route(origin, destination, waypoints)
        except Exception as e:
            print(f"GENERAL ERROR: Error getting route ({str(e)}). Using demo route.")
            return self._get_demo_route(origin, destination, waypoints)
    
    def extract_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Extract distance, duration and steps from a route."""
        if self.demo_mode:
            return self._extract_demo_route_data(route)
            
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        steps = []
        
        for leg in legs:
            total_distance += leg.get('distance', {}).get('value', 0) / 1000  # Convert to km
            total_duration += leg.get('duration', {}).get('value', 0) / 3600  # Convert to hours
            
            for step in leg.get('steps', []):
                steps.append({
                    'start_location': step.get('start_location'),
                    'end_location': step.get('end_location'),
                    'distance': step.get('distance', {}).get('value', 0) / 1000,  # km
                    'duration': step.get('duration', {}).get('value', 0) / 3600,  # hours
                })
        
        return total_distance, total_duration, steps
    
    def _get_demo_route(self, origin: str, destination: str, waypoints: List[str] = None) -> Dict[str, Any]:
        """Generate a demo route for testing without API key."""
        print("*** USING DEMO MODE FOR ROUTE - API KEY NOT WORKING OR NOT ENABLED ***")
        legs = []
        
        all_points = [origin]
        if waypoints:
            all_points.extend(waypoints)
        all_points.append(destination)
        
        for i in range(len(all_points) - 1):
            # Simple dummy estimation based on string lengths
            start = all_points[i]
            end = all_points[i+1]
            
            # Create a simple distance/duration estimation (just for demo)
            # In reality, this would come from the API
            distance = (len(start) + len(end)) * 5  # km
            duration = distance / 60  # hours
            
            legs.append({
                'distance': {'text': f"{distance} km", 'value': distance * 1000},
                'duration': {'text': f"{duration} hours", 'value': duration * 3600},
                'start_address': start,
                'end_address': end,
                'steps': [{
                    'distance': {'text': f"{distance} km", 'value': distance * 1000},
                    'duration': {'text': f"{duration} hours", 'value': duration * 3600},
                    'start_location': {'lat': 0, 'lng': 0},
                    'end_location': {'lat': 1, 'lng': 1},
                }]
            })
        
        return {
            'legs': legs,
            'warnings': ["Demo mode active - distances are estimated"]
        }
    
    def _extract_demo_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Extract data from a demo route."""
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        steps = []
        
        for leg in legs:
            distance = leg.get('distance', {}).get('value', 0) / 1000
            duration = leg.get('duration', {}).get('value', 0) / 3600
            
            total_distance += distance
            total_duration += duration
            
            steps.append({
                'start_location': {'lat': 0, 'lng': 0},
                'end_location': {'lat': 1, 'lng': 1},
                'distance': distance,
                'duration': duration
            })
        
        return total_distance, total_duration, steps