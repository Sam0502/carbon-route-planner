"""
Improved Maps Client that better handles address validation and API errors.
"""
import os
import re
import googlemaps
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

class EnhancedMapsClient:
    """Enhanced client for Google Maps APIs with better error handling."""
    
    def __init__(self, api_key: str = None):
        """Initialize Google Maps client with API key."""
        self.demo_mode = False
        self.last_route_warnings = []
        
        if api_key is None:
            api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if not api_key:
                print("Warning: No API key found. Using demo mode.")
                self.demo_mode = True
                self.client = None
                return
        
        try:
            # Initialize the client
            self.client = googlemaps.Client(key=api_key)
            print(f"Initialized Maps client with API key: {api_key[:5]}...{api_key[-4:]}")
            
            # Test the API
            test = self.client.geocode("New York, NY")
            if test:
                print("API connection test successful")
            else:
                print("Warning: API test returned no results")
        except Exception as e:
            print(f"Error initializing Maps client: {str(e)}")
            self.demo_mode = True
            self.client = None
    
    def validate_address(self, address: str) -> Dict[str, Any]:
        """Validate and format an address using Google Maps Geocoding API."""
        if not address or address.strip() == "":
            return {'valid': False, 'formatted_address': None, 'suggestions': []}
            
        if self.demo_mode:
            return {'valid': True, 'formatted_address': address, 'suggestions': []}
        
        # Clean up the address
        cleaned_address = self._clean_address(address)
        
        # Common city names that might need state/country added
        common_cities = {
            "new york": "New York, NY",
            "boston": "Boston, MA",
            "chicago": "Chicago, IL",
            "clayton": "Clayton, NC"  # Add this since it's been used
        }        # Check if this is a single-word city that needs expansion
        lower_address = cleaned_address.lower()
        if lower_address in common_cities and "," not in address:
            expanded_info = f"Expanded city name from '{address}' to '{common_cities[lower_address]}'"
            print(expanded_info)  # For local debugging
            cleaned_address = common_cities[lower_address]
        else:
            expanded_info = None
        
        try:
            # Try to geocode the address
            geocode_result = self.client.geocode(cleaned_address)
            
            if not geocode_result:
                return {'valid': False, 'formatted_address': None, 'suggestions': []}
            
            # Get the first (most relevant) result
            formatted_address = geocode_result[0]['formatted_address']
              # If we have multiple results, they could be suggestions
            suggestions = []
            if len(geocode_result) > 1:
                suggestions = [result['formatted_address'] for result in geocode_result[1:5]]
                
            result = {
                'valid': True,
                'formatted_address': formatted_address,
                'suggestions': suggestions
            }
            
            # Add expansion info if available
            if 'expanded_info' in locals() and expanded_info:
                result['expanded_info'] = expanded_info
                
            return result
            
        except Exception as e:
            print(f"Address validation error: {str(e)}")
            return {'valid': False, 'formatted_address': None, 'suggestions': []}
    
    def _clean_address(self, address: str) -> str:
        """Clean up an address string."""
        # Remove excessive spaces
        cleaned = re.sub(r'\s+', ' ', address).strip()
        # Remove excessive commas
        cleaned = re.sub(r',+', ',', cleaned).strip(',')
        return cleaned
    
    def get_route(self, origin: str, destination: str, waypoints: List[str] = None) -> Dict[str, Any]:
        """Get route information between origin and destination with optional waypoints."""
        # Reset warnings
        self.last_route_warnings = []
        
        # Check for empty addresses
        if not origin or not destination:
            print(f"Error: Empty address provided: origin='{origin}', destination='{destination}'")
            self.last_route_warnings.append("Empty address provided")
            return self._get_demo_route(origin, destination, waypoints)
        
        if self.demo_mode or self.client is None:
            print("Using demo mode: Maps client not initialized or API key invalid")
            return self._get_demo_route(origin, destination, waypoints)
        
        try:
            # Validate and format the addresses
            origin_data = self.validate_address(origin)
            destination_data = self.validate_address(destination)
            
            # Use formatted addresses if available
            formatted_origin = origin_data.get('formatted_address', origin) if origin_data.get('valid', False) else origin
            formatted_destination = destination_data.get('formatted_address', destination) if destination_data.get('valid', False) else destination
            
            print(f"Using addresses: From '{formatted_origin}' to '{formatted_destination}'")
            
            # Format waypoints if provided
            formatted_waypoints = None
            if waypoints:
                formatted_waypoints = []
                for waypoint in waypoints:
                    waypoint_data = self.validate_address(waypoint)
                    formatted_waypoint = waypoint_data.get('formatted_address', waypoint) if waypoint_data.get('valid', False) else waypoint
                    formatted_waypoints.append(formatted_waypoint)
            
            # Try to get directions
            directions = None
            try:
                now = datetime.now()
                if formatted_waypoints:
                    directions = self.client.directions(
                        origin=formatted_origin,
                        destination=formatted_destination,
                        waypoints=formatted_waypoints,
                        mode="driving",
                        departure_time=now
                    )
                else:
                    directions = self.client.directions(
                        origin=formatted_origin,
                        destination=formatted_destination,
                        mode="driving",
                        departure_time=now
                    )
                
                if directions:
                    print(f"Successfully retrieved {len(directions)} routes")
                else:
                    print("API returned no directions")
            except googlemaps.exceptions.ApiError as e:
                print(f"Directions API error: {str(e)}")
                self.last_route_warnings.append(f"API Error: {str(e)}")
                
                # Try fallback approach with Distance Matrix API
                if "NOT_FOUND" in str(e):
                    print("Location not found. Trying with enhanced location names...")
                    # Try again with enhanced location names
                    return self._get_demo_route(origin, destination, waypoints)
            
            # If we got directions, return them
            if directions:
                if 'warnings' in directions[0]:
                    self.last_route_warnings.extend(directions[0]['warnings'])
                return directions[0]
            
            # If we got here, directions failed, try distance matrix as fallback
            print("Directions API failed. Trying Distance Matrix API...")
            all_points = [formatted_origin]
            if formatted_waypoints:
                all_points.extend(formatted_waypoints)
            all_points.append(formatted_destination)
            
            # Try Distance Matrix API
            route_legs = []
            success = True
            for i in range(len(all_points) - 1):
                try:
                    matrix = self.client.distance_matrix(
                        origins=[all_points[i]],
                        destinations=[all_points[i+1]],
                        mode="driving"
                    )
                    
                    if matrix and matrix.get('status') == 'OK':
                        element = matrix['rows'][0]['elements'][0]
                        if element.get('status') == 'OK':
                            # Successfully got distance
                            route_legs.append({
                                'distance': element['distance'],
                                'duration': element['duration'],
                                'start_address': all_points[i],
                                'end_address': all_points[i+1],
                                'steps': [{
                                    'distance': element['distance'],
                                    'duration': element['duration'],
                                    'start_location': {'lat': 0, 'lng': 0},
                                    'end_location': {'lat': 1, 'lng': 1}
                                }]
                            })
                        else:
                            print(f"Distance Matrix element error: {element.get('status')}")
                            success = False
                            break
                    else:
                        print(f"Distance Matrix error: {matrix.get('status')}")
                        success = False
                        break
                except Exception as e:
                    print(f"Distance Matrix exception: {str(e)}")
                    success = False
                    break
            
            if success and len(route_legs) == len(all_points) - 1:
                self.last_route_warnings.append("Route details from Distance Matrix API")
                return {'legs': route_legs, 'warnings': self.last_route_warnings}
            
            # If all else fails, use demo route
            print("All API methods failed. Using demo route.")
            suggestions = {
                'origin_suggestions': origin_data.get('suggestions', []),
                'destination_suggestions': destination_data.get('suggestions', [])
            }
            return self._get_demo_route(origin, destination, waypoints, suggestions)
            
        except Exception as e:
            print(f"General error getting route: {str(e)}")
            self.last_route_warnings.append(f"Error: {str(e)}")
            return self._get_demo_route(origin, destination, waypoints)
    
    def _get_demo_route(self, origin: str, destination: str, waypoints: List[str] = None, 
                      suggestions: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Generate a demo route."""
        print("*** USING DEMO MODE FOR ROUTE - API KEY NOT WORKING OR NOT ENABLED ***")
        
        # Add warnings
        warnings = ["Demo mode active - distances are estimated"]
        self.last_route_warnings = warnings
        
        # Add any suggestions
        if suggestions:
            if suggestions.get('origin_suggestions'):
                warnings.append(f"Did you mean for origin: {', '.join(suggestions['origin_suggestions'][:3])}")
            if suggestions.get('destination_suggestions'):
                warnings.append(f"Did you mean for destination: {', '.join(suggestions['destination_suggestions'][:3])}")
        
        all_points = [origin]
        if waypoints:
            all_points.extend(waypoints)
        all_points.append(destination)
        
        # Generate legs
        legs = []
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i+1]
              # Get a realistic distance
            distance = self._get_realistic_distance(start, end)
              # Calculate both ground and aviation durations to support multi-modal transport options
            # This allows the optimizer to use the appropriate duration based on vehicle type
            ground_duration = distance / 60  # hours (assumed avg ground speed 60 km/h)
            air_duration = distance / 750    # hours (assumed avg air speed 750 km/h)
            
            # Use ground duration by default in the standard field for backward compatibility
            # The aviation-specific duration is provided in a separate field
            duration = ground_duration
            
            # We add both durations so the optimizer can select the appropriate one based on vehicle type
            legs.append({
                'distance': {'text': f"{distance:.1f} km", 'value': int(distance * 1000)},
                'duration': {'text': f"{duration:.2f} hours", 'value': int(duration * 3600)},
                'air_duration': {'text': f"{air_duration:.2f} hours", 'value': int(air_duration * 3600)},
                'start_address': start,
                'end_address': end,
                'steps': [{
                    'distance': {'text': f"{distance:.1f} km", 'value': int(distance * 1000)},
                    'duration': {'text': f"{duration:.2f} hours", 'value': int(duration * 3600)},
                    'start_location': {'lat': 0, 'lng': 0},
                    'end_location': {'lat': 1, 'lng': 1},
                }]
            })
        
        return {'legs': legs, 'warnings': warnings}
    
    def _get_realistic_distance(self, origin: str, destination: str) -> float:
        """Get a realistic distance between two locations."""
        # Known distances between city pairs
        known_distances = {
            frozenset({"new york", "boston"}): 350,
            frozenset({"new york", "washington"}): 365,
            frozenset({"clayton", "boston"}): 440,
            frozenset({"new york", "clayton"}): 480
        }
        
        # Normalize location names
        orig_lower = origin.lower()
        dest_lower = destination.lower()
        
        # Check for known city pairs
        for cities, distance in known_distances.items():
            cities_list = list(cities)
            if (cities_list[0] in orig_lower and cities_list[1] in dest_lower) or \
               (cities_list[1] in orig_lower and cities_list[0] in dest_lower):
                return distance
        
        # For unknown pairs, use a realistic value with some randomness
        base_distance = 100 + len(origin) + len(destination)
        return min(max(base_distance, 50), 1000)  # Between 50 and 1000 km
    
    def extract_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Extract distance, duration and segments from a route."""
        if not route or 'legs' not in route:
            return 0, 0, []
        
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        segments_data = []
        
        for leg in legs:
            distance = leg.get('distance', {}).get('value', 0) / 1000  # Convert to km
            duration = leg.get('duration', {}).get('value', 0) / 3600  # Convert to hours
              # Check if we have air_duration available (added in demo mode for aviation)
            air_duration = leg.get('air_duration', {}).get('value', 0) / 3600 if 'air_duration' in leg else None
            
            total_distance += distance
            total_duration += duration
            
            # Add segment data including air_duration if available
            segment_data = {
                'distance': distance,
                'duration': duration
            }
            
            if air_duration is not None:
                segment_data['air_duration'] = air_duration
                
            segments_data.append(segment_data)
            
            # Debug output
            print(f"Route leg: {leg.get('start_address', 'Unknown')} to {leg.get('end_address', 'Unknown')}")
            print(f"Distance: {distance:.1f} km, Duration: {duration:.2f} hours")
        
        return total_distance, total_duration, segments_data
