"""
Google Maps API integration for route planning.
"""
import os
import re
import googlemaps
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MapsClient:
    """Client for interacting with Google Maps APIs."""
      def __init__(self, api_key: str = None):
        """Initialize Google Maps client with API key."""
        self.demo_mode = False
        self.last_route_warnings = []
        
        if api_key is None:
            api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if not api_key or api_key.strip() == "":
                print("Warning: No API key found or empty API key. Using demo mode.")
                self.demo_mode = True
                self.client = None
                return
            print(f"Found API key in environment: {api_key[:5]}...{api_key[-4:]}")
        
        try:
            # Initialize the API client without skipping to demo mode
            self.client = googlemaps.Client(key=api_key)
            print(f"Initialized Maps client with API key: {api_key[:5]}...{api_key[-4:]}")
            
            # Test the API with a simple geocode request
            test_result = self.client.geocode("New York, NY")
            if test_result:
                print(f"API test successful - geocoding works")
            else:
                print(f"Warning: API returned empty result - might be restricted")
        except Exception as e:
            print(f"Error initializing Maps client: {str(e)}. Switching to demo mode.")
            self.demo_mode = True
            self.client = None
    
    def validate_address(self, address: str) -> Dict[str, Any]:
        """
        Validate and format an address using Google Maps Geocoding API.
        
        Args:
            address: The address to validate
            
        Returns:
            Dict with:
                'valid': Boolean indicating if address is valid
                'formatted_address': Formatted address if valid
                'suggestions': List of suggested addresses if original is ambiguous
        """
        if self.demo_mode:
            # In demo mode, just return the original address as valid
            return {
                'valid': True, 
                'formatted_address': address,
                'suggestions': []
            }
        
        try:
            # Clean up the address by removing excessive commas and spaces
            cleaned_address = re.sub(r'\s+', ' ', address).strip()
            cleaned_address = re.sub(r',+', ',', cleaned_address).strip(',')
            
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
                
            return {
                'valid': True,
                'formatted_address': formatted_address,
                'suggestions': suggestions
            }
            
        except Exception as e:
            print(f"Address validation error: {str(e)}")
            return {'valid': False, 'formatted_address': None, 'suggestions': []}
              def get_route(self, origin: str, destination: str, waypoints: List[str] = None) -> Dict[str, Any]:
        """Get route information between origin and destination with optional waypoints."""
        # Reset warnings from previous route requests
        self.last_route_warnings = []
        
        if self.demo_mode or self.client is None:
            print("Using demo mode: Maps API client not initialized")
            return self._get_demo_route(origin, destination, waypoints)
          try:
            # Validate and format the addresses
            print(f"Validating addresses: origin='{origin}', destination='{destination}'")
            origin_data = self.validate_address(origin)
            destination_data = self.validate_address(destination)
            
            # Check for empty or invalid addresses
            if not origin or not destination:
                print(f"ERROR: Empty address provided: origin='{origin}', destination='{destination}'")
                self.last_route_warnings.append("Empty address provided")
                return self._get_demo_route(origin, destination, waypoints)
                
            # Use formatted addresses if available, otherwise use original
            formatted_origin = origin_data.get('formatted_address', origin) if origin_data.get('valid', False) else origin
            formatted_destination = destination_data.get('formatted_address', destination) if destination_data.get('valid', False) else destination
            
            print(f"Formatted addresses: origin='{formatted_origin}', destination='{formatted_destination}'")
            
            # Format waypoints if provided
            formatted_waypoints = waypoints
            if waypoints:
                formatted_waypoints = []
                for waypoint in waypoints:
                    waypoint_data = self.validate_address(waypoint)
                    formatted_waypoint = waypoint_data.get('formatted_address', waypoint) if waypoint_data.get('valid', False) else waypoint
                    formatted_waypoints.append(formatted_waypoint)
            
            print(f"Requesting directions from {formatted_origin} to {formatted_destination}")
            now = datetime.now()
              # Try to get directions with detailed error logging
            try:
                if formatted_waypoints:
                    print(f"Requesting directions with {len(formatted_waypoints)} waypoints: {formatted_waypoints}")
                    directions = self.client.directions(
                        origin=formatted_origin,
                        destination=formatted_destination,
                        waypoints=formatted_waypoints,
                        mode="driving",
                        departure_time=now
                    )
                else:
                    print(f"Requesting direct directions without waypoints")
                    directions = self.client.directions(
                        origin=formatted_origin,
                        destination=formatted_destination,
                        mode="driving",
                        departure_time=now
                    )
                    
                if directions:
                    print(f"Success! Received {len(directions)} routes from API")
                else:
                    print(f"Warning: API returned empty directions list")
            except googlemaps.exceptions.ApiError as api_err:
                print(f"Directions API error: {str(api_err)}")
                self.last_route_warnings.append(f"Directions API error: {str(api_err)}")
                return self._get_demo_route(origin, destination, waypoints)
                
            print(f"Received {len(directions)} routes from API")
            
            if not directions:
                print(f"Warning: No route found between {formatted_origin} and {formatted_destination}. Trying Distance Matrix API...")
                
                # If Directions API failed, try using Distance Matrix API for more accuracy
                all_points = [formatted_origin]
                if formatted_waypoints:
                    all_points.extend(formatted_waypoints)
                all_points.append(formatted_destination)
                
                try:
                    # Get distances between consecutive points
                    route_legs = []
                    for i in range(len(all_points) - 1):
                        matrix = self.get_distance_matrix([all_points[i]], [all_points[i+1]])
                        if matrix and matrix.get('status') == 'OK':
                            element = matrix['rows'][0]['elements'][0]
                            if element.get('status') == 'OK':
                                # Successfully got distance from API
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
                                continue
                    
                    # If we successfully got distances for all segments
                    if len(route_legs) == len(all_points) - 1:
                        self.last_route_warnings.append("Route details unavailable, but distances obtained from Distance Matrix API")
                        return {'legs': route_legs, 'warnings': self.last_route_warnings}
                except Exception as e:
                    print(f"Distance Matrix API also failed: {str(e)}")
                
                # Fall back to demo route if both APIs fail
                suggestions = {
                    'origin_suggestions': origin_data.get('suggestions', []),
                    'destination_suggestions': destination_data.get('suggestions', [])
                }
                return self._get_demo_route(origin, destination, waypoints, suggestions=suggestions)
            
            # Store any warnings from the API response
            if 'warnings' in directions[0]:
                self.last_route_warnings = directions[0]['warnings']
                
            return directions[0]
            
        except googlemaps.exceptions.ApiError as e:
            print(f"API ERROR: Google Maps API error: {str(e)}. Using demo route.")
            self.last_route_warnings.append(f"API Error: {str(e)}")
            return self._get_demo_route(origin, destination, waypoints)
        except Exception as e:
            print(f"GENERAL ERROR: Error getting route ({str(e)}). Using demo route.")
            self.last_route_warnings.append(f"Error: {str(e)}")
            return self._get_demo_route(origin, destination, waypoints)
    
    def extract_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Extract distance, duration and steps from a route."""
        if not route or 'legs' not in route:
            return self._extract_demo_route_data(route)
            
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        segments_data = []
        
        # Process each leg (route segment between waypoints)
        for leg in legs:
            leg_distance = leg.get('distance', {}).get('value', 0) / 1000  # Convert to km
            leg_duration = leg.get('duration', {}).get('value', 0) / 3600  # Convert to hours
            
            total_distance += leg_distance
            total_duration += leg_duration
            
            # Each leg is a segment in our data model
            segments_data.append({
                'distance': leg_distance,
                'duration': leg_duration
            })
            
            # Debug information
            print(f"Route leg: {leg.get('start_address', 'Unknown')} to {leg.get('end_address', 'Unknown')}")
            print(f"Distance: {leg_distance:.1f} km, Duration: {leg_duration:.2f} hours")
        
        return total_distance, total_duration, segments_data
    
    def _get_demo_route(self, origin: str, destination: str, waypoints: List[str] = None, suggestions: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Generate a demo route for testing without API key."""
        print("*** USING DEMO MODE FOR ROUTE - API KEY NOT WORKING OR NOT ENABLED ***")
        
        # Add any suggestions to the warnings
        warnings = ["Demo mode active - distances are estimated"]
        self.last_route_warnings = warnings
        
        if suggestions:
            if suggestions.get('origin_suggestions'):
                warnings.append(f"Did you mean for origin: {', '.join(suggestions['origin_suggestions'][:3])}")
            if suggestions.get('destination_suggestions'):
                warnings.append(f"Did you mean for destination: {', '.join(suggestions['destination_suggestions'][:3])}")
        
        legs = []
        
        all_points = [origin]
        if waypoints:
            all_points.extend(waypoints)
        all_points.append(destination)
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i+1]
            
            # Improved distance estimation based on known city distances
            # This is still a simulation but provides more realistic values
            distance = self._estimate_distance_between_points(start, end)
            
            # Estimate duration based on average speed of 60 km/h
            duration = distance / 60  # hours
            
            legs.append({
                'distance': {'text': f"{distance:.1f} km", 'value': int(distance * 1000)},
                'duration': {'text': f"{duration:.2f} hours", 'value': int(duration * 3600)},
                'start_address': start,
                'end_address': end,
                'steps': [{
                    'distance': {'text': f"{distance:.1f} km", 'value': int(distance * 1000)},
                    'duration': {'text': f"{duration:.2f} hours", 'value': int(duration * 3600)},
                    'start_location': {'lat': 0, 'lng': 0},
                    'end_location': {'lat': 1, 'lng': 1},
                }]
            })
        
        return {
            'legs': legs,
            'warnings': warnings
        }
    
    def _estimate_distance_between_points(self, point1: str, point2: str) -> float:
        """
        Estimate a more realistic distance between two locations based on
        common city pairs or a reasonable default.
        
        Args:
            point1: Starting location name/address
            point2: Ending location name/address
            
        Returns:
            Estimated distance in kilometers
        """
        # Lowercase the inputs for case-insensitive matching
        p1 = point1.lower()
        p2 = point2.lower()
        
        # Create a set of the two points (order doesn't matter for lookup)
        point_pair = {p1, p2}
        
        # Dictionary of known distances between city pairs (in km)
        known_distances = {
            frozenset({"new york", "boston"}): 350,
            frozenset({"new york", "washington"}): 365,
            frozenset({"los angeles", "san francisco"}): 615,
            frozenset({"chicago", "detroit"}): 450,
            frozenset({"sydney", "melbourne"}): 880,
            frozenset({"london", "paris"}): 340,
            frozenset({"berlin", "munich"}): 585,
            frozenset({"tokyo", "osaka"}): 510,
            frozenset({"beijing", "shanghai"}): 1200,
            frozenset({"toronto", "montreal"}): 540,
            frozenset({"rome", "florence"}): 275,
            frozenset({"madrid", "barcelona"}): 620,
            frozenset({"rio de janeiro", "sao paulo"}): 430,
            frozenset({"dubai", "abu dhabi"}): 140,
            frozenset({"delhi", "mumbai"}): 1400
        }
        
        # Check for city names in the addresses
        for cities, distance in known_distances.items():
            # If both points contain parts of the city names from a known pair
            if any(city in p1 for city in cities) and any(city in p2 for city in cities):
                # Make sure we're not matching the same city
                cities_list = list(cities)
                if (cities_list[0] in p1 and cities_list[1] in p2) or (cities_list[0] in p2 and cities_list[1] in p1):
                    return distance
        
        # If no matching city pairs, use a more reasonable estimation
        # Check if the points look like they might be in the same city
        same_city = False
        for city_marker in ["new york", "los angeles", "chicago", "sydney", "london", "paris", "berlin", 
                          "tokyo", "beijing", "toronto", "rome", "madrid", "dubai", "delhi"]:
            if city_marker in p1 and city_marker in p2:
                same_city = True
                break
        
        if same_city:
            # If in the same city, distance is likely 5-30 km
            # Use length of strings to vary it a bit for demo purposes
            return min(5 + (len(p1) + len(p2)) / 10, 30)
        else:
            # For unknown city pairs, assume medium distance journey
            # Vary between 100-500 km based on string length for demo variety
            return 100 + ((len(p1) + len(p2)) % 20) * 20
    
    def _extract_demo_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Extract data from a demo route."""
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        segments_data = []
        
        for leg in legs:
            distance = leg.get('distance', {}).get('value', 0) / 1000
            duration = leg.get('duration', {}).get('value', 0) / 3600
            
            total_distance += distance
            total_duration += duration
            
            # Add this leg as a segment
            segments_data.append({
                'distance': distance,
                'duration': duration
            })
            
            # Debug output
            print(f"Demo route leg: {leg.get('start_address', 'Unknown')} to {leg.get('end_address', 'Unknown')}")
            print(f"Distance: {distance:.1f} km, Duration: {duration:.2f} hours")
        
        return total_distance, total_duration, segments_data

    def get_distance_matrix(self, origins: List[str], destinations: List[str]) -> Dict[str, Any]:
        """
        Get distance and duration between origins and destinations using the Distance Matrix API.
        
        Args:
            origins: List of origin addresses
            destinations: List of destination addresses
            
        Returns:
            Distance matrix result from Google Maps API
        """
        if self.demo_mode:
            return self._get_demo_distance_matrix(origins, destinations)
            
        try:
            # Make the API call
            result = self.client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode="driving",
                language="en",
                units="metric"
            )
            
            return result
        except Exception as e:
            print(f"Error getting distance matrix: {str(e)}")
            return self._get_demo_distance_matrix(origins, destinations)
            
    def _get_demo_distance_matrix(self, origins: List[str], destinations: List[str]) -> Dict[str, Any]:
        """
        Create a demo distance matrix for when the API is unavailable.
        
        Args:
            origins: List of origin addresses
            destinations: List of destination addresses
            
        Returns:
            Simulated distance matrix
        """
        rows = []
        for origin in origins:
            elements = []
            for destination in destinations:
                # Use our improved distance estimation
                distance = self._estimate_distance_between_points(origin, destination)
                duration = distance / 60  # hours
                
                elements.append({
                    'distance': {'text': f"{distance:.1f} km", 'value': int(distance * 1000)},
                    'duration': {'text': f"{duration:.2f} hours", 'value': int(duration * 3600)},
                    'status': 'OK'
                })
            
            rows.append({'elements': elements})
            
        return {
            'origin_addresses': origins,
            'destination_addresses': destinations,
            'rows': rows,
            'status': 'OK'
        }