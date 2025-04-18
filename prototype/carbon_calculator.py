"""
Carbon footprint calculation module.
"""
from typing import Dict, List, Tuple, Optional
from models import RouteSegment, Route
from vehicle_data import get_vehicle_by_id, get_all_vehicles

def calculate_segment_footprint(
    origin: str, 
    destination: str, 
    distance: float, 
    duration: float, 
    vehicle_type_id: str, 
    weight_tons: float = None
) -> RouteSegment:
    """
    Calculate carbon footprint and cost for a route segment.
    
    Args:
        origin: Origin address
        destination: Destination address
        distance: Distance in km
        duration: Duration in hours
        vehicle_type_id: ID of the vehicle type to use
        weight_tons: Weight of payload in tons (optional)
        
    Returns:
        RouteSegment with calculated carbon footprint and cost
    """
    vehicle = get_vehicle_by_id(vehicle_type_id)
    
    # Base emissions calculation
    co2e = distance * vehicle.co2e_per_km
    
    # Adjust for payload if provided
    if weight_tons is not None and weight_tons > 0:
        # Simplified adjustment: emissions increase proportionally 
        # with payload up to max_payload
        payload_factor = min(weight_tons / vehicle.max_payload, 1.0)
        # 70% base emissions + 30% that scales with payload
        co2e = co2e * (0.7 + (0.3 * payload_factor))
    
    # Calculate cost
    cost = distance * vehicle.cost_per_km
    
    return RouteSegment(
        origin=origin,
        destination=destination,
        distance=distance,
        duration=duration,
        vehicle_type_id=vehicle_type_id,
        co2e=co2e,
        cost=cost
    )

def calculate_route_footprint(
    waypoints: List[str],
    segments_data: List[Dict],
    vehicle_type_id: str,
    weight_tons: float = None
) -> Route:
    """
    Calculate carbon footprint for an entire route.
    
    Args:
        waypoints: List of addresses in order (including origin and destination)
        segments_data: List of segment data from Maps API (distance, duration)
        vehicle_type_id: ID of the vehicle type to use
        weight_tons: Weight of payload in tons (optional)
        
    Returns:
        Route with calculated carbon footprint and cost for all segments
    """
    route = Route()
    
    for i, segment_data in enumerate(segments_data):
        if i >= len(waypoints) - 1:
            break
            
        origin = waypoints[i]
        destination = waypoints[i + 1]
        
        segment = calculate_segment_footprint(
            origin=origin,
            destination=destination,
            distance=segment_data['distance'],
            duration=segment_data['duration'],
            vehicle_type_id=vehicle_type_id,
            weight_tons=weight_tons
        )
        
        route.add_segment(segment)
    
    return route

def calculate_optimal_route(
    waypoints: List[str],
    segments_data: List[Dict],
    max_budget: float,
    weight_tons: float = None
) -> Tuple[Optional[Route], List[Route]]:
    """
    Calculate the optimal route by evaluating all vehicle types and selecting the one 
    that minimizes carbon footprint while staying within budget.
    
    Args:
        waypoints: List of addresses in order (including origin and destination)
        segments_data: List of segment data from Maps API
        max_budget: Maximum budget constraint in currency units
        weight_tons: Weight of payload in tons (optional)
        
    Returns:
        Tuple of (optimal_route, alternative_routes) where optimal_route is the 
        route with lowest carbon footprint within budget, and alternative_routes 
        are other options sorted by increasing carbon footprint
    """
    all_vehicles = get_all_vehicles()
    all_routes = []
    
    # Calculate routes for all vehicle types
    for vehicle_id, vehicle in all_vehicles.items():
        # Check if the payload exceeds vehicle capacity
        if weight_tons and weight_tons > vehicle.max_payload:
            continue  # Skip vehicles that can't handle the payload
            
        route = calculate_route_footprint(
            waypoints=waypoints,
            segments_data=segments_data,
            vehicle_type_id=vehicle_id,
            weight_tons=weight_tons
        )
        
        # Set vehicle emission and cost factors for transparency using the new method
        route.set_vehicle_attributes(vehicle)
        
        all_routes.append(route)
    
    # Sort routes by carbon footprint (lowest first)
    all_routes.sort(key=lambda r: r.total_co2e)
    
    # Filter routes within budget
    routes_within_budget = [r for r in all_routes if r.total_cost <= max_budget]
    
    if routes_within_budget:
        # Return the lowest carbon footprint route within budget, plus alternatives
        return routes_within_budget[0], routes_within_budget[1:] + [r for r in all_routes if r.total_cost > max_budget]
    else:
        # If no routes within budget, return None and all routes as alternatives
        return None, all_routes