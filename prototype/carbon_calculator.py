"""
Carbon footprint calculation module.
"""
from typing import Dict, List, Tuple
from models import RouteSegment, Route
from vehicle_data import get_vehicle_by_id

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