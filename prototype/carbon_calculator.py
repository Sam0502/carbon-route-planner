"""
Carbon footprint calculation module.
"""
from typing import Dict, List, Tuple, Optional, Any
from models import RouteSegment, Route
from vehicle_data import get_vehicle_by_id, get_all_vehicles
from ai_predictor import get_predictor

def calculate_segment_footprint(
    origin: str, 
    destination: str, 
    distance: float, 
    duration: float, 
    vehicle_type_id: str, 
    weight_tons: float = None,
    use_ai_prediction: bool = True,
    terrain_factor: float = 1.0,
    temperature: float = 20.0,
    traffic_level: float = 0.5
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
        use_ai_prediction: Whether to use AI prediction when available
        terrain_factor: Factor representing terrain difficulty (1.0 = flat, >1 = hilly)
        temperature: Temperature in Celsius (affects fuel efficiency)
        traffic_level: Traffic congestion level (0-1, where 1 is most congested)
        
    Returns:
        RouteSegment with calculated carbon footprint and cost
    """
    vehicle = get_vehicle_by_id(vehicle_type_id)
    
    # Calculate average speed for this segment
    avg_speed = distance / duration if duration > 0 else vehicle.avg_speed
    
    # Default co2e calculation using the standard factor-based approach
    co2e = distance * vehicle.co2e_per_km
    
    # Default payload adjustment if needed
    if weight_tons is not None and weight_tons > 0:
        # Simplified adjustment: emissions increase proportionally 
        # with payload up to max_payload
        payload_factor = min(weight_tons / vehicle.max_payload, 1.0)
        # 70% base emissions + 30% that scales with payload
        co2e = co2e * (0.7 + (0.3 * payload_factor))
    
    # Enhanced calculation using AI prediction if available and enabled
    prediction_metadata = None
    if use_ai_prediction:
        try:
            # Get the AI predictor singleton
            predictor = get_predictor()
            
            # Request a prediction based on all available factors
            predicted_co2e, metadata = predictor.predict_co2e(
                vehicle_type_id=vehicle_type_id,
                distance=distance,
                avg_speed=avg_speed,
                weight_tons=weight_tons or 0,
                terrain_factor=terrain_factor,
                temperature=temperature,
                traffic_level=traffic_level
            )
            
            # Use the AI prediction instead of the factor-based calculation
            co2e = predicted_co2e
            prediction_metadata = metadata
            
        except Exception as e:
            # If AI prediction fails, we already have a fallback calculation
            print(f"AI prediction failed, using factor-based calculation: {str(e)}")
    
    # Calculate cost (not affected by AI prediction)
    cost = distance * vehicle.cost_per_km
    
    # Create the segment with the calculated values
    segment = RouteSegment(
        origin=origin,
        destination=destination,
        distance=distance,
        duration=duration,
        vehicle_type_id=vehicle_type_id,
        co2e=co2e,
        cost=cost
    )
    
    # Store prediction metadata if available
    if prediction_metadata:
        segment.prediction_metadata = prediction_metadata
    
    return segment

def calculate_route_footprint(
    waypoints: List[str],
    segments_data: List[Dict],
    vehicle_type_id: str,
    weight_tons: float = None,
    use_ai_prediction: bool = True,
    terrain_factors: List[float] = None,
    temperatures: List[float] = None,
    traffic_levels: List[float] = None
) -> Route:
    """
    Calculate carbon footprint for an entire route.
    
    Args:
        waypoints: List of addresses in order (including origin and destination)
        segments_data: List of segment data from Maps API (distance, duration)
        vehicle_type_id: ID of the vehicle type to use
        weight_tons: Weight of payload in tons (optional)
        use_ai_prediction: Whether to use AI prediction when available
        terrain_factors: List of terrain factors for each segment (optional)
        temperatures: List of temperatures for each segment (optional)
        traffic_levels: List of traffic congestion levels for each segment (optional)
        
    Returns:
        Route with calculated carbon footprint and cost for all segments
    """
    route = Route()
    
    # If batch prediction would be more efficient, collect all segment data first
    ai_prediction_batch = []
    
    for i, segment_data in enumerate(segments_data):
        if i >= len(waypoints) - 1:
            break
            
        origin = waypoints[i]
        destination = waypoints[i + 1]
        
        # Get segment-specific factors if provided, otherwise use defaults
        terrain_factor = terrain_factors[i] if terrain_factors and i < len(terrain_factors) else 1.0
        temperature = temperatures[i] if temperatures and i < len(temperatures) else 20.0
        traffic_level = traffic_levels[i] if traffic_levels and i < len(traffic_levels) else 0.5
        
        # For batch predictions, collect all data first
        if use_ai_prediction:
            # Calculate average speed for prediction
            distance = segment_data['distance']
            duration = segment_data['duration']
            avg_speed = distance / duration if duration > 0 else get_vehicle_by_id(vehicle_type_id).avg_speed
            
            ai_prediction_batch.append({
                "vehicle_type_id": vehicle_type_id,
                "distance": distance,
                "avg_speed": avg_speed,
                "weight_tons": weight_tons or 0,
                "terrain_factor": terrain_factor,
                "temperature": temperature,
                "traffic_level": traffic_level,
                "segment_index": i,
                "origin": origin,
                "destination": destination
            })
        
    # Use batch prediction if available and enabled
    if use_ai_prediction and len(ai_prediction_batch) > 1:
        try:
            predictor = get_predictor()
            batch_results = predictor.batch_predict(ai_prediction_batch)
            
            # Now create segments using the batch predictions
            for i, (predicted_co2e, metadata) in enumerate(batch_results):
                segment_info = ai_prediction_batch[i]
                
                # Calculate cost (not affected by prediction)
                vehicle = get_vehicle_by_id(vehicle_type_id)
                cost = segment_info["distance"] * vehicle.cost_per_km
                
                segment = RouteSegment(
                    origin=segment_info["origin"],
                    destination=segment_info["destination"],
                    distance=segment_info["distance"],
                    duration=segments_data[segment_info["segment_index"]]["duration"],
                    vehicle_type_id=vehicle_type_id,
                    co2e=predicted_co2e,
                    cost=cost
                )
                
                # Store prediction metadata
                segment.prediction_metadata = metadata
                
                # Add to route
                route.add_segment(segment)
                
            return route
            
        except Exception as e:
            print(f"Batch prediction failed, falling back to individual calculations: {str(e)}")
            # Continue with regular calculation below if batch fails
    
    # If batch prediction was not used or failed, calculate segments individually
    for i, segment_data in enumerate(segments_data):
        if i >= len(waypoints) - 1:
            break
            
        origin = waypoints[i]
        destination = waypoints[i + 1]
        
        # Get segment-specific factors if provided
        terrain_factor = terrain_factors[i] if terrain_factors and i < len(terrain_factors) else 1.0
        temperature = temperatures[i] if temperatures and i < len(temperatures) else 20.0
        traffic_level = traffic_levels[i] if traffic_levels and i < len(traffic_levels) else 0.5
        
        segment = calculate_segment_footprint(
            origin=origin,
            destination=destination,
            distance=segment_data['distance'],
            duration=segment_data['duration'],
            vehicle_type_id=vehicle_type_id,
            weight_tons=weight_tons,
            use_ai_prediction=use_ai_prediction,
            terrain_factor=terrain_factor,
            temperature=temperature,
            traffic_level=traffic_level
        )
        
        route.add_segment(segment)
    
    return route

def calculate_optimal_route(
    waypoints: List[str],
    segments_data: List[Dict],
    max_budget: float,
    weight_tons: float = None,
    use_ai_prediction: bool = True,
    terrain_factors: List[float] = None,
    temperatures: List[float] = None,
    traffic_levels: List[float] = None
) -> Tuple[Optional[Route], List[Route]]:
    """
    Calculate the optimal route by evaluating all vehicle types and selecting the one 
    that minimizes carbon footprint while staying within budget.
    
    Args:
        waypoints: List of addresses in order (including origin and destination)
        segments_data: List of segment data from Maps API
        max_budget: Maximum budget constraint in currency units
        weight_tons: Weight of payload in tons (optional)
        use_ai_prediction: Whether to use AI prediction when available
        terrain_factors: List of terrain factors for each segment (optional)
        temperatures: List of temperatures for each segment (optional)
        traffic_levels: List of traffic congestion levels for each segment (optional)
        
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
            weight_tons=weight_tons,
            use_ai_prediction=use_ai_prediction,
            terrain_factors=terrain_factors,
            temperatures=temperatures,
            traffic_levels=traffic_levels
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