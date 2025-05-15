"""
Route optimization module using Google OR-Tools.
"""
from typing import List, Dict, Optional, Tuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from models import Route, ShipmentRequest
from maps_client import MapsClient
from carbon_calculator import calculate_route_footprint, calculate_optimal_route
from vehicle_data import get_all_vehicles, get_vehicle_by_id

class RouteOptimizer:
    """Optimizes routes to minimize carbon footprint within budget constraints."""
    
    def __init__(self, maps_client: MapsClient):
        """Initialize the route optimizer with a maps client."""
        self.maps_client = maps_client
        self.vehicle_types = get_all_vehicles()
    
    def optimize(self, request: ShipmentRequest) -> Tuple[Optional[Route], List[Route]]:
        """
        Find the optimal route that minimizes carbon footprint within budget constraints.
        
        Args:
            request: A shipment request with origin, destination, and constraints
            
        Returns:
            Tuple of (optimal_route, alternative_routes)
        """
        all_waypoints = [request.origin]
        all_waypoints.extend(request.intermediate_stops)
        all_waypoints.append(request.destination)
        
        # Get route from Maps API
        maps_route = self.maps_client.get_route(
            origin=all_waypoints[0],
            destination=all_waypoints[-1],
            waypoints=all_waypoints[1:-1] if len(all_waypoints) > 2 else None
        )
        
        # Extract distance and duration data
        total_distance, total_duration, segments_data = self.maps_client.extract_route_data(maps_route)
        
        # If a specific vehicle is requested, calculate route only for that vehicle
        if request.vehicle_type_id:
            # Calculate route with the manually selected vehicle
            selected_route = calculate_route_footprint(
                waypoints=all_waypoints,
                segments_data=segments_data,
                vehicle_type_id=request.vehicle_type_id,
                weight_tons=request.weight_tons,
                use_ai_prediction=request.use_ai_prediction,
                terrain_factors=request.terrain_factors,
                temperatures=request.temperatures,
                traffic_levels=request.traffic_levels
            )
            
            # Set vehicle attributes
            vehicle = get_vehicle_by_id(request.vehicle_type_id)
            selected_route.set_vehicle_attributes(vehicle)
            
            # Check if route is within budget
            if selected_route.total_cost <= request.max_budget:
                return selected_route, []  # No alternatives when specific vehicle is selected
            else:
                # Return the manually selected route with a warning about budget
                return None, [selected_route]  # Return as an alternative that exceeds budget
        
        # If no specific vehicle selected, calculate the optimal route across all vehicle types
        return calculate_optimal_route(
            waypoints=all_waypoints,
            segments_data=segments_data,
            max_budget=request.max_budget,
            weight_tons=request.weight_tons,
            use_ai_prediction=request.use_ai_prediction,
            terrain_factors=request.terrain_factors,
            temperatures=request.temperatures,
            traffic_levels=request.traffic_levels
        )
    
    def _calculate_vehicle_route(
        self, 
        waypoints: List[str], 
        vehicle_type_id: str,
        weight_tons: Optional[float] = None,
        use_ai_prediction: bool = True,
        terrain_factors: List[float] = None,
        temperatures: List[float] = None,
        traffic_levels: List[float] = None
    ) -> Route:
        """
        Calculate route for a specific vehicle type.
        
        For MVP, this just calculates a simple A→B route.
        Future version would use OR-Tools for more complex optimization.
        """
        # Get route from Maps API
        maps_route = self.maps_client.get_route(
            origin=waypoints[0],
            destination=waypoints[-1],
            waypoints=waypoints[1:-1] if len(waypoints) > 2 else None
        )
        
        # Extract distance and duration data
        total_distance, total_duration, segments_data = self.maps_client.extract_route_data(maps_route)
        
        # Calculate carbon footprint for this route with AI prediction if enabled
        return calculate_route_footprint(
            waypoints=waypoints,
            segments_data=segments_data,
            vehicle_type_id=vehicle_type_id,
            weight_tons=weight_tons,
            use_ai_prediction=use_ai_prediction,
            terrain_factors=terrain_factors,
            temperatures=temperatures,
            traffic_levels=traffic_levels
        )
    
    def solve_vehicle_routing_problem(
        self,
        request: ShipmentRequest,
        distance_matrix: List[List[int]],
        vehicle_types: List[str]
    ) -> Optional[Route]:
        """
        Solve a vehicle routing problem using OR-Tools.
        
        This is a placeholder for future enhancements. In an advanced version,
        this would handle more complex scenarios with multiple stops and 
        mode changes at hubs.
        
        Args:
            request: A shipment request
            distance_matrix: Matrix of distances between all points
            vehicle_types: List of vehicle type IDs to consider
            
        Returns:
            An optimized route if found
        """
        # Create the routing model
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix), 1, 0
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Create and register a transit callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        # Return solution if found
        if solution:
            # Process the solution and create a Route object
            route = Route()
            
            index = routing.Start(0)  # Start at depot
            route_waypoints = []
            route_waypoints.append(request.origin)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                next_node_index = manager.IndexToNode(
                    solution.Value(routing.NextVar(index)))
                
                # Skip the first node (already added as origin)
                if node_index != 0:
                    if node_index < len(request.intermediate_stops) + 1:
                        # Add intermediate stop
                        if node_index == len(request.intermediate_stops):
                            waypoint = request.destination
                        else:
                            waypoint = request.intermediate_stops[node_index - 1]
                        route_waypoints.append(waypoint)
                
                # Move to next
                index = solution.Value(routing.NextVar(index))
            
            # Ensure destination is included
            if route_waypoints[-1] != request.destination:
                route_waypoints.append(request.destination)
            
            # Calculate the route with the first vehicle type (can be enhanced to try multiple)
            vehicle_type_id = vehicle_types[0] if vehicle_types else "standard_diesel"
            
            # Create segments using the optimized route
            for i in range(len(route_waypoints) - 1):
                orig = route_waypoints[i]
                dest = route_waypoints[i + 1]
                
                # Get distance and time between these points (simplified for MVP)
                distance = distance_matrix[i][i + 1] / 1000  # Convert to km
                duration = distance / self.vehicle_types[vehicle_type_id].avg_speed  # Estimated time
                
                # Create the segment with AI prediction if enabled
                from carbon_calculator import calculate_segment_footprint
                
                # Get terrain factor for this segment if available
                terrain_factor = request.terrain_factors[i] if request.terrain_factors and i < len(request.terrain_factors) else 1.0
                temperature = request.temperatures[i] if request.temperatures and i < len(request.temperatures) else 20.0
                traffic_level = request.traffic_levels[i] if request.traffic_levels and i < len(request.traffic_levels) else 0.5
                
                segment = calculate_segment_footprint(
                    origin=orig,
                    destination=dest,
                    distance=distance,
                    duration=duration,
                    vehicle_type_id=vehicle_type_id,
                    weight_tons=request.weight_tons,
                    use_ai_prediction=request.use_ai_prediction,
                    terrain_factor=terrain_factor,
                    temperature=temperature,
                    traffic_level=traffic_level
                )
                
                route.add_segment(segment)
                
            return route
        else:
            return None