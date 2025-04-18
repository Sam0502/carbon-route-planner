"""
Route optimization module using Google OR-Tools.
"""
from typing import List, Dict, Optional, Tuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from models import Route, ShipmentRequest
from maps_client import MapsClient
from carbon_calculator import calculate_route_footprint
from vehicle_data import get_all_vehicles

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
        
        # For each vehicle type, calculate a route
        all_routes = []
        for vehicle_id, vehicle in self.vehicle_types.items():
            route = self._calculate_vehicle_route(
                all_waypoints, 
                vehicle_id, 
                request.weight_tons
            )
            
            # Check if route meets budget constraint
            if route.total_cost <= request.max_budget:
                all_routes.append(route)
        
        # Sort routes by carbon footprint
        all_routes.sort(key=lambda r: r.total_co2e)
        
        # Return the best route and alternatives
        if all_routes:
            return all_routes[0], all_routes[1:]
        else:
            return None, []
    
    def _calculate_vehicle_route(
        self, 
        waypoints: List[str], 
        vehicle_type_id: str,
        weight_tons: Optional[float] = None
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
        
        # Calculate carbon footprint for this route
        return calculate_route_footprint(
            waypoints=waypoints,
            segments_data=segments_data,
            vehicle_type_id=vehicle_type_id,
            weight_tons=weight_tons
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
                
                # Create the segment
                from carbon_calculator import calculate_segment_footprint
                segment = calculate_segment_footprint(
                    origin=orig,
                    destination=dest,
                    distance=distance,
                    duration=duration,
                    vehicle_type_id=vehicle_type_id,
                    weight_tons=request.weight_tons
                )
                
                route.add_segment(segment)
                
            return route
        else:
            return None