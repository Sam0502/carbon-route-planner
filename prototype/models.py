"""
Data models for carbon-optimized logistics route planner.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

@dataclass
class VehicleType:
    """Vehicle type with associated emissions and cost factors."""
    id: str
    name: str
    co2e_per_km: float  # kg CO2e per km
    cost_per_km: float  # in currency units
    max_payload: float  # in tons
    avg_speed: float    # in km/h

@dataclass
class RouteSegment:
    """A segment of a route between two points."""
    origin: str
    destination: str
    distance: float     # in km
    duration: float     # in hours
    vehicle_type_id: str
    co2e: float         # kg CO2e
    cost: float         # in currency units
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata from AI prediction

@dataclass
class Route:
    """A complete route from origin to destination."""
    segments: List[RouteSegment] = field(default_factory=list)
    total_distance: float = 0.0
    total_duration: float = 0.0
    total_co2e: float = 0.0
    total_cost: float = 0.0
    waypoints: List[str] = field(default_factory=list)
    emission_factor: float = 0.0  # kg CO2e per km 
    cost_factor: float = 0.0      # currency per km
    max_payload: float = 0.0      # tons
    used_ai_prediction: bool = False  # Whether AI was used for predictions

    def add_segment(self, segment: RouteSegment) -> None:
        """Add a segment to the route and update totals."""
        self.segments.append(segment)
        self.total_distance += segment.distance
        self.total_duration += segment.duration
        self.total_co2e += segment.co2e
        self.total_cost += segment.cost
        
        # Add destination to waypoints if this is the first segment
        if not self.waypoints:
            self.waypoints.append(segment.origin)
        self.waypoints.append(segment.destination)
        
        # Track if AI prediction was used
        if segment.prediction_metadata:
            self.used_ai_prediction = True
        
    def set_vehicle_attributes(self, vehicle_type):
        """Set vehicle-specific attributes for transparency."""
        self.emission_factor = vehicle_type.co2e_per_km
        self.cost_factor = vehicle_type.cost_per_km
        self.max_payload = vehicle_type.max_payload

@dataclass
class ShipmentRequest:
    """User input for a shipment planning request."""
    origin: str
    destination: str
    intermediate_stops: List[str] = field(default_factory=list)
    weight_tons: Optional[float] = None
    volume_cbm: Optional[float] = None
    max_budget: float = float('inf')
    delivery_time_start: Optional[datetime] = None
    delivery_time_end: Optional[datetime] = None
    # Additional parameters for AI-enhanced prediction
    use_ai_prediction: bool = True
    terrain_factors: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    traffic_levels: List[float] = field(default_factory=list)