"""
Data models for carbon-optimized logistics route planner.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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

@dataclass
class Route:
    """A complete route from origin to destination."""
    segments: List[RouteSegment] = field(default_factory=list)
    total_distance: float = 0.0
    total_duration: float = 0.0
    total_co2e: float = 0.0
    total_cost: float = 0.0
    waypoints: List[str] = field(default_factory=list)

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