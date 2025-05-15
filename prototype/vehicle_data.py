"""
Database of vehicle types and their emission/cost factors.
"""
from models import VehicleType

# Initial vehicle types database with emission factors and costs
VEHICLE_TYPES = {
    "standard_diesel": VehicleType(
        id="standard_diesel",
        name="Standard Diesel Truck",
        co2e_per_km=0.900,  # kg CO2e per km
        cost_per_km=1.2,    # currency units per km
        max_payload=24.0,   # tons
        avg_speed=80.0      # km/h
    ),
    "eco_diesel": VehicleType(
        id="eco_diesel",
        name="Eco Diesel Truck (Euro 6)",
        co2e_per_km=0.765,  # 15% less emissions
        cost_per_km=1.32,   # 10% more expensive
        max_payload=22.0,   # tons
        avg_speed=75.0      # km/h
    ),
    "electric": VehicleType(
        id="electric",
        name="Electric Truck",
        co2e_per_km=0.450,  # 50% less emissions
        cost_per_km=1.45,   # 20% more expensive
        max_payload=18.0,   # tons
        avg_speed=70.0      # km/h
    ),
    "hybrid": VehicleType(
        id="hybrid",
        name="Hybrid Truck",
        co2e_per_km=0.675,  # 25% less emissions
        cost_per_km=1.35,   # 12% more expensive
        max_payload=20.0,   # tons
        avg_speed=75.0      # km/h
    ),
    "cng": VehicleType(
        id="cng",
        name="CNG Truck",
        co2e_per_km=0.720,  # 20% less emissions
        cost_per_km=1.25,   # 4% more expensive
        max_payload=21.0,   # tons
        avg_speed=78.0      # km/h
    ),
    "small_van_diesel": VehicleType(
        id="small_van_diesel",
        name="Small Diesel Van",
        co2e_per_km=0.250,  # kg CO2e per km
        cost_per_km=0.8,    # currency units per km
        max_payload=3.5,    # tons
        avg_speed=90.0      # km/h
    ),
    "small_van_electric": VehicleType(
        id="small_van_electric",
        name="Small Electric Van",
        co2e_per_km=0.125,  # 50% less emissions than diesel van
        cost_per_km=0.9,    # 12% more expensive
        max_payload=2.8,    # tons
        avg_speed=85.0      # km/h
    ),
    # Aviation transportation options
    "cargo_plane_medium": VehicleType(
        id="cargo_plane_medium",
        name="Medium Cargo Plane",
        co2e_per_km=12.0,   # kg CO2e per km - much higher due to aviation emissions
        cost_per_km=15.0,   # currency units per km - significantly more expensive
        max_payload=30.0,   # tons
        avg_speed=750.0     # km/h - much faster than ground transportation
    ),
    "cargo_plane_large": VehicleType(
        id="cargo_plane_large",
        name="Large Cargo Plane",
        co2e_per_km=22.0,   # kg CO2e per km
        cost_per_km=25.0,   # currency units per km
        max_payload=100.0,  # tons
        avg_speed=800.0     # km/h
    ),
    "cargo_plane_small": VehicleType(
        id="cargo_plane_small",
        name="Small Cargo Plane",
        co2e_per_km=8.0,    # kg CO2e per km
        cost_per_km=12.0,   # currency units per km
        max_payload=10.0,   # tons
        avg_speed=650.0     # km/h
    ),
    "sustainable_aviation": VehicleType(
        id="sustainable_aviation",
        name="Sustainable Aviation (SAF)",
        co2e_per_km=9.0,    # kg CO2e per km - 25% lower emissions with sustainable aviation fuel
        cost_per_km=18.0,   # currency units per km - more expensive due to SAF costs
        max_payload=30.0,   # tons
        avg_speed=750.0     # km/h
    )
}

def get_vehicle_by_id(vehicle_id: str) -> VehicleType:
    """Get a vehicle type by its ID."""
    if not vehicle_id or vehicle_id not in VEHICLE_TYPES:
        raise ValueError(f"Vehicle type '{vehicle_id}' not found")
    return VEHICLE_TYPES[vehicle_id]

def get_all_vehicles() -> dict:
    """Get all available vehicle types."""
    return VEHICLE_TYPES