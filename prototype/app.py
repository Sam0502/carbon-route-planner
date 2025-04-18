"""
Carbon-optimized logistics route planner - Web Interface
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from models import ShipmentRequest
from maps_client import MapsClient
from optimizer import RouteOptimizer
from vehicle_data import get_all_vehicles, get_vehicle_by_id

# Page configuration
st.set_page_config(
    page_title="Carbon-Optimized Logistics Planner",
    page_icon="🌍",
    layout="wide"
)

# Initialize clients
@st.cache_resource
def get_maps_client():
    return MapsClient()

@st.cache_resource
def get_route_optimizer(_maps_client):
    return RouteOptimizer(_maps_client)

maps_client = get_maps_client()
route_optimizer = get_route_optimizer(maps_client)

# Title
st.title("🌍 Carbon-Optimized Logistics Route Planner")
st.markdown("""
Plan shipments by selecting routes and transport modes that minimize carbon footprint 
while respecting budget constraints.
""")

# Sidebar for inputs
st.sidebar.header("Shipment Details")

# Origin and destination inputs
origin = st.sidebar.text_input("Origin Address", "New York, NY")
destination = st.sidebar.text_input("Destination Address", "Boston, MA")

# Intermediate stops
with st.sidebar.expander("Add Intermediate Stops", expanded=False):
    num_stops = st.number_input("Number of stops", 0, 5, 0)
    intermediate_stops = []
    for i in range(num_stops):
        stop = st.text_input(f"Stop {i+1}", "")
        if stop:
            intermediate_stops.append(stop)

# Payload characteristics
with st.sidebar.expander("Payload Characteristics", expanded=False):
    weight_tons = st.number_input("Weight (tons)", 0.0, 30.0, 5.0, 0.1)
    volume_cbm = st.number_input("Volume (cubic meters)", 0.0, 100.0, 10.0, 0.5)

# Budget constraint
max_budget = st.sidebar.number_input("Maximum Budget (€/$/₽)", 0.0, 10000.0, 1000.0, 50.0)

# Time constraint
with st.sidebar.expander("Delivery Time Window", expanded=False):
    include_time_constraint = st.checkbox("Set Delivery Time Window", False)
    
    if include_time_constraint:
        today = datetime.now().date()
        delivery_date = st.date_input("Delivery Date", today + timedelta(days=3))
        
        delivery_time_start = st.time_input("Earliest Delivery Time", datetime.strptime("08:00", "%H:%M").time())
        delivery_time_end = st.time_input("Latest Delivery Time", datetime.strptime("18:00", "%H:%M").time())
        
        delivery_time_start_dt = datetime.combine(delivery_date, delivery_time_start)
        delivery_time_end_dt = datetime.combine(delivery_date, delivery_time_end)
    else:
        delivery_time_start_dt = None
        delivery_time_end_dt = None

# Execute optimization
if st.sidebar.button("Find Optimal Route"):
    # Create shipment request
    request = ShipmentRequest(
        origin=origin,
        destination=destination,
        intermediate_stops=intermediate_stops,
        weight_tons=weight_tons,
        volume_cbm=volume_cbm,
        max_budget=max_budget,
        delivery_time_start=delivery_time_start_dt,
        delivery_time_end=delivery_time_end_dt
    )
    
    # Show spinner while processing
    with st.spinner("Calculating optimal routes..."):
        optimal_route, alternative_routes = route_optimizer.optimize(request)
    
    # Display results
    if optimal_route:
        st.success("Optimal route found!")
        
        # Main columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Optimal Route")
            
            # Map placeholder
            st.markdown("#### Route Map")
            st.info(
                "This is a placeholder for the route map. In a production version, "
                "this would display an interactive map using Google Maps API."
            )
            
            # Route summary
            st.markdown("#### Route Summary")
            
            # Get vehicle for optimal route
            vehicle_id = optimal_route.segments[0].vehicle_type_id if optimal_route.segments else None
            vehicle = get_vehicle_by_id(vehicle_id) if vehicle_id else None
            vehicle_name = vehicle.name if vehicle else "Unknown"
            
            # Format data
            optimal_route_data = {
                "Total Carbon Footprint": f"{optimal_route.total_co2e:.2f} kg CO₂e",
                "Total Cost": f"€{optimal_route.total_cost:.2f}",
                "Total Distance": f"{optimal_route.total_distance:.1f} km",
                "Total Duration": f"{optimal_route.total_duration:.2f} hours",
                "Vehicle Type": vehicle_name
            }
            
            # Display as a table
            st.table(pd.DataFrame(list(optimal_route_data.items()), columns=["Metric", "Value"]))
            
            # Route breakdown
            st.markdown("#### Route Breakdown")
            
            if optimal_route.segments:
                # Prepare segment data
                segments_data = []
                
                for segment in optimal_route.segments:
                    vehicle = get_vehicle_by_id(segment.vehicle_type_id)
                    segments_data.append({
                        "Origin": segment.origin,
                        "Destination": segment.destination,
                        "Distance (km)": f"{segment.distance:.1f}",
                        "Duration (hours)": f"{segment.duration:.2f}",
                        "Vehicle": vehicle.name,
                        "CO₂e (kg)": f"{segment.co2e:.2f}",
                        "Cost (€)": f"{segment.cost:.2f}"
                    })
                
                # Display as a table
                st.table(pd.DataFrame(segments_data))
            
        with col2:
            # Carbon footprint visualization
            st.markdown("#### Carbon Footprint Breakdown")
            
            # Prepare data for chart
            if optimal_route.segments:
                segment_labels = [f"{s.origin[:10]}→{s.destination[:10]}" for s in optimal_route.segments]
                segment_co2e = [s.co2e for s in optimal_route.segments]
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(segment_co2e, labels=segment_labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                st.pyplot(fig)
            
            # Alternative routes
            st.markdown("#### Alternative Routes")
            
            if alternative_routes:
                alt_data = []
                for i, route in enumerate(alternative_routes[:3]):  # Show up to 3 alternatives
                    vehicle_id = route.segments[0].vehicle_type_id if route.segments else None
                    vehicle = get_vehicle_by_id(vehicle_id) if vehicle_id else None
                    
                    alt_data.append({
                        "Option": i + 1,
                        "Vehicle": vehicle.name if vehicle else "Unknown",
                        "CO₂e (kg)": f"{route.total_co2e:.2f}",
                        "Cost (€)": f"{route.total_cost:.2f}",
                        "Duration (hours)": f"{route.total_duration:.2f}"
                    })
                
                st.table(pd.DataFrame(alt_data))
            else:
                st.info("No alternative routes found within budget constraints.")
    else:
        st.error(
            "No routes found within budget constraints. "
            "Please increase your budget or modify your route."
        )

# If no route has been calculated yet, show information
if 'optimal_route' not in locals():
    st.info(
        "Enter your shipment details and click 'Find Optimal Route' "
        "to calculate the most carbon-efficient route within your budget."
    )
    
    # Display example
    with st.expander("How it works"):
        st.markdown("""
        ### How the Carbon Optimization Works
        
        1. **Input your shipment details** including origin, destination, and constraints
        2. **Our system calculates multiple route options** using different vehicle types
        3. **Carbon footprint is calculated** for each option using emission factors
        4. **The route with the lowest carbon footprint** is selected, while staying within your budget
        
        ### Vehicle Emission Factors
        
        Different vehicles have different emission profiles:
        
        | Vehicle Type | CO₂e per km | Cost per km | Max Payload |
        |--------------|-------------|-------------|-------------|
        | Standard Diesel Truck | 0.900 kg | €1.20 | 24.0 tons |
        | Eco Diesel Truck (Euro 6) | 0.765 kg | €1.32 | 22.0 tons |
        | Electric Truck | 0.450 kg | €1.45 | 18.0 tons |
        | Hybrid Truck | 0.675 kg | €1.35 | 20.0 tons |
        | CNG Truck | 0.720 kg | €1.25 | 21.0 tons |
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Carbon-Optimized Logistics Planner")

if __name__ == "__main__":
    # This code will only run when the script is executed directly
    pass