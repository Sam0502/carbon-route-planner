"""
Carbon-optimized logistics route planner - Web Interface with data transparency
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json
import uuid
import os
from models import ShipmentRequest
from maps_client import MapsClient
from optimizer import RouteOptimizer
from vehicle_data import get_all_vehicles, get_vehicle_by_id
from ai_predictor import get_predictor
from carbon_calculator import calculate_route_footprint

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

# Initialize AI predictor (will run in fallback mode if no API key)
ai_predictor = get_predictor()

# Initialize a session history for data transparency
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

# Add page navigation
page = st.sidebar.radio("Navigation", ["Route Planner", "Carbon Reports", "Data Transparency"])

if page == "Route Planner":
    # Title
    st.title("🌍 Carbon-Optimized Logistics Route Planner")
    st.markdown("""
    Plan shipments by selecting routes and transport modes that minimize carbon footprint 
    while respecting budget constraints.
    """)

    # Sidebar for inputs
    st.sidebar.header("Shipment Details")

    # Origin and destination inputs
    st.sidebar.markdown("### Origin and Destination")

    origin = st.sidebar.text_input("Origin Address", "New York, NY", 
                                  help="Enter a complete address with city and country/state")

    # Add address validation for interactive feedback if the method is available
    if origin and maps_client and hasattr(maps_client, 'validate_address'):
        with st.sidebar:
            with st.spinner("Validating origin address..."):
                origin_validation = maps_client.validate_address(origin)
                if origin_validation['valid']:
                    if origin != origin_validation['formatted_address']:
                        st.success(f"Validated address: {origin_validation['formatted_address']}")
                        if st.button("Use validated origin address"):
                            origin = origin_validation['formatted_address']
                else:
                    st.warning("Could not validate the origin address. Please check for typos.")
                
                # Show suggestions if available
                if origin_validation.get('suggestions'):
                    with st.expander("Did you mean one of these addresses?"):
                        for i, suggestion in enumerate(origin_validation['suggestions'][:3]):
                            if st.button(f"Use: {suggestion}", key=f"origin_sugg_{i}"):
                                origin = suggestion

    destination = st.sidebar.text_input("Destination Address", "Boston, MA",
                                       help="Enter a complete address with city and country/state")

    # Add address validation for interactive feedback
    if destination and maps_client and not maps_client.demo_mode:
        with st.sidebar:
            with st.spinner("Validating destination address..."):
                dest_validation = maps_client.validate_address(destination)
                if dest_validation['valid']:
                    if destination != dest_validation['formatted_address']:
                        st.success(f"Validated address: {dest_validation['formatted_address']}")
                        if st.button("Use validated destination address"):
                            destination = dest_validation['formatted_address']
                else:
                    st.warning("Could not validate the destination address. Please check for typos.")
                
                # Show suggestions if available
                if dest_validation.get('suggestions'):
                    with st.expander("Did you mean one of these addresses?"):
                        for i, suggestion in enumerate(dest_validation['suggestions'][:3]):
                            if st.button(f"Use: {suggestion}", key=f"dest_sugg_{i}"):
                                destination = suggestion

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
    max_budget = st.sidebar.number_input("Maximum Budget (€)", 0.0, 10000.0, 1000.0, 50.0)

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

    # AI-Enhanced CO2e calculation options
    with st.sidebar.expander("AI-Enhanced CO2e Calculation", expanded=True):
        use_ai_prediction = st.checkbox("Use AI for more accurate CO2e predictions", True,
                                        help="Uses AI to predict emissions considering multiple dynamic factors")
        
        # Only show these options if AI prediction is enabled
        if use_ai_prediction:
            # Add terrain factors for each segment
            st.markdown("#### Terrain Factors")
            st.caption("Terrain difficulty affects fuel consumption (1.0 = flat, >1 = hilly)")
            
            # Create terrain factors for main route segment
            terrain_factors = [1.0]  # Default value for the main segment
            
            # A simple UI to set terrain factor for the main route
            if not intermediate_stops:
                terrain_main = st.slider("Main Route", 1.0, 2.0, 1.0, 0.1, 
                                       help="1.0 = flat terrain, 2.0 = very mountainous")
                terrain_factors = [terrain_main]
            else:
                # If there are intermediate stops, offer to set terrain for each segment
                for i in range(len(intermediate_stops) + 1):
                    if i == 0:
                        segment_name = f"{origin.split(',')[0]} → {intermediate_stops[0].split(',')[0]}"
                    elif i == len(intermediate_stops):
                        segment_name = f"{intermediate_stops[-1].split(',')[0]} → {destination.split(',')[0]}"
                    else:
                        segment_name = f"{intermediate_stops[i-1].split(',')[0]} → {intermediate_stops[i].split(',')[0]}"
                    
                    terrain = st.slider(segment_name, 1.0, 2.0, 1.0, 0.1, 
                                       help="1.0 = flat terrain, 2.0 = very mountainous")
                    terrain_factors.append(terrain)
            
            # Temperature factors
            st.markdown("#### Average Temperatures (°C)")
            
            temperature_factors = [20.0]  # Default value
            
            if not intermediate_stops:
                temp_main = st.slider("Main Route Temperature", -10.0, 40.0, 20.0, 1.0, 
                                    help="Temperature affects fuel efficiency")
                temperature_factors = [temp_main]
            else:
                # If there are intermediate stops, offer to set temperature for each segment
                for i in range(len(intermediate_stops) + 1):
                    if i == 0:
                        segment_name = f"{origin.split(',')[0]} → {intermediate_stops[0].split(',')[0]}"
                    elif i == len(intermediate_stops):
                        segment_name = f"{intermediate_stops[-1].split(',')[0]} → {destination.split(',')[0]}"
                    else:
                        segment_name = f"{intermediate_stops[i-1].split(',')[0]} → {intermediate_stops[i].split(',')[0]}"
                    
                    temp = st.slider(f"{segment_name} Temperature", -10.0, 40.0, 20.0, 1.0)
                    temperature_factors.append(temp)
            
            # Traffic congestion
            st.markdown("#### Traffic Congestion")
            
            traffic_levels = [0.5]  # Default value (0-1 scale)
            
            if not intermediate_stops:
                traffic_main = st.slider("Main Route Traffic", 0.0, 1.0, 0.5, 0.1, 
                                       help="0 = no traffic, 1 = heavy congestion")
                traffic_levels = [traffic_main]
            else:
                # If there are intermediate stops, offer to set traffic for each segment
                for i in range(len(intermediate_stops) + 1):
                    if i == 0:
                        segment_name = f"{origin.split(',')[0]} → {intermediate_stops[0].split(',')[0]}"
                    elif i == len(intermediate_stops):
                        segment_name = f"{intermediate_stops[-1].split(',')[0]} → {destination.split(',')[0]}"
                    else:
                        segment_name = f"{intermediate_stops[i-1].split(',')[0]} → {intermediate_stops[i].split(',')[0]}"
                    
                    traffic = st.slider(f"{segment_name} Traffic", 0.0, 1.0, 0.5, 0.1)
                    traffic_levels.append(traffic)
        else:
            # Empty lists when AI is disabled
            terrain_factors = []
            temperature_factors = []
            traffic_levels = []

    # Execute optimization
    if st.sidebar.button("Find Optimal Route"):
        # Create shipment request with AI prediction settings
        request = ShipmentRequest(
            origin=origin,
            destination=destination,
            intermediate_stops=intermediate_stops,
            weight_tons=weight_tons,
            volume_cbm=volume_cbm,
            max_budget=max_budget,
            delivery_time_start=delivery_time_start_dt,
            delivery_time_end=delivery_time_end_dt,
            # AI prediction parameters
            use_ai_prediction=use_ai_prediction,
            terrain_factors=terrain_factors,
            temperatures=temperature_factors,
            traffic_levels=traffic_levels
        )
        
        # Show spinner while processing
        with st.spinner("Calculating optimal routes..."):
            optimal_route, alternative_routes = route_optimizer.optimize(request)
            
            # If AI prediction is enabled, also calculate with standard method for comparison
            standard_prediction_route = None
            if use_ai_prediction and optimal_route:
                # Extract the route data
                maps_client = route_optimizer.maps_client
                all_waypoints = [request.origin]
                all_waypoints.extend(request.intermediate_stops)
                all_waypoints.append(request.destination)
                
                # Get the same route data but calculate with standard method
                if hasattr(maps_client, 'last_route_data'):
                    segments_data = maps_client.last_route_data
                    
                    # Calculate using standard method
                    standard_prediction_route = calculate_route_footprint(
                        waypoints=all_waypoints,
                        segments_data=segments_data,
                        vehicle_type_id=optimal_route.segments[0].vehicle_type_id if optimal_route.segments else "standard_diesel",
                        weight_tons=request.weight_tons,
                        use_ai_prediction=False  # Force standard calculation
                    )
                    
                    # Set vehicle attributes
                    if standard_prediction_route and standard_prediction_route.segments:
                        vehicle = get_vehicle_by_id(standard_prediction_route.segments[0].vehicle_type_id)
                        standard_prediction_route.set_vehicle_attributes(vehicle)
            
            # Store calculation data for transparency
            if optimal_route:
                calculation_record = {
                    "id": str(uuid.uuid4())[:8],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "route": {
                        "origin": origin,
                        "destination": destination,
                        "intermediate_stops": intermediate_stops,
                        "total_distance": optimal_route.total_distance,
                        "total_co2e": optimal_route.total_co2e,
                        "vehicle_type": optimal_route.segments[0].vehicle_type_id if optimal_route.segments else None
                    },
                    "parameters": {
                        "weight_tons": weight_tons,
                        "use_ai_prediction": use_ai_prediction,
                        "terrain_factors": terrain_factors if use_ai_prediction else [],
                        "temperatures": temperature_factors if use_ai_prediction else [],
                        "traffic_levels": traffic_levels if use_ai_prediction else []
                    },
                    "ai_metadata": {}
                }
                
                # Add AI metadata if available
                if optimal_route.used_ai_prediction and optimal_route.segments and hasattr(optimal_route.segments[0], 'prediction_metadata'):
                    segment = optimal_route.segments[0]
                    if segment.prediction_metadata:
                        # Clean up the metadata to ensure it's serializable
                        metadata = segment.prediction_metadata.copy()
                        if "input_parameters" in metadata:
                            del metadata["input_parameters"]
                        calculation_record["ai_metadata"] = metadata
                
                # Append to history
                st.session_state.calculation_history.append(calculation_record)
                
                # Optional: Save to file for permanent storage
                try:
                    history_dir = os.path.join(os.path.dirname(__file__), "calculation_history")
                    if not os.path.exists(history_dir):
                        os.makedirs(history_dir)
                    
                    history_file = os.path.join(history_dir, f"calculation_{calculation_record['id']}.json")
                    with open(history_file, 'w') as f:
                        json.dump(calculation_record, f, indent=2)
                except Exception as e:
                    st.warning(f"Could not save calculation history to file: {str(e)}")
        
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
                    "Vehicle Type": vehicle_name,
                    "Emission Factor": f"{optimal_route.emission_factor:.3f} kg CO₂e/km",
                    "Cost Factor": f"€{optimal_route.cost_factor:.2f}/km",
                    "Max Payload": f"{optimal_route.max_payload:.1f} tons"
                }
                
                # Add AI prediction badge if used
                if optimal_route.used_ai_prediction:
                    st.info("🧠 **AI-Enhanced Prediction**: CO₂e calculated using AI model considering terrain, temperature, and traffic conditions")
                
                # Display as a table
                st.table(pd.DataFrame(list(optimal_route_data.items()), columns=["Metric", "Value"]))
                
                # Route breakdown
                st.markdown("#### Route Breakdown")
                
                if optimal_route.segments:
                    # Prepare segment data
                    segments_data = []
                    
                    for segment in optimal_route.segments:
                        vehicle = get_vehicle_by_id(segment.vehicle_type_id)
                        
                        # Base segment data
                        segment_data = {
                            "Origin": segment.origin,
                            "Destination": segment.destination,
                            "Distance (km)": f"{segment.distance:.1f}",
                            "Duration (hours)": f"{segment.duration:.2f}",
                            "Vehicle": vehicle.name,
                            "CO₂e (kg)": f"{segment.co2e:.2f}",
                            "Cost (€)": f"{segment.cost:.2f}"
                        }
                        
                        # Add AI prediction factors if available
                        if hasattr(segment, 'prediction_metadata') and segment.prediction_metadata:
                            if "adjustments" in segment.prediction_metadata:
                                adjustments = segment.prediction_metadata["adjustments"]
                                # Add relevant factors to display
                                for factor, value in adjustments.items():
                                    factor_name = factor.replace("_factor", "").capitalize()
                                    segment_data[factor_name] = f"{value:.2f}"
                        
                        segments_data.append(segment_data)
                    
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
                
                # AI vs Standard prediction comparison if both are available
                if optimal_route.used_ai_prediction and standard_prediction_route:
                    st.markdown("#### AI vs Standard Prediction")
                    
                    # Calculate the differences
                    ai_co2e = optimal_route.total_co2e
                    std_co2e = standard_prediction_route.total_co2e
                    diff_pct = ((ai_co2e / std_co2e) - 1) * 100 if std_co2e > 0 else 0
                    
                    # Display comparison stats
                    st.markdown(f"""
                    **Standard calculation**: {std_co2e:.2f} kg CO₂e  
                    **AI-enhanced calculation**: {ai_co2e:.2f} kg CO₂e  
                    **Difference**: {ai_co2e - std_co2e:.2f} kg CO₂e ({diff_pct:+.1f}%)
                    """)
                    
                    # Create comparison bar chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # Prepare segment-by-segment comparison
                    segment_labels = [f"Segment {i+1}" for i in range(len(optimal_route.segments))]
                    ai_segment_co2e = [s.co2e for s in optimal_route.segments]
                    std_segment_co2e = [s.co2e for s in standard_prediction_route.segments]
                    
                    x = np.arange(len(segment_labels))
                    width = 0.35
                    
                    # Plot bars
                    bars1 = ax.bar(x - width/2, std_segment_co2e, width, label='Standard')
                    bars2 = ax.bar(x + width/2, ai_segment_co2e, width, label='AI Enhanced')
                    
                    # Add text and labels
                    ax.set_ylabel('CO₂e (kg)')
                    ax.set_title('CO₂e by Segment: AI vs Standard')
                    ax.set_xticks(x)
                    ax.set_xticklabels(segment_labels)
                    ax.legend()
                    
                    # Add percentage difference labels
                    for i, (ai, std) in enumerate(zip(ai_segment_co2e, std_segment_co2e)):
                        diff = ((ai / std) - 1) * 100 if std > 0 else 0
                        ax.annotate(f"{diff:+.1f}%", 
                                    xy=(i + width/2, ai + 5),
                                    ha='center', va='bottom',
                                    fontsize=9, fontweight='bold',
                                    color='green' if diff < 0 else 'red')
                    
                    # Display the chart
                    st.pyplot(fig)
                    
                    with st.expander("Why are the predictions different?"):
                        st.markdown("""
                        ### Factors that AI prediction considers:
                        
                        1. **Terrain effects**: Uphill driving can increase fuel consumption by 20-40%
                        2. **Temperature impact**: Extreme temperatures reduce efficiency (heating/cooling use)
                        3. **Traffic congestion**: Stop-start driving increases emissions substantially
                        4. **Non-linear payload effects**: Emissions don't scale linearly with weight
                        5. **Speed efficiency curve**: Vehicles have an optimal speed range for efficiency
                        
                        The standard calculation uses fixed emission factors that don't account for these dynamic conditions.
                        """)
                
                # AI prediction details if available
                if optimal_route.used_ai_prediction:
                    with st.expander("AI Prediction Details"):
                        st.markdown("#### How the AI calculates emissions")
                        st.markdown("""
                        The AI prediction model considers multiple factors to provide a more accurate CO₂e estimate:
                        
                        1. **Vehicle type & base emissions**: Starting point for calculation
                        2. **Payload impact**: How cargo weight affects fuel consumption
                        3. **Terrain difficulty**: Hilly routes increase fuel consumption
                        4. **Temperature**: Extreme temperatures (hot or cold) increase fuel use
                        5. **Traffic conditions**: Stop-and-go traffic increases emissions
                        
                        The model combines these factors to estimate the total carbon footprint.
                        """)
                        
                        # Show any prediction metadata from the first segment
                        if optimal_route.segments and hasattr(optimal_route.segments[0], 'prediction_metadata'):
                            metadata = optimal_route.segments[0].prediction_metadata
                            if metadata.get("method") == "gemini_api":
                                st.success("✅ Using Gemini AI model for predictions")
                                if "explanation" in metadata:
                                    st.markdown(f"**Model explanation**: {metadata['explanation']}")
                            else:
                                st.info("ℹ️ Using enhanced factor-based calculation (AI fallback mode)")
                
                # Alternative routes
                st.markdown("#### Alternative Routes")
                
                if alternative_routes:
                    alt_data = []
                    for i, route in enumerate(alternative_routes[:3]):  # Show up to 3 alternatives
                        vehicle_id = route.segments[0].vehicle_type_id if route.segments else None
                        vehicle = get_vehicle_by_id(vehicle_id) if vehicle_id else None
                        
                        ai_tag = " (AI)" if route.used_ai_prediction else ""
                        
                        alt_data.append({
                            "Option": i + 1,
                            "Vehicle": f"{vehicle.name if vehicle else 'Unknown'}{ai_tag}",
                            "CO₂e (kg)": f"{route.total_co2e:.2f}",
                            "Cost (€)": f"{route.total_cost:.2f}",
                            "Emission Factor": f"{route.emission_factor:.3f} kg CO₂e/km",
                            "Within Budget": "✅" if route.total_cost <= max_budget else "❌",
                            "Distance (km)": f"{route.total_distance:.1f}"
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
            3. **Carbon footprint is calculated** for each option using emission factors or AI prediction
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
            
            ### AI-Enhanced Prediction
            
            When enabled, the AI prediction system provides more accurate CO₂e estimates by considering:
            
            - **Vehicle type and payload**: How the specific vehicle and cargo weight affect emissions
            - **Terrain**: Hillier routes require more fuel and produce more emissions
            - **Temperature**: Extreme temperatures affect vehicle efficiency
            - **Traffic conditions**: Stop-and-go traffic significantly increases emissions
            
            This gives you a more realistic picture of your shipment's environmental impact.
            """)

elif page == "Carbon Reports":
    st.title("📊 Carbon Emission Reports")
    
    if not st.session_state.calculation_history:
        st.info("No calculations have been made yet. Use the Route Planner to generate carbon reports.")
    else:
        st.write(f"You have {len(st.session_state.calculation_history)} route calculations in your session.")
        
        # Summary statistics
        total_distance = sum(calc["route"]["total_distance"] for calc in st.session_state.calculation_history)
        total_emissions = sum(calc["route"]["total_co2e"] for calc in st.session_state.calculation_history)
        avg_emissions_per_km = total_emissions / total_distance if total_distance > 0 else 0
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Distance", f"{total_distance:.1f} km")
        col2.metric("Total CO₂e", f"{total_emissions:.2f} kg")
        col3.metric("Avg. CO₂e/km", f"{avg_emissions_per_km:.3f} kg/km")
        
        # Route history table
        st.subheader("Route History")
        
        # Prepare data for table
        history_data = []
        for calc in st.session_state.calculation_history:
            # Get vehicle name
            vehicle_type_id = calc["route"]["vehicle_type"]
            vehicle = get_vehicle_by_id(vehicle_type_id) if vehicle_type_id else None
            vehicle_name = vehicle.name if vehicle else "Unknown"
            
            # Get prediction method
            ai_method = "Standard"
            if calc["parameters"]["use_ai_prediction"]:
                if "method" in calc["ai_metadata"]:
                    if calc["ai_metadata"]["method"] == "gemini_api":
                        ai_method = "Gemini AI"
                    elif calc["ai_metadata"]["method"] == "enhanced_factor_model":
                        ai_method = "Enhanced Model"
            
            history_data.append({
                "ID": calc["id"],
                "Date": calc["timestamp"],
                "Route": f"{calc['route']['origin'].split(',')[0]} → {calc['route']['destination'].split(',')[0]}",
                "Vehicle": vehicle_name,
                "Distance (km)": f"{calc['route']['total_distance']:.1f}",
                "CO₂e (kg)": f"{calc['route']['total_co2e']:.2f}",
                "Method": ai_method
            })
        
        # Display the table
        st.dataframe(pd.DataFrame(history_data))
        
        # Generate charts
        st.subheader("Emissions Analysis")
        
        # Chart 1: Emissions by route
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        route_names = [f"{h['id']}: {h['route']['origin'].split(',')[0]} → {h['route']['destination'].split(',')[0]}" 
                      for h in st.session_state.calculation_history[-10:]]  # Show last 10
        co2e_values = [h["route"]["total_co2e"] for h in st.session_state.calculation_history[-10:]]
        distances = [h["route"]["total_distance"] for h in st.session_state.calculation_history[-10:]]
        
        x = np.arange(len(route_names))
        width = 0.35
        
        ax1.bar(x - width/2, co2e_values, width, label='CO₂e (kg)')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, distances, 'r-o', label='Distance (km)')
        
        ax1.set_xlabel('Routes')
        ax1.set_ylabel('CO₂e (kg)')
        ax1_twin.set_ylabel('Distance (km)')
        ax1.set_title('Emissions by Route')
        ax1.set_xticks(x)
        ax1.set_xticklabels(route_names, rotation=45, ha='right')
        
        fig1.tight_layout()
        st.pyplot(fig1)
        
        # Chart 2: Emissions by vehicle type (if there's variety)
        vehicle_types = {}
        for calc in st.session_state.calculation_history:
            vehicle_id = calc["route"]["vehicle_type"]
            if vehicle_id:
                vehicle = get_vehicle_by_id(vehicle_id)
                vehicle_name = vehicle.name if vehicle else vehicle_id
                if vehicle_name not in vehicle_types:
                    vehicle_types[vehicle_name] = {"total_co2e": 0, "total_distance": 0, "count": 0}
                
                vehicle_types[vehicle_name]["total_co2e"] += calc["route"]["total_co2e"]
                vehicle_types[vehicle_name]["total_distance"] += calc["route"]["total_distance"]
                vehicle_types[vehicle_name]["count"] += 1
        
        if len(vehicle_types) > 1:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Calculate average emissions per km for each vehicle type
            vehicle_names = list(vehicle_types.keys())
            avg_emissions = [vt["total_co2e"]/vt["total_distance"] if vt["total_distance"] > 0 else 0 
                           for vt in vehicle_types.values()]
            total_emissions = [vt["total_co2e"] for vt in vehicle_types.values()]
            
            # Create grouped bar chart
            x = np.arange(len(vehicle_names))
            width = 0.35
            
            ax2.bar(x - width/2, avg_emissions, width, label='Avg CO₂e/km (kg)')
            ax2_twin = ax2.twinx()
            ax2_twin.bar(x + width/2, total_emissions, width, color='orange', label='Total CO₂e (kg)')
            
            ax2.set_xlabel('Vehicle Types')
            ax2.set_ylabel('Avg CO₂e/km (kg)')
            ax2_twin.set_ylabel('Total CO₂e (kg)')
            ax2.set_title('Emissions by Vehicle Type')
            ax2.set_xticks(x)
            ax2.set_xticklabels(vehicle_names)
            
            # Add legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            fig2.tight_layout()
            st.pyplot(fig2)
        
        # Export options
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                # Convert history to DataFrame
                df = pd.DataFrame(history_data)
                
                # Create a download link
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"carbon_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export as JSON"):
                # Prepare JSON export
                export_data = json.dumps(st.session_state.calculation_history, indent=2)
                
                # Create a download link
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"carbon_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

elif page == "Data Transparency":
    st.title("🔍 Emissions Data Transparency")
    
    st.markdown("""
    ### Why Data Transparency Matters
    
    Transparent carbon emissions data is crucial for:
    
    - **Regulatory compliance**: Many jurisdictions now require emissions reporting
    - **Customer trust**: Demonstrating your commitment to accurate environmental reporting
    - **Supply chain transparency**: Providing verifiable data to partners and customers
    - **Continuous improvement**: Using accurate data to identify reduction opportunities
    
    Our system provides complete transparency into how emissions are calculated, what factors are considered,
    and how AI enhances the accuracy of predictions.
    """)
    
    # Show calculation methods
    with st.expander("Calculation Methods", expanded=True):
        st.markdown("""
        #### How We Calculate Carbon Emissions
        
        Our system uses two primary calculation methods:
        
        1. **Standard Factor-Based Method**
           - Uses fixed emission factors per km for each vehicle type
           - Based on industry-standard emission factors
           - Simple multiplication: Distance × Emission Factor
           
        2. **AI-Enhanced Prediction**
           - Uses Google's Gemini AI to provide more accurate estimates
           - Considers multiple dynamic factors like terrain, temperature, and traffic
           - Applies non-linear adjustments for real-world conditions
           - Falls back to an enhanced factor model when AI is unavailable
        """)
        
        # Show vehicle emission factors
        st.markdown("#### Vehicle Emission Factors")
        
        # Get all vehicles
        vehicles = get_all_vehicles().values()
        vehicle_data = []
        
        for v in vehicles:
            vehicle_data.append({
                "Vehicle Type": v.name,
                "Base Emission Factor (kg CO₂e/km)": v.co2e_per_km,
                "Cost (€/km)": v.cost_per_km,
                "Max Payload (tons)": v.max_payload
            })
        
        st.dataframe(pd.DataFrame(vehicle_data))
        
        # AI prediction details
        if not ai_predictor.demo_mode:
            st.success("✅ Using Gemini AI for enhanced predictions")
        else:
            st.info("ℹ️ Using enhanced factor-based calculation (AI fallback mode)")
    
    # AI Factor Explanation
    with st.expander("AI Factor Adjustments", expanded=True):
        st.markdown("""
        #### How the AI adjusts emissions based on factors
        
        The AI considers multiple factors when making predictions:
        """)
        
        # Create tabs for different factors
        factor1, factor2, factor3, factor4, factor5 = st.tabs([
            "Payload", "Terrain", "Temperature", "Traffic", "Speed"
        ])
        
        with factor1:
            st.markdown("""
            ### Payload Impact
            
            Payload weight significantly affects fuel consumption and emissions, but not in a linear way:
            
            - An empty truck still produces about 60-70% of the emissions of a fully loaded truck
            - Emissions increase non-linearly with payload weight
            - Different vehicle types have different sensitivity to payload
            
            The AI models this complex relationship to provide more accurate estimates than simple linear calculations.
            """)
            
            # Create a simple visualization
            payload_weights = [0, 5, 10, 15, 20, 24]  # tons
            empty_factor = 0.65
            base_emissions = 100  # arbitrary base for visualization
            
            # Calculate emissions for different payloads
            simple_emissions = [base_emissions * (empty_factor + 0.35 * (w/24)) for w in payload_weights]
            ai_emissions = [base_emissions * (empty_factor + 0.35 * (w/24)**0.8) for w in payload_weights]
            
            # Create chart
            fig, ax = plt.subplots()
            ax.plot(payload_weights, simple_emissions, 'b-', label='Simple Linear Model')
            ax.plot(payload_weights, ai_emissions, 'r-', label='AI Non-Linear Model')
            ax.set_xlabel('Payload Weight (tons)')
            ax.set_ylabel('Relative Emissions')
            ax.set_title('Impact of Payload on Emissions')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
        
        with factor2:
            st.markdown("""
            ### Terrain Effects
            
            Terrain has a major impact on fuel consumption:
            
            - Uphill driving can increase consumption by 20-40%
            - Downhill sections may reduce consumption but not enough to offset uphill sections
            - The effect is more pronounced for heavily loaded vehicles
            
            The AI model accounts for terrain difficulty based on the provided terrain factor.
            """)
            
            # Create a simple visualization
            terrain_factors = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            base_emissions = 100  # arbitrary base for visualization
            
            # Calculate emissions for different terrains
            terrain_emissions = [base_emissions * tf for tf in terrain_factors]
            
            # Create chart
            fig, ax = plt.subplots()
            ax.bar(terrain_factors, terrain_emissions, width=0.05, color='green')
            ax.set_xlabel('Terrain Factor (1.0 = flat)')
            ax.set_ylabel('Relative Emissions')
            ax.set_title('Impact of Terrain on Emissions')
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            st.pyplot(fig)
        
        # Similar tabs for other factors
        # ...
    
    # Verification and Auditing
    with st.expander("Verification & Auditing", expanded=False):
        st.markdown("""
        ### Emissions Calculation Verification
        
        Our system is designed to support verification and auditing of carbon calculations:
        
        1. **Detailed Calculation History**: Every calculation is recorded with all parameters
        2. **Complete Metadata**: The system captures prediction factors, methods and adjustments
        3. **Export Options**: Data can be exported in standard formats for verification
        4. **Comparison Tools**: Standard and AI predictions can be compared
        
        These features support compliance with reporting standards and carbon accounting audits.
        """)
        
    # Industry Standards
    with st.expander("Emissions Standards & Compliance", expanded=False):
        st.markdown("""
        ### Industry Standards & Compliance
        
        Our carbon calculation methodology aligns with key industry standards:
        
        - **GLEC Framework**: Global Logistics Emissions Council methodology
        - **GHG Protocol**: Greenhouse Gas Protocol for Scope 3 emissions
        - **ISO 14083**: Quantification and reporting of greenhouse gas emissions in transport chains
        - **EN 16258**: European standard for calculation of transport emissions
        
        The AI-enhanced predictions provide more accurate data for these reporting frameworks.
        """)
        
        st.info("Note: For regulatory compliance, we recommend consulting with environmental compliance experts to ensure all reporting requirements are met.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Carbon-Optimized Logistics Planner")

if __name__ == "__main__":
    # This code will only run when the script is executed directly
    pass