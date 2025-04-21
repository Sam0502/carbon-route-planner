# Carbon-Optimized Logistics Route Planner

A prototype application for logistics companies to plan shipments by selecting routes and transport modes that minimize carbon footprint while respecting budget and time constraints.

## Overview

This application helps logistics planners:

- Calculate and compare carbon footprints of different transport options
- Find routes that minimize emissions while staying within budget constraints
- Visualize the carbon footprint breakdown of each route segment
- Compare alternative routes with different vehicles and emissions profiles
- Use AI to predict more accurate CO2e estimates based on dynamic factors

## Features

- **Input**: Origin, destination, optional intermediate stops, payload characteristics, budget, and time constraints
- **Processing**: Calculates multiple route options using different vehicle types and their emissions factors
- **Optimization**: Finds the route with the lowest carbon footprint that stays within budget
- **Output**: Displays the optimal route, its carbon footprint, cost, and breakdown by segment
- **AI-Enhanced Prediction**: Uses Vertex AI to predict CO2e emissions based on vehicle type, distance, speed, payload weight, terrain, temperature, and traffic conditions

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`
- Google Cloud account with Vertex AI API enabled (for AI prediction features)

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd carbon-route-planner
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Google Maps API key** (optional for full functionality):
   - Create a `.env` file in the project root and add your API key:
     ```
     GOOGLE_MAPS_API_KEY=your_api_key_here
     ```
   - Without an API key, the application will run in demo mode with estimated distances

6. **Set up Vertex AI** (optional for AI prediction):
   - Create a Google Cloud project and enable the Vertex AI API
   - Create a service account with Vertex AI access and download the JSON key
   - Add to your `.env` file:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
     GOOGLE_CLOUD_PROJECT=your-project-id
     VERTEX_AI_MODEL_ENDPOINT=your-model-endpoint-id
     ```
   - Without Vertex AI setup, the application will run an enhanced factor-based prediction model

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface** at http://localhost:8501

3. **Enter shipment details**:
   - Origin and destination addresses
   - Intermediate stops (if any)
   - Payload characteristics (weight/volume)
   - Maximum budget
   - Required delivery time window (optional)
   - Environmental factors (terrain, temperature, traffic) for AI prediction

4. **Click "Find Optimal Route"** to calculate and display the most carbon-efficient route options

## Using AI Prediction

The application supports two modes of CO2e calculation:

1. **Standard Factor-Based Calculation**: Uses fixed emission factors per km for each vehicle type
2. **AI-Enhanced Prediction**: Uses a Vertex AI model to predict emissions based on multiple dynamic factors

To use the AI prediction feature:

1. Set `use_ai_prediction=True` in your ShipmentRequest
2. Optionally provide additional environmental factors:
   - `terrain_factors`: List of terrain difficulty values (1.0=flat, >1.0=hilly)
   - `temperatures`: List of temperatures in Celsius
   - `traffic_levels`: List of traffic congestion levels (0-1, 1=heavy traffic)

Example using the API:

```python
from models import ShipmentRequest
from optimizer import RouteOptimizer
from maps_client import MapsClient

# Create a request with AI prediction enabled
request = ShipmentRequest(
    origin="Berlin, Germany",
    destination="Munich, Germany",
    intermediate_stops=["Leipzig, Germany", "Nuremberg, Germany"],
    weight_tons=15.0,
    max_budget=1000.0,
    use_ai_prediction=True,
    terrain_factors=[1.0, 1.2, 1.1],  # Terrain factors for each segment
    temperatures=[22.0, 18.0, 20.0],   # Temperatures for each segment
    traffic_levels=[0.6, 0.4, 0.7]     # Traffic levels for each segment
)

# Process the route
maps_client = MapsClient()
optimizer = RouteOptimizer(maps_client)
optimal_route, alternatives = optimizer.optimize(request)

# The routes will use AI-enhanced prediction
print(f"Total CO2e: {optimal_route.total_co2e:.2f} kg")
```

You can run a demo of the AI prediction features:

```bash
python ai_demo.py
```

## Project Structure

- `app.py` - Streamlit web interface
- `models.py` - Data models and structures
- `vehicle_data.py` - Database of vehicle types with emission factors
- `maps_client.py` - Google Maps API integration for route calculation
- `carbon_calculator.py` - Carbon footprint calculation logic
- `optimizer.py` - Route optimization using OR-Tools
- `ai_predictor.py` - AI-based carbon footprint prediction using Vertex AI
- `ai_demo.py` - Demonstration of AI prediction capabilities

## Future Enhancements

- Integration with real vehicle emissions databases (GLEC, DEFRA)
- Support for multiple transport modes (rail, sea, air)
- Advanced optimization for multi-modal transport chains
- Machine learning model to predict emissions based on more factors
- Real-time carbon footprint tracking and reporting
- Training custom Vertex AI models on proprietary emissions data

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please contact [your-email@example.com]