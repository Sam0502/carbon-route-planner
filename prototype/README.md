# Carbon-Optimized Logistics Route Planner

A prototype application for logistics companies to plan shipments by selecting routes and transport modes that minimize carbon footprint while respecting budget and time constraints.

## Overview

This application helps logistics planners:

- Calculate and compare carbon footprints of different transport options
- Find routes that minimize emissions while staying within budget constraints
- Visualize the carbon footprint breakdown of each route segment
- Compare alternative routes with different vehicles and emissions profiles

## Features

- **Input**: Origin, destination, optional intermediate stops, payload characteristics, budget, and time constraints
- **Processing**: Calculates multiple route options using different vehicle types and their emissions factors
- **Optimization**: Finds the route with the lowest carbon footprint that stays within budget
- **Output**: Displays the optimal route, its carbon footprint, cost, and breakdown by segment

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

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

4. **Click "Find Optimal Route"** to calculate and display the most carbon-efficient route options

## Project Structure

- `app.py` - Streamlit web interface
- `models.py` - Data models and structures
- `vehicle_data.py` - Database of vehicle types with emission factors
- `maps_client.py` - Google Maps API integration for route calculation
- `carbon_calculator.py` - Carbon footprint calculation logic
- `optimizer.py` - Route optimization using OR-Tools

## Future Enhancements

- Integration with real vehicle emissions databases (GLEC, DEFRA)
- Support for multiple transport modes (rail, sea, air)
- Advanced optimization for multi-modal transport chains
- Machine learning model to predict emissions based on more factors
- Real-time carbon footprint tracking and reporting

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please contact [your-email@example.com]