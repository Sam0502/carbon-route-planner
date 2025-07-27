# Google Cloud Solution Challenge Submission

## Team Details
- **Team Name**: EcoRoute Logistics
- **Team Leader**: Your Name
- **Problem Statement**: Carbon footprint optimization for logistics companies is increasingly important but lacks accessible tools that balance environmental impact with business constraints.

## Solution Overview
EcoRoute is a carbon-optimized logistics route planner that helps companies minimize their carbon footprint while respecting budget constraints. Our solution uses AI-enhanced prediction models to calculate accurate emissions for different routes and transportation modes, enabling logistics managers to make environmentally responsible shipping decisions.

## Opportunities
- **Differentiation**: Unlike existing solutions that focus solely on cost or time optimization, EcoRoute prioritizes carbon footprint while still respecting business constraints.
- **Problem Solving**: We address the challenge of accurate carbon emissions calculation by using Gemini API to consider real-world factors like terrain, temperature, and traffic.
- **USP**: Our solution offers the unique ability to compare multiple transportation modes (ground, air) with transparent emissions data and AI-enhanced prediction accuracy.

## Features List
- Carbon-optimized route planning across multiple transportation modes
- AI-enhanced carbon emissions prediction using Google Gemini
- Multi-route comparison with detailed analytics
- Data transparency and emissions reporting dashboard
- Real-time address validation via Google Maps API
- Aviation and ground transportation options with realistic travel times
- Terrain, temperature, and traffic factor adjustments
- Exportable reports for sustainability documentation

## Process / Use-case Flow
1. User enters origin and destination addresses
2. System validates addresses using Google Maps API
3. User adds any intermediate stops required
4. User selects transportation preferences (automatic optimization or manual vehicle selection)
5. User adjusts environmental factors (terrain, temperature, traffic)
6. System calculates optimal route using Gemini API for emissions prediction
7. System displays route with emissions data, cost, and duration
8. User can compare multiple routes and export reports

## Design Concepts
- Streamlined sidebar for input parameters
- Interactive map displaying optimized routes
- Tabbed interface for different functions (planning, comparison, reporting)
- Data visualizations showing emissions comparisons
- Mobile-responsive design for field use

## Architecture Diagram
- Frontend: Streamlit web interface
- Backend: Python processing engine
- APIs: Google Maps Platform (Directions, Geocoding), Google Gemini AI
- Data Storage: Session-based with CSV/JSON export options
- Deployment: Streamlit Cloud with GitHub integration

## Technology Stack
- Python for core logic and data processing
- Streamlit for interactive web interface
- Google Maps Platform APIs for geocoding and directions
- Google Gemini API for AI-enhanced emissions prediction
- Matplotlib and Pandas for data visualization and manipulation
- NumPy for numerical computations
- GitHub for version control and deployment

## Gemini API Integration
Gemini is used to enhance the accuracy of carbon emissions predictions by:
1. Analyzing the impact of terrain difficulty on fuel consumption
2. Calculating temperature effects on vehicle efficiency
3. Assessing traffic congestion impact on emissions
4. Providing more accurate radiative forcing effects for aviation
5. Generating natural language explanations of carbon calculation factors

## Prototype Evidence
Our prototype demonstrates:
- Working route optimization with multiple transportation options
- AI-enhanced carbon footprint calculation
- Multi-route comparison analytics
- Data transparency reporting
- Address validation and geocoding
- Aviation transport with realistic travel times
- Export functionality for reports

## Demo Links
- Demo Video: [Link to your 3-minute video]
- Working Prototype: [Link to your Streamlit Cloud deployment]

## Support
For any queries related to this submission, please contact:
- Team Email: [Your Email]
- Solution Challenge Support: apacsolutionchallenge@hack2skill.com

## Recent Improvements
We've recently made several important improvements to the platform:

1. **Fixed Aviation Transport Duration Calculation**:
   - Added separate calculation for aviation transport using realistic speeds (750 km/h vs 60 km/h for ground transport)
   - Implemented detection logic for aviation vehicle types (`cargo_plane_*` and `sustainable_aviation`)
   - Result: Aviation transport now shows appropriate travel times (~6x faster than ground)

2. **Enhanced Budget Constraints**:
   - Updated maximum budget parameter to allow virtually unlimited values (up to 1e308)
   - Maintained default budget at 1000€ for usability
   - Added help text explaining the unlimited budget option

3. **Improved Address Validation UI**:
   - Added city name expansion information display
   - Fixed validation message visibility when running on Streamlit Cloud
   - Enhanced user experience with clear feedback about address processing

These improvements have significantly enhanced the accuracy and usability of our solution, particularly for comparing air and ground transport options based on realistic time calculations.
