"""
AI-based carbon footprint prediction module using Google Gemini API.

This module enhances carbon calculations by using an AI model to predict
emissions based on dynamic factors like vehicle type, distance, speed,
payload, and terrain.
"""
import os
import logging
import json
import requests
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for Gemini API availability
GEMINI_API_AVAILABLE = bool(os.environ.get("GOOGLE_AI_API_KEY"))

class AIPredictor:
    """Uses Google Gemini API to predict carbon emissions based on dynamic factors."""
    
    def __init__(self):
        """Initialize the AI predictor with a Google Gemini API key."""
        self.api_key = os.environ.get("GOOGLE_AI_API_KEY")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
        
        if not self.api_key:
            self.demo_mode = True
            logger.warning("Google Gemini API key not found. AI prediction will use fallback mode.")
        else:
            self.demo_mode = False
            logger.info("AI Predictor initialized with Google Gemini API")
    
    def predict_co2e(self, 
                    vehicle_type_id: str, 
                    distance: float, 
                    avg_speed: float, 
                    weight_tons: float = 0, 
                    terrain_factor: float = 1.0, 
                    temperature: float = 20.0,
                    traffic_level: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        Predict CO2e emissions using the Google Gemini API.
        
        Args:
            vehicle_type_id: The ID of the vehicle type
            distance: Distance in km
            avg_speed: Average speed in km/h
            weight_tons: Weight of payload in tons (default: 0)
            terrain_factor: Factor representing terrain difficulty (1.0 = flat, >1 = hilly)
            temperature: Temperature in Celsius (affects fuel efficiency)
            traffic_level: Traffic congestion level (0-1, where 1 is most congested)
            
        Returns:
            Tuple of (predicted_co2e, prediction_metadata)
        """
        if self.demo_mode:
            return self._fallback_predict_co2e(
                vehicle_type_id, 
                distance, 
                avg_speed, 
                weight_tons, 
                terrain_factor, 
                temperature, 
                traffic_level
            )
            
        try:
            # Prepare input features for the Gemini API
            from vehicle_data import get_vehicle_by_id
            vehicle = get_vehicle_by_id(vehicle_type_id)
            
            # Create a prompt for Gemini to predict CO2 emissions
            prompt = f"""
            You are an expert carbon emissions prediction system. Calculate the CO2e emissions for a vehicle with the following parameters:
            
            - Vehicle type: {vehicle_type_id} 
              (base emission factor: {vehicle.co2e_per_km} kg CO2e/km)
              (max payload: {vehicle.max_payload} tons)
            - Distance: {distance} km
            - Average speed: {avg_speed} km/h
            - Payload weight: {weight_tons} tons
            - Terrain factor: {terrain_factor} (1.0 = flat, >1 = hilly)
            - Temperature: {temperature}°C
            - Traffic congestion: {traffic_level} (0-1, where 1 is heavy traffic)
            
            Calculate the total CO2e emissions in kg for this journey, accounting for all factors.
            Respond ONLY with a JSON object containing:
            1. "co2e_kg": the predicted emissions in kg
            2. "factors": a sub-object with the adjustment factors you applied
            3. "explanation": a brief explanation of your calculation
            """
            
            # Configure request to Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 1,
                    "topP": 0.1,
                    "maxOutputTokens": 1024
                }
            }
            
            # Make API request
            url = f"{self.gemini_url}?key={self.api_key}"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            try:
                # Extract text and parse JSON from the response
                text_response = result["candidates"][0]["content"]["parts"][0]["text"]
                # Extract JSON part from the response (handling potential markdown formatting)
                json_str = text_response
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                prediction_data = json.loads(json_str)
                
                # Extract the CO2e prediction
                predicted_co2e = float(prediction_data.get("co2e_kg", 0))
                
                # Create metadata
                metadata = {
                    "method": "gemini_api",
                    "model": "gemini-1.5-pro",
                    "adjustments": prediction_data.get("factors", {}),
                    "explanation": prediction_data.get("explanation", ""),
                    "input_parameters": {
                        "vehicle_type_id": vehicle_type_id,
                        "distance_km": distance,
                        "avg_speed_kmh": avg_speed,
                        "payload_tons": weight_tons,
                        "terrain_factor": terrain_factor,
                        "temperature_celsius": temperature,
                        "traffic_congestion": traffic_level
                    }
                }
                
                logger.info(f"Gemini API predicted {predicted_co2e:.2f} kg CO2e")
                return predicted_co2e, metadata
                
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing Gemini API response: {str(e)}")
                # If we can't parse the response properly, fall back to enhanced calculation
                return self._fallback_predict_co2e(
                    vehicle_type_id, 
                    distance, 
                    avg_speed, 
                    weight_tons, 
                    terrain_factor, 
                    temperature, 
                    traffic_level
                )
                
        except Exception as e:
            logger.error(f"Error during Gemini API prediction: {str(e)}")
            # Fall back to factor-based calculation
            return self._fallback_predict_co2e(
                vehicle_type_id, 
                distance, 
                avg_speed, 
                weight_tons, 
                terrain_factor, 
                temperature, 
                traffic_level
            )
    
    def _fallback_predict_co2e(self, 
                             vehicle_type_id: str, 
                             distance: float, 
                             avg_speed: float, 
                             weight_tons: float = 0, 
                             terrain_factor: float = 1.0, 
                             temperature: float = 20.0,
                             traffic_level: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        Fallback method using an enhanced factor-based calculation when AI is unavailable.
        This method provides a more sophisticated calculation than the basic factor model,
        simulating what an AI might predict.
        
        Args:
            Same as predict_co2e()
            
        Returns:
            Tuple of (estimated_co2e, calculation_metadata)
        """
        from vehicle_data import get_vehicle_by_id
        
        try:
            # Get the base vehicle data
            vehicle = get_vehicle_by_id(vehicle_type_id)
            
            # Get base emissions using the standard factor
            base_co2e = distance * vehicle.co2e_per_km
            
            # Enhanced adjustments that an AI might learn:
            
            # 1. Speed efficiency adjustment (U-curve: efficiency drops at very low and very high speeds)
            optimal_speed = 65.0  # km/h - generally most efficient speed
            speed_factor = 1.0 + 0.008 * ((avg_speed - optimal_speed) / 10) ** 2
            
            # 2. Payload adjustment - more sophisticated than the basic model
            if weight_tons > 0:
                # Emissions don't scale linearly with payload
                payload_ratio = min(weight_tons / vehicle.max_payload, 1.0)
                # Empty truck is about 60-70% of full truck emissions
                payload_factor = 0.65 + (0.35 * (payload_ratio ** 0.8))
            else:
                payload_factor = 0.65  # Empty truck baseline
            
            # 3. Terrain adjustment - hills/mountains increase consumption
            # terrain_factor provided as input (1.0 = flat, >1 = hilly)
            
            # 4. Temperature impact - extreme temps increase fuel use (heating/cooling)
            temp_optimal = 20.0  # Celsius
            temp_factor = 1.0 + 0.005 * abs(temperature - temp_optimal) / 10
            
            # 5. Traffic impact - stop/start driving increases emissions
            traffic_factor = 1.0 + 0.15 * traffic_level
            
            # Combine all factors
            adjusted_co2e = base_co2e * speed_factor * payload_factor * terrain_factor * temp_factor * traffic_factor
            
            # Create metadata about how we calculated this estimate
            metadata = {
                "method": "enhanced_factor_model",
                "base_co2e": base_co2e,
                "adjustments": {
                    "speed_factor": speed_factor,
                    "payload_factor": payload_factor,
                    "terrain_factor": terrain_factor,
                    "temperature_factor": temp_factor,
                    "traffic_factor": traffic_factor
                },
                "note": "Using enhanced factor-based model as fallback for AI prediction"
            }
            
            logger.info(f"Enhanced factor model estimated {adjusted_co2e:.2f} kg CO2e (base was {base_co2e:.2f})")
            return adjusted_co2e, metadata
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {str(e)}")
            # If even the fallback fails, use the most basic calculation
            if not vehicle_type_id.startswith("electric"):
                return distance * 0.9, {"method": "basic_fallback", "error": str(e)}
            else:
                return distance * 0.45, {"method": "basic_fallback", "error": str(e)}
    
    def batch_predict(self, prediction_requests: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Predict CO2e emissions for multiple segments in batch.
        
        Args:
            prediction_requests: List of dictionaries with prediction parameters
                               Each dict should have the same parameters as predict_co2e()
        
        Returns:
            List of (predicted_co2e, prediction_metadata) tuples
        """
        results = []
        
        # For Gemini API, process each request individually
        # (We could optimize this in the future by sending multiple requests in parallel)
        for req in prediction_requests:
            result = self.predict_co2e(
                vehicle_type_id=req.get("vehicle_type_id", ""),
                distance=req.get("distance", 0),
                avg_speed=req.get("avg_speed", 0),
                weight_tons=req.get("weight_tons", 0),
                terrain_factor=req.get("terrain_factor", 1.0),
                temperature=req.get("temperature", 20.0),
                traffic_level=req.get("traffic_level", 0.5)
            )
            results.append(result)
        
        return results

# Singleton instance
_predictor_instance = None

def get_predictor() -> AIPredictor:
    """Get or create the singleton AIPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AIPredictor()
    return _predictor_instance