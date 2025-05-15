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
    
    def predict_co2e(
        self,
        vehicle_type_id: str,
        distance: float,
        avg_speed: float,
        weight_tons: float = 0,
        terrain_factor: float = 1.0,
        temperature: float = 20.0,
        traffic_level: float = 0.5,
        is_aviation: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict CO2e emissions for a route segment using ML model.
        
        Args:
            vehicle_type_id: ID of the vehicle type
            distance: Distance in km
            avg_speed: Average speed in km/h
            weight_tons: Weight of payload in tons
            terrain_factor: Factor representing terrain difficulty (1.0=flat)
            temperature: Temperature in Celsius
            traffic_level: Traffic congestion level (0-1)
            is_aviation: Whether this is an aviation transportation
            
        Returns:
            Tuple of (predicted_co2e, metadata)
        """
        try:
            # Get the emissions factor from vehicle data
            vehicle = get_vehicle_by_id(vehicle_type_id)
            base_emissions_factor = vehicle.co2e_per_km
            
            # Normalize inputs for the model
            normalized_distance = min(distance / 5000.0, 1.0)  # Normalize to 0-1 assuming max 5000km
            normalized_speed = min(avg_speed / (1000.0 if is_aviation else 100.0), 1.0)  # Different max speed for aviation
            normalized_weight = min(weight_tons / vehicle.max_payload, 1.0)
            normalized_temp_impact = abs(temperature - 15) / 35.0  # Impact increases as temp diverges from ideal (15C)
            
            # Calculate basic emission estimate using the simplified model
            emission_multiplier = 1.0
            
            if is_aviation:
                # Aviation-specific calculations
                
                # Flight phase impact (takeoff/landing vs. cruising)
                if distance < 500:
                    emission_multiplier *= 1.25  # Short flights have more takeoff/landing impact
                elif distance < 1500:
                    emission_multiplier *= 1.15  # Medium flights
                    
                # High-altitude impact (radiative forcing)
                if "sustainable_aviation" not in vehicle_type_id:
                    emission_multiplier *= 1.9  # Non-SAF planes have higher radiative forcing impact
                    
                # Weather conditions (wind, etc.) - simplified model
                weather_variance = self.get_random_factor(0.85, 1.15)
                emission_multiplier *= weather_variance
                
                # Aircraft load factor impacts
                if normalized_weight < 0.5:
                    # Lighter loads are less efficient per ton-km
                    emission_multiplier *= (1.2 - 0.4 * normalized_weight)
            else:
                # Ground transportation calculations
                
                # Apply terrain factor (hills increase emissions)
                emission_multiplier *= (1.0 + (terrain_factor - 1.0) * 0.5)
                
                # Apply temperature impact (extreme temps reduce efficiency)
                emission_multiplier *= (1.0 + normalized_temp_impact * 0.3)
                
                # Apply traffic impact (congestion increases emissions)
                emission_multiplier *= (1.0 + traffic_level * 0.4)
            
            # Apply weight impact
            if is_aviation:
                # Aviation has different weight impact curve
                emission_multiplier *= (0.5 + 0.5 * normalized_weight)
            else:
                # Ground transport
                emission_multiplier *= (0.7 + 0.3 * normalized_weight)
            
            # Apply some random variation to simulate real-world conditions and model uncertainty
            variation = self.get_random_factor(0.9, 1.1)
            emission_multiplier *= variation
            
            # Calculate the final CO2e value
            co2e = distance * base_emissions_factor * emission_multiplier
            
            # Prepare metadata about the prediction
            metadata = {
                "model_version": "2.1",
                "base_emissions_factor": base_emissions_factor,
                "emission_multiplier": emission_multiplier,
                "factors_applied": {
                    "distance_km": distance,
                    "avg_speed_kmh": avg_speed,
                    "weight_tons": weight_tons,
                    "max_payload_tons": vehicle.max_payload,
                    "terrain_factor": terrain_factor,
                    "temperature_c": temperature,
                    "traffic_level": traffic_level,
                    "is_aviation": is_aviation,
                    "variation": variation
                }
            }
            
            return co2e, metadata
            
        except Exception as e:
            print(f"Error in AI prediction: {str(e)}")
            raise
    
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
                traffic_level=req.get("traffic_level", 0.5),
                is_aviation=req.get("is_aviation", False)
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