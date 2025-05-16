"""
Test script to verify all fixes are working correctly:
1. Aviation duration calculation
2. Budget constraint
3. Address validation messages
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fix_verification")

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_aviation_duration():
    """Test that aviation durations are calculated with proper speed."""
    from enhanced_maps_client import EnhancedMapsClient
    
    logger.info("Testing aviation duration calculation...")
    
    client = EnhancedMapsClient()
    
    # Example with a 1000km distance
    test_distance = 1000  # km
    
    # Get demo route with this distance
    route_data = client._get_demo_route("Origin", "Destination", test_distance)
    
    # Check if air_duration is in the route data
    if 'air_duration' not in route_data['legs'][0]:
        logger.error("❌ air_duration not found in route data")
        return False
    
    # Calculate expected air duration (distance / 750 km/h)
    expected_air_duration_hours = test_distance / 750
    expected_air_duration_seconds = expected_air_duration_hours * 3600
    
    actual_air_duration_seconds = route_data['legs'][0]['air_duration']['value']
    
    # Check if the values match (allowing for small floating-point differences)
    if abs(expected_air_duration_seconds - actual_air_duration_seconds) > 1:
        logger.error(f"❌ Air duration calculation is incorrect. Expected ~{expected_air_duration_seconds}s, got {actual_air_duration_seconds}s")
        return False
    
    logger.info(f"✅ Air duration correctly calculated: {expected_air_duration_hours:.2f} hours")
    return True

def test_budget_constraint():
    """Test that the budget constraint can be set to a very large value."""
    # For this test, we would need to mock the Streamlit UI
    # Since we can't do that easily in this script, we'll just check the code
    
    logger.info("Testing budget constraint setting...")
    
    # Read the app.py file
    with open("app.py", "r") as f:
        app_code = f.read()
    
    # Check if the max_budget has been updated to use 1e308
    if "max_budget = st.sidebar.number_input(\"Maximum Budget (€)\", 0.0, 1e308, 1000.0" in app_code:
        logger.info("✅ Budget constraint has been updated to allow virtually unlimited values")
        return True
    else:
        logger.error("❌ Budget constraint has not been properly updated")
        return False

def test_address_validation():
    """Test that expanded_info is properly returned in address validation."""
    from enhanced_maps_client import EnhancedMapsClient
    
    logger.info("Testing address validation expanded_info...")
    
    client = EnhancedMapsClient()
    
    # Test with a city that needs expansion
    result = client.validate_address("Boston")
    
    if 'expanded_info' not in result:
        logger.error("❌ expanded_info not found in validation result")
        return False
    
    if not result['expanded_info'] or "Expanded city name" not in result['expanded_info']:
        logger.error("❌ expanded_info does not contain expansion message")
        return False
    
    logger.info(f"✅ Address validation expanded_info working: {result['expanded_info']}")
    return True

def run_all_tests():
    """Run all verification tests."""
    logger.info("Starting verification of all fixes...")
    
    tests = [
        ("Aviation Duration", test_aviation_duration),
        ("Budget Constraint", test_budget_constraint),
        ("Address Validation", test_address_validation)
    ]
    
    results = []
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Error running test {name}: {str(e)}")
            results.append((name, False))
    
    # Print summary
    logger.info("\n=== TEST RESULTS SUMMARY ===")
    all_pass = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{name}: {status}")
        if not result:
            all_pass = False
    
    if all_pass:
        logger.info("\n✅ All fixes successfully verified!")
    else:
        logger.error("\n❌ Some tests failed. Please check the details above.")
    
if __name__ == "__main__":
    run_all_tests()
