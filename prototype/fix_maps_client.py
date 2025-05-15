"""
Fix for Google Maps API access issues.

This script patches the maps_client.py file to improve error handling 
and address validation for better handling of various place names.
"""
import os
import re

def patch_maps_client():
    """
    Adds improved error handling and address validation to the maps_client.py file.
    """
    # Get the path to the maps_client.py file
    maps_client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps_client.py")
    
    # Check if the file exists
    if not os.path.exists(maps_client_path):
        print(f"Error: Could not find maps_client.py at {maps_client_path}")
        return False
    
    # Read the current content
    with open(maps_client_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add improved location validation logic
    if "_validate_location_name" not in content:
        # Add the method to the MapsClient class
        new_method = """
    def _validate_location_name(self, location: str) -> str:
        """
        Validate and format a location name to improve geocoding accuracy.
        
        Args:
            location: The location name or address to validate
            
        Returns:
            Cleaned and potentially expanded location string
        """
        if not location or location.strip() == "":
            print(f"Warning: Empty location provided")
            return location
            
        # Clean up the location string
        location = location.strip()
        
        # Common city names that might need state/country added
        common_cities = {
            "new york": "New York, NY, USA",
            "boston": "Boston, MA, USA",
            "chicago": "Chicago, IL, USA",
            "los angeles": "Los Angeles, CA, USA",
            "san francisco": "San Francisco, CA, USA",
            "dallas": "Dallas, TX, USA",
            "houston": "Houston, TX, USA",
            "miami": "Miami, FL, USA",
            "seattle": "Seattle, WA, USA",
            "detroit": "Detroit, MI, USA",
            "philadelphia": "Philadelphia, PA, USA",
            "london": "London, UK",
            "paris": "Paris, France",
            "rome": "Rome, Italy",
            "madrid": "Madrid, Spain",
            "berlin": "Berlin, Germany",
            "sydney": "Sydney, Australia",
            "tokyo": "Tokyo, Japan",
            "beijing": "Beijing, China",
            "moscow": "Moscow, Russia"
        }
        
        # Check if this is a single-word location that matches a common city
        if "," not in location and location.lower() in common_cities:
            print(f"Expanding city name '{location}' to '{common_cities[location.lower()]}'")
            return common_cities[location.lower()]
        
        # If it's a short string without commas, it might be ambiguous
        if len(location) < 10 and "," not in location:
            print(f"Warning: '{location}' might be ambiguous - consider adding state/country")
        
        return location
"""
        
        # Find the class definition to insert after
        class_pattern = r"class MapsClient:"
        match = re.search(class_pattern, content)
        if match:
            # Find the end of the class (last method)
            methods = re.findall(r"    def .*?\(", content)
            if methods:
                last_method = methods[-1]
                insert_pos = content.rfind(last_method)
                
                # Insert the new method after the last method
                if insert_pos > 0:
                    new_content = content[:insert_pos] + new_method + content[insert_pos:]
                    # Write the updated content
                    with open(maps_client_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print("✅ Added improved location validation logic")
                    return True
    else:
        print("Location validation logic already present")
    
    return False

if __name__ == "__main__":
    print("Applying maps client fixes...")
    if patch_maps_client():
        print("✅ Maps client successfully patched")
    else:
        print("❌ Failed to patch maps client")
