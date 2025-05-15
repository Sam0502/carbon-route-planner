"""
Simple script to replace the extract_route_data method in the enhanced_maps_client.py file.
This will fix indentation issues.
"""
import re

def fix_extract_route_data_method():
    with open("enhanced_maps_client.py", "r") as f:
        content = f.read()
    
    # Find the start of the method definition
    pattern = r"def extract_route_data\(self,[^{]*?:"
    match = re.search(pattern, content)
    
    if not match:
        print("Could not find the extract_route_data method.")
        return False
    
    # Find the method body
    start_pos = match.start()
    
    # Find the next method definition after extract_route_data
    next_method = re.search(r"\n\s+def [a-zA-Z_]", content[start_pos+10:])
    if next_method:
        end_pos = start_pos + 10 + next_method.start()
    else:
        end_pos = len(content)
    
    # Build the replacement method with correct indentation
    new_method = """    def extract_route_data(self, route: Dict[str, Any]) -> Tuple[float, float, List[Dict[str, Any]]]:
        \"\"\"Extract distance, duration and segments from a route.\"\"\"
        if not route or 'legs' not in route:
            return 0, 0, []
        
        legs = route.get('legs', [])
        
        total_distance = 0
        total_duration = 0
        segments_data = []
        
        for leg in legs:
            distance = leg.get('distance', {}).get('value', 0) / 1000  # Convert to km
            duration = leg.get('duration', {}).get('value', 0) / 3600  # Convert to hours
            
            # Check if we have air_duration available (added in demo mode for aviation)
            air_duration = leg.get('air_duration', {}).get('value', 0) / 3600 if 'air_duration' in leg else None
            
            total_distance += distance
            total_duration += duration
            
            # Add segment data including air_duration if available
            segment_data = {
                'distance': distance,
                'duration': duration
            }
            
            if air_duration is not None:
                segment_data['air_duration'] = air_duration
                
            segments_data.append(segment_data)
            
            # Debug output
            print(f"Route leg: {leg.get('start_address', 'Unknown')} to {leg.get('end_address', 'Unknown')}")
            print(f"Distance: {distance:.1f} km, Duration: {duration:.2f} hours")
        
        return total_distance, total_duration, segments_data
"""
    
    # Replace the old method with the new one
    new_content = content[:start_pos] + new_method + content[end_pos:]
    
    with open("enhanced_maps_client.py", "w") as f:
        f.write(new_content)
    
    print("✅ The extract_route_data method has been replaced.")
    return True

if __name__ == "__main__":
    fix_extract_route_data_method()
