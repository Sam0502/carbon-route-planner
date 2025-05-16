"""
Script to fix address validation display in the app.py file
and fix indentation issues in enhanced_maps_client.py
"""
import re

def fix_enhanced_maps_client():
    """Fix indentation in enhanced_maps_client.py"""
    file_path = "d:/Test file/prototype/enhanced_maps_client.py"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix indentation for the geocode result block
    pattern = r'            # If we have multiple results, they could be suggestions\s+suggestions = \[\]\s+if len\(geocode_result\) > 1:\s+suggestions = \[result\[\'formatted_address\'\] for result in geocode_result\[1:5\]\]\s+.*?result = \{'
    replacement = """            # If we have multiple results, they could be suggestions
            suggestions = []
            if len(geocode_result) > 1:
                suggestions = [result['formatted_address'] for result in geocode_result[1:5]]
                
            result = {"""
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Ensure the expansion info is correctly implemented
    pattern = r'        # Check if this is a single-word city that needs expansion.*?expanded_info = None'
    replacement = """        # Check if this is a single-word city that needs expansion
        lower_address = cleaned_address.lower()
        if lower_address in common_cities and "," not in address:
            expanded_info = f"Expanded city name from '{address}' to '{common_cities[lower_address]}'"
            print(expanded_info)  # For local debugging
            cleaned_address = common_cities[lower_address]
        else:
            expanded_info = None"""
    
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Save the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print("✅ Enhanced maps client fixed")

def fix_app_py():
    """Add expanded_info display to app.py for all address validation sections"""
    file_path = "d:/Test file/prototype/app.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Process the file
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        
        # Look for address validation blocks
        if "_validation = maps_client.validate_address" in line:
            address_type = line.split("_validation")[0].strip()
            
            # Find the end of the if block for valid address
            j = i + 1
            block_indent = None
            while j < len(lines):
                if block_indent is None and lines[j].strip().startswith("if"):
                    # Capture the indentation of the block
                    block_indent = len(lines[j]) - len(lines[j].lstrip())
                
                if block_indent is not None and lines[j].strip() == "else:":
                    # Found the else clause, insert our code before it
                    expanded_info_code = " " * block_indent + "    # Display address expansion info if available\n"
                    expanded_info_code += " " * block_indent + f"    if 'expanded_info' in {address_type}_validation:\n"
                    expanded_info_code += " " * block_indent + f"        st.info({address_type}_validation['expanded_info'])\n"
                    new_lines.append(expanded_info_code)
                    break
                j += 1
        i += 1
    
    # Write back the modified file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)
    
    print("✅ App.py expanded_info display added")

if __name__ == "__main__":
    try:
        fix_enhanced_maps_client()
        fix_app_py()
        print("✅ All fixes completed successfully!")
    except Exception as e:
        print(f"❌ Error during fix: {str(e)}")
