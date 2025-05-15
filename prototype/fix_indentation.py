"""
Script to fix indentation issues in the enhanced_maps_client.py file
"""
import os

def fix_indentation():
    """Fix indentation in the enhanced_maps_client.py file"""
    file_path = "enhanced_maps_client.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix indentation problems
    fixed_content = content.replace('  def extract_route_data', '    def extract_route_data')
    fixed_content = fixed_content.replace('  # Add segment data', '            # Add segment data')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Indentation fixed")

if __name__ == "__main__":
    fix_indentation()
