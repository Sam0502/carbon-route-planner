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
    fixed_content = fixed_content.replace('  total_distance', '            total_distance')
    fixed_content = fixed_content.replace('  segment_data', '            segment_data')
    
    # Fix any other inconsistent indentation
    import re
    lines = fixed_content.split('\n')
    fixed_lines = []
    
    in_function = False
    indentation_level = 0
    
    for line in lines:
        if line.strip().startswith('def ') and not line.startswith('    '):
            # Fix method definition indentation
            fixed_lines.append('    ' + line)
            in_function = True
            indentation_level = 4
        elif in_function and line.strip() and not line.startswith(' '):
            # Fix indentation within methods
            fixed_lines.append(' ' * (indentation_level + 4) + line)
        else:
            fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("✅ Indentation fixed")

if __name__ == "__main__":
    fix_indentation()
