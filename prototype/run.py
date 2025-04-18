"""
Script to run the Streamlit app with Python caching disabled
to ensure all code changes are loaded.
"""
import os
import sys
import subprocess

# Disable Python bytecode caching
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# Get the path to the app.py file
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

# Build the command
cmd = [sys.executable, "-m", "streamlit", "run", app_path]

print(f"Starting Streamlit with command: {' '.join(cmd)}")
print("Python caching disabled to ensure fresh module loading")

# Run Streamlit
subprocess.call(cmd)