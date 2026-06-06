import sys
from pathlib import Path
import runpy

# Ensure safevision/ is on sys.path so internal imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent / "safevision"))

# Execute the main streamlit application
runpy.run_path(str(Path(__file__).resolve().parent / "safevision" / "dashboard" / "app.py"), run_name="__main__")
