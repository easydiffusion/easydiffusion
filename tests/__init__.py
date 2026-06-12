import os
import sys

# link easydiffusion (for tests)
dep_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.exists(dep_path):
    if dep_path not in sys.path:
        sys.path.append(dep_path)
