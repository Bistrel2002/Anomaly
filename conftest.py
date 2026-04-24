import sys
import os

# Add the project root to sys.path so that `tests.*` imports resolve correctly
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(__file__))
