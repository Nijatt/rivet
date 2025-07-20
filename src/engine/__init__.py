# src/engine/__init__.py

# Package version (optional)
__version__ = "0.1.0"

# Expose the main API at package level
from .body      import RigidBody
from .core      import XPBDSolver
from .collision import resolve_collision
from .renderer  import Renderer

# Define what 'import *' will bring in
__all__ = [
    "RigidBody",
    "XPBDSolver",
    "resolve_collision",
    "Renderer",
    "__version__",
]