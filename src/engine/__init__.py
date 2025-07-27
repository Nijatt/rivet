# src/engine/__init__.py

# Package version (optional)
__version__ = "0.1.0"

# Expose the main API at package level
from .body      import RigidBody
from .body      import Particle
from .core      import XPBDSolver
from .core import PBDSolver
from .collision import resolve_collision
from .renderer  import Renderer
from .transform import Transform
from .gl_camera import Camera
from .opengl_renderer import OpenGLRenderer
from .rod_system import ElasticEdge
from .rod_system import OrientationElement
from .rod_system import ElasticRod
from .rod_utils import RodUtils


# Define what 'import *' will bring in
__all__ = [
    "RigidBody",
    "Particle",
    "XPBDSolver",
    "PBDSolver",
    "resolve_collision",
    "Renderer",
    "Transform",
    "Camera",
    "OpenGLRenderer",
    "ElasticEdge",
    "ElasticRod",
    "RodUtils"
    "__version__",
]