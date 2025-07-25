# -----------------------------------------------------------------------------
# mathutils.py · float3 / float3x3 implemented as true NumPy subclasses
# -----------------------------------------------------------------------------
from __future__ import annotations
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# float3 – thin subclass of np.ndarray (shape (3,))
# ═════════════════════════════════════════════════════════════════════════════
class float3(np.ndarray):
    """A NumPy row‑vector of length 3 with swizzle helpers.•

    Because it *is* an ndarray, you can pass it directly to any NumPy/SciPy
    function without copying or casting, while still enjoying attribute access
    (v.x) and small utility methods (norm, cross, etc.)."""

    # ---- constructor ------------------------------------------------------
    def __new__(cls, x=0.0, y=0.0, z=0.0):
        obj = np.asarray([x, y, z], dtype=float).view(cls)
        return obj

    @staticmethod
    def from_iter(it) -> "float3":
        it = list(it)
        if len(it) != 3:
            raise ValueError("float3 expects exactly 3 elements")
        return float3(*it)

    # ---- swizzle properties ----------------------------------------------
    x = property(lambda self: self[0], lambda self, v: self.__setitem__(0, v))
    y = property(lambda self: self[1], lambda self, v: self.__setitem__(1, v))
    z = property(lambda self: self[2], lambda self, v: self.__setitem__(2, v))

    # ---- magnitude helpers -------------------------------------------------
    def norm(self) -> float:
        return float(np.linalg.norm(self))

    def normalized(self) -> "float3":
        n = self.norm()
        return self if n < 1e-12 else float3(*(self / n))

    # dot / cross remain via numpy: v @ w  (dot) , np.cross(v,w) -------------
    def dot(self, other) -> float:
        return float(np.dot(self, other))

    def cross(self, other) -> "float3":
        return float3(*np.cross(self, other))

    # nice repr -------------------------------------------------------------
    def __repr__(self):
        return f"float3({self[0]:.4g}, {self[1]:.4g}, {self[2]:.4g})"


# helpers --------------------------------------------------------------------
zeros3 = lambda: float3(0.0, 0.0, 0.0)
ones3  = lambda: float3(1.0, 1.0, 1.0)

def ensure_float3(v) -> float3:
    return v if isinstance(v, float3) else float3.from_iter(v)


# ═════════════════════════════════════════════════════════════════════════════
# float3x3 – thin subclass of np.ndarray (shape (3,3))
# ═════════════════════════════════════════════════════════════════════════════
class float3x3(np.ndarray):
    """3×3 matrix stored in row‑major order (matches NumPy convention)."""

    # ---- constructor ------------------------------------------------------
    def __new__(cls, arr=None):
        if arr is None:
            arr = np.eye(3, dtype=float)
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (3, 3):
            raise ValueError("float3x3 must be initialised with a 3×3 array")
        return arr.view(cls)

    # class factories -------------------------------------------------------
    @staticmethod
    def identity():
        return float3x3(np.eye(3, dtype=float))

    @staticmethod
    def zeros():
        return float3x3(np.zeros((3, 3), dtype=float))

    @staticmethod
    def from_rows(r0: float3, r1: float3, r2: float3):
        return float3x3(np.vstack([r0, r1, r2]))

    @staticmethod
    def from_cols(c0: float3, c1: float3, c2: float3):
        return float3x3(np.column_stack([c0, c1, c2]))

    # linear‑algebra sugar ---------------------------------------------------
    def transpose(self):
        return float3x3(super().T)

    def det(self) -> float:
        return float(np.linalg.det(self))

    def inverse(self):
        return float3x3(np.linalg.inv(self))

    # matrix‑vector and matrix‑matrix use base ndarray @ operator ------------

    def __repr__(self):
        rows = ["  [" + ", ".join(f"{v:.4g}" for v in row) + "]" for row in self]
        return "float3x3(\n" + ",\n".join(rows) + "\n)"


# conversion helper ----------------------------------------------------------

def ensure_np(arr):
    return np.asarray(arr, dtype=float)
