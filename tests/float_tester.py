# tests/test_mathutils.py
import unittest
import numpy as np
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from engine.math import float3, float3x3, ensure_float3
from engine.transform  import Transform
from engine.quaternion import (
    quat_identity, quat_from_axis_angle, quat_mul, quat_normalize
)


# ───────────────────────────────────────────── float3
class TestFloat3(unittest.TestCase):
    def test_construction_and_repr(self):
        v = float3(1, 2, 3)
        self.assertEqual(repr(v), "float3(1, 2, 3)")
        self.assertTrue(isinstance(v, np.ndarray))
        self.assertEqual(v.shape, (3,))

    def test_arithmetic(self):
        a = float3(1, 2, 3)
        b = float3(3, -2, 1)
        self.assertTrue(np.allclose(a + b, [4, 0, 4]))
        self.assertTrue(np.allclose(a - b, [-2, 4, 2]))
        self.assertTrue(np.allclose(2 * a, [2, 4, 6]))
        self.assertTrue(np.allclose(a * 0.5, [0.5, 1, 1.5]))

    def test_dot_cross_norm(self):
        a = float3(1, 0, 0)
        b = float3(0, 1, 0)
        self.assertAlmostEqual(a.dot(b), 0.0)
        self.assertTrue(np.allclose(a.cross(b), [0, 0, 1]))
        self.assertAlmostEqual(float3(3, 4, 0).norm(), 5.0)
        self.assertAlmostEqual(float3(3, 0, 4).normalized().norm(), 1.0, places=6)

    def test_numpy_ufuncs(self):
        v = float3(0, np.pi, 2*np.pi)
        self.assertTrue(np.allclose(np.cos(v), [1, -1, 1]))  # ufunc works in‑place
        self.assertTrue(np.allclose(v + np.array([1,1,1]), [1, np.pi+1, 2*np.pi+1]))


# ───────────────────────────────────────────── float3x3
class TestFloat3x3(unittest.TestCase):
    def test_identity_and_mul(self):
        I = float3x3.identity()
        v = float3(5, -1, 2)
        self.assertTrue(np.allclose(I @ v, v))
        self.assertTrue(np.allclose(I @ I, I))

    def test_row_vs_col_constructors(self):
        r0, r1, r2 = float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)
        A = float3x3.from_rows(r0, r1, r2)
        B = float3x3.from_cols(r0, r1, r2)
        self.assertTrue(np.allclose(A, np.eye(3)))
        self.assertTrue(np.allclose(B, np.eye(3)))

    def test_inverse(self):
        M = float3x3.from_rows(float3(2, 0, 0),
                               float3(0, 3, 0),
                               float3(0, 0, 4))
        Minv = M.inverse()
        self.assertTrue(np.allclose(M @ Minv, np.eye(3)))


# ───────────────────────────────────────────── Transform
class TestTransform(unittest.TestCase):
    def test_setters_and_translate(self):
        t = Transform()
        t.set_position(float3(1, 2, 3))
        self.assertTrue(np.allclose(t.position, [1, 2, 3]))
        q = quat_from_axis_angle((0, 1, 0), np.pi / 4)
        t.set_rotation(q)
        self.assertTrue(np.allclose(t.rotation, quat_normalize(q)))

        t.translate(float3(0, 0, 1))
        self.assertTrue(np.allclose(t.position, [1, 2, 4]))

    def test_rotate_about_axis(self):
        t = Transform()
        t.rotate_about_axis((0, 0, 1), np.pi)
        expected = quat_from_axis_angle((0, 0, 1), np.pi)
        self.assertTrue(np.allclose(t.rotation, quat_normalize(expected)))


if __name__ == "__main__":
    unittest.main()
