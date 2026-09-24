"""
Unit tests for Vector2D implementation.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics.vector2d import Vector2D, vec2, zero_vector, lerp


def test_vector2d_creation():
    """Test basic vector creation."""
    v = Vector2D(3.0, 4.0)
    assert v.x == 3.0
    assert v.y == 4.0

    v2 = vec2(1.0, 2.0)
    assert v2.x == 1.0
    assert v2.y == 2.0

    zero = zero_vector()
    assert zero.x == 0.0
    assert zero.y == 0.0


def test_vector2d_operations():
    """Test vector arithmetic operations."""
    v1 = Vector2D(3.0, 4.0)
    v2 = Vector2D(1.0, 2.0)

    # Addition
    v3 = v1 + v2
    assert v3.x == 4.0
    assert v3.y == 6.0

    # Subtraction
    v4 = v1 - v2
    assert v4.x == 2.0
    assert v4.y == 2.0

    # Scalar multiplication
    v5 = v1 * 2.0
    assert v5.x == 6.0
    assert v5.y == 8.0

    # Scalar division
    v6 = v1 / 2.0
    assert v6.x == 1.5
    assert v6.y == 2.0

    # Negation
    v7 = -v1
    assert v7.x == -3.0
    assert v7.y == -4.0


def test_vector2d_magnitude():
    """Test vector magnitude calculations."""
    v = Vector2D(3.0, 4.0)
    assert v.magnitude() == 5.0  # 3-4-5 triangle
    assert v.magnitude_squared() == 25.0

    zero = zero_vector()
    assert zero.magnitude() == 0.0
    assert zero.magnitude_squared() == 0.0


def test_vector2d_normalize():
    """Test vector normalization."""
    v = Vector2D(3.0, 4.0)
    v_norm = v.normalize()
    assert abs(v_norm.magnitude() - 1.0) < 1e-10

    # Zero vector should remain zero
    zero = zero_vector()
    zero_norm = zero.normalize()
    assert zero_norm.x == 0.0
    assert zero_norm.y == 0.0


def test_vector2d_dot_product():
    """Test dot product calculation."""
    v1 = Vector2D(1.0, 0.0)
    v2 = Vector2D(0.0, 1.0)
    assert v1.dot(v2) == 0.0  # Perpendicular vectors

    v3 = Vector2D(1.0, 0.0)
    v4 = Vector2D(2.0, 0.0)
    assert v3.dot(v4) == 2.0  # Parallel vectors

    v5 = Vector2D(1.0, 2.0)
    v6 = Vector2D(3.0, 4.0)
    assert v5.dot(v6) == 11.0  # 1*3 + 2*4


def test_vector2d_distance():
    """Test distance calculations."""
    v1 = Vector2D(0.0, 0.0)
    v2 = Vector2D(3.0, 4.0)
    assert v1.distance_to(v2) == 5.0
    assert v1.distance_squared_to(v2) == 25.0


def test_vector2d_limits():
    """Test vector magnitude limiting."""
    v = Vector2D(3.0, 4.0)  # Magnitude 5.0
    v_limited = v.limit_magnitude(2.5)
    assert abs(v_limited.magnitude() - 2.5) < 1e-10
    # Should be in same direction
    assert abs(v_limited.x / v_limited.y - v.x / v.y) < 1e-10

    # Vector already under limit should be unchanged
    v_small = Vector2D(1.0, 1.0)
    v_limited_small = v_small.limit_magnitude(10.0)
    assert v_limited_small.x == v_small.x
    assert v_limited_small.y == v_small.y


def test_lerp():
    """Test linear interpolation."""
    v1 = Vector2D(0.0, 0.0)
    v2 = Vector2D(10.0, 20.0)

    # t=0 should return start
    assert lerp(v1, v2, 0.0) == v1

    # t=1 should return end
    result = lerp(v1, v2, 1.0)
    assert result.x == 10.0
    assert result.y == 20.0

    # t=0.5 should return midpoint
    result = lerp(v1, v2, 0.5)
    assert result.x == 5.0
    assert result.y == 10.0

    # t values outside [0,1] should be clamped
    result = lerp(v1, v2, -1.0)  # Should clamp to 0
    assert result == v1

    result = lerp(v1, v2, 2.0)  # Should clamp to 1
    assert result.x == 10.0
    assert result.y == 20.0


def test_vector2d_equality():
    """Test vector equality comparison."""
    v1 = Vector2D(1.0, 2.0)
    v2 = Vector2D(1.0, 2.0)
    v3 = Vector2D(1.0, 2.0000000001)  # Very close but not equal

    assert v1 == v2
    assert v1 != v3
    assert v1 != "not a vector"