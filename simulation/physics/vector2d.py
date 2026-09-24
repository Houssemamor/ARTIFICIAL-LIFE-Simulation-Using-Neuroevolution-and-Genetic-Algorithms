"""
2D Vector mathematics for physics calculations.
Provides basic vector operations used throughout the simulation.
"""

import math
from typing import Tuple


class Vector2D:
    """A 2D vector for position, velocity, acceleration, and forces."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float]) -> 'Vector2D':
        """Create Vector2D from a tuple (x, y)."""
        return cls(coords[0], coords[1])

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector addition."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        """Vector subtraction."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        """Scalar multiplication."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Vector2D':
        """Scalar division."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Vector2D':
        """Vector negation."""
        return Vector2D(-self.x, -self.y)

    def magnitude(self) -> float:
        """Calculate the magnitude (length) of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (avoids sqrt for efficiency)."""
        return self.x * self.x + self.y * self.y

    def normalize(self) -> 'Vector2D':
        """Return a normalized version of this vector (length = 1)."""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def dot(self, other: 'Vector2D') -> float:
        """Calculate the dot product with another vector."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return self.x * other.x + self.y * other.y

    def angle_to(self, other: 'Vector2D') -> float:
        """Calculate the angle to another vector in radians."""
        dot_product = self.dot(other)
        magnitude_product = self.magnitude() * other.magnitude()
        if magnitude_product == 0:
            return 0.0
        # Clamp to [-1, 1] to avoid floating point errors
        clamped = max(-1.0, min(1.0, dot_product / magnitude_product))
        return math.acos(clamped)

    def distance_to(self, other: 'Vector2D') -> float:
        """Calculate the Euclidean distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_squared_to(self, other: 'Vector2D') -> float:
        """Calculate the squared distance to another vector (avoids sqrt)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def limit_magnitude(self, max_mag: float) -> 'Vector2D':
        """Return a vector with magnitude limited to max_mag."""
        if self.magnitude() <= max_mag:
            return Vector2D(self.x, self.y)
        return self.normalize() * max_mag

    def __str__(self) -> str:
        return f"Vector2D({self.x:.3f}, {self.y:.3f})"

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return False
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10


# Convenience functions
def vec2(x: float, y: float) -> Vector2D:
    """Create a Vector2D instance."""
    return Vector2D(x, y)


def zero_vector() -> Vector2D:
    """Create a zero vector."""
    return Vector2D(0.0, 0.0)


def lerp(start: Vector2D, end: Vector2D, t: float) -> Vector2D:
    """Linear interpolation between two vectors."""
    return start + (end - start) * max(0.0, min(1.0, t))