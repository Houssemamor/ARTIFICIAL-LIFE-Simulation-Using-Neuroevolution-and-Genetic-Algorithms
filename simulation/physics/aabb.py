"""
Axis-Aligned Bounding Box for spatial partitioning and collision detection.
"""

from __future__ import annotations
from typing import Union
from .vector2d import Vector2D


class AABB:
    """Axis-Aligned Bounding Box for spatial partitioning and collision detection."""

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)

    @classmethod
    def from_center_size(cls, center_x: float, center_y: float,
                         width: float, height: float) -> 'AABB':
        """Create AABB from center point and size."""
        half_width = width / 2
        half_height = height / 2
        return cls(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height
        )

    @classmethod
    def from_points(cls, points) -> 'AABB':
        """Create AABB that encompasses all given points."""
        if not points:
            return cls(0, 0, 0, 0)

        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        return cls(min_x, min_y, max_x, max_y)

    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.max_x - self.min_x

    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.max_y - self.min_y

    def center(self) -> Vector2D:
        """Get the center point of the bounding box."""
        return Vector2D(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)

    def contains_vector(self, vec: Vector2D) -> bool:
        """Check if a Vector2D is inside the bounding box."""
        return self.contains(vec.x, vec.y)

    def intersects(self, other: 'AABB') -> bool:
        """Check if this AABB intersects with another."""
        return not (self.max_x < other.min_x or
                   self.min_x > other.max_x or
                   self.max_y < other.min_y or
                   self.min_y > other.max_y)

    def intersection(self, other: 'AABB') -> 'AABB':
        """Get the intersection of two AABBs."""
        if not self.intersects(other):
            return AABB(0, 0, 0, 0)  # Empty intersection

        return AABB(
            max(self.min_x, other.min_x),
            max(self.min_y, other.min_y),
            min(self.max_x, other.max_x),
            min(self.max_y, other.max_y)
        )

    def __str__(self) -> str:
        return f"AABB({self.min_x:.2f},{self.min_y:.2f} to {self.max_x:.2f},{self.max_y:.2f})"

    def __eq__(self, other) -> bool:
        """Check if two AABBs are equal."""
        if not isinstance(other, AABB):
            return False
        return (self.min_x == other.min_x and
                self.min_y == other.min_y and
                self.max_x == other.max_x and
                self.max_y == other.max_y)

def __repr__(self) -> str:
        return f"AABB({self.min_x}, {self.min_y}, {self.max_x}, {self.max_y})"