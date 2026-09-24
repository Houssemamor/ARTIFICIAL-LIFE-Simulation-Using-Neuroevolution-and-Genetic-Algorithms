"""
Unit tests for spatial hash implementation.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import SpatialHash, Vector2D, AABB


def test_spatial_hash_creation():
    """Test spatial hash creation."""
    sh = SpatialHash(cell_size=10.0)
    assert sh.cell_size == 10.0
    assert len(sh.get_all_object_ids()) == 0

    # With bounds
    bounds = AABB(0, 0, 100, 100)
    sh2 = SpatialHash(cell_size=10.0, bounds=bounds)
    assert sh2.cell_size == 10.0
    assert sh2.bounds == bounds


def test_spatial_hash_add_remove():
    """Test adding and removing objects."""
    sh = SpatialHash(cell_size=10.0)

    # Add point object
    obj_id = sh.add_object(Vector2D(15.0, 25.0))
    assert obj_id == 0
    assert sh.get_object_position(obj_id) == Vector2D(15.0, 25.0)
    assert len(sh.get_all_object_ids()) == 1

    # Remove object
    sh.remove_object(obj_id)
    assert len(sh.get_all_object_ids()) == 0
    assert sh.get_object_position(obj_id) is None

    # Add AABB object
    bounds = AABB(10, 10, 20, 20)
    obj_id2 = sh.add_object(Vector2D(15.0, 15.0), bounds)
    assert obj_id2 == 0  # IDs reset after removal
    assert sh.get_object_position(obj_id2) == Vector2D(15.0, 15.0)
    assert sh.get_object_bounds(obj_id2) == bounds


def test_spatial_hash_update():
    """Test updating object positions."""
    sh = SpatialHash(cell_size=10.0)

    # Add object
    obj_id = sh.add_object(Vector2D(15.0, 25.0))
    original_pos = sh.get_object_position(obj_id)

    # Update to same cell (should be efficient)
    sh.update_object_position(obj_id, Vector2D(12.0, 22.0))  # Still in cell (1,2)
    assert sh.get_object_position(obj_id) == Vector2D(12.0, 22.0)

    # Update to different cell
    sh.update_object_position(obj_id, Vector2D(35.0, 45.0))  # Cell (3,4)
    assert sh.get_object_position(obj_id) == Vector2D(35.0, 45.0)

    # Test with AABB object
    bounds = AABB(0, 0, 5, 5)
    obj_id2 = sh.add_object(Vector2D(10.0, 10.0), bounds)
    original_bounds = sh.get_object_bounds(obj_id2)

    # Update position
    sh.update_object_position(obj_id2, Vector2D(20.0, 20.0))
    new_pos = sh.get_object_position(obj_id2)
    new_bounds = sh.get_object_bounds(obj_id2)
    assert new_pos == Vector2D(20.0, 20.0)
    # Bounds should move with the object
    expected_bounds = AABB(20, 20, 25, 25)
    assert new_bounds.min_x == expected_bounds.min_x
    assert new_bounds.min_y == expected_bounds.min_y
    assert new_bounds.max_x == expected_bounds.max_x
    assert new_bounds.max_y == expected_bounds.max_y


def test_spatial_hash_queries():
    """Test spatial queries."""
    sh = SpatialHash(cell_size=10.0)

    # Add some test objects
    sh.add_object(Vector2D(5.0, 5.0))  # ID 0, cell (0,0)
    sh.add_object(Vector2D(15.0, 25.0))  # ID 1, cell (1,2)
    sh.add_object(Vector2D(15.0, 15.0))  # ID 2, cell (1,1)
    sh.add_object(Vector2D(35.0, 45.0))  # ID 3, cell (3,4)

    # Add an AABB object
    bounds = AABB(12, 12, 18, 18)
    sh.add_object(Vector2D(15.0, 15.0), bounds)  # ID 4

    # Test point query
    points_at_origin = sh.query_point(Vector2D(0.0, 0.0))
    assert len(points_at_origin) == 0

    points_at_5_5 = sh.query_point(Vector2D(5.0, 5.0))
    assert len(points_at_5_5) == 1
    assert 0 in points_at_5_5  # Object 0

    # Test rectangle query
    # Rect must span y up to 30 to contain object 1 at (15.0, 25.0)
    query_rect = AABB(0, 0, 30, 30)  # Should catch objects 0, 1, 2, 4
    objects_in_rect = sh.query_rect(query_rect)
    assert len(objects_in_rect) >= 4  # At least these 4 objects
    assert 0 in objects_in_rect
    assert 1 in objects_in_rect
    assert 2 in objects_in_rect
    assert 4 in objects_in_rect

    # Test circle query
    objects_in_circle = sh.query_circle(Vector2D(0.0, 0.0), 10.0)
    assert len(objects_in_circle) >= 1
    assert 0 in objects_in_circle  # Object 0 is at distance ~7.07 from origin

    # Test empty queries
    empty_points = sh.query_point(Vector2D(100.0, 100.0))
    assert len(empty_points) == 0

    empty_rect = sh.query_rect(AABB(100.0, 100.0, 110.0, 110.0))
    assert len(empty_rect) == 0

    empty_circle = sh.query_circle(Vector2D(100.0, 100.0), 10.0)
    assert len(empty_circle) == 0


def test_spatial_hash_clear():
    """Test clearing the spatial hash."""
    sh = SpatialHash(cell_size=10.0)

    # Add some objects
    sh.add_object(Vector2D(5.0, 5.0))
    sh.add_object(Vector2D(15.0, 15.0))
    assert len(sh.get_all_object_ids()) == 2

    # Clear and verify
    sh.clear()
    assert len(sh.get_all_object_ids()) == 0
    assert sh.get_object_position(0) is None
    assert sh.get_object_position(1) is None