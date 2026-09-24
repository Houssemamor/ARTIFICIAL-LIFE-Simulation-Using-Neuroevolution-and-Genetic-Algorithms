"""
Unit tests for physics engine implementation.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import Vector2D, PhysicsObject, integrate_euler, integrate_verlet, AABB


def test_aabb_creation():
    """Test AABB creation and basic properties."""
    # From min/max
    box = AABB(0.0, 0.0, 10.0, 5.0)
    assert box.min_x == 0.0
    assert box.min_y == 0.0
    assert box.max_x == 10.0
    assert box.max_y == 5.0
    assert box.width() == 10.0
    assert box.height() == 5.0
    assert box.center().x == 5.0
    assert box.center().y == 2.5

    # From center/size
    box2 = AABB.from_center_size(5.0, 2.5, 10.0, 5.0)
    assert box2.min_x == 0.0
    assert box2.min_y == 0.0
    assert box2.max_x == 10.0
    assert box2.max_y == 5.0
    assert box2.width() == 10.0
    assert box2.height() == 5.0
    assert box2.center().x == 5.0
    assert box2.center().y == 2.5

    # From points
    points = [(0, 0), (10, 5), (5, 3)]
    box3 = AABB.from_points(points)
    assert box3.min_x == 0.0
    assert box3.min_y == 0.0
    assert box3.max_x == 10.0
    assert box3.max_y == 5.0


def test_aabb_operations():
    """Test AABB operations like intersection and containment."""
    box1 = AABB(0.0, 0.0, 5.0, 5.0)
    box2 = AABB(3.0, 3.0, 8.0, 8.0)
    box3 = AABB(10.0, 10.0, 15.0, 15.0)  # No overlap

    # Intersection tests
    assert box1.intersects(box2) == True
    assert box1.intersects(box3) == False
    assert box2.intersects(box3) == False

    # Intersection area
    intersection = box1.intersection(box2)
    assert intersection.min_x == 3.0
    assert intersection.min_y == 3.0
    assert intersection.max_x == 5.0
    assert intersection.max_y == 5.0

    # Containment tests
    assert box1.contains(2.0, 2.0) == True
    assert box1.contains(5.0, 5.0) == True  # Edge case
    assert box1.contains(6.0, 6.0) == False
    assert box1.contains(-1.0, 2.0) == False

    # Vector containment
    vec_inside = Vector2D(2.0, 2.0)
    vec_outside = Vector2D(6.0, 6.0)
    assert box1.contains_vector(vec_inside) == True
    assert box1.contains_vector(vec_outside) == False


def test_physics_object():
    """Test basic physics object functionality."""
    obj = PhysicsObject(
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(1.0, 0.0),
        mass=2.0
    )

    assert obj.position.x == 0.0
    assert obj.position.y == 0.0
    assert obj.velocity.x == 1.0
    assert obj.velocity.y == 0.0
    assert obj.mass == 2.0
    assert obj.is_fixed() == False

    # Test force application
    obj.apply_force(Vector2D(10.0, 0.0))  # 10N force
    assert obj.force_accumulator.x == 10.0
    assert obj.force_accumulator.y == 0.0

    # Test impulse
    obj.apply_impulse(Vector2D(0.0, 5.0))  # 5 Ns impulse
    assert obj.velocity.x == 1.0  # Unchanged (impulse was in y)
    assert obj.velocity.y == 0.0 + 5.0 / 2.0  # 5 Ns / 2kg = 2.5 m/s

    # Test fixed object
    obj.set_fixed(True)
    assert obj.is_fixed() == True
    obj.apply_force(Vector2D(100.0, 0.0))  # Should have no effect
    assert obj.force_accumulator.x == 0.0  # Forces should be cleared when fixed
    assert obj.velocity.x == 1.0  # Velocity should remain unchanged when fixed
    obj.clear_forces()  # This should not affect fixed object's velocity
    assert obj.velocity.x == 1.0

    # Test energy and momentum
    obj.set_fixed(False)
    obj.velocity = Vector2D(3.0, 4.0)  # Speed 5 m/s
    assert obj.get_kinetic_energy() == 0.5 * 2.0 * 5.0 * 5.0  # 0.5 * m * v^2
    assert obj.get_momentum() == Vector2D(6.0, 8.0)  # m * v


def test_euler_integration():
    """Test Euler integration."""
    obj = PhysicsObject(
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        mass=1.0
    )

    # Apply constant force
    obj.apply_force(Vector2D(10.0, 0.0))  # 10N force on 1kg mass = 10 m/s^2 acceleration

    # Integrate for 1 second
    integrate_euler(obj, 1.0)

    # After 1s: v = a*t = 10*1 = 10 m/s
    # x = 0.5*a*t^2 = 0.5*10*1*1 = 5 m (starting from rest)
    assert abs(obj.velocity.x - 10.0) < 0.01  # Allow for damping
    assert abs(obj.position.x - 5.0) < 0.1
    assert abs(obj.velocity.y - 0.0) < 0.01
    assert abs(obj.position.y - 0.0) < 0.1

    # Test with initial velocity
    obj2 = PhysicsObject(
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(5.0, 0.0),  # 5 m/s initial velocity
        mass=1.0
    )
    obj2.apply_force(Vector2D(0.0, 10.0))  # 10N upward force

    integrate_euler(obj2, 1.0)
    # After 1s: vx = 5 m/s (unchanged), vy = 0 + 10*1 = 10 m/s
    # x = 5*1 = 5 m, y = 0.5*10*1^2 = 5 m
    assert abs(obj2.velocity.x - 5.0) < 0.1
    assert abs(obj2.velocity.y - 10.0) < 0.1
    assert abs(obj2.position.x - 5.0) < 0.1
    assert abs(obj2.position.y - 5.0) < 0.1


def test_verlet_integration():
    """Test Verlet integration."""
    obj = PhysicsObject(
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        mass=1.0
    )

    # Apply constant force
    obj.apply_force(Vector2D(10.0, 0.0))  # 10N force

    # First call needs previous position - we'll simulate it
    prev_pos = Vector2D(-5.0, 0.0)  # If moving at 5 m/s for 1s, we'd be at -5 -> 0
    integrate_verlet(obj, 1.0, prev_pos)
    # Should behave similarly to Euler but more stable

    # Test with zero initial conditions
    obj2 = PhysicsObject(
        position=Vector2D(0.0, 0.0),
        velocity=Vector2D(0.0, 0.0),
        mass=1.0
    )
    obj2.apply_force(Vector2D(10.0, 0.0))

    # First Verlet call falls back to Euler when no previous position
    integrate_verlet(obj2, 1.0)  # Should be same as Euler for first step
    assert abs(obj2.position.x - 5.0) < 0.1
    assert abs(obj2.velocity.x - 10.0) < 0.1
    assert abs(obj2.position.y - 0.0) < 0.1
    assert abs(obj2.velocity.y - 0.0) < 0.1

    # Second call with stored previous position
    prev_pos = Vector2D(obj2.position.x, obj2.position.y)  # Current position
    integrate_verlet(obj2, 1.0, prev_pos)  # Now we have proper previous position
    # After another second: with no force applied, position should stay at 5, velocity 0
    assert abs(obj2.position.x - 5.0) < 0.1
    assert abs(obj2.velocity.x - 0.0) < 0.1