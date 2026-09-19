#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import PhysicsObject, integrate_verlet, integrate_euler, Vector2D

print("Debugging verlet integration (which falls back to euler)...")

# Test with zero initial conditions - this is what the verlet test does
obj2 = PhysicsObject(
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    mass=1.0
)
obj2.apply_force(Vector2D(10.0, 0.0))

print(f"Before integration:")
print(f"  position: {obj2.position}")
print(f"  velocity: {obj2.velocity}")
print(f"  force_accumulator: {obj2.force_accumulator}")

# First Verlet call falls back to Euler when no previous position
result = integrate_verlet(obj2, 1.0)  # Should be same as Euler for first step

print(f"After integration:")
print(f"  position: {obj2.position}")
print(f"  velocity: {obj2.velocity}")
print(f"  force_accumulator: {obj2.force_accumulator}")
print(f"  return value: {result}")

# Test assertions
print(f"\nTest assertions:")
position_diff = abs(obj2.position.x - 5.0)
print(f"  abs(obj2.position.x - 5.0) = {position_diff}")
print(f"  Expected: < 0.1")
print(f"  Actual: {position_diff} {'<' if position_diff < 0.1 else '>='} 0.1")

velocity_diff = abs(obj2.velocity.x - 10.0)
print(f"  abs(obj2.velocity.x - 10.0) = {velocity_diff}")
print(f"  Expected: < 0.1")
print(f"  Actual: {velocity_diff} {'<' if velocity_diff < 0.1 else '>='} 0.1")

# Let's also see what pure euler gives
print(f"\nFor comparison, pure Euler integration:")
obj3 = PhysicsObject(
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    mass=1.0
)
obj3.apply_force(Vector2D(10.0, 0.0))
integrate_euler(obj3, 1.0)
print(f"  position: {obj3.position}")
print(f"  velocity: {obj3.velocity}")