#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import PhysicsObject, integrate_euler, Vector2D

print("Replicating exact test_euler_integration...")

obj = PhysicsObject(
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    mass=1.0
)

print(f"After creation:")
print(f"  position: {obj.position}")
print(f"  velocity: {obj.velocity}")
print(f"  mass: {obj.mass}")
print(f"  force_accumulator: {obj.force_accumulator}")

# Apply constant force
obj.apply_force(Vector2D(10.0, 0.0))  # 10N force on 1kg mass = 10 m/s^2 acceleration
print(f"\nAfter apply_force(Vector2D(10.0, 0.0)):")
print(f"  force_accumulator: {obj.force_accumulator}")

# Integrate for 1 second
integrate_euler(obj, 1.0)
print(f"\nAfter integrate_euler(obj, 1.0):")
print(f"  position: {obj.position}")
print(f"  velocity: {obj.velocity}")
print(f"  force_accumulator: {obj.force_accumulator}")

# Test assertions
print(f"\nTest assertions:")
velocity_diff = abs(obj.velocity.x - 10.0)
print(f"  abs(obj.velocity.x - 10.0) = {velocity_diff}")
print(f"  Expected: < 0.01")
print(f"  Actual: {velocity_diff} {'<' if velocity_diff < 0.01 else '>='} 0.01")

position_diff = abs(obj.position.x - 5.0)
print(f"  abs(obj.position.x - 5.0) = {position_diff}")
print(f"  Expected: < 0.1")
print(f"  Actual: {position_diff} {'<' if position_diff < 0.1 else '>='} 0.1")

# Let's also check what the test comment says the expected values should be
print(f"\nAccording to test comment:")
print(f"  After 1s: v = a*t = 10*1 = 10 m/s")
print(f"  x = 0.5*a*t^2 = 0.5*10*1*1 = 5 m (starting from rest)")
print(f"  Allow for damping in velocity check")