#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import PhysicsObject, integrate_euler, Vector2D

print("Creating physics object...")
obj = PhysicsObject(
    position=Vector2D(0.0, 0.0),
    velocity=Vector2D(0.0, 0.0),
    mass=1.0
)

print(f"Initial state:")
print(f"  position: {obj.position}")
print(f"  velocity: {obj.velocity}")
print(f"  mass: {obj.mass}")
print(f"  fixed: {obj.fixed}")
print(f"  force_accumulator: {obj.force_accumulator}")

print("\nApplying force (10, 0)...")
obj.apply_force(Vector2D(10.0, 0.0))
print(f"  force_accumulator: {obj.force_accumulator}")

print("\nIntegrating with dt=1.0...")
print(f"  Before integration:")
print(f"    position: {obj.position}")
print(f"    velocity: {obj.velocity}")
print(f"    force_accumulator: {obj.force_accumulator}")

integrate_euler(obj, 1.0)

print(f"  After integration:")
print(f"    position: {obj.position}")
print(f"    velocity: {obj.velocity}")
print(f"    force_accumulator: {obj.force_accumulator}")

# Calculate expected values
print(f"\nExpected (without damping):")
print(f"  acceleration = force / mass = {obj.force_accumulator} / {obj.mass} = {obj.force_accumulator / obj.mass}")
print(f"  velocity_change = acceleration * dt = {(obj.force_accumulator / obj.mass)} * 1.0 = {(obj.force_accumulator / obj.mass) * 1.0}")
print(f"  final_velocity = initial_velocity + velocity_change = {Vector2D(0,0)} + {(obj.force_accumulator / obj.mass) * 1.0} = {(obj.force_accumulator / obj.mass) * 1.0}")
print(f"  position_change = 0.5 * acceleration * dt^2 = 0.5 * {(obj.force_accumulator / obj.mass)} * 1.0^2 = {0.5 * (obj.force_accumulator / obj.mass)}")
print(f"  final_position = initial_position + position_change = {Vector2D(0,0)} + {0.5 * (obj.force_accumulator / obj.mass)} = {0.5 * (obj.force_accumulator / obj.mass)}")