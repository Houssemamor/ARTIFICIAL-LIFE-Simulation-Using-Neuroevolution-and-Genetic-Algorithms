#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import SpatialHash, Vector2D, AABB

print("=== Debugging test_spatial_hash_update step by step ===")

sh = SpatialHash(cell_size=10.0)

print("\n1. Creating AABB object:")
bounds = AABB(0, 0, 5, 5)
print(f"   bounds = {bounds}")
obj_id2 = sh.add_object(Vector2D(10.0, 10.0), bounds)
print(f"   Returned obj_id: {obj_id2}")

print("\n2. State after add_object:")
pos = sh.get_object_position(obj_id2)
bnds = sh.get_object_bounds(obj_id2)
print(f"   position = {pos}")
print(f"   bounds = {bnds}")

print("\n3. Calculating expected state after update:")
expected_new_pos = Vector2D(20.0, 20.0)
delta = Vector2D(expected_new_pos.x - pos.x, expected_new_pos.y - pos.y)
print(f"   delta = {delta}")
expected_new_bounds = AABB(
    bnds.min_x + delta.x,
    bnds.min_y + delta.y,
    bnds.max_x + delta.x,
    bnds.max_y + delta.y
)
print(f"   expected new bounds = {expected_new_bounds}")
print(f"     min_x: {expected_new_bounds.min_x}")
print(f"     min_y: {expected_new_bounds.min_y}")
print(f"     max_x: {expected_new_bounds.max_x}")
print(f"     max_y: {expected_new_bounds.max_y}")

print("\n4. Calling update_object_position:")
sh.update_object_position(obj_id2, expected_new_pos)

print("\n5. State after update_object_position:")
new_pos = sh.get_object_position(obj_id2)
new_bounds = sh.get_object_bounds(obj_id2)
print(f"   position = {new_pos}")
print(f"   bounds = {new_bounds}")
print(f"     min_x: {new_bounds.min_x}")
print(f"     min_y: {new_bounds.min_y}")
print(f"     max_x: {new_bounds.max_x}")
print(f"     max_y: {new_bounds.max_y}")

print("\n6. Comparison:")
print(f"   position correct? {new_pos == expected_new_pos}")
print(f"   bounds min_x correct? {new_bounds.min_x == expected_new_bounds.min_x}")
print(f"   bounds min_y correct? {new_bounds.min_y == expected_new_bounds.min_y}")
print(f"   bounds max_x correct? {new_bounds.max_x == expected_new_bounds.max_x}")
print(f"   bounds max_y correct? {new_bounds.max_y == expected_new_bounds.max_y}")

# Also test equality directly
print(f"   bounds equal? {new_bounds == expected_new_bounds}")

# Let's also check what the test is specifically asserting
print(f"\nTest assertion: new_bounds.min_x == {expected_new_bounds.min_x}")
print(f"Actual: new_bounds.min_x = {new_bounds.min_x}")
print(f"Test passes? {new_bounds.min_x == expected_new_bounds.min_x}")