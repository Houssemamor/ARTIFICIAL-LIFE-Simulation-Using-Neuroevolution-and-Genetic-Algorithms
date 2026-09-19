#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import SpatialHash, Vector2D, AABB

print("=== Debugging test_spatial_hash_update ===")

sh = SpatialHash(cell_size=10.0)

# Add object
bounds = AABB(0, 0, 5, 5)
print(f"Adding object with position=(10,10) and bounds={bounds}")
obj_id2 = sh.add_object(Vector2D(10.0, 10.0), bounds)
print(f"After add_object:")
print(f"  position: {sh.get_object_position(obj_id2)}")
print(f"  bounds: {sh.get_object_bounds(obj_id2)}")

# Update position
print(f"\nUpdating position to (20,20)")
sh.update_object_position(obj_id2, Vector2D(20.0, 20.0))
print(f"After update_object_position:")
print(f"  position: {sh.get_object_position(obj_id2)}")
print(f"  bounds: {sh.get_object_bounds(obj_id2)}")

# Check expectations
new_pos = sh.get_object_position(obj_id2)
new_bounds = sh.get_object_bounds(obj_id2)
expected_bounds = AABB(20, 20, 25, 25)

print(f"\nResults:")
print(f"  new_pos == Vector2D(20,20)? {new_pos == Vector2D(20.0, 20.0)}")
print(f"  new_bounds == AABB(20,20,25,25)? {new_bounds == expected_bounds}")
print(f"  Expected bounds: {expected_bounds}")
print(f"  Actual bounds: {new_bounds}")