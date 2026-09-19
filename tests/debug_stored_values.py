#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import SpatialHash, Vector2D, AABB

print("Checking stored values...")

sh = SpatialHash(cell_size=10.0)

# Add some test objects
sh.add_object(Vector2D(5.0, 5.0))  # ID 0
sh.add_object(Vector2D(15.0, 25.0))  # ID 1
sh.add_object(Vector2D(15.0, 15.0))  # ID 2
sh.add_object(Vector2D(35.0, 45.0))  # ID 3

# Add an AABB object
bounds = AABB(12, 12, 18, 18)
sh.add_object(Vector2D(15.0, 15.0), bounds)  # ID 4

print("Stored positions:")
for i in range(5):
    if i in sh.object_positions:
        pos = sh.object_positions[i]
        print(f"  Object {i}: {pos}")
    else:
        print(f"  Object {i}: not in positions")

print("\nStored bounds:")
for i in range(5):
    if i in sh.object_bounds and sh.object_bounds[i] is not None:
        bounds = sh.object_bounds[i]
        print(f"  Object {i}: {bounds}")
    elif i in sh.object_bounds:
        print(f"  Object {i}: None (point object)")
    else:
        print(f"  Object {i}: not in bounds dict")

# Let's also check what the grid contains
print(f"\nGrid contents:")
for cell, objects in sh.grid.items():
    print(f"  Cell {cell}: {objects}")