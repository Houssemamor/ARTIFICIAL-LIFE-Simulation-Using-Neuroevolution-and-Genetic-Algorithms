#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.physics import SpatialHash, Vector2D, AABB

print("Debugging rectangle query...")

sh = SpatialHash(cell_size=10.0)

# Add some test objects
sh.add_object(Vector2D(5.0, 5.0))  # ID 0, cell (0,0)
print(f"Added object 0: position={sh.get_object_position(0)}")

sh.add_object(Vector2D(15.0, 25.0))  # ID 1, cell (1,2)
print(f"Added object 1: position={sh.get_object_position(1)}")

sh.add_object(Vector2D(15.0, 15.0))  # ID 2, cell (1,1)
print(f"Added object 2: position={sh.get_object_position(2)}")

sh.add_object(Vector2D(35.0, 45.0))  # ID 3, cell (3,4)
print(f"Added object 3: position={sh.get_object_position(3)}")

# Add an AABB object
bounds = AABB(12, 12, 18, 18)
sh.add_object(Vector2D(15.0, 15.0), bounds)  # ID 4
print(f"Added object 4: position={sh.get_object_position(4)}, bounds={sh.get_object_bounds(4)}")

# Test rectangle query
query_rect = AABB(0, 0, 20, 20)
print(f"\nQuery rectangle: {query_rect}")

# Let's manually check what should be in each step
print(f"\nStep 1: Calculate cell range")
min_cell = sh._get_cell_coords(Vector2D(query_rect.min_x, query_rect.min_y))
max_cell = sh._get_cell_coords(Vector2D(query_rect.max_x, query_rect.max_y))
print(f"  min_cell: {min_cell}  # for ({query_rect.min_x}, {query_rect.min_y})")
print(f"  max_cell: {max_cell}  # for ({query_rect.max_x}, {query_rect.max_y})")

print(f"  Cells to check: x from {min_cell[0]} to {max_cell[0]}, y from {min_cell[1]} to {max_cell[1]}")

print(f"\nStep 2: Collect candidates from overlapping cells")
candidates = set()
for cell_x in range(min_cell[0], max_cell[0] + 1):
    for cell_y in range(min_cell[1], max_cell[1] + 1):
        cell_coords = (cell_x, cell_y)
        if cell_coords in sh.grid:
            print(f"    Cell {cell_coords} contains objects: {sh.grid[cell_coords]}")
            candidates.update(sh.grid[cell_coords])
        else:
            print(f"    Cell {cell_coords} is empty")

print(f"  Final candidates: {candidates}")

print(f"\nStep 3: Filter by actual intersection")
result = []
for obj_id in candidates:
    print(f"  Checking object {obj_id}:")
    if obj_id in sh.object_bounds and sh.object_bounds[obj_id] is not None:
        bounds_obj = sh.object_bounds[obj_id]
        intersects = bounds_obj.intersects(query_rect)
        print(f"    AABB object with bounds {bounds_obj}")
        print(f"    Intersects with {query_rect}? {intersects}")
        if intersects:
            result.append(obj_id)
            print(f"    -> ADDED to result")
    elif obj_id in sh.object_positions:
        pos_obj = sh.object_positions[obj_id]
        contains = query_rect.contains_vector(pos_obj)
        print(f"    Point object at {pos_obj}")
        print(f"    Contains point {pos_obj}? {contains}")
        if contains:
            result.append(obj_id)
            print(f"    -> ADDED to result")
    else:
        print(f"    Object not found in positions or bounds!")

print(f"\nFinal result: {result}")
print(f"Expected to find at least objects 0, 1, 2, 4")

# Let's also check each object individually against the query rect
print(f"\nIndividual checks:")
for i in range(5):
    if i in sh.object_positions:
        pos = sh.object_positions[i]
        in_rect = query_rect.contains_vector(pos)
        print(f"  Object {i} at {pos}: in rect = {in_rect}")
    elif i in sh.object_bounds and sh.object_bounds[i] is not None:
        bounds = sh.object_bounds[i]
        # For AABB object, check if bounds intersect query rect
        intersects = bounds.intersects(query_rect)
        print(f"  Object {i} with bounds {bounds}: intersects rect = {intersects}")
    else:
        print(f"  Object {i}: not found")