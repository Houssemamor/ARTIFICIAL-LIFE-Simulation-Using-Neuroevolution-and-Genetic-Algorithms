"""
Spatial partitioning using a grid-based hash for efficient broad-phase collision detection.
"""

from typing import Dict, List, Tuple, Set, Optional
from .vector2d import Vector2D
from .aabb import AABB


class SpatialHash:
    """
    Grid-based spatial hash for efficient spatial queries.
    Divides space into equal-sized cells and tracks which objects occupy each cell.
    """

    def __init__(self, cell_size: float, bounds: AABB = None):
        """
        Initialize spatial hash.

        Args:
            cell_size: Size of each grid cell (should be roughly average object size)
            bounds: Optional bounding box for the entire space (if None, grows dynamically)
        """
        self.cell_size = max(cell_size, 0.1)  # Prevent division by zero
        self.bounds = bounds
        self.grid: Dict[Tuple[int, int], Set[int]] = {}
        self.object_positions: Dict[int, Vector2D] = {}
        self.object_bounds: Dict[int, AABB] = {}  # Stores whatever was passed as bounds parameter
        self.next_object_id = 0
        self.free_ids: List[int] = []  # Stack of free IDs for reuse

    def _get_cell_coords(self, position: Vector2D) -> Tuple[int, int]:
        """Get grid cell coordinates for a position."""
        cell_x = int(position.x // self.cell_size)
        cell_y = int(position.y // self.cell_size)
        return (cell_x, cell_y)

    def _get_cell_range(self, aabb: AABB) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get range of grid cells that an AABB overlaps."""
        min_cell = self._get_cell_coords(Vector2D(aabb.min_x, aabb.min_y))
        max_cell = self._get_cell_coords(Vector2D(aabb.max_x, aabb.max_y))
        return (min_cell, max_cell)

    def _add_to_grid(self, obj_id: int, position: Vector2D):
        """Add object to grid cell based on position."""
        cell_coords = self._get_cell_coords(position)
        if cell_coords not in self.grid:
            self.grid[cell_coords] = set()
        self.grid[cell_coords].add(obj_id)

    def _remove_from_grid(self, obj_id: int, position: Vector2D):
        """Remove object from grid cell based on position."""
        cell_coords = self._get_cell_coords(position)
        if cell_coords in self.grid:
            self.grid[cell_coords].discard(obj_id)
            # Clean up empty cells
            if not self.grid[cell_coords]:
                del self.grid[cell_coords]

    def _allocate_object_id(self) -> int:
        """Allocate an object ID, reusing free IDs if available."""
        if self.free_ids:
            return self.free_ids.pop()
        else:
            obj_id = self.next_object_id
            self.next_object_id += 1
            return obj_id

    def _free_object_id(self, obj_id: int):
        """Free an object ID for future reuse."""
        self.free_ids.append(obj_id)

    def add_object(self, position: Vector2D, bounds: AABB = None) -> int:
        """
        Add an object to the spatial hash.

        Args:
            position: Position of the object (reference point)
            bounds: Optional bounding box (if None, treated as point object)

        Returns:
            Object ID for later reference/update/removal
        """
        obj_id = self._allocate_object_id()

        self.object_positions[obj_id] = Vector2D(position.x, position.y)
        # Store bounds as-is (they represent the object's world bounds)
        self.object_bounds[obj_id] = bounds

        # Add to all relevant grid cells
        if bounds is None:
            # Point object - only in one cell
            self._add_to_grid(obj_id, position)
        else:
            # AABB object - add to all overlapping cells
            min_cell, max_cell = self._get_cell_range(bounds)
            for cell_x in range(min_cell[0], max_cell[0] + 1):
                for cell_y in range(min_cell[1], max_cell[1] + 1):
                    self._add_to_grid(obj_id, Vector2D(cell_x * self.cell_size,
                                                       cell_y * self.cell_size))

        return obj_id

    def remove_object(self, obj_id: int):
        """Remove an object from the spatial hash."""
        if obj_id in self.object_positions:
            position = self.object_positions[obj_id]
            # Note: We don't need the bounds for removal since we store them

            # Remove from grid
            bounds = self.object_bounds[obj_id]
            if bounds is None:
                # Point object
                self._remove_from_grid(obj_id, position)
            else:
                # AABB object
                min_cell, max_cell = self._get_cell_range(bounds)
                for cell_x in range(min_cell[0], max_cell[0] + 1):
                    for cell_y in range(min_cell[1], max_cell[1] + 1):
                        self._remove_from_grid(obj_id, Vector2D(cell_x * self.cell_size,
                                                               cell_y * self.cell_size))

            # Clean up object data
            del self.object_positions[obj_id]
            if obj_id in self.object_bounds:
                del self.object_bounds[obj_id]

            # Free the ID for reuse
            self._free_object_id(obj_id)

    def update_object_position(self, obj_id: int, new_position: Vector2D):
        """Update an object's position in the spatial hash."""
        if obj_id not in self.object_positions:
            raise KeyError(f"Object {obj_id} not found in spatial hash")

        old_position = self.object_positions[obj_id]
        old_bounds = self.object_bounds[obj_id]

        # If object hasn't moved cells, no update needed
        if old_bounds is None:
            # Point object
            old_cell = self._get_cell_coords(old_position)
            new_cell = self._get_cell_coords(new_position)
            if old_cell == new_cell:
                self.object_positions[obj_id] = Vector2D(new_position.x, new_position.y)
                return
        else:
            # AABB object - check if bounding box changed cells significantly
            # Calculate delta in position
            delta_x = new_position.x - old_position.x
            delta_y = new_position.y - old_position.y

            # Calculate what the new bounds would be if we move with the position
            # The bounds move twice as fast as the position reference point
            new_bounds = AABB(
                old_bounds.min_x + 2 * delta_x,
                old_bounds.min_y + 2 * delta_y,
                old_bounds.max_x + 2 * delta_x,
                old_bounds.max_y + 2 * delta_y
            )

            # Check if the new bounds would be in the same cells as the old bounds
            old_min_cell, old_max_cell = self._get_cell_range(old_bounds)
            new_min_cell, new_max_cell = self._get_cell_range(new_bounds)

            if (old_min_cell == new_min_cell and old_max_cell == new_max_cell):
                # Bounds haven't changed cells significantly - just update position and bounds
                self.object_positions[obj_id] = Vector2D(new_position.x, new_position.y)
                self.object_bounds[obj_id] = new_bounds
                return

        # Object moved cells significantly - remove and re-add
        self.remove_object(obj_id)
        if old_bounds is None:
            # Point object
            self.add_object(new_position, None)
        else:
            # AABB object - use the already calculated new_bounds
            self.add_object(new_position, new_bounds)

    def query_point(self, position: Vector2D) -> List[int]:
        """Find all objects that occupy the given point (exact position for points, containment for bounds)."""
        cell_coords = self._get_cell_coords(position)
        if cell_coords not in self.grid:
            return []

        result = []
        for obj_id in self.grid[cell_coords]:
            # Check point objects: exact position match
            if obj_id in self.object_positions and self.object_bounds.get(obj_id) is None:
                obj_pos = self.object_positions[obj_id]
                if obj_pos.x == position.x and obj_pos.y == position.y:
                    result.append(obj_id)
            # Check AABB objects: point containment
            elif obj_id in self.object_bounds and self.object_bounds[obj_id] is not None:
                if self.object_bounds[obj_id].contains_vector(position):
                    result.append(obj_id)

        return result

    def query_rect(self, aabb: AABB) -> List[int]:
        """Find all objects that intersect with the given AABB."""
        if not self.grid:
            return []

        min_cell, max_cell = self._get_cell_range(aabb)
        candidates: Set[int] = set()

        # Collect all objects in overlapping cells
        for cell_x in range(min_cell[0], max_cell[0] + 1):
            for cell_y in range(min_cell[1], max_cell[1] + 1):
                cell_coords = (cell_x, cell_y)
                if cell_coords in self.grid:
                    candidates.update(self.grid[cell_coords])

        # Filter by actual AABB intersection
        result = []
        for obj_id in candidates:
            if obj_id in self.object_bounds and self.object_bounds[obj_id] is not None:
                if self.object_bounds[obj_id].intersects(aabb):
                    result.append(obj_id)
            elif obj_id in self.object_positions:
                # Point object - check if point is in AABB
                if aabb.contains_vector(self.object_positions[obj_id]):
                    result.append(obj_id)

        return result

    def query_circle(self, center: Vector2D, radius: float) -> List[int]:
        """Find all objects within a circular area."""
        # Create bounding box for the circle and then filter precisely
        search_aabb = AABB(
            center.x - radius, center.y - radius,
            center.x + radius, center.y + radius
        )
        candidates = self.query_rect(search_aabb)

        result = []
        radius_squared = radius * radius
        for obj_id in candidates:
            if obj_id in self.object_positions:
                obj_pos = self.object_positions[obj_id]
                dx = obj_pos.x - center.x
                dy = obj_pos.y - center.y
                if dx * dx + dy * dy <= radius_squared:
                    result.append(obj_id)
            elif obj_id in self.object_bounds and self.object_bounds[obj_id] is not None:
                # For AABB objects, check if any corner is within radius (conservative)
                bounds = self.object_bounds[obj_id]
                corners = [
                    Vector2D(bounds.min_x, bounds.min_y),
                    Vector2D(bounds.min_x, bounds.max_y),
                    Vector2D(bounds.max_x, bounds.min_y),
                    Vector2D(bounds.max_x, bounds.max_y)
                ]
                if any(corner.distance_squared_to(center) <= radius_squared for corner in corners):
                    result.append(obj_id)

        return result

    def get_all_object_ids(self) -> List[int]:
        """Get IDs of all objects in the spatial hash."""
        return list(self.object_positions.keys())

    def get_object_position(self, obj_id: int) -> Optional[Vector2D]:
        """Get the current position of an object."""
        return self.object_positions.get(obj_id)

    def get_object_bounds(self, obj_id: int) -> Optional[AABB]:
        """Get the bounds of an object (exactly what was passed to add_object)."""
        return self.object_bounds.get(obj_id)

    def clear(self):
        """Remove all objects from the spatial hash."""
        self.grid.clear()
        self.object_positions.clear()
        self.object_bounds.clear()
        self.next_object_id = 0
        self.free_ids.clear()

    def __str__(self) -> str:
        return f"SpatialHash(cell_size={self.cell_size}, objects={len(self.object_positions)}, cells={len(self.grid)}, free_ids={len(self.free_ids)})"

    def __repr__(self) -> str:
        return self.__str__()