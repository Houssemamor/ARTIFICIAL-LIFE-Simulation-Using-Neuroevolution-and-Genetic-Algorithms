"""
Physics engine for Artificial Life Neuroevolution Simulation.
Handles vector mathematics, basic physics integration, collision detection, and spatial partitioning.
"""

from .vector2d import Vector2D, vec2, zero_vector, lerp
from .aabb import AABB
from .spatial_hash import SpatialHash

# Make commonly used classes available at module level
__all__ = [
    'Vector2D',
    'vec2',
    'zero_vector',
    'lerp',
    'AABB',
    'SpatialHash',
    'PhysicsObject',
    'integrate_euler',
    'integrate_verlet'
]


class PhysicsObject:
    """Base class for objects that participate in physics simulation."""

    def __init__(self, position: Vector2D = None,
                 velocity: Vector2D = None,
                 mass: float = 1.0):
        self.position = position or Vector2D(0, 0)
        self.velocity = velocity or Vector2D(0, 0)
        self.acceleration = Vector2D(0, 0)
        self.force_accumulator = Vector2D(0, 0)
        self.mass = max(mass, 0.001)  # Prevent division by zero
        self.damping = 0.999  # Velocity damping per second
        self.fixed = False  # If True, object doesn't move

    def apply_force(self, force: Vector2D):
        """Apply a force to this object (accumulates until integration)."""
        if self.fixed:
            self.force_accumulator = Vector2D(0, 0)  # Clear forces when fixed
        else:
            self.force_accumulator += force

    def apply_impulse(self, impulse: Vector2D):
        """Apply an instantaneous impulse (changes velocity directly)."""
        if not self.fixed and self.mass > 0:
            self.velocity += impulse / self.mass

    def clear_forces(self):
        """Clear accumulated forces (called after integration step)."""
        self.force_accumulator = Vector2D(0, 0)

    def set_fixed(self, fixed: bool):
        """Set whether this object is fixed in place."""
        self.fixed = fixed

    def is_fixed(self) -> bool:
        """Check if this object is fixed."""
        return self.fixed

    def get_kinetic_energy(self) -> float:
        """Calculate kinetic energy: 0.5 * m * v^2."""
        return 0.5 * self.mass * self.velocity.magnitude_squared()

    def get_momentum(self) -> Vector2D:
        """Calculate momentum: m * v."""
        return self.velocity * self.mass


def integrate_euler(obj: PhysicsObject, dt: float):
    """
    Integrate physics using Euler method.
    Simple but less stable - good for small time steps and low speeds.
    """
    if obj.fixed:
        obj.velocity = Vector2D(0, 0)
        obj.acceleration = Vector2D(0, 0)
        return

    # F = m * a, so a = F / m
    if obj.mass > 0:
        acceleration = obj.force_accumulator / obj.mass
    else:
        acceleration = Vector2D(0, 0)

    # Update position using: x = x + v*dt + 0.5*a*dt^2
    obj.position += obj.velocity * dt + acceleration * (0.5 * dt * dt)

    # Update velocity: v = v + a * dt
    obj.velocity += acceleration * dt

    # Apply damping (air resistance)
    obj.velocity *= obj.damping ** dt

    # Clear forces for next frame
    obj.clear_forces()


def integrate_verlet(obj: PhysicsObject, dt: float,
                    previous_position: Vector2D = None):
    """
    Integrate physics using Verlet method.
    More stable than Euler, especially for constrained systems.
    """
    if obj.fixed:
        if previous_position is not None:
            # For fixed objects, previous position equals current
            pass
        obj.velocity = Vector2D(0, 0)
        obj.acceleration = Vector2D(0, 0)
        return obj.position

    # Store current position for next iteration
    current_position = Vector2D(obj.position.x, obj.position.y)

    # Calculate acceleration from accumulated forces
    if obj.mass > 0:
        acceleration = obj.force_accumulator / obj.mass
    else:
        acceleration = Vector2D(0, 0)

    # Verlet integration:
    # x_new = 2*x_current - x_old + a * dt^2
    if previous_position is not None:
        obj.position = Vector2D(
            2 * obj.position.x - previous_position.x + acceleration.x * dt * dt,
            2 * obj.position.y - previous_position.y + acceleration.y * dt * dt
        )
        # Calculate velocity: v = (x_new - x_old) / (2 * dt)
        obj.velocity = Vector2D(
            (obj.position.x - previous_position.x) / (2 * dt),
            (obj.position.y - previous_position.y) / (2 * dt)
        )
        # Apply damping
        obj.velocity *= obj.damping ** dt
    else:
        # Fallback to Euler if no previous position (Euler already applies damping)
        integrate_euler(obj, dt)

    # Clear forces for next frame
    obj.clear_forces()

    return current_position


# Export commonly used functions
__all__.extend(['integrate_euler', 'integrate_verlet'])