"""
Shared body geometry for the simulation.

Every subsystem that reasons about physical bodies must read these
constants instead of hard-coding its own sizes. Before this module
existed the obstacle body was a 10 px square in the collision engine and
renderer, while the ray caster treated it as a 5.5 px circle; the two
descriptions drifted and agents could pass through geometry they could
see. Keeping one source of truth makes that class of bug impossible.

Units are world pixels. The obstacle side is 4x the original 10 px
baseline, the middle of the requested 3x-5x range.
"""

from __future__ import annotations

# Agent collision body and drawn triangle size (see visualization.renderer).
AGENT_BODY_SIZE: float = 8.0
AGENT_HALF_SIZE: float = AGENT_BODY_SIZE / 2.0

# Obstacle collision body, ray-cast target and drawn square.
OBSTACLE_SIZE: float = 40.0
OBSTACLE_HALF_SIZE: float = OBSTACLE_SIZE / 2.0

# Clearance used when spawning agents so their body does not start
# overlapping an obstacle. One extra pixel keeps the initial AABB test
# unambiguous.
AGENT_SPAWN_CLEARANCE: float = AGENT_HALF_SIZE + 1.0

# Clearance used when spawning food. Food is drawn as a 4 px circle, so
# keeping its center 4 px off an obstacle face also keeps the sprite out.
FOOD_SPAWN_CLEARANCE: float = 4.0
