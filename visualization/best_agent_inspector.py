"""
Best-agent inspector (PLAN.md Section 12.2).

Click any agent in the live world view to open this panel: vitals, the
seven-ray sensor overlay drawn on top of the world, the controller's
node-and-edge graph with outputs highlighted by current activation,
and a genome fingerprint standing in for lineage (the fixed-topology
controller has no breeding history; NEAT genomes do, and the
fingerprint still identifies a controller uniquely).

Pure formatting/drawing helpers over an agent - no simulation logic.
"""

from __future__ import annotations
import hashlib
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from agents.organism import Organism
from agents.sensors import MAX_RANGE, RAY_ANGLES_DEG
from neural.network import ARCHITECTURE
from neural.genome import unpack_genome

PANEL_WIDTH = 260
LINE_SPACING = 16
MARGIN = 12
RAY_COLORS = {
    "far": (90, 90, 110),
    "near": (240, 200, 80),
}


def format_vitals(agent: Organism) -> List[str]:
    """The vitals block: identity, state, and current behavior."""
    speed = math.hypot(agent.velocity.x, agent.velocity.y)
    return [
        f"agent #{agent.id} ({agent.role})",
        f"fitness {agent.fitness:.3f}   age {agent.age}",
        f"energy {agent.energy:.1f}   health {agent.health:.1f}",
        f"speed {speed:.2f} px/step",
        f"food/captures {agent.food_eaten}   collisions {agent.collisions}",
        f"alive: {agent.is_alive}",
    ]


def genome_fingerprint(agent: Organism) -> str:
    """
    Short stable id for the agent's controller. Hashing the genome
    bytes gives a lineage stand-in that is unique per controller and
    identical across runs for the same genome.
    """
    if agent.genome is None:
        return "no-genome"
    return hashlib.sha256(
        np.asarray(agent.genome, dtype=np.float32).tobytes()
    ).hexdigest()[:12]


def ray_segments(agent: Organism) -> List[Tuple[float, float, float, float, str]]:
    """
    The seven sensor rays as screen segments with a proximity class.

    Distances come from the agent's last observation row (the same
    12 values the controller saw), normalized against the sensor range,
    so the overlay shows what the agent actually perceives, not a
    fresh cast.

    Returns:
        List of (x1, y1, x2, y2, "far"|"near") segments.
    """
    if agent.last_observation is None:
        return []
    segments = []
    for index, angle_deg in enumerate(RAY_ANGLES_DEG):
        angle = agent.heading + math.radians(angle_deg)
        distance = float(agent.last_observation[index]) * MAX_RANGE
        end_x = agent.position.x + math.cos(angle) * distance
        end_y = agent.position.y + math.sin(angle) * distance
        band = "near" if distance < MAX_RANGE * 0.5 else "far"
        segments.append((agent.position.x, agent.position.y, end_x, end_y, band))
    return segments


def network_layout(columns: Sequence[int], left: int, top: int, width: int,
                   height: int) -> List[List[Tuple[int, int]]]:
    """
    Node coordinates for the controller graph: layers left to right
    inside the (left, top, width, height) viewport, each layer centered
    vertically. Node spacing shrinks to fit the tallest layer, so a
    32-node layer stays on screen instead of overflowing it.
    """
    layer_width = (width - 2 * MARGIN) // max(len(columns), 1)
    tallest = max(columns)
    spacing = min(LINE_SPACING * 2, max(height // max(tallest, 1), 6))
    positions = []
    for layer, count in enumerate(columns):
        x = left + MARGIN + layer * layer_width
        center_y = top + height // 2
        layer_positions = []
        for node in range(count):
            y = center_y + (node - (count - 1) / 2) * spacing
            layer_positions.append((x, int(y)))
        positions.append(layer_positions)
    return positions


def draw_sensor_overlay(screen, agent: Organism) -> None:
    """Draw the seven rays over the live world, near hits in amber."""
    for x1, y1, x2, y2, band in ray_segments(agent):
        pygame.draw.line(screen, RAY_COLORS[band], (x1, y1), (x2, y2), 2)


def draw_network_graph(screen, font, agent: Organism, left: int,
                       top: int, width: int, height: int,
                       last_action: Optional[Sequence[float]] = None) -> None:
    """
    Draw the controller as a node-and-edge diagram.

    Edge opacity scales with |weight| (the genome's stored weights);
    output nodes are highlighted by the last decoded action when one
    is supplied, otherwise by weight magnitude.
    """
    if agent.genome is None:
        surface = font.render("no genome to draw", True, (200, 200, 200))
        screen.blit(surface, (left, top))
        return

    layers = network_layout(ARCHITECTURE, left, top, width, height)
    layer_weights, _ = unpack_genome(np.asarray(agent.genome, dtype=np.float32))

    # Edges first so nodes sit on top
    for layer in range(len(ARCHITECTURE) - 1):
        in_dim, out_dim = ARCHITECTURE[layer], ARCHITECTURE[layer + 1]
        weights = layer_weights[layer].reshape(in_dim, out_dim)
        for in_node, (x1, y1) in enumerate(layers[layer]):
            for out_node, (x2, y2) in enumerate(layers[layer + 1]):
                weight = float(weights[in_node, out_node])
                shade = int(120 + 135 * min(1.0, abs(weight)))
                pygame.draw.line(screen, (shade, shade, 150),
                                 (x1, y1), (x2, y2), 1)

    # Nodes
    for layer, layer_positions in enumerate(layers):
        for node, (x, y) in enumerate(layer_positions):
            if layer == len(layers) - 1 and last_action is not None:
                activation = abs(float(last_action[node]))
                radius = 4 + int(5 * min(1.0, activation))
                color = (255, int(200 * (1 - min(1.0, activation))), 60)
            else:
                radius = 4
                color = (200, 200, 210)
            pygame.draw.circle(screen, color, (x, y), radius)
            pygame.draw.circle(screen, (30, 30, 30), (x, y), radius, 1)


def draw_inspector(screen, font, agent: Organism,
                   last_action: Optional[Sequence[float]] = None) -> None:
    """
    Draw the full inspector panel on the right edge of the screen:
    vitals, genome fingerprint, and the controller graph.
    """
    width, height = screen.get_size()
    left = width - PANEL_WIDTH
    panel = pygame.Surface((PANEL_WIDTH, height), pygame.SRCALPHA)
    panel.fill((20, 20, 28, 235))
    screen.blit(panel, (left, 0))
    pygame.draw.line(screen, (90, 90, 110), (left, 0), (left, height), 2)

    y = MARGIN
    for line in format_vitals(agent):
        surface = font.render(line, True, (225, 225, 235))
        screen.blit(surface, (left + MARGIN, y))
        y += LINE_SPACING

    fingerprint = genome_fingerprint(agent)
    surface = font.render(f"genome {fingerprint}", True, (150, 200, 255))
    screen.blit(surface, (left + MARGIN, y))
    y += LINE_SPACING + 6

    title = font.render("controller", True, (200, 200, 210))
    screen.blit(title, (left + MARGIN, y))
    graph_top = y + LINE_SPACING + 4
    draw_network_graph(screen, font, agent, left, graph_top,
                       PANEL_WIDTH - 2 * MARGIN, height - graph_top - MARGIN,
                       last_action)
