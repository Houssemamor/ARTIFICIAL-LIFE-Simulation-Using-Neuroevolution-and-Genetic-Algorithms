"""
Unit tests for the Phase 9 GUI overlays (control bar, agent inspector).

Both modules are drawable with SDL's dummy video driver, so the
draw paths run in CI headless. The layout regression is real: the
32-node layer once spanned y=422..1414 on a 700-pixel screen, and
the graph was drawn at the screen's left margin instead of inside
the right-edge panel.
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import pygame  # noqa: E402

from agents.organism import Organism  # noqa: E402
from neural.genome import genome_size  # noqa: E402
from visualization.best_agent_inspector import (  # noqa: E402
    ARCHITECTURE, MARGIN, PANEL_WIDTH, draw_inspector, network_layout)
from visualization.control_bar import (  # noqa: E402
    ACTIONS, action_at_position, button_rects, draw_control_bar)


def _agent(agent_id=0):
    agent = Organism(agent_id, 100.0, 100.0, initial_energy=100.0)
    agent.initialize_genome(genome_size())
    agent.last_observation = [0.1] * 12
    agent.food_eaten = 2
    return agent


def test_hit_test_agrees_with_drawn_geometry():
    width, height = 800, 600
    for action, (bx, by, bw, bh) in button_rects(width, height).items():
        center = (bx + bw // 2, by + bh // 2)
        assert action_at_position(*center, width, height) == action
    assert action_at_position(2, 2, 800, 600) is None


def test_network_layout_stays_inside_the_viewport():
    width, height = 1200, 700
    left = width - PANEL_WIDTH
    top = 240
    viewport_height = height - top - MARGIN
    layers = network_layout(ARCHITECTURE, left, top,
                            PANEL_WIDTH - 2 * MARGIN, viewport_height)
    assert len(layers) == len(ARCHITECTURE)
    for count, layer in zip(ARCHITECTURE, layers):
        assert len(layer) == count
        for x, y in layer:
            assert left <= x <= left + PANEL_WIDTH, f"node x={x} escapes panel"
            assert top <= y <= top + viewport_height, (
                f"node y={y} escapes viewport [{top}, "
                f"{top + viewport_height}]")


def test_control_bar_and_inspector_draw_headless():
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    font = pygame.font.SysFont("consolas", 14)
    try:
        draw_control_bar(screen, font,
                         {action: action for action in ACTIONS})
        draw_inspector(screen, font, _agent(),
                       last_action=[0.1, 0.2, 0.3])
    finally:
        pygame.quit()


def test_draw_network_graph_handles_missing_genome():
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    font = pygame.font.SysFont("consolas", 14)
    agent = Organism(1, 50.0, 50.0)
    try:
        agent.genome = None
        draw_inspector(screen, font, agent)
    finally:
        pygame.quit()
