"""
Renderer module for Artificial Life Neuroevolution Simulation.

Provides PyGame-based rendering of the world, food, obstacles, and agents.
"""

from __future__ import annotations
from typing import List
import pygame
import numpy as np

from simulation.environment import World
from agents.organism import Organism


class Renderer:
    """
    PyGame renderer for the simulation world.

    Attributes:
        width (int): Width of the display window in pixels
        height (int): Height of the display window in pixels
        screen (pygame.Surface): The PyGame display surface
        clock (pygame.time.Clock): Clock to control frame rate
        font (pygame.font.Font): Font for rendering text
    """

    def __init__(self, width: int = 1200, height: int = 700):
        """
        Initialize the renderer.

        Args:
            width (int): Width of the display window (default: 1200)
            height (int): Height of the display window (default: 700)
        """
        self.width = width
        self.height = height

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Artificial Life Neuroevolution Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        # Set when the user presses F9; the main loop reads and clears it
        # (single event pump: this class owns pygame.event.get())
        self.panel_toggled = False

    def draw_world(self, world: World, agents: List[Organism]) -> None:
        """
        Render the world, food, obstacles, and agents.

        Args:
            world (World): The world instance containing food and obstacles
            agents (List[Organism]): List of agents to render
        """
        # Fill background with dark gray
        self.screen.fill((30, 30, 30))

        # Draw world border (optional)
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.screen, (50, 50, 50), border_rect, 2)

        # Draw food items (green circles)
        for food in world.food:
            pos = (int(food.x), int(food.y))
            pygame.draw.circle(self.screen, (0, 200, 0), pos, 4)

        # Draw obstacles (gray rectangles)
        for obstacle in world.obstacles:
            # Obstacles are 10x10 pixels
            rect = pygame.Rect(
                int(obstacle.x - 5),
                int(obstacle.y - 5),
                10,
                10
            )
            pygame.draw.rect(self.screen, (100, 100, 100), rect)

        # Draw agents (triangles oriented by heading)
        for agent in agents:
            if not agent.is_alive:
                continue

            # Agent size
            size = 8
            # Calculate triangle points based on heading
            # Point 0: front of the triangle
            front_x = agent.position.x + size * np.cos(agent.heading)
            front_y = agent.position.y + size * np.sin(agent.heading)
            # Point 1: rear left
            rear_left_x = agent.position.x - size * 0.5 * np.cos(agent.heading) - size * 0.5 * np.sin(agent.heading)
            rear_left_y = agent.position.y - size * 0.5 * np.sin(agent.heading) + size * 0.5 * np.cos(agent.heading)
            # Point 2: rear right
            rear_right_x = agent.position.x - size * 0.5 * np.cos(agent.heading) + size * 0.5 * np.sin(agent.heading)
            rear_right_y = agent.position.y - size * 0.5 * np.sin(agent.heading) - size * 0.5 * np.cos(agent.heading)

            # Convert to integers for PyGame
            points = [
                (int(front_x), int(front_y)),
                (int(rear_left_x), int(rear_left_y)),
                (int(rear_right_x), int(rear_right_y))
            ]

            # Color: prey keeps the energy green->red gradient; predators
            # draw red-tinted with brightness by energy so the two roles
            # are visually distinct (Phase 6)
            energy_ratio = max(0.0, min(1.0, agent.energy / 100.0))
            if getattr(agent, 'role', 'prey') == 'predator':
                color = (200, int(80 * (1 - energy_ratio)), 0)
            else:
                color = (
                    int(255 * (1 - energy_ratio)),  # Red
                    int(255 * energy_ratio),        # Green
                    0                               # Blue
                )
            pygame.draw.polygon(self.screen, color, points)

        # Optional: draw agent ID or energy text (for debugging)
        # for agent in agents:
        #     if agent.is_alive:
        #         text_surface = self.font.render(
        #             f"{agent.id}:{agent.energy:.0f}", True, (255, 255, 255)
        #         )
        #     self.screen.blit(text_surface, (agent.position.x, agent.position.y))

        # Update the display
        pygame.display.flip()

    def handle_events(self) -> bool:
        """
        Handle PyGame events.

        Returns:
            False if the user requests to quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_F9:
                    # Phase 6 dashboard toggle; the main loop consumes
                    # the flag so this class stays presentation-only
                    self.panel_toggled = True
        return True

    def quit(self) -> None:
        """Clean up PyGame resources."""
        pygame.quit()