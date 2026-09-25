"""
Bottom control bar for the live GUI (PLAN.md Section 12).

Buttons: [Pause] [x1] [x10] [Save] [Reset]. One shared geometry
function serves both drawing and click hit-testing, so the two can
never disagree.

The plan's "Next Gen" control is deliberately absent: the GUI is a
free-running sandbox, not a generation loop - generations advance in
the headless runners (`experiments/run.py`, `experiments/run_neat.py`).
"Save" snapshots the best-fitness controller through
`neural/checkpoint.py`; "Reset" reinitializes the population from the
active config.
"""

from __future__ import annotations
from typing import Dict, Tuple

BAR_HEIGHT = 34
BUTTON_WIDTH = 74
BUTTON_GAP = 8
ACTIONS = ("pause", "x1", "x10", "save", "reset")


def button_rects(width: int, height: int) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Button rectangles (x, y, w, h) centered along the bottom edge.

    Shared by draw_control_bar and hit-testing so drawing and clicks
    can never diverge.
    """
    total = len(ACTIONS) * BUTTON_WIDTH + (len(ACTIONS) - 1) * BUTTON_GAP
    start_x = (width - total) // 2
    y = height - BAR_HEIGHT + 4
    return {
        action: (start_x + index * (BUTTON_WIDTH + BUTTON_GAP), y,
                 BUTTON_WIDTH, BAR_HEIGHT - 10)
        for index, action in enumerate(ACTIONS)
    }


def action_at_position(x: int, y: int, width: int, height: int
                       ) -> str | None:
    """
    Which button (if any) contains the point. Clicking elsewhere in
    the world is handled by the caller (agent selection), not here.
    """
    for action, (bx, by, bw, bh) in button_rects(width, height).items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return action
    return None


def draw_control_bar(screen, font, states: Dict[str, str]) -> None:
    """
    Draw the bar. `states` maps action -> label of the current state
    (e.g. "pause" -> "Resume" when paused), so the buttons reflect
    live state instead of a static legend.
    """
    import pygame
    width, height = screen.get_size()
    bar = pygame.Surface((width, BAR_HEIGHT), pygame.SRCALPHA)
    bar.fill((18, 18, 24, 220))
    screen.blit(bar, (0, height - BAR_HEIGHT))
    for action, (bx, by, bw, bh) in button_rects(width, height).items():
        rect = pygame.Rect(bx, by, bw, bh)
        pygame.draw.rect(screen, (60, 64, 80), rect, border_radius=4)
        pygame.draw.rect(screen, (110, 115, 140), rect, 1, border_radius=4)
        label = states.get(action, action)
        text = font.render(label, True, (225, 225, 235))
        screen.blit(text, (rect.centerx - text.get_width() // 2,
                          rect.centery - text.get_height() // 2))
