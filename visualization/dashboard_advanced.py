"""
Advanced dashboard panels for Artificial Life Neuroevolution Simulation.

Phase 6: renders the live ecosystem counters (prey/predator counts,
captures, births) and the hall-of-fame delta as in-GUI text panels.
Pure text formatting over a pygame font - no plotting dependencies.

Design note: this module formats data it is given; it never runs
simulation logic. Real data comes from either the live GUI state
(ecosystem counters) or the latest co-evolution run
(experiments/EXP-COEV/coevolution_summary.json).
"""

from __future__ import annotations
from typing import List, Optional

# Panel placement: top-left corner of the world window, below any HUD
PANEL_MARGIN = 12
LINE_SPACING = 18


def format_ecosystem_lines(prey_alive: int, predator_alive: int,
                           captures: int, births: int,
                           total_agents: int) -> List[str]:
    """Format the live ecosystem counter lines for the GUI panel."""
    return [
        f"prey: {prey_alive}  predators: {predator_alive}",
        f"captures: {captures}  births: {births}",
        f"agents: {total_agents}",
    ]


def format_hof_lines(hof_evaluations: Optional[List[dict]] = None) -> List[str]:
    """
    Format the hall-of-fame delta lines from the latest co-evolution
    run's hof_evaluations list. None (no run on disk yet) renders a
    short 'no data' line rather than a placeholder number.
    """
    if not hof_evaluations:
        return ["hall-of-fame: no co-evolution run yet"]
    lines = ["hall-of-fame (win rate vs frozen archive):"]
    # Show the latest evaluation: the most recent generation measured
    latest = hof_evaluations[-1]
    lines.append(
        f"  gen {latest['generation']}: "
        f"predator {latest['predator_win_rate_vs_archive']:.2f} / "
        f"prey {latest['prey_win_rate_vs_archive']:.2f}")
    return lines


def draw_panel(renderer, lines: List[str]) -> None:
    """
    Blit a list of text lines onto the renderer's screen, top-left.

    Args:
        renderer: Renderer instance (provides screen and font).
        lines: Pre-formatted text lines.
    """
    for index, line in enumerate(lines):
        surface = renderer.font.render(line, True, (220, 220, 220))
        y = PANEL_MARGIN + index * LINE_SPACING
        renderer.screen.blit(surface, (PANEL_MARGIN, y))
