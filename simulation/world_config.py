"""
World configuration module for Artificial Life Neuroevolution Simulation.

Provides Pydantic models for validating configuration files.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class WorldConfig(BaseModel):
    """World configuration."""
    width: int = Field(1200, gt=0)
    height: int = Field(700, gt=0)
    food_count: int = Field(20, ge=0)
    obstacle_count: int = Field(10, ge=0)


class BrainConfig(BaseModel):
    """Neural network architecture configuration."""
    architecture: List[int] = Field([12, 32, 16, 3], min_length=2)


class AgentConfig(BaseModel):
    """Agent configuration."""
    sensors: int = Field(7, gt=0)
    brain: BrainConfig


class PopulationConfig(BaseModel):
    """Population configuration."""
    size: int = Field(250, gt=0)
    agent: AgentConfig


class EvolutionConfig(BaseModel):
    """Evolution configuration."""
    generations: int = Field(1000, gt=0)
    mutation_rate: float = Field(0.05, ge=0, le=1)
    crossover_rate: float = Field(0.7, ge=0, le=1)
    elitism_count: int = Field(2, ge=0)


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    seeds: int = Field(10, gt=0)
    save_interval: int = Field(50, gt=0)


class BaselineConfig(BaseModel):
    """Baseline configuration matching the design document's Section 18.2 example."""
    experiment_name: str = "baseline"
    mode: str = Field("gui", pattern="^(gui|headless)$")
    device: str = Field("cpu", pattern="^(cpu|cuda|auto)$")
    world: WorldConfig
    population: PopulationConfig
    evolution: EvolutionConfig
    experiment: ExperimentConfig
    # Additional fields from the design document's Section 18.2 example
    crossover_method: str = Field("blend", pattern="^(blend|uniform|none)$")
    fitness_weights: Dict[str, float] = Field(default_factory=lambda: {"survival": 0.4, "food": 0.3, "exploration": 0.2, "collision": 0.1})
    seed: Dict[str, Optional[int]] = Field(default_factory=lambda: {"torch": None, "numpy": None, "random": None})
    reproducibility_tier: str = Field("cpu-deterministic", pattern="^(cpu-deterministic|non-deterministic)$")

    @validator('fitness_weights')
    def weights_sum_to_one(cls, v):
        """Ensure fitness weights sum to 1.0 (optional, but good practice)."""
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Fitness weights must sum to 1.0, got {total}')
        return v


def load_config(config_path: str) -> BaselineConfig:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Validated BaselineConfig object
    """
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return BaselineConfig(**config_dict)