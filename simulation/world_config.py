"""
World configuration module for Artificial Life Neuroevolution Simulation.

Provides Pydantic models for validating configuration files.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator


class WorldConfig(BaseModel):
    """World configuration."""
    width: int = Field(1200, gt=0)
    height: int = Field(700, gt=0)
    food_count: int = Field(20, ge=0)
    obstacle_count: int = Field(10, ge=0)
    # Seeds the global numpy RNG before food/obstacle placement so each
    # layout (A1-A3 train / B1-B3 test in Phase 5) is reproducible and
    # distinct. None = unseeded (legacy behavior).
    layout_seed: Optional[int] = None
    # Food regrowth rate per simulation step (Phase 6): consumed items
    # respawn at random positions up to the food_count target. 0.0 =
    # finite food supply, the legacy Phase 1-5 behavior (stability
    # comparisons in earlier phases are unaffected).
    food_regrowth_per_step: float = Field(0.0, ge=0)


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
    # Simulation steps each genome is evaluated for per generation.
    # Kept configurable so Phase 5 experiments can trade fidelity for
    # wall-clock time (ray casting is O(agents x objects) per step).
    evaluation_steps: int = Field(1000, gt=0)


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    seeds: int = Field(10, gt=0)
    save_interval: int = Field(50, gt=0)


class PredationConfig(BaseModel):
    """
    Predator/prey ecosystem parameters (Phase 6).

    predator_count = 0 means legacy prey-only mode: every existing
    config without a 'predation' section behaves exactly as before.
    """
    predator_count: int = Field(0, ge=0)
    # Distance in pixels within which a firing eat-gate captures prey.
    # This and capture_energy_transfer are the Phase 6 step-5 tuning
    # knobs for ecosystem stability.
    capture_radius: float = Field(16.0, gt=0)
    # Energy gained by a predator per successful capture. Deliberately
    # separate from eaten_energy_value so prey-plant and predator-meat
    # energy flows can be tuned independently.
    capture_energy_transfer: float = Field(60.0, gt=0)


class ReproductionConfig(BaseModel):
    """
    In-episode reproduction parameters (Phase 6).

    Offspring are asexual clone+mutate copies of an eligible parent,
    during an episode. The parent pays reproduction_cost energy per
    birth, which throttles the birth rate (a parent that just gave
    birth falls below min_energy and must recover before breeding
    again) and couples predator birth rate to capture success. This is
    separate from generation-level GA replacement in
    genetic_algorithm.run_generation / run_coevolution_generation.

    Eligibility follows the plan's rule:
        age >= min_age and energy >= min_energy and fitness >= min_fitness
    Per-role carrying capacities (max_population_prey /
    max_population_predator) keep the predator population a small
    fraction of the prey population, preventing predator overshoot.
    """
    enabled: bool = False
    min_age: int = Field(40, ge=0)
    min_energy: float = Field(60.0, ge=0)
    min_fitness: float = Field(0.15, ge=0)
    # Per-role carrying capacities. Predators must be capped well below
    # prey: a shared cap (earlier design) let the predator population
    # overshoot to prey-crushing levels, and ecologically a predator's
    # carrying capacity is inherently a fraction of its prey's.
    max_population_prey: int = Field(40, gt=0)
    max_population_predator: int = Field(8, gt=0)
    mutation_rate: float = Field(0.05, ge=0, le=1)
    mutation_strength: float = Field(0.1, gt=0)
    # Energy the parent pays per birth. Without a cost every eligible
    # agent breeds every step and the population explodes to the cap
    # (observed in the first ecosystem smoke run), so the cost is a
    # stability requirement, not decoration.
    reproduction_cost: float = Field(50.0, ge=0)


class NeatConfig(BaseModel):
    """
    NEAT topology-evolution parameters (Phase 7, Appendix C).

    The compatibility coefficients and threshold are the Appendix C
    baseline (Stanley & Miikkulainen 2002): c1=1.0 (enabled genes),
    c2=1.0 (gene-count difference), c3=0.4 (mean weight difference),
    compatibility threshold 3.0.
    """
    # Compatibility distance: delta = (c1*E + c2*D)/N + c3*Wbar
    c1_enabled_gene_coefficient: float = Field(1.0, ge=0)
    c2_gene_count_coefficient: float = Field(1.0, ge=0)
    c3_weight_coefficient: float = Field(0.4, ge=0)
    compatibility_threshold: float = Field(3.0, gt=0)

    # Structural mutation rates (probability per reproduced offspring)
    add_node_probability: float = Field(0.05, ge=0, le=1)
    add_connection_probability: float = Field(0.05, ge=0, le=1)
    # Per-connection probability of Gaussian weight perturbation
    weight_mutation_probability: float = Field(0.2, ge=0, le=1)
    weight_mutation_sigma: float = Field(0.5, gt=0)

    # Speciation dynamics
    stagnation_generations: int = Field(15, ge=1)
    min_species_size: int = Field(1, ge=1)
    # Probability a reproduced offspring is an innovation-aligned
    # crossover child of two parents from the same species
    crossover_rate: float = Field(0.5, ge=0, le=1)
    # Every species with an improved member keeps at least this many
    # offspring so rare species are not reaped before they can recover
    min_offspring_per_species: int = Field(1, ge=0)

    # Network shape (matches the engine's controller contract)
    n_inputs: int = Field(12, gt=0)
    n_outputs: int = Field(3, gt=0)

    # Initial weight range for new connections
    initial_weight_min: float = -1.0
    initial_weight_max: float = 1.0


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
    # Phase 6 sections. None = absent in legacy configs, which keeps
    # every pre-Phase-6 config valid and prey-only.
    predation: Optional[PredationConfig] = None
    reproduction: Optional[ReproductionConfig] = None
    # Phase 7 NEAT extension; None = fixed-topology evolution
    neat: Optional[NeatConfig] = None

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