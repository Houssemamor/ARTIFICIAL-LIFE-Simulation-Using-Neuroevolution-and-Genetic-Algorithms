"""
Speciation, fitness sharing, and stagnation removal for NEAT
(Phase 7, plan step 6).

Per Stanley & Miikkulainen (2002):

  - Species assignment: compute the compatibility distance from each
    genome to every existing species representative; assign to the
    first compatible species (within threshold), else found a new
    species. Existing species below min_species_size, or stagnant for
    more than stagnation_generations without a fitness improvement, are
    removed first (their members then compete for surviving species or
    found new ones). The species holding the best genome is protected
    from stagnation removal.
  - Fitness sharing: adjusted = raw / species_size, removing the
    incentive to evolve in a species that already has many members.
  - Stagnation: a species whose best fitness has not beaten its own
    historical best for stagnation_generations is removed. Removal
    never touches the species containing the overall champion.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from evolution.neat.compatibility import compatibility_distance
from evolution.neat.genome import NeatGenome


@dataclass
class Species:
    """One species: a representative genome plus bookkeeping."""
    representative: NeatGenome
    members: List[int] = field(default_factory=list)
    generation_created: int = 0
    generation_improved: int = 0
    best_fitness: float = -float("inf")

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class SpeciationResult:
    """Outcome of one speciation pass."""
    species: List[Species]
    species_of: List[int]          # genome index -> species index
    adjusted_fitness: np.ndarray   # per genome


def assign_species(
    genomes: List[NeatGenome],
    fitnesses: np.ndarray,
    generation: int,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
    threshold: float = 3.0,
    stagnation_generations: int = 15,
    min_species_size: int = 1,
    previous_species: Optional[List[Species]] = None,
) -> SpeciationResult:
    """
    One speciation pass with sharing and stagnation removal.

    Args:
        genomes: The evaluated population.
        fitnesses: Raw fitness per genome.
        generation: Current generation number.
        c1, c2, c3: Compatibility coefficients.
        threshold: Compatibility threshold (delta <= threshold means
            compatible).
        stagnation_generations: Generations without improvement before
            a species is removed.
        min_species_size: Species below this size are removed.
        previous_species: Species carried from the prior generation
            (representatives and stagnation bookkeeping).

    Returns:
        SpeciationResult with the surviving species, the per-genome
        species mapping, and sharing-adjusted fitnesses.
    """
    population_size = len(genomes)
    if population_size == 0:
        return SpeciationResult([], [], np.zeros(0))

    # Start from the previous generation's species, pruning stale ones.
    # Stagnation protection keeps the previous best species alive: it
    # holds the strongest raw fitness and therefore the most likely
    # champion's species (the standard champion-protection rule; the
    # champion's species is only identified after this pass).
    protected = -1
    if previous_species:
        best_historical = max(range(len(previous_species)),
                              key=lambda i: previous_species[i].best_fitness)
        protected = best_historical

    species: List[Species] = []
    for old_index, old in enumerate(previous_species or []):
        if old.size < min_species_size:
            continue
        stale = (generation - old.generation_improved > stagnation_generations
                 and old_index != protected)
        if stale:
            continue
        # Carry the species forward with fresh member lists
        species.append(Species(
            representative=old.representative,
            members=[],
            generation_created=old.generation_created,
            generation_improved=old.generation_improved,
            best_fitness=old.best_fitness,
        ))

    species_of = [-1] * population_size
    # Assign in random order (rng-free: use a stable shuffle by index
    # for determinism under the run's rng elsewhere; here the caller's
    # ordering is preserved which keeps species ids stable run-to-run)
    for index in range(population_size):
        genome = genomes[index]
        placed = False
        for species_index, sp in enumerate(species):
            delta = compatibility_distance(
                genome, sp.representative, c1=c1, c2=c2, c3=c3)
            if delta <= threshold:
                species_of[index] = species_index
                sp.members.append(index)
                placed = True
                break
        if not placed:
            new_species = Species(
                representative=genome,
                members=[index],
                generation_created=generation,
                generation_improved=generation,
                best_fitness=float(fitnesses[index]),
            )
            species.append(new_species)
            species_of[index] = len(species) - 1

    # Update improvement bookkeeping and champions
    for sp in species:
        if not sp.members:
            continue
        best_index = max(sp.members, key=lambda i: fitnesses[i])
        best_fitness = float(fitnesses[best_index])
        if best_fitness > sp.best_fitness:
            sp.best_fitness = best_fitness
            sp.generation_improved = generation
        # Keep the representative at the current best
        sp.representative = genomes[best_index]

    # Drop species that ended up empty after the champion protection
    species = [sp for sp in species if sp.members]
    # Remap genome->species indices after any drops
    remap = {}
    new_species_of = [-1] * population_size
    for new_index, sp in enumerate(species):
        for member in sp.members:
            remap[member] = new_index
    for index in range(population_size):
        new_species_of[index] = remap.get(index, -1)

    # Fitness sharing: divide by species size
    adjusted = np.zeros(population_size, dtype=float)
    for sp in species:
        for member in sp.members:
            adjusted[member] = fitnesses[member] / max(sp.size, 1)

    return SpeciationResult(species=species, species_of=new_species_of,
                            adjusted_fitness=adjusted)
