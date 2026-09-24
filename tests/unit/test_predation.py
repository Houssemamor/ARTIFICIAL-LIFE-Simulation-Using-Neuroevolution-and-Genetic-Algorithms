"""Unit tests for Phase 6 predation mechanics (capture resolution)."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from agents.organism import Organism
from simulation.engine import resolve_captures, step_simulation
from simulation.environment import World
from agents.sensors import RayCaster
from neural.genome import genome_size


def make_predator(x, y, flag=False):
    agent = Organism(0, x, y, role='predator')
    agent._capture_attempt = flag
    return agent


def make_prey(x, y, agent_id=1):
    agent = Organism(agent_id, x, y, role='prey')
    return agent


def test_capture_kills_prey_and_transfers_energy():
    predator = make_predator(100.0, 100.0, flag=True)
    predator.energy = 40.0
    prey = make_prey(110.0, 100.0)

    captures = resolve_captures([predator, prey], capture_radius=16.0,
                                energy_transfer=60.0)

    assert captures == 1
    assert not prey.is_alive
    assert prey.energy == 0.0
    assert predator.energy == 100.0  # 40 + 60 clamped at max_energy
    assert predator.food_eaten == 1


def test_capture_energy_not_clamped_below_max():
    predator = make_predator(100.0, 100.0, flag=True)
    predator.energy = 20.0
    prey = make_prey(105.0, 100.0)

    resolve_captures([predator, prey], capture_radius=16.0,
                     energy_transfer=60.0)

    # 20 + 60 = 80, below the max_energy clamp
    assert predator.energy == 80.0


def test_no_flag_means_no_capture_even_in_range():
    predator = make_predator(100.0, 100.0, flag=False)
    prey = make_prey(105.0, 100.0)

    captures = resolve_captures([predator, prey], capture_radius=16.0,
                                energy_transfer=60.0)

    assert captures == 0
    assert prey.is_alive
    assert predator.food_eaten == 0


def test_prey_beyond_radius_not_captured():
    predator = make_predator(100.0, 100.0, flag=True)
    prey = make_prey(200.0, 200.0)

    captures = resolve_captures([predator, prey], capture_radius=16.0,
                                energy_transfer=60.0)

    assert captures == 0
    assert prey.is_alive
    # The spent attempt flag must be cleared either way
    assert predator._capture_attempt is False


def test_two_predators_contest_one_prey_deterministically():
    first = make_predator(100.0, 100.0, flag=True)
    second = make_predator(102.0, 100.0, flag=True)
    prey = make_prey(101.0, 100.0)

    captures = resolve_captures([first, second, prey], capture_radius=16.0,
                                energy_transfer=60.0)

    # One prey, one death; list order decides the winner
    assert captures == 1
    assert first.food_eaten == 1
    assert second.food_eaten == 0


def test_dead_prey_never_targeted():
    predator = make_predator(100.0, 100.0, flag=True)
    prey = make_prey(105.0, 100.0)
    prey.is_alive = False

    captures = resolve_captures([predator, prey], capture_radius=16.0,
                                energy_transfer=60.0)

    assert captures == 0


def test_step_simulation_predator_defers_eat_gate_and_ignores_plants():
    # All-positive genome biases the eat-gate sigmoid above the 0.5
    # threshold deterministically, so the capture flag is guaranteed set
    world = World(200, 200)
    world.food = []
    from simulation.environment import Food
    world.food.append(Food(100.0, 100.0))

    predator = make_predator(100.0, 100.0)
    predator.genome = (np.ones(genome_size()) * 0.5).astype(np.float32)

    step_simulation(world, [predator], RayCaster())

    # Predator flagged a capture attempt and did not eat the plant food
    assert predator._capture_attempt is True
    assert len(world.food) == 1


def test_step_simulation_prey_eats_plant_food():
    world = World(200, 200)
    world.food = []
    from simulation.environment import Food
    world.food.append(Food(100.0, 100.0))

    prey = make_prey(100.0, 100.0)
    prey.genome = (np.ones(genome_size()) * 0.5).astype(np.float32)

    step_simulation(world, [prey], RayCaster())

    assert prey._capture_attempt is False
    assert len(world.food) == 0
    assert prey.food_eaten == 1


def test_prey_role_default_and_validation():
    agent = Organism(0, 10.0, 10.0)
    assert agent.role == 'prey'
    try:
        Organism(0, 10.0, 10.0, role='parasite')
        assert False, "invalid role must raise"
    except ValueError:
        pass


def test_organism_state_round_trip_preserves_role():
    predator = make_predator(100.0, 100.0)
    predator.food_eaten = 3
    state = predator.get_state()

    restored = Organism(0, 0.0, 0.0)
    restored.load_state(state)

    assert restored.role == 'predator'
    assert restored.food_eaten == 3


def test_organism_state_restores_legacy_default_role():
    # States serialized before Phase 6 have no role key; restoring them
    # must keep the pre-Phase-6 behavior (every agent is prey)
    legacy = make_prey(10.0, 10.0)
    state = legacy.get_state()
    del state['role']
    del state['food_eaten']

    restored = Organism(0, 0.0, 0.0)
    restored.load_state(state)

    assert restored.role == 'prey'
    assert restored.food_eaten == 0
