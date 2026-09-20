"""
Unit tests for all three crossover methods (Phase 3).

Verifies blend, uniform, and none crossover produce valid offspring.
"""

from __future__ import annotations
import numpy as np
import pytest

from evolution.crossover import (
    blend_crossover,
    uniform_crossover,
    no_crossover,
    crossover,
    crossover_population,
    CrossoverMethod,
)


class TestBlendCrossover:
    def test_mean_of_parents(self) -> None:
        p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        child = blend_crossover(p1, p2, alpha=0.5, rng=np.random.default_rng(0))
        assert np.allclose(child, 0.5, atol=0.1)

    def test_alpha_zero_returns_parent1(self) -> None:
        p1 = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        p2 = np.array([0.8, 0.9, 1.0], dtype=np.float32)
        child = blend_crossover(p1, p2, alpha=0.0, rng=np.random.default_rng(0))
        assert np.allclose(child, p1, atol=0.1)

    def test_alpha_one_returns_parent2(self) -> None:
        p1 = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        p2 = np.array([0.8, 0.9, 1.0], dtype=np.float32)
        child = blend_crossover(p1, p2, alpha=1.0, rng=np.random.default_rng(0))
        assert np.allclose(child, p2, atol=0.1)

    def test_preserves_shape_and_dtype(self) -> None:
        p1 = np.random.uniform(-1, 1, 995).astype(np.float32)
        p2 = np.random.uniform(-1, 1, 995).astype(np.float32)
        child = blend_crossover(p1, p2, rng=np.random.default_rng(0))
        assert child.shape == p1.shape
        assert child.dtype == p1.dtype


class TestUniformCrossover:
    def test_genes_from_either_parent(self) -> None:
        p1 = np.zeros(100, dtype=np.float32)
        p2 = np.ones(100, dtype=np.float32)
        child = uniform_crossover(p1, p2, rng=np.random.default_rng(0))
        assert np.all((child == 0) | (child == 1))

    def test_preserves_shape_and_dtype(self) -> None:
        p1 = np.random.uniform(-1, 1, 995).astype(np.float32)
        p2 = np.random.uniform(-1, 1, 995).astype(np.float32)
        child = uniform_crossover(p1, p2, rng=np.random.default_rng(0))
        assert child.shape == p1.shape
        assert child.dtype == p1.dtype


class TestNoCrossover:
    def test_returns_copy_of_parent1(self) -> None:
        p1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        p2 = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        child = no_crossover(p1, p2, rng=np.random.default_rng(0))
        assert np.array_equal(child, p1)
        assert child is not p1

    def test_preserves_shape_and_dtype(self) -> None:
        p1 = np.random.uniform(-1, 1, 995).astype(np.float32)
        p2 = np.random.uniform(-1, 1, 995).astype(np.float32)
        child = no_crossover(p1, p2, rng=np.random.default_rng(0))
        assert child.shape == p1.shape
        assert child.dtype == p1.dtype


class TestCrossoverDispatch:
    def test_blend_method(self) -> None:
        p1 = np.array([0.0, 0.0], dtype=np.float32)
        p2 = np.array([1.0, 1.0], dtype=np.float32)
        child = crossover(p1, p2, method="blend", rng=np.random.default_rng(0))
        assert np.allclose(child, 0.5, atol=0.1)

    def test_uniform_method(self) -> None:
        p1 = np.zeros(10, dtype=np.float32)
        p2 = np.ones(10, dtype=np.float32)
        child = crossover(p1, p2, method="uniform", rng=np.random.default_rng(0))
        assert np.all((child == 0) | (child == 1))

    def test_none_method(self) -> None:
        p1 = np.array([0.5, 0.5], dtype=np.float32)
        p2 = np.array([0.1, 0.1], dtype=np.float32)
        child = crossover(p1, p2, method="none", rng=np.random.default_rng(0))
        assert np.array_equal(child, p1)

    def test_unknown_method_raises(self) -> None:
        p1 = np.array([0.0], dtype=np.float32)
        p2 = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError):
            crossover(p1, p2, method="invalid", rng=np.random.default_rng(0))


class TestCrossoverPopulation:
    def test_produces_correct_number_of_offspring(self) -> None:
        parents = [
            np.random.uniform(-1, 1, 50).astype(np.float32)
            for _ in range(10)
        ]
        fitnesses = np.random.rand(10)
        offspring = crossover_population(
            parents, fitnesses, offspring_count=20,
            crossover_rate=1.0, method="blend", rng=np.random.default_rng(0)
        )
        assert len(offspring) == 20

    def test_crossover_rate_zero_clones_parents(self) -> None:
        parents = [np.array([0.1, 0.2], dtype=np.float32),
                   np.array([0.3, 0.4], dtype=np.float32)]
        fitnesses = np.array([1.0, 0.5])
        offspring = crossover_population(
            parents, fitnesses, offspring_count=5,
            crossover_rate=0.0, method="blend", rng=np.random.default_rng(0)
        )
        for child in offspring:
            assert np.array_equal(child, parents[0]) or np.array_equal(child, parents[1])

    def test_single_parent_fallback(self) -> None:
        parents = [np.array([0.1, 0.2], dtype=np.float32)]
        fitnesses = np.array([1.0])
        offspring = crossover_population(
            parents, fitnesses, offspring_count=3,
            crossover_rate=1.0, method="blend", rng=np.random.default_rng(0)
        )
        assert len(offspring) == 3
        for child in offspring:
            assert np.array_equal(child, parents[0])