"""Optimizer adapters for robot-control experiments."""
from __future__ import annotations

from typing import Protocol

import nevergrad as ng
import numpy as np


class OptimizerAdapter(Protocol):
    """Minimal optimizer interface used by training loops."""

    def ask(self) -> list[np.ndarray]:
        ...

    def tell(self, vectors: list[np.ndarray], scores: list[float]) -> None:
        ...

    def recommendation(self) -> np.ndarray:
        ...


class NevergradCMAESOptimizer:
    """Nevergrad ParametrizedCMA behind the shared optimizer interface."""

    def __init__(
        self,
        initial_guess: np.ndarray,
        sigma: float,
        population: int,
        generations: int,
    ) -> None:
        self._population = int(population)
        parametrization = ng.p.Array(init=initial_guess)
        parametrization.set_mutation(sigma=sigma)
        self._optimizer = ng.optimizers.ParametrizedCMA(popsize=population)(
            parametrization=parametrization,
            budget=generations * population,
            num_workers=population,
        )
        self._pending = []

    def ask(self) -> list[np.ndarray]:
        self._pending = [self._optimizer.ask() for _ in range(self.population)]
        return [
            np.asarray(candidate.value, dtype=np.float32).copy()
            for candidate in self._pending
        ]

    def tell(self, vectors: list[np.ndarray], scores: list[float]) -> None:
        if len(scores) != len(self._pending):
            raise ValueError(
                f"Got {len(scores)} scores for {len(self._pending)} pending candidates."
            )
        for candidate, score in zip(self._pending, scores):
            self._optimizer.tell(candidate, float(score))
        self._pending = []

    def recommendation(self) -> np.ndarray:
        return np.asarray(
            self._optimizer.provide_recommendation().value,
            dtype=np.float32,
        )

    @property
    def population(self) -> int:
        return self._population
