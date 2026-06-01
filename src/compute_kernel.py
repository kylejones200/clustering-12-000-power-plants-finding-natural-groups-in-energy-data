"""Synthetic plant features (numpy reference; RNG differs from Rust LCG)."""

from __future__ import annotations

import numpy as np


def generate_plant_features(n_plants: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    capacity = 50.0 + rng.random(n_plants) * 950.0
    heat_rate = 8.0 + rng.random(n_plants) * 4.0
    emissions = 0.4 + rng.random(n_plants) * 0.8
    return capacity, heat_rate, emissions
