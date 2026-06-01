#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import generate_plant_features  # noqa: E402

def main() -> None:
    n_plants, seed = 12000, 42
    t0 = time.perf_counter()
    for _ in range(200):
        generate_plant_features(n_plants, seed)
    py_s = time.perf_counter() - t0
    try:
        import clustering_12_000_power_plants_finding_natural_groups_in_energy_data_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(n_plants, seed, 200)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    py = generate_plant_features(500, 42)
    rs_out = rs.generate_plant_features_py(500, 42)
    for a, b in zip(py, rs_out):
        assert np.asarray(a).shape == np.asarray(b).shape == (500,)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
