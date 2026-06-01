# Clustering 12 000 Power Plants Finding Natural Groups in Energy Data

Published: 2025-10-06
Medium: [https://medium.com/@kyle-t-jones/clustering-12-000-power-plants-finding-natural-groups-in-energy-data-0e58e8803b03](https://medium.com/@kyle-t-jones/clustering-12-000-power-plants-finding-natural-groups-in-energy-data-0e58e8803b03)

## Business context

Not all power plants are created equal. A 1,000 MW nuclear plant operating at 92% capacity has nothing in common with a 10 MW solar farm running at 25% capacity. Yet both exist in the same dataset, and comparing them directly is meaningless.

Clustering solves this. It automatically groups similar plants together, then we can analyze the groups. This reveals insights impossible to see in aggregate data --- like identifying that certain natural gas plants are 20% more efficient than their peers, or that some states have systematically cleaner electricity portfolios.

This article demonstrates five clustering methods on 12,613 U.S. power plants, showing how to find natural groupings, profile clusters, and apply results to real-world problems like benchmarking and policy targeting.



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — synthetic plant feature generation. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p clustering_12_000_power_plants_finding_natural_groups_in_energy_data_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).