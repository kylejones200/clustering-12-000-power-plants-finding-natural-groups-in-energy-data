use clustering_12_000_power_plants_finding_natural_groups_in_energy_data_core::generate_plant_features;
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (n_plants, seed=42))]
fn generate_plant_features_py<'py>(
    py: Python<'py>,
    n_plants: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (cap, hr, em) = generate_plant_features(n_plants, seed);
    Ok((cap.into_pyarray(py), hr.into_pyarray(py), em.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (n_plants, seed=42, iterations=200))]
fn bench_kernel_py(n_plants: usize, seed: u64, iterations: usize) -> PyResult<f64> {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = generate_plant_features(n_plants, seed);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn clustering_12_000_power_plants_finding_natural_groups_in_energy_data_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_plant_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
