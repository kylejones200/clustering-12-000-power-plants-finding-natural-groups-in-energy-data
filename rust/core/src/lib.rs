//! Synthetic plant feature vectors for clustering benchmarks.

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }
}

pub fn generate_plant_features(n_plants: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let mut capacity = Vec::with_capacity(n_plants);
    let mut heat_rate = Vec::with_capacity(n_plants);
    let mut emissions = Vec::with_capacity(n_plants);
    for _ in 0..n_plants {
        capacity.push(50.0 + rng.next_f64() * 950.0);
        heat_rate.push(8.0 + rng.next_f64() * 4.0);
        emissions.push(0.4 + rng.next_f64() * 0.8);
    }
    (capacity, heat_rate, emissions)
}
