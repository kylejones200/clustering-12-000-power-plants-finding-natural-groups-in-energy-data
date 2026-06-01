use clustering_12_000_power_plants_finding_natural_groups_in_energy_data_core::generate_plant_features;

fn main() {
    for _ in 0..200 {
        let _ = generate_plant_features(12000, 42);
    }
}
