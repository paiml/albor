//! Benchmarks for sovereign tool response times.
//!
//! Validates that key sovereign tool operations complete within
//! acceptable time bounds (Jidoka performance gate).

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_yaml_parse(c: &mut Criterion) {
    c.bench_function("yaml_parse_contracts", |b| {
        b.iter(|| {
            for entry in std::fs::read_dir("contracts").unwrap().flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "yaml") {
                    let content = std::fs::read_to_string(&path).unwrap();
                    let _: serde_yaml::Value = serde_yaml::from_str(&content).unwrap();
                }
            }
        })
    });
}

criterion_group!(benches, bench_yaml_parse);
criterion_main!(benches);
