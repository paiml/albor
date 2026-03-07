//! Integration tests for the albor sovereign stack.
//!
//! These tests verify that sovereign tools (pv, batuta, bashrs) produce
//! correct results on albor's configs and contracts.

use std::process::Command;

// ─── Contract Validation ─────────────────────────────────────────────

#[test]
fn pv_validates_all_contracts() {
    let entries: Vec<_> = std::fs::read_dir("contracts")
        .expect("contracts/ directory must exist")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "yaml")
        })
        .collect();

    assert!(!entries.is_empty(), "no contract YAML files found");

    for entry in &entries {
        let path = entry.path();
        let output = match Command::new("pv")
            .args(["validate", path.to_str().unwrap()])
            .output()
        {
            Ok(o) => o,
            Err(_) => {
                eprintln!("pv not installed, skipping test");
                return;
            }
        };

        assert!(
            output.status.success(),
            "pv validate failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn pv_audits_all_contracts_clean() {
    let entries: Vec<_> = std::fs::read_dir("contracts")
        .expect("contracts/ directory must exist")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "yaml")
        })
        .collect();

    for entry in &entries {
        let path = entry.path();
        let output = match Command::new("pv")
            .args(["audit", path.to_str().unwrap()])
            .output()
        {
            Ok(o) => o,
            Err(_) => {
                eprintln!("pv not installed, skipping test");
                return;
            }
        };

        assert!(
            output.status.success(),
            "pv audit failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("No audit findings"),
            "pv audit found issues in {}: {}",
            path.display(),
            stdout
        );
    }
}

// ─── YAML Config Validation ──────────────────────────────────────────

#[test]
fn all_yaml_configs_parse() {
    let config_dirs = ["configs/train", "configs/eval", "configs/pipeline"];

    for dir in &config_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let ext = path.extension().and_then(|e| e.to_str());
                if ext == Some("yaml") || ext == Some("yml") {
                    let content = std::fs::read_to_string(&path)
                        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
                    let _: serde_yaml::Value = serde_yaml::from_str(&content)
                        .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));
                }
            }
        }
    }
}

#[test]
fn all_contract_yamls_parse() {
    for entry in std::fs::read_dir("contracts").expect("contracts/").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "yaml") {
            let content = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
            let _: serde_yaml::Value = serde_yaml::from_str(&content)
                .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));
        }
    }
}

// ─── Makefile Linting ────────────────────────────────────────────────

#[test]
fn bashrs_lints_makefile() {
    let output = match Command::new("bashrs")
        .args(["make", "lint", "Makefile"])
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            eprintln!("bashrs not installed, skipping test");
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    // bashrs changed output format: older versions print "0 error(s)",
    // newer versions print "✓ No issues found"
    assert!(
        combined.contains("0 error(s)") || combined.contains("No issues found"),
        "bashrs make lint found errors: {combined}",
    );
}

// ─── Falsification Tests ─────────────────────────────────────────────

#[test]
fn falsify_test_yamls_parse() {
    for entry in std::fs::read_dir("tests/falsify").expect("tests/falsify/").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "yaml") {
            let content = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
            let _: serde_yaml::Value = serde_yaml::from_str(&content)
                .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e));
        }
    }
}

// ─── Provenance ──────────────────────────────────────────────────────

#[test]
fn provenance_file_exists() {
    assert!(
        std::path::Path::new("docs/PROVENANCE.md").exists(),
        "docs/PROVENANCE.md must exist for data provenance tracking"
    );
}

#[test]
fn model_cards_exist() {
    let cards = [
        "docs/book/src/model-cards/albor-base-50m.md",
        "docs/book/src/model-cards/albor-base-350m.md",
    ];
    for card in &cards {
        assert!(
            std::path::Path::new(card).exists(),
            "model card {} must exist",
            card
        );
    }
}

// ─── Spec Integrity ─────────────────────────────────────────────────

#[test]
fn spec_chapters_exist() {
    let required = [
        "docs/book/src/spec/03-architecture.md",
        "docs/book/src/spec/06-training.md",
        "docs/book/src/spec/08-evaluation.md",
        "docs/book/src/spec/11-gaps.md",
        "docs/book/src/spec/12-quality-contracts.md",
        "docs/book/src/spec/15-phases.md",
        "docs/book/src/spec/17-success.md",
    ];
    for ch in &required {
        assert!(
            std::path::Path::new(ch).exists(),
            "spec chapter {} must exist",
            ch
        );
    }
}

/// Pre-registered effect size thresholds (HDD-07 statistical significance)
#[allow(dead_code)]
const EFFECT_SIZE_THRESHOLD: f64 = 0.8; // Cohen's d, large effect
