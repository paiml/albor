//! Albor LLM — sovereign Python code completion model
//!
//! This crate provides integration tests for the albor training pipeline,
//! validating contracts, configs, and sovereign stack tool interoperability.

#![deny(dead_code)]
#![deny(unused_imports)]

/// Sovereign stack configuration module
pub mod config {
    /// Validate a YAML config file
    pub fn validate_yaml(path: &std::path::Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        let _: serde_yaml::Value = serde_yaml::from_str(&content)
            .map_err(|e| format!("parse {}: {}", path.display(), e))?;
        Ok(())
    }
}

/// Contract validation module
pub mod contracts {
    /// Number of provable contracts in the project
    pub const CONTRACT_COUNT: usize = 7;
}
