use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn validate_file(path: &str) -> Result<()> {
    let file = File::open(path).with_context(|| format!("Failed to open {}", path))?;
    let reader = BufReader::new(file);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        
        let _v: Value = serde_json::from_str(&line)
            .with_context(|| format!("Invalid JSON on line {}", i + 1))?;
    }
    Ok(())
}
