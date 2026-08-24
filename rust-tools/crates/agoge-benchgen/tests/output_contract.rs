//! The on-disk file contract that Python and Julia read back.
//!
//! Per `docs/contracts/polyglot_boundaries.md` there is no FFI: these two
//! files *are* the interface, so their shape is part of the public API.

use std::fs;
use std::path::{Path, PathBuf};

use agoge_benchgen::{
    generate_workload, write_workload, WorkloadSpec, GENERATOR_NAME, MAX_REQUEST_COUNT,
    WORKLOAD_FILE_NAME, WORKLOAD_MANIFEST_FILE_NAME, WORKLOAD_SCHEMA_VERSION,
};
use serde_json::Value;

/// A fresh, uniquely named runs root under `CARGO_TARGET_TMPDIR`.
///
/// Tests run as parallel threads in one process, so every test needs its own
/// directory; `CARGO_TARGET_TMPDIR` keeps that under `rust-tools/target/`
/// instead of the system temp filesystem.
fn scratch_root(label: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(label);
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch root");
    }
    fs::create_dir_all(&dir).expect("create scratch root");
    dir
}

fn spec_for(run_name: &str, count: usize) -> WorkloadSpec {
    WorkloadSpec {
        run_name: run_name.to_string(),
        workload: "eval".to_string(),
        count,
        seed: 11,
        max_tokens: 64,
        stream: true,
    }
}

#[test]
fn write_workload_returns_the_jsonl_path_with_the_manifest_beside_it() {
    let root = scratch_root("bench-contract-paths");
    let spec = spec_for("shape", 5);

    let workload_path = write_workload(&spec, &root).expect("write_workload");

    assert_eq!(
        workload_path.file_name().and_then(|name| name.to_str()),
        Some(WORKLOAD_FILE_NAME)
    );
    assert!(workload_path.is_file());
    assert!(
        workload_path
            .with_file_name(WORKLOAD_MANIFEST_FILE_NAME)
            .is_file(),
        "the manifest must sit next to the workload, not elsewhere"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn every_line_is_a_valid_json_object_and_no_line_is_blank() {
    let root = scratch_root("bench-contract-lines");
    let spec = spec_for("lines", 12);
    let workload_path = write_workload(&spec, &root).expect("write_workload");
    let text = fs::read_to_string(&workload_path).expect("read workload");

    // A blank line is legal JSONL for a lenient reader but not for a strict
    // one, and an unterminated last line silently drops a row when files are
    // concatenated — so both are checked explicitly rather than implied.
    assert!(text.ends_with('\n'), "the file must end with a newline");
    assert!(
        !text.contains("\n\n"),
        "the file must not contain a blank line"
    );
    assert!(!text.contains('\r'), "line endings must be bare \\n");

    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), spec.count);
    for (index, line) in lines.iter().enumerate() {
        let row: Value = serde_json::from_str(line)
            .unwrap_or_else(|err| panic!("line {} is not valid JSON: {err}", index + 1));
        assert!(row.is_object(), "line {} is not an object", index + 1);
    }

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn rows_are_numbered_one_to_count_in_order_and_echo_the_spec() {
    let root = scratch_root("bench-contract-rows");
    let spec = spec_for("rows", 9);
    let workload_path = write_workload(&spec, &root).expect("write_workload");
    let text = fs::read_to_string(&workload_path).expect("read workload");

    for (index, line) in text.lines().enumerate() {
        let row: Value = serde_json::from_str(line).expect("row parses");
        assert_eq!(row["request_id"], Value::from(index as u64 + 1));
        assert_eq!(row["workload"], Value::from(spec.workload.as_str()));
        assert_eq!(row["seed"], Value::from(spec.seed));
        assert_eq!(row["max_tokens"], Value::from(spec.max_tokens));
        assert_eq!(row["stream"], Value::from(spec.stream));
        assert_eq!(row["schema_version"], Value::from(WORKLOAD_SCHEMA_VERSION));
        assert!(
            row["prompt"].as_str().is_some_and(|p| !p.is_empty()),
            "row {} has no prompt",
            index + 1
        );
    }

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn the_manifest_records_the_spec_and_carries_no_timestamp() {
    let root = scratch_root("bench-contract-manifest");
    let spec = spec_for("manifest", 6);
    let workload_path = write_workload(&spec, &root).expect("write_workload");
    let manifest_text =
        fs::read_to_string(workload_path.with_file_name(WORKLOAD_MANIFEST_FILE_NAME))
            .expect("read manifest");
    let manifest: Value = serde_json::from_str(&manifest_text).expect("manifest parses");

    assert_eq!(manifest["run_name"], Value::from(spec.run_name.as_str()));
    assert_eq!(manifest["workload"], Value::from(spec.workload.as_str()));
    assert_eq!(manifest["count"], Value::from(spec.count));
    assert_eq!(manifest["seed"], Value::from(spec.seed));
    assert_eq!(manifest["max_tokens"], Value::from(spec.max_tokens));
    assert_eq!(manifest["stream"], Value::from(spec.stream));
    assert_eq!(
        manifest["schema_version"],
        Value::from(WORKLOAD_SCHEMA_VERSION)
    );
    assert_eq!(manifest["generator"], Value::from(GENERATOR_NAME));
    assert!(manifest["generator_version"].as_str().is_some());

    // Asserted by name, not merely by "the two runs matched": a timestamp is
    // the single most tempting field to add here and the one that would break
    // byte-identical replay for everyone downstream.
    let object = manifest.as_object().expect("manifest is an object");
    assert!(
        !object.contains_key("timestamp"),
        "the manifest must not carry a timestamp"
    );
    assert!(!manifest_text.contains("timestamp"));
    assert!(
        manifest_text.ends_with("}\n"),
        "the manifest must be newline-terminated"
    );
    assert!(
        manifest_text.contains("\n  \"run_name\""),
        "the manifest must be pretty-printed with a 2-space indent"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_count_of_one_is_accepted_as_the_lower_boundary() {
    let root = scratch_root("bench-contract-count-one");
    let workload_path = write_workload(&spec_for("boundary", 1), &root).expect("write_workload");
    let text = fs::read_to_string(&workload_path).expect("read workload");

    assert_eq!(text.lines().count(), 1);
    assert!(text.ends_with('\n'));

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_count_of_zero_is_rejected_and_writes_nothing() {
    let root = scratch_root("bench-contract-count-zero");
    let spec = spec_for("empty", 0);

    let error = write_workload(&spec, &root).expect_err("count 0 must be rejected");

    assert!(
        format!("{error:#}").contains("count must be greater than 0"),
        "unexpected error: {error:#}"
    );
    // The spec is validated before the run directory is created, so a rejected
    // request must not leave a half-built run behind.
    assert!(!root.join("empty").exists());

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_count_above_the_maximum_is_rejected_before_any_allocation() {
    let root = scratch_root("bench-contract-count-max");
    let spec = spec_for("too-big", MAX_REQUEST_COUNT + 1);

    // If validation ever moved after `Vec::with_capacity(count)`, this test
    // would try to reserve ~1M rows instead of failing fast.
    let error = write_workload(&spec, &root).expect_err("oversized count must be rejected");

    assert!(
        format!("{error:#}").contains("exceeds the maximum"),
        "unexpected error: {error:#}"
    );
    assert!(!root.join("too-big").exists());
    assert!(generate_workload(&spec).is_err());

    fs::remove_dir_all(&root).expect("cleanup");
}
