//! The byte-for-byte replay guarantee.
//!
//! Every assertion here compares raw file bytes rather than deserialized
//! structs. Struct equality would still hold if a refactor changed JSON field
//! order, number formatting, or line terminators — and those are exactly the
//! things a downstream Python or Julia reader diffs against a stored baseline.

use std::fs;
use std::path::{Path, PathBuf};

use agoge_benchgen::{
    write_workload, WorkloadSpec, WORKLOAD_FILE_NAME, WORKLOAD_MANIFEST_FILE_NAME,
};

/// A fresh, uniquely named runs root under `CARGO_TARGET_TMPDIR`.
///
/// `CARGO_TARGET_TMPDIR` sits under `rust-tools/target/`, which keeps the
/// suite off the system temp filesystem. Each test passes its own `label`
/// because `cargo test` runs tests as parallel threads in a single process:
/// two tests sharing a runs root would race on the same `workload.jsonl`.
fn scratch_root(label: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(label);
    // Leftovers from an aborted run could otherwise mask a write that never
    // happened, so start from a guaranteed-empty directory.
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch root");
    }
    fs::create_dir_all(&dir).expect("create scratch root");
    dir
}

fn spec_for(run_name: &str, workload: &str, count: usize, seed: u64) -> WorkloadSpec {
    WorkloadSpec {
        run_name: run_name.to_string(),
        workload: workload.to_string(),
        count,
        seed,
        max_tokens: 128,
        stream: false,
    }
}

/// Writes `spec` into `root` and returns the raw `(workload, manifest)` bytes.
fn write_and_read(root: &Path, spec: &WorkloadSpec) -> (Vec<u8>, Vec<u8>) {
    let workload_path = write_workload(spec, root).expect("write_workload");
    let manifest_path = workload_path.with_file_name(WORKLOAD_MANIFEST_FILE_NAME);
    (
        fs::read(&workload_path).expect("read workload"),
        fs::read(&manifest_path).expect("read manifest"),
    )
}

/// FNV-1a, reimplemented here so the digest pin does not depend on any hash
/// the crate under test might change.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn first_prompt(workload_bytes: &[u8]) -> String {
    let text = String::from_utf8(workload_bytes.to_vec()).expect("workload is utf-8");
    let line = text.lines().next().expect("at least one row");
    let row: serde_json::Value = serde_json::from_str(line).expect("row parses");
    row["prompt"]
        .as_str()
        .expect("prompt is a string")
        .to_string()
}

#[test]
fn same_spec_written_to_two_roots_produces_identical_bytes() {
    let root_a = scratch_root("bench-determinism-same-spec-a");
    let root_b = scratch_root("bench-determinism-same-spec-b");
    let spec = spec_for("replay", "inference", 16, 7);

    let (workload_a, manifest_a) = write_and_read(&root_a, &spec);
    let (workload_b, manifest_b) = write_and_read(&root_b, &spec);

    assert_eq!(workload_a, workload_b, "workload.jsonl bytes differ");
    // The manifest carries the same promise: it deliberately holds no
    // timestamp, hostname, or path, so two runs of one spec must match too.
    assert_eq!(
        manifest_a, manifest_b,
        "workload_manifest.json bytes differ"
    );

    fs::remove_dir_all(&root_a).expect("cleanup");
    fs::remove_dir_all(&root_b).expect("cleanup");
}

#[test]
fn a_different_seed_produces_different_bytes() {
    let root = scratch_root("bench-determinism-seed");
    let (workload_42, _) = write_and_read(&root, &spec_for("seed-42", "inference", 16, 42));
    let (workload_43, _) = write_and_read(&root, &spec_for("seed-43", "inference", 16, 43));

    assert_ne!(
        workload_42, workload_43,
        "changing the seed must change the generated stream"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_different_workload_label_changes_the_prompt_stream() {
    let root = scratch_root("bench-determinism-label");
    let (inference, _) = write_and_read(&root, &spec_for("label-inference", "inference", 4, 42));
    let (eval, _) = write_and_read(&root, &spec_for("label-eval", "eval", 4, 42));

    // The label is echoed on every row, so the files would differ even if the
    // label were ignored by the PRNG. Compare the prompts instead: that is the
    // part the label is supposed to reach, via the seed mix.
    assert_ne!(
        first_prompt(&inference),
        first_prompt(&eval),
        "the workload label must be mixed into the PRNG seed"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_short_run_is_a_byte_exact_prefix_of_a_longer_run() {
    let root = scratch_root("bench-determinism-prefix");
    let (short, _) = write_and_read(&root, &spec_for("count-8", "inference", 8, 42));
    let (long, _) = write_and_read(&root, &spec_for("count-32", "inference", 32, 42));

    // Rows are drawn sequentially from one PRNG stream, so raising --count
    // must only append. If this fails, a bigger benchmark can no longer be
    // compared against a smaller one recorded earlier.
    assert!(long.len() > short.len());
    assert_eq!(
        &long[..short.len()],
        &short[..],
        "count=8 output is not a prefix of count=32 output"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn seed_42_inference_matches_the_pinned_reference_bytes() {
    // Self-consistency checks (write twice, compare) pass happily even if the
    // PRNG stream shifts under a refactor — both sides move together. This
    // literal is the only thing that pins the stream itself, so a changed
    // SplitMix64 constant, a reordered prompt table, or a swapped field order
    // fails here instead of silently invalidating every recorded benchmark.
    const EXPECTED: &str = concat!(
        r#"{"request_id":1,"workload":"inference","seed":42,"prompt":"Write a short guide to learning-rate warmup schedules.","max_tokens":128,"stream":false,"schema_version":1}"#,
        "\n",
        r#"{"request_id":2,"workload":"inference","seed":42,"prompt":"Draft a changelog entry for 4-bit weight quantization; call out anything that would corrupt a run; keep it under 100 words; answer as a bulleted list.","max_tokens":128,"stream":false,"schema_version":1}"#,
        "\n",
        r#"{"request_id":3,"workload":"inference","seed":42,"prompt":"Summarize KV cache eviction policies; cite concrete numbers.","max_tokens":128,"stream":false,"schema_version":1}"#,
        "\n",
    );

    let root = scratch_root("bench-determinism-pinned-rows");
    let (workload, _) = write_and_read(&root, &spec_for("pinned", "inference", 3, 42));

    assert_eq!(String::from_utf8(workload).expect("utf-8"), EXPECTED);

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn a_longer_pinned_run_matches_its_reference_digest() {
    // The three pinned rows above only cover the head of the stream; this
    // digest covers all 32 rows, including the rejection-sampling branch in
    // `next_range` that a short run may never take.
    const EXPECTED_DIGEST: u64 = 0x19a5_2a9a_77da_5a9c;
    const EXPECTED_LEN: usize = 6717;

    let root = scratch_root("bench-determinism-pinned-digest");
    let (workload, _) = write_and_read(&root, &spec_for("pinned-32", "inference", 32, 42));

    assert_eq!(workload.len(), EXPECTED_LEN, "workload byte length changed");
    assert_eq!(
        fnv1a_64(&workload),
        EXPECTED_DIGEST,
        "the 32-row stream for seed 42 changed"
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn rewriting_a_run_directory_overwrites_rather_than_appends() {
    // Regenerating into an existing run directory is the normal way to redo a
    // benchmark; a partial overwrite would leave stale rows behind the new
    // ones and quietly break the byte comparison downstream.
    let root = scratch_root("bench-determinism-rewrite");
    let long_spec = spec_for("rerun", "inference", 32, 42);
    let short_spec = spec_for("rerun", "inference", 4, 42);

    let (_, _) = write_and_read(&root, &long_spec);
    let (rewritten, _) = write_and_read(&root, &short_spec);

    let expected = {
        let fresh_root = scratch_root("bench-determinism-rewrite-fresh");
        let (bytes, _) = write_and_read(&fresh_root, &short_spec);
        fs::remove_dir_all(&fresh_root).expect("cleanup");
        bytes
    };

    assert_eq!(rewritten, expected);
    assert_eq!(
        String::from_utf8(rewritten).expect("utf-8").lines().count(),
        4
    );

    fs::remove_dir_all(&root).expect("cleanup");
}

#[test]
fn the_workload_file_lands_at_the_documented_path() {
    let root = scratch_root("bench-determinism-path");
    let spec = spec_for("named-run", "inference", 2, 1);
    let workload_path = write_workload(&spec, &root).expect("write_workload");

    assert_eq!(
        workload_path,
        root.join("named-run").join(WORKLOAD_FILE_NAME)
    );

    fs::remove_dir_all(&root).expect("cleanup");
}
