//! `validate_file` is JSONL *syntax* validation only.
//!
//! Row-schema validation stays Python-owned in `datasets.py:normalize_row`
//! (see `docs/contracts/polyglot_boundaries.md`), so these tests deliberately
//! assert that a syntactically valid line is accepted regardless of its shape.

use std::fs;
use std::path::{Path, PathBuf};

/// A fresh, uniquely named scratch directory under `CARGO_TARGET_TMPDIR`.
///
/// That directory lives under `rust-tools/target/`, not the system temp dir,
/// and is only defined for integration tests. The per-test `label` keeps the
/// parallel test threads from sharing a fixture file.
fn scratch_dir(label: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(label);
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch dir");
    }
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write_fixture(dir: &Path, contents: &str) -> String {
    let path = dir.join("dataset.jsonl");
    fs::write(&path, contents).expect("write fixture");
    path.to_str().expect("utf-8 path").to_string()
}

#[test]
fn accepts_a_well_formed_jsonl_file() {
    let dir = scratch_dir("jsonl-accepts-valid");
    let path = write_fixture(
        &dir,
        "{\"text\":\"hello\"}\n{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n",
    );

    agoge_jsonl::validate_file(&path).expect("valid file should pass");

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn accepts_a_final_line_with_no_trailing_newline() {
    // Hand-edited datasets routinely lose the final newline; dropping the last
    // row from validation would let a broken row through unchecked.
    let dir = scratch_dir("jsonl-accepts-no-trailing-newline");
    let path = write_fixture(&dir, "{\"text\":\"a\"}\n{\"text\":\"b\"}");

    agoge_jsonl::validate_file(&path).expect("missing trailing newline should pass");

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn skips_blank_and_whitespace_only_lines() {
    let dir = scratch_dir("jsonl-skips-blank");
    let path = write_fixture(&dir, "{\"text\":\"a\"}\n\n   \n\t\n{\"text\":\"b\"}\n");

    agoge_jsonl::validate_file(&path).expect("blank lines should be skipped");

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn accepts_any_valid_json_value_because_schema_checks_are_python_owned() {
    let dir = scratch_dir("jsonl-accepts-scalars");
    let path = write_fixture(&dir, "42\n\"a string\"\nnull\n[1,2,3]\n");

    agoge_jsonl::validate_file(&path)
        .expect("syntax-only validation must not reject non-object rows");

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn rejects_malformed_json_and_reports_the_one_indexed_line() {
    // The line number is what a user acts on, and off-by-one here sends them
    // to the wrong row of a million-line dataset — hence the exact assertion.
    let dir = scratch_dir("jsonl-rejects-malformed");
    let path = write_fixture(&dir, "{\"text\":\"a\"}\n{\"text\":\"b\"}\n{oops}\n");

    let error = agoge_jsonl::validate_file(&path).expect_err("malformed JSON must fail");
    let rendered = format!("{error:#}");

    assert!(
        rendered.contains("Invalid JSON on line 3"),
        "unexpected error: {rendered}"
    );

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn blank_lines_still_count_toward_the_reported_line_number() {
    // Skipped lines are skipped for validation, not for counting; if they were
    // dropped from the index the reported line would drift.
    let dir = scratch_dir("jsonl-line-number-with-blanks");
    let path = write_fixture(&dir, "\n\n{\"text\":\"a\"}\nnot json\n");

    let error = agoge_jsonl::validate_file(&path).expect_err("malformed JSON must fail");
    let rendered = format!("{error:#}");

    assert!(
        rendered.contains("Invalid JSON on line 4"),
        "unexpected error: {rendered}"
    );

    fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn reports_the_path_when_the_file_is_missing() {
    let dir = scratch_dir("jsonl-missing-file");
    let missing = dir.join("absent.jsonl");
    let missing_path = missing.to_str().expect("utf-8 path");

    let error = agoge_jsonl::validate_file(missing_path).expect_err("missing file must fail");
    let rendered = format!("{error:#}");

    assert!(rendered.contains("Failed to open"), "{rendered}");
    assert!(rendered.contains(missing_path), "{rendered}");

    fs::remove_dir_all(&dir).expect("cleanup");
}
