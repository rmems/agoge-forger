//! `run_name` is untrusted input joined onto a caller-supplied root, so it is
//! held to the same rule as the Python `path_safety.py`: the check runs on the
//! pre-resolution string, because canonicalizing first would consume the very
//! `..` segments it is meant to catch.

use std::fs;
use std::path::{Path, PathBuf};

use agoge_benchgen::{write_workload, WorkloadSpec};

/// A fresh, uniquely named scratch directory under `CARGO_TARGET_TMPDIR`.
///
/// Each test needs its own because `cargo test` runs them as parallel threads
/// in a single process.
fn scratch_dir(label: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(label);
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("clear scratch dir");
    }
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn spec_named(run_name: &str) -> WorkloadSpec {
    let mut spec = WorkloadSpec::new(run_name);
    spec.count = 2;
    spec
}

#[test]
fn unsafe_run_names_are_rejected() {
    let root = scratch_dir("bench-path-unsafe").join("runs");

    // Every shape that must never reach `runs_root.join(..)`: parent
    // traversal, nested paths, both separators, an absolute path (which
    // `Path::join` would substitute wholesale for the root), and the names
    // that are not a directory component at all.
    let rejected = [
        "..",
        "../escape",
        "..\\escape",
        "a/b",
        "a\\b",
        "/etc/agoge",
        ".",
        "./here",
        "",
        "   ",
        "\t",
    ];

    for run_name in rejected {
        let result = write_workload(&spec_named(run_name), &root);
        assert!(
            result.is_err(),
            "run_name {run_name:?} must be rejected, got {result:?}"
        );
    }

    fs::remove_dir_all(root.parent().expect("scratch parent")).expect("cleanup");
}

#[test]
fn a_traversing_run_name_creates_nothing_outside_the_runs_root() {
    let scratch = scratch_dir("bench-path-traversal");
    // `runs_root` deliberately does not exist yet: a rejected run must not
    // even create the root, let alone the sibling it was aiming at.
    let runs_root = scratch.join("runs");
    let traversal_target = scratch.join("escape");

    let error = write_workload(&spec_named("../escape"), &runs_root)
        .expect_err("traversal must be rejected");

    assert!(
        format!("{error:#}").contains("path separator"),
        "unexpected error: {error:#}"
    );
    assert!(
        !traversal_target.exists(),
        "a rejected run_name wrote outside the runs root"
    );
    assert!(!runs_root.exists(), "a rejected run_name created the root");

    fs::remove_dir_all(&scratch).expect("cleanup");
}

#[test]
fn an_absolute_run_name_does_not_hijack_the_runs_root() {
    let scratch = scratch_dir("bench-path-absolute");
    let runs_root = scratch.join("runs");
    // An absolute name is the dangerous case `Path::join` handles silently:
    // joining it discards `runs_root` entirely. Point it inside the scratch
    // directory so a regression is contained and still visible.
    let hijack_target = scratch.join("absolute-target");
    let absolute_name = hijack_target.to_str().expect("utf-8 scratch path");

    let error =
        write_workload(&spec_named(absolute_name), &runs_root).expect_err("absolute must reject");

    assert!(format!("{error:#}").contains("run_name"), "{error:#}");
    assert!(
        !hijack_target.exists(),
        "an absolute run_name escaped the runs root"
    );
    assert!(!runs_root.exists());

    fs::remove_dir_all(&scratch).expect("cleanup");
}

#[test]
fn a_plain_run_name_is_accepted_and_stays_inside_the_runs_root() {
    let scratch = scratch_dir("bench-path-plain");
    let runs_root = scratch.join("runs");

    // The mirror image of the rejection table: names that merely look
    // suspicious (leading dot, dots inside) are ordinary directory names and
    // must keep working, or the check is over-tightened into a nuisance.
    for run_name in ["plain-run", "run.2026-08-23", ".hidden", "a...b"] {
        let workload_path =
            write_workload(&spec_named(run_name), &runs_root).expect("plain name accepted");
        let expected_dir = runs_root.join(run_name);
        assert_eq!(workload_path.parent(), Some(expected_dir.as_path()));
        assert!(workload_path.is_file());
    }

    fs::remove_dir_all(&scratch).expect("cleanup");
}

#[test]
fn a_blank_workload_label_is_rejected_before_any_directory_is_created() {
    let scratch = scratch_dir("bench-path-blank-label");
    let runs_root = scratch.join("runs");
    let mut spec = spec_named("blank-label");
    spec.workload = "   ".to_string();

    let error = write_workload(&spec, &runs_root).expect_err("blank label must be rejected");

    assert!(
        format!("{error:#}").contains("workload label"),
        "unexpected error: {error:#}"
    );
    assert!(!runs_root.exists());

    fs::remove_dir_all(&scratch).expect("cleanup");
}

/// The case the `run_name` string check cannot see: the name is a single,
/// perfectly legal component, but the destination already exists as a symlink
/// pointing out of the root. `create_dir_all` accepts a symlink-to-directory
/// and `fs::write` follows it, so without an explicit check both artifacts
/// land outside `runs_root`.
#[cfg(unix)]
#[test]
fn a_symlinked_run_directory_is_rejected() {
    let scratch = scratch_dir("bench-path-symlink");
    let runs_root = scratch.join("runs");
    let outside = scratch.join("outside");
    fs::create_dir_all(&runs_root).expect("create runs root");
    fs::create_dir_all(&outside).expect("create outside dir");
    std::os::unix::fs::symlink(&outside, runs_root.join("escape")).expect("create symlink");

    let error =
        write_workload(&spec_named("escape"), &runs_root).expect_err("symlink must be rejected");

    assert!(
        format!("{error:#}").contains("symlink"),
        "unexpected error: {error:#}"
    );
    for artifact in [
        agoge_benchgen::WORKLOAD_FILE_NAME,
        agoge_benchgen::WORKLOAD_MANIFEST_FILE_NAME,
    ] {
        assert!(
            !outside.join(artifact).exists(),
            "{artifact} escaped the runs root through a symlinked run directory"
        );
    }

    fs::remove_dir_all(&scratch).expect("cleanup");
}

/// One level below the directory case: the run directory is genuine, but an
/// artifact *inside* it is a symlink. `fs::write` follows a symlinked file and
/// truncates whatever it points at, so checking only `run_dir` is not enough.
///
/// Runs once per artifact, since the two are written by different functions.
#[cfg(unix)]
#[test]
fn a_symlinked_artifact_file_is_rejected() {
    for (label, artifact) in [
        ("workload", agoge_benchgen::WORKLOAD_FILE_NAME),
        ("manifest", agoge_benchgen::WORKLOAD_MANIFEST_FILE_NAME),
    ] {
        let scratch = scratch_dir(&format!("bench-path-symlink-{label}"));
        let runs_root = scratch.join("runs");
        let run_dir = runs_root.join("planted");
        fs::create_dir_all(&run_dir).expect("create run dir");

        // The file the attacker wants truncated, well outside the runs root.
        let victim = scratch.join("victim.txt");
        fs::write(&victim, b"precious").expect("create victim");
        std::os::unix::fs::symlink(&victim, run_dir.join(artifact)).expect("create symlink");

        let error = write_workload(&spec_named("planted"), &runs_root)
            .expect_err("symlinked {label} must be rejected");

        assert!(
            format!("{error:#}").contains("symlink"),
            "unexpected error for {label}: {error:#}"
        );
        assert_eq!(
            fs::read(&victim).expect("victim readable"),
            b"precious",
            "a symlinked {label} let the write truncate its target"
        );

        fs::remove_dir_all(&scratch).expect("cleanup");
    }
}
