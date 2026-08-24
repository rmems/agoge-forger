use std::path::PathBuf;

use agoge_benchgen::WorkloadSpec;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Validates a JSONL dataset
    Validate { file: String },
    /// Generates a deterministic, seeded benchmark workload as JSONL
    ///
    /// Writes <runs-root>/<run-name>/workload.jsonl plus a companion
    /// workload_manifest.json. The same --seed, --count and --workload always
    /// produce byte-identical output.
    Benchgen {
        /// Run name; becomes the <runs-root>/<run-name>/ output directory
        #[arg(long)]
        run_name: String,
        /// Number of requests to generate
        #[arg(long, default_value_t = 32)]
        count: usize,
        /// PRNG seed; identical seeds produce byte-identical output
        #[arg(long, default_value_t = agoge_benchgen::DEFAULT_SEED)]
        seed: u64,
        /// Workload label recorded on every row (e.g. inference, eval)
        #[arg(long, default_value = "inference")]
        workload: String,
        /// max_tokens recorded on every row
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Record stream=true on every row
        #[arg(long)]
        stream: bool,
        /// Root directory holding per-run output directories
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    match args.cmd {
        Commands::Validate { file } => {
            println!("Validating {} ...", file);
            agoge_jsonl::validate_file(&file)?;
            println!("Validation successful.");
        }
        Commands::Benchgen {
            run_name,
            count,
            seed,
            workload,
            max_tokens,
            stream,
            runs_root,
        } => {
            println!("Generating workload for run {run_name} ...");
            let spec = WorkloadSpec {
                run_name,
                workload,
                count,
                seed,
                max_tokens,
                stream,
            };
            let workload_path = agoge_benchgen::write_workload(&spec, &runs_root)?;
            let manifest_path =
                workload_path.with_file_name(agoge_benchgen::WORKLOAD_MANIFEST_FILE_NAME);
            println!(
                "Wrote {} ({count} requests, seed {seed}).",
                workload_path.display()
            );
            println!("Wrote {}.", manifest_path.display());
        }
    }

    Ok(())
}
