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
    Validate {
        file: String,
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
    }

    Ok(())
}
