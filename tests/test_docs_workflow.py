import json


def test_readme_distinguishes_checkpoint_adapter_and_merged_artifact():
    with open("README.md") as handle:
        readme = handle.read()

    assert "`checkpoint-*` directories are trainer recovery snapshots" in readme
    assert "The adapter saved at `adapters/<run_name>` is the LoRA output" in readme
    assert "The merged model under `merged/<run_name>` is the single final artifact" in readme


def test_devcontainer_matches_python_only_repository_boundary():
    with open(".devcontainer/devcontainer.json") as handle:
        devcontainer = json.load(handle)
    with open(".devcontainer/Dockerfile") as handle:
        dockerfile = handle.read()

    assert "Rust" not in devcontainer["name"]
    assert "rust-tools" not in devcontainer["postCreateCommand"]
    assert "rust-lang.rust-analyzer" not in devcontainer["customizations"]["vscode"]["extensions"]
    assert "rustup" not in dockerfile.lower()
    assert "cargo" not in dockerfile.lower()
