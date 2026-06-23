def test_readme_distinguishes_checkpoint_adapter_and_merged_artifact():
    with open("README.md") as handle:
        readme = handle.read()

    assert "`checkpoint-*` directories are trainer recovery snapshots" in readme
    assert "The adapter saved at `adapters/<run_name>` is the LoRA output" in readme
    assert "The merged model under `merged/<run_name>` is the single final artifact" in readme
