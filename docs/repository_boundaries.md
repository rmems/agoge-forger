# Repository boundaries

Agoge Forger owns a single Python/PyTorch post-training path. It may consume
versioned packages, manifests, datasets, or benchmark results from sibling
repositories, but it does not duplicate their implementation trees.

| Responsibility | Source of truth |
| --- | --- |
| Synthetic generation, curation, and public-dataset admission | [`rmems/synthetic-factory`](https://github.com/rmems/synthetic-factory) |
| Real engineering trajectories and derived SFT/DPO datasets | [`rmems/operation-prometheus`](https://github.com/rmems/operation-prometheus) |
| Training, evaluation, checkpoints, manifests, and model releases | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| Custom CUDA kernels and GPU-performance investigations | [`rmems/blackwell-kernel-lab`](https://github.com/rmems/blackwell-kernel-lab) |
| Terraform, cloud jobs, costs, and provider runbooks | [`rmems/Dioscuri-Cloud`](https://github.com/rmems/Dioscuri-Cloud) |

## Integration rules

- Dataset inputs must identify an immutable upstream revision and pass Agoge's
  local schema validation before training.
- Kernel research crosses the boundary only through a versioned package or a
  recorded benchmark result. Agoge does not carry first-party CUDA sources.
- Cloud launchers consume Agoge's documented CLI, container, configuration, and
  artifact contracts. Provider authentication and Terraform remain in
  Dioscuri-Cloud.
- Experiment and artifact contracts are Python-owned here. A second runtime must
  not be introduced as an alternative training or evaluation implementation.
- A boundary change requires an issue that identifies the new source of truth,
  migration steps, and compatibility impact.

## GitHub repository metadata

The GitHub settings should use this exact description:

> Python/PyTorch post-training platform for reproducible SFT, evaluation,
> checkpoints, experiment manifests, and Hugging Face model releases.

Recommended topics, in the intended display order:

1. `pytorch`
2. `llm`
3. `post-training`
4. `supervised-fine-tuning`
5. `peft`
6. `trl`
7. `qlora`
8. `evaluation`
9. `hugging-face`
10. `reproducible-research`

Repository settings are changed through GitHub administration, not by this
document. Keep the description, topics, README, package metadata, and active
issue scopes aligned with this boundary contract.
