# Artifacts and Safetensors

Agoge-Forger strictly prefers `safetensors` over `pickle` (e.g., `.bin`, `.pt`) for saving model weights and PEFT adapters. 

## Why Safetensors?
1. **Security**: Safetensors prevents arbitrary code execution vulnerabilities common in pickle-based weights.
2. **Speed**: It allows for zero-copy memory mapping, making model loading significantly faster.
3. **Lazy Loading**: Enables inspecting metadata without loading the entire multi-GB payload into memory.

## Artifact Indexing
Every training and merge run generates an `artifact_index.json`. This file contains the relative paths, sizes in bytes, and SHA256 hashes of all output files. This ensures reproducibility and integrity of exported artifacts.

## Blocking Unsafe Binaries
By default, the framework will raise an error if any `pytorch_model.bin` or `adapter_model.bin` is created. You must explicitly configure `allow_unsafe_serialization: true` in your run config if an older model architecture strictly requires legacy saving formats.
