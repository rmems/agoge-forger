# Limen-Neural Library Integrations

This document outlines planned optional integrations with the Limen-Neural ecosystem. These are not hard dependencies in the first release to maintain a minimal, stable base.

## 1. Limen-Neural/engram-parser
**Purpose:** Advanced GGUF metadata inspection and expert tensor extraction after export.
**Integration Plan:** Optional Rust crate dependency in `agoge-gguf` or Python bindings via PyO3.

## 2. Limen-Neural/corpus-ipc
**Purpose:** High-performance cross-process communication (Rust/Python/Julia) for future hybrid training telemetry and distributed orchestration.
**Integration Plan:** Will wrap `agoge_forger/logging.py` telemetry for streaming to external monitors.

## 3. Limen-Neural/axon-encoder
**Purpose:** Converting numeric/sensor/time-series data into spike-like training features.
**Integration Plan:** Dataset preprocessing step within `agoge_forger/datasets.py` when handling temporal modalities.

## 4. Limen-Neural/SpikenautDistill.jl
**Purpose:** Future Spiking Neural Network (SNN) distillation or hybrid SNN+LLM training.
**Integration Plan:** Hooking via JAX/Julia interop or exported ONNX/GGUF models.

## 5. Limen-Neural/metabolic-ledger
**Purpose:** Possible JSONL decision-log dataset source for Reinforcement Learning or DPO.
**Integration Plan:** Direct parsing in `datasets/samples` pipelines.
