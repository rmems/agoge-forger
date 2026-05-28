"""
Notes on GGUF Export for Agoge Forger

Automatic GGUF conversion is intentionally NOT included for custom or hybrid architectures 
(like GKA-HQwen3, Nemotron, or SSM variants). 

Why?
Llama.cpp requires explicit C++ implementations for every new architecture's forward pass.
If the architecture is unknown to llama.cpp, conversion scripts (like convert-hf-to-gguf.py) 
will fail or produce unrunnable binaries.

Workflow for GGUF:
1. Merge adapter via `agoge merge-adapter`
2. Check if llama.cpp supports the architecture.
3. If yes, clone llama.cpp and run their conversion script manually.
"""
