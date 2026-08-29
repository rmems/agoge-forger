# Public SWE/DevOps dataset candidate manifest

**Scout:** Public Dataset Scout  
**Date:** 2026-08-21 (America/Chicago)  
**Method:** HF Dataset Viewer + official cards/READMEs/LICENSE files. **No corpus downloads.**  
**Rank:** training signal × legal clarity × realism × complementarity to Operation Prometheus + the four-family factory.

## Locked constraints (do not relitigate)

- Public sets join **only at compose**, never factory `outputs/raw/`. Never rehost raw on Hub.
- **SWE-bench / Verified / HumanEval: out of train.**
- **Eval board:** Multi-SWE-bench Rust, 10 repos / 239 instances. Do not filter the board to clean train.
- **Train denylist:** those 10 + Prometheus sources (`corinth-canal`, `grok-ozempic`, `myelin-accelerator`, `Limen-Neural/axon-encoder`) + first-party freeze (`Theseus-Quarry`, `worktrees-hives`, `xai-dissect`) + GPL / no-SPDX locals.
- Multi-SWE Rust 10: `BurntSushi/ripgrep`, `clap-rs/clap`, `nushell/nushell`, `rayon-rs/rayon`, `serde-rs/serde`, `sharkdp/bat`, `sharkdp/fd`, `tokio-rs/bytes`, `tokio-rs/tokio`, `tokio-rs/tracing`.
- Already accepted compose: **R2E-Gym-V1** (Apache-2.0), **CommitPackFT** (MIT, spice only after denylist).
- Prometheus **32 unique PRs** are **neither-until-split**. **REAL_ONLY is empty.**
- Hidden CoT / `thought` / `internal_reasoning` is never training material.
- Never recommend a set solely because it is large.

## Ranked compose candidates (train-side)

Scores 1–5. Product is the rank key. Assume denylist + license filter at compose.

| Rank | Dataset | n (verified) | License | Task | Code | PR/issue | REAL_ONLY | Eval | Signal | Legal | Real | Comp | Product | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | `R2E-Gym/R2E-Gym-V1` | 8,101 | Apache-2.0 | SWE-Gen commit→env | patch + docker | commit, no issue URL | 2 | 2 | 4 | 5 | 4 | 3 | 240 | **Accepted.** Python gym. No Rust. Prefer `R2E-Gym-Subset` (4,578) if SWE-bench-repo leak matters. |
| A0 | `bigcode/commitpackft` | 702,062 (rust 2,996) | MIT + per-sample | commit + 1 file | snippet pair | commit only | 1 | 0 | 2 | 4 | 2 | 2 | 32 | **Accepted spice.** Drop AGPL/LGPL/`unknown`. Rust slice is **not** REAL_ONLY. Known ∩ eval: clap, serde, fd, tokio = 173/239. |
| 1 | `nebius/SWE-rebench` | 27,878 | CC-BY-4.0 + `license_name` | issue→patch + docker | patch + test_patch | instance_id + issue text | 4 | 4 | 5 | 4 | 5 | 4 | 400 | **Best Python REAL_ONLY unlock.** Filter Apache/MIT/BSD. Check ∩ SWE-bench 12 before compose. Does **not** unlock Rust. |
| 2 | `SWE-Gym/SWE-Gym` | 2,438 (11 Python repos) | HF MIT / GitHub Apache-2.0 | issue→patch + env | patch + test_patch | yes | 4 | 2 | 4 | 4 | 5 | 4 | 320 | **Clean small Python gym.** Complements R2E. Raw 64,689 has no env. Trajectories carry model ToS. |
| 3 | `internlm/SWE-Fixer-Train-110K` | 115,406 | MIT (no per-repo field) | issue→patch, no env | patch + test_patch | yes | 3 | 0 | 3 | 3 | 4 | 3 | 108 | Volume Python compose. **Eval sibling is SWE-bench leak — do not train that.** Denylist unknown overlap. |
| 4 | `JetBrains-Research/lca-bug-localization` | 14,958 | Apache-2.0 + `repo_license` | issue→diff localize | diff | issue/PR URLs | 3 | 3 | 3 | 4 | 3 | 3 | 108 | Java/Kotlin/Python localization, not executable SWE. Filter `repo_license`. |
| 5 | `JetBrains-Research/lca-ci-builds-repair` | 68 (default) | source claimed MIT/Apache/BSD; compilation SPDX **UNKNOWN** | GHA fail→pass | workflow + logs + diff | commit URL | 4 | 4 | 3 | 3 | 5 | 5 | 225 | **Best real CI seed.** Tiny. Verify each repo LICENSE. Train **or** eval, not both. |
| 6 | `PrimeIntellect/Multi-SWE-RL-Verified` | 2,232 (rust n **UNKNOWN**) | CC0 + ByteDance IP + per-repo | issue→patch, 7 langs | fix_patch + test_patch | yes | 3 | 0 | 4 | 2 | 5 | 3 | 120 | Same *family* as the eval board. Train only after `lang=rust` minus the 10 repos + permissive per-repo. Official `ByteDance-Seed/Multi-SWE-RL` has 0 parquet rows. |
| 7 | `user2f86/rustbench` + `raw_dataset` | 500 / 1,040 | **UNKNOWN** (no YAML, GitHub LICENSE 404) | Rust issue→PR→patch + F2P | patch + test_patch | yes | 4 *if* license | 3 *if* disjoint | 5 | 1 | 5 | 5 | 125 | **Only public Rust issue→patch set.** Blocked until SPDX. Raw row0 is `clap-rs/clap` — denylist the 10. Survivors would be the first REAL_ONLY Rust arm. |
| 8 | `Dorothydu/SWE-Dev` | 14k + 500 (Viewer empty) | Apache-2.0 | feature-dev, not repair | GT ~190 LOC (paper) | PRD, not always issues | 2 | 3 | 3 | 4 | 4 | 3 | 144 | Complementary task family. Keep test 500 out of train. |
| 9 | `SWE-bench/SWE-smith` | 59,136 | MIT | **synthetic** bugs on real repos | synthetic patch + docker | no real issues | 0 | 0 | 2 | 5 | 2 | 2 | 40 | Synth spice only. Trajectories are Claude ToS. Not REAL_ONLY. Complements factory, does not unlock it. |
| 10 | `Helmcode/stack-v3-devops` permissive slice | helm 65k / tf 780k / df 4.6M / gha 3.4M | ODC-By 1.0 + original; `license_types` **lies** | IaC/Docker/GHA units | full multi-file units | repo+commit only | 1 unfiltered / 4 if enriched | 1 | 4 | 1 | 4 | 5 | 80 | **Do not ingest unfiltered.** Only large real IaC dump. Needs external GitHub LICENSE on `repo_path`. Not recommended for size. |
| 11 | `ynotbhatc/rego_policy_libraries` | 505 (README) | Apache-2.0 (README; verify LICENSE) | OPA/Rego policies | yes | no | 3 | 2 | 3 | 4 | 3 | 4 | 144 | Rare real Rego. Complements factory synthetic policies. |
| 12 | `PagerDuty/incident-response-docs` | docs tree | Apache-2.0 | IR process text | no | no | 2 | 1 | 2 | 5 | 2 | 3 | 60 | Only license-clear real IR text. No incident traces exist publicly. |
| 13 | `trace-commons/agent-traces` | 30 | CC-BY-4.0 compilation; per-trace audit | donated agent sessions | traces | n/a | 2 | 1 | 2 | 3 | 4 | 3 | 72 | n=30. One Grafana/Loki. Audit each trace. |
| 14 | `nebius/SWE-bench-extra` | 6,376 | CC-BY-4.0 + per-instance `license` | issue→patch | patch + test_patch | yes | 3 | 1 | 3 | 4 | 5 | 3 | 180 | Older/smaller than rebench. Prefer rebench unless unique repos. |
| 15 | Trajectories (SWE-Gym OpenHands 6,055; nebius SWE-agent 80,036; R2E SFT 3,231; SWE-smith 76k) | see n | mixed / often YAML-missing | agent traces | patches + messages | via parent | 1–2 | 0 | 3 | 2 | 3 | 2 | 36 | Spice / SFT only. Model ToS (GPT-4o, Claude, Kimi). Hidden CoT strip. |

## Eval-only (do not train)

| Dataset | n | License | Why eval, not train | Eval usefulness |
|---|---|---|---|---|
| `ByteDance-Seed/Multi-SWE-bench` Rust | 239 / 10 repos | CC0 + ByteDance IP + per-repo | **This is the v0 board** | 5 |
| SWE-bench / Verified / Lite / Multimodal / Multilingual | 2,519 / 500 / 323 / 580 / 300 | MIT toolkit; several YAML missing | Locked contamination | 5 community / 0 our Rust board |
| `AmazonScience/SWE-PolyBench` | 2,110 (JS/TS/Py/Java; **no Rust**) | MIT | Public multilingual board | 4 (non-Rust complement) |
| `autoiac-project/iac-eval` | 458 | CC-BY-4.0 (code MIT) | Only clean Terraform+Rego bench | 5 ops |
| `microsoft/AIOpsLab` | interactive (grows) | MIT | K8s fault-injection harness, not a dump | 4 ops |
| `phamquiluan/RCAEval` | 735 cases | MIT | Lab-injected RCA; ~3.4 GB telemetry — do not download wholesale | 4 RCA |
| Terminal-Bench (git, no official HF) | ~89 TB 2.0 (exact N UNKNOWN) | Apache-2.0 | Long-horizon terminal | 4 |
| TheAgentCompany (git, no official HF) | 175 | MIT | Workplace sim | 4 |
| LiveCodeBench | 121+ live | YAML `cc` (SPDX vague) | Contest codegen, not SWE | 5 codegen / 0 SWE |
| `SWE-bench/SWE-bench_Multilingual` | 300 | MIT | SWE-bench family; rust presence UNKNOWN | 3 |
| SWE-Lancer | 1,488 (Diamond 502 public) | **UNKNOWN** (LICENSE 404) | Freelance eval; agents can overwrite tests | 3 |
| Defects4J | 854 Java | MIT | Saturated APR | 4 Java / 1 train |
| `ScaleAI/SWE-bench_Pro` | 731 public | **GPL-by-design** | Contamination-resistant **eval**; train = poison | 4 eval / 0 train |

Pre-registered idea (Eval Architect Q2, not pinned): CPT-disjoint 66-instance Multi-SWE slice if CommitPackFT spice is used.

## Rejects (do not compose)

| Candidate | Reason |
|---|---|
| SWE-bench family as train, `internlm/SWE-Fixer-Eval`, `R2E-Gym/SWE-Bench-Verified` | Eval contamination |
| `bigcode/humanevalpack` | HumanEval family |
| `bigcode/the-stack-v2` | Gated BigCode TOS; size-only |
| `Helmcode/stack-v3-devops` **unfiltered** | Header `license_types` is not a grant; 98.2% repos unlabeled |
| `ScaleAI/SWE-bench_Pro` as train | Copyleft selected on purpose |
| SWE-Factory **code** | AGPL-3.0 or paid commercial |
| `facebookresearch/SWE-RL` data | Unreleased + CC-BY-NC code |
| DebugBench | LeetCode + GPT-4 implanted bugs |
| QuixBugs / DeepFix | Toy, saturated |
| Stack / CommitPack full 4 TB | Size-only |
| Review-only CodeReviewer as REAL_ONLY | Diff+comment, not executable repair |
| Guessed IDs that 404 | CI-bench, BuildHopper, CommentBench, CodeReviewBench, RepoDebug HF, official OpenHands/SWE-agent traj HF |
| AutoCodeRover / Agentless / Moatless train dumps | Not found as standalone public train sets |
| Mozilla/Chromium bug dumps | Not licensed issue→patch SWE |
| Google SRE book | Copyright; do not scrape |
| ToolBench / API-Bank / τ-bench | Not coding-agent SWE/DevOps |
| Factory `rmems/*` Grok slugs | Already ours; synthetic; not public-real |
| Any GPL/AGPL/LGPL/`unknown` slice | Copyleft poison |
| Hidden CoT in any trajectory | Never trains |

## Required fields (high-signal rows)

### `nebius/SWE-rebench` — rank 1 Python REAL_ONLY
- **source:** Badertdinov et al. 2025, arXiv 2505.20411
- **schema:** instance_id, base_commit, patch, test_patch, problem_statement, repo, FAIL_TO_PASS / PASS_TO_PASS / …, license_name, docker_image
- **redistribution:** CC-BY-4.0 packaging; honor `license_name`
- **training-use:** commercial OK with attribution after license filter
- **contamination:** high vs GitHub-2023 pretrain; designed for live decontaminated eval
- **Prometheus:** same task family; instance overlap UNKNOWN
- **overlap benches:** must check `repo` vs SWE-bench 12 + Multi-SWE 10 (Python, so Rust 10 likely empty)

### `SWE-Gym/SWE-Gym` — rank 2
- **source:** Pan et al. 2024, arXiv 2412.21139
- **schema:** instance_id, patch, test_patch, problem_statement, repo, base_commit, PASS_TO_PASS, FAIL_TO_PASS
- **per-repo license:** not in schema (UNKNOWN)
- **Prometheus:** same family; instance UNKNOWN

### `user2f86/rustbench` — blocked REAL_ONLY Rust
- **source:** arXiv 2602.22764 (2026); 34 repos; harness `GhabiX/Rust-SWE-Bench`
- **schema:** SWE-bench protocol (instance_id, repo, pull_number, issue_numbers, patch, test_patch, FAIL_TO_PASS, …)
- **redistribution / training-use:** UNKNOWN until SPDX
- **contamination:** confirmed `clap-rs/clap` in raw; likely more of the 10
- **Prometheus:** same family; instance UNKNOWN

### `lca-ci-builds-repair` — best real DevOps
- **source:** Bogomolov et al., Long Code Arena, arXiv 2406.11612
- **schema:** language, repo_owner, repo_name, workflow, logs, diff, sha_fail, sha_success, commit_link
- **redistribution:** source licenses claimed permissive; compilation SPDX UNKNOWN
- **Prometheus:** CI task family; instance UNKNOWN

## Next compose actions (no huge downloads)

1. **License-unblock rustbench.** Ask authors / HF discussion for SPDX. Then Viewer-enumerate `repo`, drop the 10 + GPL. Survivors = first REAL_ONLY Rust arm (EXP-016).
2. **When Viewer filter works:** `Multi-SWE-RL-Verified` `lang=rust` minus the 10, permissive per-repo only. Train-only, never score.
3. **Python REAL_ONLY:** compose `SWE-Gym` and/or license-filtered `SWE-rebench`. Do not call that the Rust unlock.
4. **CI seed:** LCA-CI 68 after live LICENSE check. Pick train *or* held-out.
5. **Do not** ingest SWE-bench*, HumanEval*, Multi-SWE-bench, SWE-bench Pro, Stack v2, unfiltered stack-v3-devops, SWE-Factory code, facebook SWE-RL.
6. Prometheus 32 stay neither-until-split.

Cite paths: `/workspace/swe-candidates.md`, `/workspace/devops-candidates.md`, `/workspace/prometheus-coverage.md`, `/workspace/gap-map.md`.
