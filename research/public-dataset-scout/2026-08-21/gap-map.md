# Gap map — public data vs Prometheus + four-family factory

**Date:** 2026-08-21 (America/Chicago)  
**Local inventory:** Prometheus 32 unique PRs (neither-until-split). Grok-4.6 178,461 factory / 139,855 published (not training_ready). Fable-5 189 rows (metadata-only on Hub). GPT-5.6-Sol and Muse Spark 1.2: **no corpora**. No public SWE dump on disk. Eval board = Multi-SWE Rust 239. BASE cannot be assembled today.

## What we already have (do not re-scout as if missing)

| Lane | Local / decided | Public already accepted |
|---|---|---|
| Synthetic agentic SWE/DevOps episodes | Grok 140k episodes (k8s, terraform, docker, incident, RAG, long-horizon, tool-use, …) | — |
| Synthetic prefs | Grok 11.8k (hold until traj-pref contract) | UltraFeedback / HelpSteer2 named by DR, not factory-ingested |
| Synthetic safety / multi-agent | Grok hold | — |
| Real issue→patch (ours) | Prometheus 32, no split | none in train |
| Public Python SWE gym | none on disk | R2E-Gym-Subset |
| Public single-file commits | none on disk | CommitPackFT spice |
| Eval execution board | none implemented in-repo | Multi-SWE Rust 10/239 |
| First-party freeze | Theseus / worktrees-hives / xai-dissect: 0 executable F2P boards | — |

## Gaps that block REAL_ONLY

| ID | Gap | Why it matters | Closest public candidate | Why it does not close the gap yet |
|---|---|---|---|---|
| G1 | **Licensed Rust issue→PR→patch disjoint from Multi-SWE 10** | EXP-016 / REAL_ONLY unlock. Prometheus is 29 Rust / 4 Python and cannot enter until split. CommitPackFT rust is single-file spice. | `user2f86/rustbench` (500) / `raw_dataset` (1,040) | License UNKNOWN. Confirmed clap overlap. Repo list not enumerable this pass (Viewer 429). |
| G2 | **Multi-SWE-RL rust count after denylist** | Only other multilingual issue→patch with Rust | `PrimeIntellect/Multi-SWE-RL-Verified` (2,232) | rust n UNKNOWN (filter 422). ByteDance IP clause. Same family as the board. |
| G3 | **Python REAL_ONLY not on disk** | Five-way ablation needs a REAL arm | SWE-rebench 27,878 / SWE-Gym 2,438 | Not downloaded (correct). Need license + SWE-bench-12 denylist before compose. |
| G4 | **Prometheus 32 have no train/eval split** | One repo one side. 4 source repos. | — | Policy, not a public-data gap. |

## Task-family holes (factory is synthetic-only here)

| ID | Target from the brief | Factory coverage | Public real coverage | Verdict |
|---|---|---|---|---|
| T1 | Code repair / issue→patch | Synth episodes | Python: rebench, SWE-Gym, SWE-Fixer, R2E. Rust: rustbench blocked | Python unlockable. Rust blocked. |
| T2 | PR review / review comments | Synth `code-review-preference` (mixed-kind #30) | CodeReviewer (Zenodo, confirm Apache). Martian review bench is eval. Guessed HF IDs 404 | **No clean executable review→repair train set.** |
| T3 | CI failures / test repair | Synth flaky-test, eval-harness | LCA-CI **68** rows. Guessed CI-bench/BuildHopper 404 | **Real CI is a seed, not an arm.** |
| T4 | Debugging | Synth observability / cascading-error | DebugBench = LeetCode+GPT-4. RepoDebug HF 404 | **No real repo-debug train set found.** |
| T5 | Git histories | — | CommitPackFT spice only | Not SWE. |
| T6 | Docker / K8s / Terraform / IaC | Heavy Grok synth (docker-build-cache, k8s-crashloop, infra-as-code) | stack-v3-devops legally dirty unless external LICENSE lookup. IaC-Eval 458 is **eval-only**. No Pulumi train set. | **Real IaC train is gated on license enrichment, not on finding a dump.** |
| T7 | Incident response | Synth 7.7k oncall trajectories | PagerDuty **docs** Apache-2.0. **No public incident/RCA event dump.** RCAEval/AIOpsLab are lab injections | **Hard hole.** Do not scrape SRE book. |
| T8 | Tool use / agentic coding | Grok tool-use prefs + episodes | Trajectories exist (SWE-Gym, Nebius, SWE-smith) but model-ToS and often SWE-bench-family | Spice only. n=30 donated traces. |
| T9 | RAG debugging | Synth rag-retrieval-debug | No verified public *RAG-debug* SWE set this pass (CodeRepoQA ID 404) | Hole. |
| T10 | Long-horizon SWE | Synth long-horizon + sparse-reward | Terminal-Bench / TheAgentCompany / SWE-Lancer = **eval** | Do not burn as train. |
| T11 | Second generator | Sol + Muse empty. Fable 189 is not a MULTI-GENERATOR arm | Public data cannot create a second *generator* | Factory gap, not scout gap. |

## Contamination / overlap map

```text
OUT OF TRAIN (locked)
  SWE-bench family, HumanEval / HumanEvalPack, LiveCodeBench (as train)

EVAL BOARD — denylist these 10 from every train set
  ripgrep, clap, nushell, rayon, serde, bat, fd, bytes, tokio, tracing

SOFT LEAK if CommitPackFT rust spice is used
  clap, serde, fd, tokio = 173/239 instances
  → do not shrink the board; optionally pin a 66-instance CPT-disjoint slice

SAME FAMILY AS BOARD (train only after denylist, never score)
  Multi-SWE-RL / Verified / Reupload

CONFIRMED rustbench ∩ board
  clap-rs/clap (raw row0). Full overlap UNKNOWN

GPL POISON
  SWE-bench Pro (by design)
  CommitPackFT agpl-3.0 / lgpl-2.1 (93 rust after prior count)
  SWE-Factory code (AGPL)

TOS / NC
  Stack v2, facebook SWE-RL, unfiltered Stack v3 contents
```

## What would change the ranking

1. rustbench SPDX = Apache-2.0 or MIT **and** ≥ ~200 instances after dropping the 10 → becomes rank-0 REAL_ONLY Rust.
2. Multi-SWE-RL rust after denylist is non-empty + permissive per-repo → second Rust arm (still same family as eval).
3. External LICENSE pass on stack-v3-devops `repo_path` → first real IaC arm (still not a volume play).
4. A public incident/RCA set with Apache/MIT and real tickets (not lab inject) → closes T7.
5. A review→repair set with gold patches + tests + permissive license → closes T2.

Until (1) or (2), **Rust REAL_ONLY stays empty.** Python REAL_ONLY can unlock without it. DevOps REAL_ONLY is a 68-row CI seed plus process docs.

## Honesty check

The factory already covers almost every *topic* in the brief synthetically. Public data is scarce where we need it most (licensed Rust SWE, real incidents, real CI, real IaC with a grant). Ranking by size would have put Stack v3 Dockerfiles (4.6M) first. That set is legally the worst of the large options. We did not.
