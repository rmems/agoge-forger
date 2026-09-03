# Public Dataset Scout research log

Ranked list is ready. REAL\_ONLY is still empty.

Python can unlock without Rust: license-filtered [SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) (27,878, CC-BY-4.0 with a per-row license field) and [SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (2,438, MIT/Apache). R2E-Gym-V1 and CommitPackFT stay as already accepted compose, spice only on the rust slice.

The Rust unlock is [rustbench](https://huggingface.co/datasets/user2f86/rustbench) (500 issue→PR→patch + F2P). It is the only public set of that shape. License is unknown, and the raw sibling already contains `clap-rs/clap`, which is on the Multi-SWE denylist. Until SPDX lands and those 10 repos are dropped, it does not enter compose.

9:13 PM📚 Public Dataset Scout

DevOps is thinner than the topic list. Factory already covers K8s/Terraform/Docker/incident synthetically. Real public CI is [LCA CI-repair](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair) (68 rows). There is no public incident-trace dump. The huge Stack-v3 DevOps slice is legally dirty unless someone does an external GitHub LICENSE pass. IaC-Eval, AIOpsLab, Terminal-Bench, SWE-PolyBench stay eval-only.

Out of train, still: SWE-bench family, HumanEval, Multi-SWE-bench itself, SWE-bench Pro (GPL on purpose), Stack v2, SWE-Factory code (AGPL).

9:13 PM📚 Public Dataset Scoutgap-map.mdCouldn't read file9:13 PM📚 Public Dataset Scoutcandidate-manifest.mdCouldn't read file9:13 PM📚 Public Dataset Scout

Full cards are in those two files. No corpora downloaded. Next unlock is rustbench SPDX, then a repo denylist count. I can do that from Viewer + authors without pulling the dump.

9:13 PM📚 Public Dataset Scout

Full DevOps cards are in. Ranking does not change. One extra Rust note I am checking: Multi-SWE-RL lists repos outside the 239 board (alacritty, helix, fish-shell). That only matters if they survive a license + denylist pass.

9:15 PM📚 Public Dataset Scout

Files updated. Two legal dead-ends are now explicit: OpenRCA (real telemetry, CC BY-NC) and LogHub (academic only). That is why the incident lane stays empty.

I am not treating SWE-Gym as a SWE-bench leak. It is the same protocol on 11 other Python repos, same as the already-accepted R2E-Gym-V1. The locked reject is SWE-bench instances, not every gym that looks like it.

9:16 PMFri, Aug 21 9:37 PMCreated routine9:37 PM📚 Public Dataset Scout

Got the standing hunt. I’ll run it against the baseline we already have, one unresolved candidate per pass, not a fresh full scout every time.

First up when it fires: rustbench SPDX, then the extra Multi-SWE-RL Rust repos. No big downloads.

9:37 PM📚 Public Dataset Scout

Checked rustbench. License is still a blank: no SPDX on the Hub cards, GitHub `license` is null, LICENSE file 404. The ICSE paper is CC-BY for the PDF, not for the dataset.

What did change: overlap with the Multi-SWE Rust board is no longer a guess. Paper Table 1 has all 10 of those repos. **145 of 500** instances sit on the denylist (clap alone is 56). The other **355 from 24 repos** could be a complementary Rust eval *if* SPDX appears. Not compose, and not a replacement for the 239 board.

9:44 PM📚 Public Dataset Scout

**Dataset:** `user2f86/rustbench` (500) / `raw_dataset` (1,040)
**License:** UNKNOWN
**Training value:** shape is right, usable value is 0 until SPDX
**Contamination:** 145/500 on our eval board
**Prometheus overlap:** task family only
**Decision:** do not compose
**Gap filled:** Rust-board overlap counted
**Next:** `nebius/SWE-rebench-V2`

9:44 PM📚 Public Dataset Scout

Looked at `nebius/SWE-rebench-V2`. This is the one that actually moves REAL\_ONLY.

32,079 issue→patch rows, docker images, F2P tests, CC-BY-4.0 on the pack plus a per-row SPDX `license` field. MIT 13,117 / Apache-2.0 10,407 / BSD-family \~2.5k. Drop AGPL (54) and GPL (5). Hold `custom-check-github` (5,038) until a live LICENSE check.

Languages include **rust 3,123**, python 7,243, go 6,144, **julia 793**, plus C/C++. First row is a real TS MIT issue (`elastic/synthetics`) with patch + test\_patch + image.

Did not download the parquet (\~409 MB). Viewer search for the Multi-SWE 10 500’d, so denylist overlap is still unknown. Compose only after that filter. This beats V1 for planning because V1 was Python-only.

9:53 PM📚 Public Dataset Scout

**Dataset:** `nebius/SWE-rebench-V2`
**License:** CC-BY-4.0 + per-instance SPDX
**Training value:** 4/5 after license+denylist (rust arm pending overlap count)
**Contamination:** SWE-bench-family construction; board overlap unknown this pass
**Prometheus overlap:** task family only
**Decision:** compose candidate, no download yet
**Gap filled:** licensed public Rust issue→patch with SPDX (n=3123 before denylist)
**Next:** denylist overlap on those 3,123, or Multi-SWE-RL off-board Rust

9:53 PM📚 Public Dataset Scout

Multi-SWE-RL rust is counted now, and it is not the unlock.

Verified has **0 rust**. Reupload has **215 rust**, of which **173 sit on the eval-board repos** (clap 62, nushell 36, bat 35, plus tokio/fd/ripgrep/tracing/bytes/serde). Rayon is the only one of the 10 that is missing.

The leftover 42 are alacritty 23 (Apache-2.0), mdBook 11 (MPL), fish-shell 4 (no SPDX), helix 3 (MPL), rusqlite 1 (MIT). Apache/MIT-only that is **24 rows**. Same pipeline as the board, plus the ByteDance IP clause. Spice at most, not a REAL\_ONLY arm.

V2’s 3,123 rust rows are still the actual unlock. Viewer search for that denylist 500’d again.

10:05 PM📚 Public Dataset Scout

**Dataset:** `PrimeIntellect/Multi-SWE-RL-Reupload` rust slice (Verified has none)
**License:** CC0 + ByteDance IP + per-repo
**Training value:** 1/5 (24 permissive off-board)
**Contamination:** 173/215 are the eval board
**Prometheus overlap:** task family only
**Decision:** do not compose the 173; 24 is optional spice
**Gap filled:** MSRL rust n is known
**Next:** V2 rust ∩ the 10, or V2 Julia 793

10:05 PM📚 Public Dataset Scout

V2 rust vs the eval board is counted now, no parquet download.

**Dataset:** `nebius/SWE-rebench-V2` rust slice
**License:** CC-BY-4.0 packaging + per-row SPDX. Off-board keep: Apache-2.0 1,905 / MIT 659 / CC0 55 / BSD-3 49 / ISC 2
**Training value:** 2,670 issue→patch+test+docker rows, 259 repos. This is the Rust REAL\_ONLY compose recipe.
**Contamination:** 112 hits on the 10 (clap 82, fd 18, ripgrep 5, bat 4, bytes 2, rayon 1). Dropped. Train only, never score.
**Prometheus overlap:** none (a `myelin-ai/mockiato` false positive, 2 rows, kept)
**Decision:** compose-ready, still empty on disk. Do not fetch the \~429 MB file unless you say so.
**Gap filled:** G1, licensed Rust disjoint from Multi-SWE 10
**Next:** V2 python 7,243 vs SWE-bench 12

rustbench is no longer the blocker. I folded this into the candidate manifest.

10:20 PM📚 Public Dataset Scout

V2 python vs SWE-bench 12 is counted. Overlap is zero.

**Dataset:** `nebius/SWE-rebench-V2` python slice
**License:** CC-BY-4.0 + per-row SPDX. Keep 5,495 (MIT 1,940 / Apache 1,854 / BSD-3 1,366 / MIT-0 166 / BSD-2 122 / ISC 25 / Unlicense 22). HOLD custom-check-github 1,706.
**Training value:** 5,495 issue→patch+test+docker rows, 536 repos. Python REAL\_ONLY compose recipe.
**Contamination:** classic SWE-bench 12 = 0, including name variants. Train only. If SWE-Gym is also in the mix, dedup `getmoto/moto` (249).
**Prometheus overlap:** none
**Decision:** compose-ready, still empty on disk. Same 429 MB parquet gate as the rust slice. Do not fetch unless you say so.
**Gap filled:** G3 planning unlock. One approval now unlocks both language arms (rust 2,670 + python 5,495).
**Next:** V2 ∩ R2E-Gym repos, or the 311 rust `custom-check-github` LICENSE sample.

10:25 PM📚 Public Dataset Scout

Counted the Python gym overlap so we don’t double-train the same repos.

**Dataset:** V2 python ∩ SWE-Gym 11 ∩ R2E-Gym names
**License:** unchanged (V2 SPDX filter)
**Training value:** SWE-Gym is 11 repos / 2,438. V2 shares 537 rows on 7 of them (moto, pandas, modin, dask, dvc, conan, hydra). If both are composed, V2 still has **4,958 / 529**. Extra R2E overlap is only 29 perm rows (scrapy/tornado/statsmodels).
**Contamination:** train-train, not eval. R2E-V1 includes sympy (SWE-bench); keep preferring Subset. V2 still has 0 SWE-bench 12.
**Prometheus overlap:** none
**Decision:** keep the 5,495 recipe; drop the 7 SWE-Gym repos at compose if both arms are used. Do not fetch V2 or R2E parquets.
**Gap filled:** G3 dedup numbers
**Next:** the 311 rust `custom-check-github` LICENSE sample, or V2 C/C++. CUDA still missing.

10:34 PMFri, Aug 21 10:49 PM📚 Public Dataset Scout

Sampled the 311 rust rows Nebius left as `custom-check-github` (35 repos, HEAD licenses, no parquet dump).

**Dataset:** V2 rust `custom-check-github`
**License:** DROP 60 that are source-available at HEAD: meilisearch BUSL (32), fuel-vm BUSL (16), sentry-cli FSL-1.1-MIT (12). 240 look OSI (Apache/MIT/BSD) at HEAD. 11 stay HOLD (MPL or no file).
**Training value:** those 240 could become 2,910 with the existing 2,670, but HEAD ≠ `base_commit`, so they stay HOLD.
**Contamination:** none of the 35 are on the Multi-SWE 10
**Prometheus overlap:** none
**Decision:** do not enlarge the 2,670 recipe. Permanently exclude the 60 BUSL/FSL. Side note: V2 C/C++ permissive is only 127 rows / 18 repos (spice). No CUDA label in V2.
**Gap filled:** G1 custom bucket triaged, not unlocked
**Next:** a real CUDA issue→patch set, or V2 Go / Julia spice

10:49 PM📚 Public Dataset Scout

Looked for a real CUDA issue→patch set. There isn't one.

**Dataset:** public CUDA/kernel dumps (KernelBench, Nemotron-SFT-CUDA, CUDA-Agent-Ops, Sakana, GPURepair, V2)
**License:** KernelBench is MIT on GitHub (HF card unlabeled). Nemotron and CUDA-Agent-Ops are CC-BY-4.0 but **synthetic**. CUDA-L1 is GPL. V2 has two `NVIDIA/cccl` rows, still `custom-check-github`.
**Training value:** 0 for REAL\_ONLY. Closest spice is Nemotron-SFT-CUDA-v1 (2,276 GLM-4.7 traces), which is not GitHub issues.
**Contamination:** KernelBench is the CUDA eval board; Sakana/Agent-Ops/traces sit in that orbit. Do not train it.
**Prometheus overlap:** none
**Decision:** hard hole. Optional later eval board: KernelBench 270. Do not fetch the 65–83 MB synth archives.
**Gap filled:** T12 CUDA documented, not closed
**Next:** V2 Go 6,144, or Julia 293

10:55 PM📚 Public Dataset Scout

V2 Go is counted. It’s a real third language arm, plus some actual Helm/k8s issues.

**Dataset:** `nebius/SWE-rebench-V2` go slice
**License:** CC-BY-4.0 + SPDX. Keep Apache 3,151 / MIT 2,193 / BSD-3 234 / BSD-2 60 / ISC 13. HOLD custom 485.
**Training value:** 5,228 rows / 887 repos after dropping Multi-SWE Go (68) and `vaskoz/dailycodingproblem-go` (355 interview problems). helm 139, argo 89, kops 62, istio 33 are in the keep set.
**Contamination:** v0 board is still Rust-only; dropped the 3 Multi-SWE Go repos anyway. Train only.
**Prometheus overlap:** none
**Decision:** compose-ready behind the same 429 MB parquet gate. Does not replace rust 2,670 or python 5,495.
**Gap filled:** T6 partial (Helm/k8s issue→patch, not Terraform)
**Next:** V2 Julia 293, or TypeScript 4,204

11:04 PM📚 Public Dataset Scout

Julia’s counted. It’s spice, not an arm.

**Dataset:** `nebius/SWE-rebench-V2` julia slice
**License:** CC-BY-4.0 + SPDX. Keep MIT 289 + Unlicense 4 (`MPI.jl`) = **293 / 39 repos**. HOLD custom-check-github 500 (that’s where JuMP/Nemo/Symbolics sit).
**Training value:** real scientific/ML packages (Makie, MLJ, DataStructures, JuliaFormatter). No interview-dump equivalent. Too small to be a language arm.
**Contamination:** Multi-SWE has no Julia. Train only.
**Prometheus overlap:** none
**Decision:** optional spice behind the same parquet gate. Do not promote the 500 custom rows.
**Gap filled:** Julia slot in the brief, as spice
**Next:** V2 TypeScript 4,204 vs the Multi-SWE JS/TS repos

11:16 PM📚 Public Dataset Scout

V2 TypeScript and JavaScript are counted.

**Dataset:** `nebius/SWE-rebench-V2` ts+js
**License:** CC-BY-4.0 + SPDX. ts perm 3,931; js perm 3,317 (js AGPL 53 already out). HOLD custom ts 168 / js 699.
**Training value:** **7,214** rows after dropping 28 Multi-SWE hits and `vuejs/vue-next` (old `vuejs/core` name). ts 3,923 / 412 + js 3,297 / 370. `sveltejs/kit` 117 kept; it is not `sveltejs/svelte`.
**Contamination:** Multi-SWE JS/TS mostly absent (express/insomnia/svelte/MUI/vue core = 0). SWE-PolyBench overlap still unknown. Train only.
**Prometheus overlap:** none
**Decision:** web arm, same parquet gate. Does not replace rust/python/go.
**Gap filled:** TypeScript/JS issue→patch at arm scale
**Next:** V2 Java vs Multi-SWE Java, or clean the C/C++ 127 spice

11:26 PM📚 Public Dataset Scout

Java and C/C++ vs Multi-SWE are counted.

**Dataset:** V2 java + c + cpp
**License:** CC-BY-4.0 + SPDX. Copyleft 0. Java perm 1,437; C 88; C++ 39. HOLD custom java 255 / c 142 / cpp 136.
**Training value:** Java **1,435 / 145** after dropping fastjson2 (2). Modest arm (kestra 132 is workflow orchestration). C stays 88; C++ **37** after simdjson. C/C++ is spice, and several `c` labels are really Go/Ruby bindings.
**Contamination:** Multi-SWE barely shows (4 perm rows). logstash/mockito/jq/Catch2/fmt = 0. PolyBench∩java still unknown.
**Prometheus overlap:** none
**Decision:** Java compose-ready; do not call C/C++ an arm. Same parquet gate.
**Gap filled:** Java modest arm. Brief C/C++ still spice-only.
**Next:** PolyBench repo overlap, or a one-page V2 rollup of every language recipe

11:36 PM📚 Public Dataset Scout

PolyBench vs V2 is counted, and the one-parquet recipe is now a single card.

**Dataset:** `AmazonScience/SWE-PolyBench` (2,110 / 21 repos, MIT, eval-only) ∩ V2
**License:** PolyBench MIT toolkit; do not train it
**Training value:** none as train. V2 perm overlap is only **93** rows (keras 46, tailwind 24, yt-dlp 18, transformers 5). `serverless` 272 is already HOLD. svelte and MUI, the big PolyBench slices, are 0 in V2.
**Contamination:** drop those 4 repos if we might score PolyBench later
**Prometheus overlap:** none
**Decision:** PolyBench stays eval. Optional −93. Rollup behind the same 429 MB file: rust 2,670 + python 5,495 + go 5,228 + ts/js 7,214 + java 1,435 = **22,042** core, or **22,460** with julia/C spice.
**Gap filled:** V2 language map is done enough to stop slicing it
**Next:** holes V2 cannot fill (Terraform issue→patch, review→repair, incidents, CUDA)

Card is `v2-compose-recipe.md`.

11:44 PM📚 Public Dataset Scout

Terraform-core is a bust. The only licensed issue→patch spice is already inside V2.

**Dataset:** dedicated TF dumps + V2 terraform/pulumi/helm scan
**License:** `hashicorp/terraform` is **BUSL 1.1** (V2 n=3, drop). `actaylor/github-terraform-provider-issues` has **no SPDX and no patches** (1,224 issue JSON).
**Training value:** \~51 TF-provider rows and \~26 Pulumi rows already in the V2 Go/TS recipes, plus helm 139 / cfn-lint 135. Not a Terraform arm.
**Contamination:** none vs the rust board. Keep IaC-Eval as eval.
**Prometheus overlap:** none
**Decision:** no new compose dataset. Do not fetch actaylor.
**Gap filled:** T6 sharpened — Helm/k8s/cfn/providers yes, Terraform-core no
**Next:** review→repair (T2), still outside V2

11:53 PMSat, Aug 22 12:10 AM📚 Public Dataset Scout

T2 is still open for REAL\_ONLY. CodeReviewer on Zenodo is CC-BY-4.0 (snippet spice, no tests). The new dump is Lego-X/SWE-Review-Traj: 8,914 Apache-2.0 traces, executable via SWE-rebench images, but they are GLM-5 teacher reviews of model patches, not human comments, and every row ships a think tool.

12:10 AM📚 Public Dataset Scout

Overlap: Prometheus 0, Multi-SWE Rust 10 = 0, SWE-bench 12 repos = 36 rows, SWE-Gym repos = 299. Bench (1,384) is SWE-bench Verified, so eval-only. Traj stays HOLD spice pending CoT strip and a V2 instance dedup. Did not pull the 595 MB parquet or the 1–3 GB Zenodo zips. Next up is the 68-row LCA-CI SPDX pass.

12:10 AM📚 Public Dataset Scout

LCA-CI is not the clean permissive seed the card claims. Of the 68 rows, 10 are GPL or Elastic-2.0 (cloud-init, errbot, mindsdb, pylint), cowrie has no LICENSE file, and 57 are actually OSI/Unlicense.

12:13 AM📚 Public Dataset Scout

It is also the public Long Code Arena test split, so I am treating the 57 as a held-out CI eval seed, not REAL\_ONLY train. Prometheus overlap is 0. Next I will look for a CI-repair train dump that is not an LCA test set.

12:13 AM📚 Public Dataset Scout

The larger CI dump (`ci-repair-bench`, 567 rows) is a no. No license on HF or GitHub, almost half the rows are the author's own forks, and first-rows still contain the same LCA pairs plus GPL repos. Did not pull the 340 MB parquet. T3 still has no train set; LCA 57 stays eval-only.

12:22 AM📚 Public Dataset Scout

RepoDebug finally has a public home: GitHub `BUAA-IRIP-LLM/RepoDebug`, MIT. Still no HF dump. The 116 MB JSON is implanted bugs (22 tree-sitter error types on real PR files), not natural debugging, so it is eval-only. Listed repos miss Multi-SWE 10, SWE-bench 12, and Prometheus. Did not pull the 85 MB Java file. T4 still has no real debug train set.

12:36 AM📚 Public Dataset Scout

Still no real incident tickets. The new dump is OpenSRE: Apache-2.0, 197 LLM traces over 19 public postmortems with fake k8s tools, Claude/Kimi in the loop, so eval-only. ManySStuBs4J on Zenodo is finally CC-BY-4.0; it is still 2019 Java one-liners, and I did not pull the 510 MB file.
