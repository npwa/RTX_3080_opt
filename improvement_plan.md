# RTX_3080_opt — Project Plan

**Working title:** *From Roofline to Agent: Closing the Loop on llama.cpp Decode Performance, with a
Guard-Railed Agent That Profiles, Hypothesizes, Patches and Validates*
(upgrade of `RTX_3080_opt`, currently "CUDA kernel optimization take-home follow-up")

**Status of this document:** planning only, nothing implemented. Written 2026-10-11 against the repo
state at commit `42f0ae6` (read-only review of `~/work/RTX_3080_opt`; related: `inf_opt`,
`TensorRT_plugin`, `Agentic_platform`).

**Target roles:** JR2024262 (AI Inference Performance), JR2022556 (Agentic AI, HP code-gen),
JR2023179 (Agent Architecture and Evaluation), JR2024664 (High Performance AI, CUDA agents).
Already applied: JR2021914.

---

## 1. Where the project stands today

| Area | Current state |
|---|---|
| Target | `mul_mat_vec_q<12,1,1,0>` (Q4_K decode matvec) in llama.cpp (tag `b10326`), Qwen2.5-7B-Instruct Q4_K_M, RTX 3080 10 GB |
| Baseline | pp512 5229.88 tok/s; tg128 134.05 tok/s; kernel = 26.4 % of decode GPU time; 532.4 GB/s = 72.1 % of 760 GB/s peak; SM 45.7 % of peak; sector efficiency 34.0 % |
| Attempts | (1) int4 broadcast load: no effect; (2) int2 + `__shfl_sync`: correctness passed (`test-backend-ops` 13,054/13,054, fixed-seed byte-identical), sector efficiency 32.87→36.47 %, but DRAM bytes unchanged (93.65 vs 93.69 MB), decode -2.84 %; (3) `GGML_CUDA_GRAPH_OPT=1`: decode -2.98 %, prefill -5.93 % |
| Re-tests under `-np 4` | graph-opt flag: +0.90 % (overlapping); patch: +0.80 % (weak); batched decode shifts to `<12,2>`/`<12,4>` variants (41.9 % of GPU time) |
| Pending | **Core continuous-batching test: PENDING** (the one experiment the author calls the "real lever") |
| Scripts | one-off shell scripts per phase (`run_*.sh`), `concurrent_bench.py` (~80 lines), raw `.ncu-rep`/`.nsys-rep`/CSV in phase folders |

Strengths to preserve: closed-loop causal analysis, honest negative results, correctness gates
before performance claims. Weaknesses for the target roles: (a) no roofline/speed-of-light model
that *predicts* the ceiling, (b) no latency-distribution view (TTFT / ITL / P99) or KV-cache study,
(c) one-off scripts rather than a repeatable harness, (d) no agentic component even though the
author's own write-up says the work was done with Claude Code, (e) the key experiment unfinished.

## 2. Fit to the four roles

| Posting asks for | How this plan answers it |
|---|---|
| JR2024262: "build speed-of-light and roofline models"; "profile with Nsight Systems, Nsight Compute, PyTorch Profiler and custom instrumentation"; "optimize TTFT, ITL, P99, KV-cache capacity"; "tune batching, KV-cache management, quantization, speculative decoding, CUDA Graphs"; **"AI-agent-supported performance workflows that automatically gather and analyze profiles, identify bottlenecks… validate generated changes through reproducible, human-reviewed tests"** | Workstreams A (roofline), B (serving study incl. CUDA graphs, KV type, spec-decoding, batching), C (perf agent with correctness gate and human approval) map one-to-one |
| JR2022556: coding agents + AI evaluation for non-deterministic behavior; CUDA kernels + Nsight Compute/Systems; vLLM/SGLang/TRT-LLM optimization | Workstream C builds the agent and an evaluation of it; workstream D (optional) adds a vLLM/TRT-LLM cross-check |
| JR2023179: agent architecture and evaluation, inference dynamics of agents, coding agents (Codex/Claude Code), open-source showcase | Agent architecture (tools, state, guardrails) + evaluation on tasks with ground truth + public repo |
| JR2024664: multi-agent systems for the CUDA ecosystem; reproducible evaluation frameworks (performance, quality, productivity); resource-constrained platforms | Planner/profiler/patcher/validator roles; metrics for quality (correct diagnosis), performance (speed-up), productivity (human minutes, iterations, tokens) |

Resume impact: `RTX_3080_opt` is already listed; after the plan its bullet changes from "profiled a
kernel and reported a negative result" to "built and validated a guard-railed agentic profiling
workflow and a roofline model that predicts decode throughput within X %".

## 3. Goals and non-goals

### Goals
- **A. Model** — a roofline/speed-of-light model for decode and prefill on the 3080 that predicts measured throughput and explains the 72 % bandwidth figure.
- **B. Study** — finish the pending continuous-batching experiment and extend into a serving-level study (TTFT/ITL/P99 under load, KV-cache type and size, CUDA graphs, speculative decoding, flash-attention).
- **C. Agent** — a profiling-and-optimization agent whose actions are profile → diagnose → hypothesize → patch → validate → decide, with automatic correctness and statistical gates and a human approval step; evaluated for diagnostic accuracy and for false-win rate.
- **D. Cross-check (stretch)** — one comparison point on a second runtime (vLLM or TensorRT-LLM if it fits 10 GB) to show the method is not llama.cpp-specific.
- **E. Repro harness** — turn per-phase shell scripts into one parameterised, tested harness.

### Non-goals
- Shipping a faster kernel. A confirmed negative result remains a valid outcome; the agent's value is
  in rigor and speed of finding that out.
- Multi-GPU / NCCL studies (hardware not available); stated as a limitation.
- Fine-tuning models. Agent LLM is used as-is.

## 4. Requirements

### 4.1 Workstream A — Roofline and speed-of-light model

| ID | Requirement |
|---|---|
| A1 | Measure achievable peaks on this card: DRAM bandwidth (device-to-device copy, `nvbandwidth` or a custom kernel), FP16/INT8 tensor-core and FMA throughput (microbenchmarks); record vs spec (760 GB/s) |
| A2 | Compute per-token **bytes moved** and **FLOPs** for Qwen2.5-7B-Instruct Q4_K_M (weights per token = total weight bytes read once per decode step; KV bytes grow with context) and the resulting arithmetic intensity |
| A3 | Predict decode tok/s = achievable BW / bytes-per-token at batch sizes 1…16 and prefill tok/s from the compute roof; compare with measured; report the efficiency gap per kernel (use per-kernel `ncu` DRAM and SM numbers) |
| A4 | Deliver a roofline plot (log–log) with each top-10 kernel placed by measured intensity and throughput |
| A5 | State the model's assumptions and error bars; the prediction for batch-1 decode must be within ±15 % of measured (134.05 tok/s) or the discrepancy must be explained |

### 4.2 Workstream B — Serving-level study

| ID | Requirement |
|---|---|
| B1 | **Continuous batching test (pending item):** `llama-server -np ∈ {1,2,4,8,16}` with context scaled; aggregate decode tok/s, per-request tok/s, and `ncu` `dram__bytes.sum` **per request** vs batch size (tests amortization of weight reads) |
| B2 | **Load generator** supporting closed-loop (N concurrent users) and open-loop (Poisson arrivals at rate λ) traffic with fixed and sampled prompt/output lengths; records per-request TTFT, ITL distribution, E2E latency, tokens |
| B3 | Report TTFT, ITL (P50/P95/P99), E2E P99 and goodput at a latency SLO for each configuration |
| B4 | Factors (documented design): CUDA graphs on/off, flash-attention on/off, KV-cache type {f16, q8_0}, `-c`/`-np` sizing, `-ub`/`-b` batch sizes, speculative decoding with a vocabulary-verified same-family draft model |
| B5 | **KV-cache capacity study:** max concurrent sequences and context before VRAM exhaustion for each KV type; quality impact via perplexity/KL divergence on a fixed slice (use KL, not perplexity deltas, per the lesson in `Evol_inference`) |
| B6 | **Prefill/decode split analysis** for long prompts (pp512…pp8192): chunked prefill interaction with concurrent decode (ITL spikes) |
| B7 | Every comparison reports median, 95 % CI (bootstrap) and n ≥ 5 reps; server restarted between flag states (the flag is read once, as noted in the existing re-test) |

### 4.3 Workstream C — Guard-railed performance agent

**Purpose:** automate the manual loop the project has already demonstrated by hand, and measure how
well it works.

| ID | Requirement |
|---|---|
| C1 | **Tool layer** (typed functions, exposed as plain Python and as an MCP server so Claude Code/OpenCode can drive them): `build(commit|patch)`, `bench(config)`, `nsys_profile(cmd)`, `ncu_profile(kernel_regex, metrics)`, `parse_profile(path)`, `run_correctness(suite)` (`test-backend-ops` + fixed-seed byte-identical output), `compare(a,b)` (CI test), `git_worktree_*`, `write_report` |
| C2 | **Agent graph** (LangGraph, reusing the planner/executor/validator and interrupt-node patterns from `Agentic_platform`): nodes `collect_baseline → diagnose → hypothesize → patch → correctness_gate → perf_gate → human_approval → report`; explicit state (hypotheses, evidence, patches, results); retry limits; rollback to clean worktree |
| C3 | **Guardrails:** patches only in an isolated git worktree; no network; per-tool timeouts; correctness gate must pass **before** any performance run is counted; perf gate requires CI excluding zero **and** ≥ 3 reps **and** kernel-level counters that agree with the hypothesis (e.g. DRAM bytes or sector efficiency moved as predicted) — this directly encodes the author's earlier finding that a better kernel proxy (sector efficiency) did not mean better end-to-end throughput; human approval before anything is merged or reported as a win |
| C4 | **Hypothesis discipline:** the agent must write the expected counter change *before* the run (as the existing re-test docs do: "Expectation, stated before running"); verdict compares predicted vs observed |
| C5 | **Observability:** every LLM call and tool call traced (Langfuse as in `Agentic_platform`); every run stores prompts, tool I/O, artifacts; costs (tokens, wall-clock, human minutes) recorded |
| C6 | **LLM backends:** (i) local `qwen3.6` via Ollama/OpenCode (consistent with the local-first theme), (ii) a hosted coding agent (Claude Code) as a reference configuration; same tool layer and tasks; hosted usage budget capped and declared |
| C7 | **Evaluation suite with ground truth** (see §5.3): ≥ 8 tasks across three difficulty tiers |
| C8 | **Metrics:** diagnosis accuracy (identified true bottleneck?), counter-prediction accuracy, success (speed-up ≥ target with correctness), iterations to success, tokens and cost, **false-win rate** (claims a win that fails the independent re-measure), harmful-patch rate (breaks correctness; must be 0 reaching approval) |
| C9 | Agent must produce a report in the repo's established style (hypothesis, evidence table, verdict, negative results kept) |

### 4.4 Workstream D — Cross-check (stretch)
D1. Same Qwen2.5-7B (or nearest) on vLLM *or* TRT-LLM within 10 GB; same load generator; compare aggregate tok/s and P99 vs llama.cpp at `np` 1…8. Report as a comparison, not a contest (different quantization formats must be controlled and disclosed).

### 4.5 Workstream E — Reproducible harness

| ID | Requirement |
|---|---|
| E1 | Single CLI (`rtxopt run <experiment.yaml>`) replacing per-phase shell scripts; experiment YAML records commit hashes, flags, env vars, driver/CUDA versions, clocks (lock clocks or record them), GPU temperature and power |
| E2 | Raw outputs stored under `results/<date>_<name>/` with a manifest; summaries regenerate from raw |
| E3 | Thermal/clock control: warm-up, interleaved A/B ordering, temperature log; discard-and-rerun rule if clocks differ > 3 % between A and B |
| E4 | Unit tests for parsers, statistics, experiment-file validation; CI runs them without a GPU using stored `ncu`/`nsys` CSV fixtures |

### 4.6 Non-functional
- Python 3.12; analysis code type-hinted and linted; ≥ 85 % coverage on parsers/stats.
- Full B1–B6 study ≤ 12 h machine time; agent evaluation ≤ 20 h.
- Every number in README traceable to a raw file path.
- AI-assistance disclosure kept and updated (agent and human roles distinguished).

### 4.7 Project-level success criteria
1. Roofline model predicts batch-1 decode within ±15 % (or the miss is explained) and ranks the top-10 kernels correctly by headroom.
2. Continuous-batching experiment completed: per-request DRAM bytes vs `np` with a clear confirm/refute of weight-read amortization.
3. TTFT/ITL/P99 table for ≥ 6 configurations with CIs; ≥ 1 configuration shown to improve P99 or goodput, or a documented negative.
4. Agent evaluated on ≥ 8 tasks × ≥ 3 runs × 2 backends: report diagnosis accuracy, false-win rate (target 0 reaching approval), iterations, cost.
5. Agent independently re-derives the three known findings (72.1 % BW / 34 % sector efficiency; int2+shuffle no DRAM change; graph-opt batch-1 regression) **without being told the answers**.
6. Deck, README, resume bullet delivered.

## 5. Design

### 5.1 Repository layout (additions)
```
RTX_3080_opt/
  rtxopt/                 # harness package (E)
    experiments/          # YAML specs
    runners/              # build, bench, load generator, nsys/ncu wrappers
    analysis/             # parsers, stats, roofline
    agent/                # tools, MCP server, LangGraph graph, prompts (C)
  tasks/                  # agent evaluation tasks with ground truth (C7)
  results/<date>_<name>/  # raw + manifest
  tests/ fixtures/        # CSV/ncu/nsys fixtures
  docs/ (roofline.md, agent.md)
```
Existing phase folders stay untouched (history), referenced from the new README.

### 5.2 Agent architecture

```
          ┌────────────── LangGraph state (hypotheses, evidence, patches, results) ──────────────┐
 baseline ─► diagnose ─► hypothesize(+predicted counters) ─► patch(worktree) ─► correctness gate ─┐
    ▲           ▲                                                                 fail ◄──────────┤ (rollback, retry ≤ k)
    │           └──────── new evidence ◄──── perf gate (CI + counters vs prediction) ◄─ pass ──────┘
    └── report ◄── human approval (interrupt) ◄── win / no-win / negative-result classification
```
Tool results are structured JSON (parsed counters), so the LLM never reads raw multi-MB profiler output;
parsed summaries (top-N kernels, DRAM bytes, sector efficiency, occupancy, stall reasons) are given
instead. This keeps the context within what a 10 GB-local model can handle.

### 5.3 Agent evaluation tasks (ground truth)

| Tier | Task | Ground truth available from |
|---|---|---|
| 1 Diagnose | "Find the dominant decode kernel and say whether it is memory- or compute-bound, with evidence" | Phase 0 numbers (26.4 %, 72.1 % BW, 45.7 % SM) |
| 1 | "Is `GGML_CUDA_GRAPH_OPT=1` beneficial at batch 1? Measure." | Phase 3 (-2.98 % decode, -5.93 % prefill) |
| 2 Hypothesis test | "Evaluate the int2+shuffle patch (given as diff): correct? faster? why/why not?" | Phase 2 data (DRAM bytes unchanged, L1 absorbs) |
| 2 | "Does batching amortise weight reads? Design and run the test" | completed in workstream B |
| 2 | "Choose the best of 4 server configs for P99 ITL at 4 users" | Workstream B results (after they exist) |
| 3 Fix | Seeded regression: reintroduce an uncoalesced access pattern into the naive kernel from `TensorRT_plugin` (known fix gave 4.3x; 86 % wasted sectors) — agent must find and fix | `TensorRT_plugin` Phase 2/3 data |
| 3 | Seeded regression: wrong group-size stride in dequant (correctness bug) — agent must be stopped by the correctness gate | unit tests |
| 3 | "Negative control": already-optimal kernel — agent must report no actionable win rather than invent one | known from Phase 0–2 |

Tier 3 tasks use the small self-contained kernel in `TensorRT_plugin` (fast build, deterministic) so
evaluation runs take minutes, not hours, while llama.cpp tasks remain for tiers 1–2.

## 6. Implementation plan

~5–6 weeks at 10–12 h/week; milestones are independently presentable.

### M0 — Freeze and scaffold (2 days)
1. Tag `v0-takehome`; no changes to existing phase folders.
2. Create package skeleton, `pyproject.toml`, CI, fixtures directory; copy a small set of existing CSV/`ncu` exports into `tests/fixtures/`.
3. Record environment manifest tool (driver, CUDA, llama.cpp commit, GPU clocks/temperature, ECC/persistence mode).

### M1 — Harness (E) and parsers (4–5 days)
1. Parsers for `ncu --csv`, `nsys stats` CSV, `llama-bench` output and `llama-server` logs; tests against fixtures from phases 0–3.
2. Experiment YAML + runner; interleaved A/B with restart; clock/temperature guard (E3).
3. Port `concurrent_bench.py` into the load generator (B2): closed-loop first, then open-loop Poisson; per-request timing via streaming responses (TTFT = first SSE chunk, ITL = inter-chunk gaps).
4. Statistics module (bootstrap CI, ratio CI, Mann–Whitney).

### M2 — Roofline model (A) (4 days)
1. Microbenchmarks for bandwidth and compute peaks (A1); document achieved vs spec.
2. Count bytes/FLOPs per token from the GGUF tensor list and architecture (28 layers, GQA KV heads; weights read once per decode step; KV bytes by context length) (A2).
3. Predict decode/prefill throughput; compare to measured at batch 1; compute per-kernel headroom from `ncu` (A3, A4).
4. Write `docs/roofline.md` including failure analysis if prediction misses.

### M3 — Serving study (B) (8–9 days incl. unattended runs)
1. **B1 first** (the pending experiment): `-np` sweep; `ncu` DRAM bytes per request (profile one decode window with `--launch-skip/--launch-count` or NVTX-free time-slicing by kernel regex); aggregate vs `N ×` single-stream.
2. Open-loop load tests (B2–B3): determine saturation point, then P99 ITL/TTFT at 50/80/95 % of saturation.
3. Factor study (B4): CUDA graphs, flash-attn, KV type, `-ub`; fractional design or OFAT (documented).
4. KV capacity + quality (B5): KL divergence vs f16 KV for q8_0 on a fixed slice (≥ 8 k tokens to avoid the noise-floor issue identified in `Evol_inference`).
5. Prefill/decode interference (B6): long-prompt injection while 3 decodes run; measure ITL spikes; test `-ub` smaller chunks.
6. Speculative decoding (B4): verify draft vocab, output equivalence first; measure acceptance and speed at `np` = 1 and 4 (prediction: gains shrink with batching because the GPU is no longer idle).
7. Write findings in the repo's "hypothesis → expectation → result" format.

### M4 — Agent tool layer and graph (C1–C5) (8–9 days)
1. Implement tool functions with strict JSON schemas and unit tests (fixtures); sandboxed worktree manager.
2. Profile summarizers (top-N kernels; compute/memory-bound classification using the A-workstream roofline; stall breakdown).
3. LangGraph graph with interrupt node, retry limits, Langfuse tracing; config in YAML (models, thresholds).
4. MCP server exposing the same tools, so Claude Code/OpenCode can drive the identical workflow (C6).
5. Perf-gate and correctness-gate modules (C3) with their own tests, including adversarial cases: noise-only difference, counter moves but throughput does not (the int2 case), correctness failure.

### M5 — Agent evaluation (C7–C8) (6–7 days, mostly unattended)
1. Implement the 8 tasks with ground-truth checkers; seed regressions by patch files.
2. Runs: 2 backends (local qwen3.6, Claude Code) × 8 tasks × 3 runs; token/wall-clock/human-minute accounting.
3. Analysis: diagnosis accuracy, prediction accuracy, false-win rate, iterations, cost; failure taxonomy (mis-diagnosis, tool misuse, over-claiming).
4. Write-up includes failures; the negative-control task is the headline integrity check.

### M6 — Cross-check and write-up (D, deliverables) (4–5 days)
1. Optional vLLM/TRT-LLM comparison (D1) if memory allows; otherwise document why it was infeasible.
2. README restructure (results first), `docs/`, AI-disclosure update, deck, resume bullet, blog post draft ("what the profiler told me vs what the benchmark told me").

## 7. Test plan

### 7.1 Unit tests (CI, no GPU)

| Test | Req | Pass condition |
|---|---|---|
| `ncu`/`nsys`/`llama-bench` parsers on fixtures from phases 0–3 | M1 | extracted DRAM bytes, sector efficiency, durations equal the values in the existing READMEs (e.g. 93.65 vs 93.69 MB; 34.0 %) |
| Bytes/FLOPs counter on a toy model with hand-computed totals | A2 | exact match |
| Bootstrap/ratio CI coverage on synthetic data | B7 | 93–97 % coverage |
| Experiment YAML validator rejects incomplete specs (missing commit, flags) | E1 | negative cases rejected |
| Clock/temperature guard rejects A/B pair with clock mismatch > 3 % | E3 | rejected |
| Tool JSON-schema conformance (every tool output validates) | C1 | all schemas pass |
| Perf gate: noise-only data → "no win"; real 5 % shift → "win"; counter/throughput disagreement → "inconclusive" | C3 | 3 of 3 correct classifications over 100 synthetic trials each ≥ 95 % |
| Correctness gate: deliberately corrupted kernel output → fail; identical → pass | C3 | 100 % |
| Worktree manager never writes outside worktree (path-traversal tests) | C3 | all attempts blocked |

### 7.2 Integration tests (GPU box)

| Test | Pass condition |
|---|---|
| Reproduce Phase 0 baseline with the new harness | pp512 and tg128 within ±3 % of 5229.88 / 134.05 tok/s; top kernel identified as `mul_mat_vec_q<12,1,1,0>` |
| Reproduce Phase 3 graph-opt batch-1 regression | sign and rough magnitude (−1…−5 %) reproduced, CI excludes 0 |
| Load generator sanity: synthetic constant-time server | measured TTFT/ITL equal injected delays within 2 ms |
| Open-loop arrival generator | observed arrival rate within 3 % of λ; inter-arrival distribution passes KS test vs exponential |
| Speculative decoding equivalence | greedy output token-identical with/without draft |
| `ncu` per-request DRAM measurement method validated | batch-1 bytes/request equals bytes/token × tokens within ±5 % |
| End-to-end agent smoke test on Tier-1 task | produces report with all required sections; no gate bypass |

### 7.3 Agent evaluation protocol and acceptance
- 3 independent runs per (task × backend), fresh worktree, temperature fixed, seeds recorded.
- Ground-truth checkers are code; no LLM judge for success. (An LLM may summarise, never grade.)
- **Acceptance:** (i) zero harmful patches reach the approval step; (ii) false-win rate ≤ 10 % on all tasks and 0 on the negative control for at least one backend; (iii) Tier-1 diagnosis accuracy ≥ 80 % for the reference backend; (iv) the local backend's results are reported even if worse — the comparison is part of the story.
- Report per-task pass rates with Wilson intervals, never only means.

### 7.4 Experiment validity checks (all workstreams)
- Server restarted between flag states; A/B interleaved; warm-up discarded.
- Clock, temperature and power logged; rerun on violation.
- Every surprising result is first re-measured with the simplest independent method (e.g. `llama-bench`) before interpretation.
- Raw files referenced by path in each README table.

### 7.5 Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Local 7–8 B-class model cannot drive the agent reliably (tool misuse, as seen in `agent_opt`) | high | structured tool outputs, narrow schemas, mandatory tool naming in prompts; Claude Code reference backend; report the gap |
| `ncu` overhead distorts batched runs | medium | profile short windows, cross-check bytes with `nsys`/`llama-bench`; use counters not timings from `ncu` runs |
| Roofline prediction misses by > 15 % | medium | treat as a result: decompose miss into launch gaps, attention/KV, sampling; document |
| Per-request DRAM attribution under concurrency is ambiguous | medium | measure total bytes per decode step vs batch size and divide by active slots; validate with kernel-level `dram__bytes` |
| vLLM/TRT-LLM does not fit in 10 GB at comparable quality | medium | stretch item only; document |
| Scope creep | high | milestones M0–M3 are sufficient for a strong story; M4–M5 are the differentiator; M6 optional parts can be dropped |

## 8. Deliverables
1. `rtxopt/` harness with tests and CI; `results/` with raw data and manifests.
2. `docs/roofline.md`, `docs/serving_study.md`, `docs/agent.md`.
3. Updated `README.md` (dashboard first), `Summary.md`, `AI-disclosure.md`.
4. Deck `presentation/RTX_3080_opt.pptx`.
5. Resume bullet and a short blog post draft.

### Draft resume entry (fill with measured numbers)
> **RTX_3080_opt:** CUDA kernel- and serving-level optimization study of llama.cpp (Qwen2.5-7B, RTX 3080 10 GB) with Nsight Systems/Compute: roofline model predicts decode throughput within **X %**; continuous-batching study quantifies weight-read amortization (**Y×** aggregate throughput at 8 slots, **Z** MB DRAM/request); TTFT/ITL/P99 under open-loop load; built a guard-railed LangGraph/MCP profiling agent (correctness gate, CI-based perf gate, human approval) that re-derived the manual findings on **k/n** tasks with **0** false wins reaching approval.

## 9. Slide deck concept (≈ 11 main slides + backups)

1. **Title / thesis** — "Predict, measure, then let an agent do the loop — and keep it honest." Hardware strip.
2. **Starting point** — the manual result in one table: baseline, three attempts, all negative; "what I wanted to know next".
3. **Roofline** — headline figure: roofline plot with kernels placed; text: decode is memory-bound; achievable BW X GB/s; predicted vs measured tok/s bar (batch 1…16).
4. **Batching is the lever (or not)** — chart: aggregate tok/s vs `np` against the `N×` line; second axis DRAM MB per request. One-line verdict on amortization.
5. **Serving latency** — TTFT / ITL P99 vs offered load (open-loop) for the best and baseline config with CI bands; saturation knee annotated.
6. **Knobs that matter (and ones that don't)** — tornado chart of factor effects (CUDA graphs, KV type, flash-attn, `-ub`, spec-decoding) with CIs; highlight surprising nulls.
7. **KV cache capacity vs quality** — max sequences vs KV type; KL divergence cost; recommendation.
8. **Agent architecture** — graph diagram (§5.2) with the guardrails (worktree, correctness gate, perf gate, human approval) colour-coded; "hypothesis before run" rule.
9. **Does the agent work?** — task × backend matrix of pass rates with Wilson intervals; diagnosis accuracy; iterations and cost.
10. **Integrity check** — false-win analysis: noise-only, counter-moves-but-throughput-does-not (int2 case), negative control; what the gates blocked.
11. **Takeaways and next steps** — 3 bullets; what it means for NVIDIA workflows (TensorRT-LLM/vLLM/Dynamo parity, Nsight automation, DGX/RTX).
- Backup A: methodology & validity checks (clocks, interleaving, restarts).
- Backup B: Roofline derivation (bytes/FLOPs table).
- Backup C: agent prompts/tools/schemas; failure taxonomy.
- Backup D: reproduction commands.

Visual rules: finding in the title, CI on every comparison, raw-data path in speaker notes, consistent colour per configuration across slides.

## 10. Dependencies between the two plans

- `agent_opt` supplies the **evaluation machinery** (task specs with executable verifiers, statistics, parity checks, citation/claim verification). `RTX_3080_opt` Workstream C reuses its statistics module and its "did a tool call precede the claim?" check to catch unverified claims by the perf agent. Build `agent_opt` M0–M3 first, or extract the stats/verifier utilities into a small shared package.
- `RTX_3080_opt` supplies a **realistic GPU workload and profiling tools** for the `agent_opt` Nsight Systems mode (F9) and for the speculative-decoding experiment (F12).
- Suggested order: `agent_opt` M0–M4 → `RTX_3080_opt` M0–M3 → `agent_opt` M5–M6 ∥ `RTX_3080_opt` M4 → `RTX_3080_opt` M5 → write-ups. Minimum viable story for applications: both projects' M0–M3 plus one lever result each.

## 11. Open questions for the author

1. Is a hosted agent (Claude Code) acceptable as a *reference backend* for the evaluation (usage cost, disclosure)? The plan assumes yes, capped, and reported separately.
2. Which second runtime (vLLM vs TensorRT-LLM) is realistic on the 10 GB card for the stretch cross-check?
3. Should the Tier-3 seeded-regression tasks live in this repo or in `TensorRT_plugin` (where the kernels are)? Default: here, importing the kernel as a git submodule/fixture.
