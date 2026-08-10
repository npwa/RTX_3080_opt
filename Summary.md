## Abstract

This is an attempt to profile and optimize the dominant decode-phase CUDA kernel
(`mul_mat_vec_q`, Q4_K matrix-vector multiply) in llama.cpp running
Qwen2.5-7B-Instruct on an RTX 3080.  Baseline profiling (`nsys`/`ncu`) shows this
kernel at 26.4% of total decode GPU time, achieving 72.1% of peak DRAM bandwidth
against only 45.7% of peak compute — a memory-bound regime.  Source tracing
(`vecdotq.cuh`) locates the cause: two 16-byte-apart scalar loads per thread, each
independently triggering a 32-byte sector fetch while using half of it, measured at
32.87–34.0% sector efficiency.

Two kernel modifications were implemented and verified.  The first (broadcast wide
load) redistributed bytes across lanes without reducing transaction count and showed
no effect.  The second (cooperative `int2` load + warp-shuffle) genuinely improved
sector efficiency (36.47%) and cut sectors fetched (-12.4%), passing full correctness
verification (`test-backend-ops`, fixed-seed byte-identical output).  End-to-end
throughput regressed anyway — decode tok/s fell 2.84%, real and repeatable.  A
follow-up pass closed the loop: instruction count rose 13.45% with occupancy
unchanged, and a direct DRAM-level measurement showed total bytes moved from HBM were
statistically identical between builds (93.65 vs. 93.69 MB, +0.048%, noise).  The
L1-level gain never became a real DRAM saving — L1 was already absorbing that waste
before the fix ran.

Two follow-ups extended this finding.  A review of llama.cpp's fusion infrastructure
found two of three plausible fusions around this kernel already active in baseline
(gate+up+GLU, RMSNorm+mul); the one untested lever — an opt-in Q/K/V
stream-concurrency flag — produced a second real regression (-2.98% decode, -5.93%
prefill) when enabled.  A targeted kernel-level `ncu` comparison then confirmed the
kernel itself is unaffected by the flag, narrowing the cause to host-side overhead or
CUDA-graph interaction rather than anything visible at the single-kernel level.

This is reported as a negative result with a fully quantified, closed-loop root
cause, per the assignment's allowance for such outcomes.  Three independent
investigations now converge on the same conclusion: this kernel sits near a local
optimum for its load pattern and available fusions, and further tuning here is
unlikely to pay off.  The more promising remaining lever is continuous batching —
amortizing weight reads across concurrent requests rather than optimizing a
single-token decode path that has nothing left to amortize against — identified but
not yet pursued.
