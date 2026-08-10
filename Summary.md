## Abstract

This is an attempt to profile and optimize the dominant decode-phase CUDA kernel
(`mul_mat_vec_q`, Q4_K matrix-vector multiply) in llama.cpp running
Qwen2.5-7B-Instruct on an RTX 3080.  Baseline profiling (`nsys`/`ncu`) shows this
kernel at 26.4% of total decode GPU time, achieving 72.1% of peak DRAM bandwidth
against only 45.7% of peak compute; a memory-bound regime.  Source tracing
(`vecdotq.cuh`) locates the cause: two 16-byte-apart scalar loads per thread, each
independently triggering a 32-byte sector fetch while using half of it, measured at
32.87–34.0% sector efficiency.

Two kernel modifications were implemented and verified.  The first showed no effect.
The second (cooperative `int2` load plus warp-shuffle) improved sector efficiency
(36.47%) and cut sectors fetched (-12.4%), passing full correctness verification
(`test-backend-ops`, fixed-seed byte-identical output).  At batch-1, decode throughput
still regressed 2.84%.  A follow-up pass closed the loop: instructions executed rose
13.45% with occupancy unchanged, and DRAM bytes moved were statistically identical
between builds (93.65 vs. 93.69 MB).  The L1-level gain never became a real DRAM
saving; L1 was already absorbing that waste before the fix ran.

A separate lever, an opt-in Q/K/V stream-concurrency flag, produced a second batch-1
regression (-2.98% decode, -5.93% prefill), also confirmed unaffected at the
kernel level by `ncu`.

Both the flag and the patch were then re-tested under real concurrent load (`-np 4`)
instead of single-stream batch-1. Both reversed direction. The flag's regression
disappeared, with flag-off/on throughput overlapping within noise, consistent with a
batch-1-specific, idle-GPU artifact rather than a fixed per-request cost. The patch
flipped from a clean -2.84% regression to a small +0.80% improvement, though with a
weaker, overlapping-sample signal, not a confirmed win. In both cases DRAM bytes
moved stayed unchanged, so neither reversal is explained by reduced memory traffic;
the likely cause is a small fixed per-launch overhead, exposed in isolation at
batch-1, that gets absorbed once the GPU has other concurrent work to schedule
around.

This is reported as a negative result with a fully quantified, closed-loop root
cause, per the assignment's allowance for such outcomes. Every batch-1 finding
converges cleanly. Under real concurrency the picture is more mixed, but neither
reversal is a confirmed win. The core question, whether continuous batching itself
improves throughput by amortizing weight reads across concurrent requests, remains
the one experiment not yet run.
