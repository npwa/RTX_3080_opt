## Abstract

This is an attempt to profile and optimize the dominant decode-phase CUDA kernel
(`mul_mat_vec_q`, Q4_K quantized matrix-vector multiply) in llama.cpp running
Qwen2.5-7B-Instruct on an RTX 3080.  Baseline profiling (`nsys`/`ncu`) identifies this
kernel as 26.4% of total decode GPU time, achieving 72.1% of theoretical peak DRAM
bandwidth against only 45.7% of peak compute throughput.  This confirms a memory-bound
regime.  Source-level tracing (`vecdotq.cuh`) attributes this to a specific access
pattern: two 16-byte-apart scalar loads per thread, each independently triggering a
32-byte sector fetch while using only half of it, measured directly at 32.87–34.0% sector
efficiency (`ncu`, hardware counters).

Two kernel modifications were implemented and verified.  The first (broadcast-style wide
load) redistributed bytes across lanes without reducing transaction count, and showed no
measurable effect.  The second (cooperative `int2` load with warp-shuffle redistribution)
genuinely improved sector efficiency (36.47%) and reduced sectors fetched (-12.4%) at this
load site, while passing full correctness verification (`test-backend-ops`, fixed-seed
byte-identical generation).  Despite this, end-to-end throughput *regressed* and decode
tok/s fell 2.84%, a repeatable, non-noise result.

A follow-up profiling pass closed the loop on why.  Instruction count rose 13.45% per
kernel launch with occupancy unchanged, so the fix wasn't free.  More importantly, a
direct DRAM-level measurement (`dram__bytes.sum`) showed total bytes moved from HBM were
statistically identical between builds (93.65 vs.  93.69 MB, +0.048%, noise).  The
sector-efficiency gain was real at the L1 request level but never became a real DRAM
saving — L1 was already absorbing that "waste" before the fix ever ran.  The full causal
chain: L1 efficiency improves -> DRAM bytes moved: unchanged -> instructions executed:
+13.45% -> decode tok/s: -2.84%.  The fix solved a problem that didn't cost real cycles at
the level that actually gates this kernel's performance, and the attempt to solve it cost
cycles of its own.

This is reported as a negative result with a fully quantified, closed-loop root cause,
per the assignment's explicit allowance for such outcomes.  With DRAM-level headroom at
this specific access pattern now confirmed near zero — not just assumed — further
load-pattern tuning here is unlikely to pay off; this kernel appears to already sit near
a local optimum for its L1/DRAM behavior on this architecture.  More promising directions
identified but not pursued: kernel fusion around the matvec (assignment-sanctioned), or
at the deployment level, increasing effective batch size via continuous batching to shift
traffic onto the codebase's existing, more bandwidth-efficient n>1 kernel variants.
