# Phase 2 re-test: int2+shuffle patch under batched decode

This follows up on the third "Next Steps" item from the main README - the prerequisite
check for whether the int2+shuffle patch (Phase 2) is even relevant once real
concurrency shifts decode onto different kernel templates - and then, since the check
came back positive, extends the actual patch test to that batched setting.

## Purpose

Phase 2 measured the int2+shuffle patch against `mul_mat_vec_q<12,1,1,0>` (batch-1,
fused) and found a real regression: decode -2.84%, prefill -0.44%, root-caused to
+13.45% instruction overhead with DRAM bytes moved unchanged (waste absorbed by L1,
never reached HBM). That result was specific to batch-1. Two open questions before
trusting it as the final word on this patch:

1. **Does the patch even apply once real concurrency is involved?** It touches
   `vec_dot_q4_K_q8_1`, called by every `mul_mat_vec_q<12,N,...>` template regardless of
   `N` - but if concurrent decode shifts dominant GPU time onto a different kernel
   entirely (e.g. `mul_mat_q`, a separate code path in `mmq.cuh`), the patch would be
   moot for that share of decode time.
2. **If it does apply, does the same regression hold, or was batch-1's overhead-vs-savings
   trade-off itself a batch-1-specific artifact** - the way `GGML_CUDA_GRAPH_OPT`'s
   batch-1 regression turned out to be (see `next-steps-graphopt-concurrency.md`)?

## Setup

**Step 1 - prerequisite check.** Captured one `nsys` trace under `-np 4` (4 slots,
`-c 16384`), 4 concurrent completion requests, and inspected the kernel breakdown.
Needed a custom script (`run_nsys_concurrency.sh`) since the standard `nsys profile`
signal-based shutdown deadlocked when combined with concurrent background curl
requests (a bare `wait` blocks on the `nsys` job too, not just the requests) and,
separately, `kill -INT`/`kill -9` on the `nsys profile` PID never triggered a proper
report export under a 4-stream concurrent trace - `nsys shutdown
--session=<name> --kill=sigterm` was the reliable fix, not `nsys stop`.

**Step 2 - correctness + throughput, pre/post patch.** `run_patch_concurrency_test.sh`
launches the server once per state (`-np 4 -c 16384`), fires one round of 4 concurrent
requests to capture generated text per slot (`concurrent_bench.py
--correctness-capture`), then runs the aggregate-throughput benchmark
(`concurrent_bench.py`, 4 concurrent requests, `n_predict=128`, 5 reps - same tool and
convention as `next-steps-graphopt-concurrency.md`). Patch applied via `git apply
phase-2/int2_shuffle_attempt.patch`, rebuilt, tested, then `git checkout --` to revert
before moving to the next step - repo never left at a dirty state between runs.

**Step 3 - DRAM-bytes check, pre/post patch.** Same `ncu` metrics as Phase 2's original
DRAM check (`dram__bytes.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`),
`--kernel-name regex:mul_mat_vec_q`, but under `-np 4` with 4 concurrent requests
instead of one, and `--launch-count 20` (vs. Phase 2's 5) to get reasonable coverage
given decode now interleaves multiple kernel templates instead of one dominant one.

## Results

### Step 1: kernel dominance shifts under concurrency

| Kernel | % of GPU time | Instances |
|---|---|---|
| `mul_mat_q<12,64,0>` | 23.9% | 332 |
| `mul_mat_vec_q<12,4,0,0>` | 21.9% | 336 |
| `mul_mat_vec_q<12,2,0,0>` | 20.0% | 506 |
| `mul_mat_vec_q<12,1,1,0>` (Phase 2's original target) | not in top 20 | negligible |

The batch-1 fused kernel Phase 2 tested barely runs under real concurrency. But
`vec_dot_q4_K_q8_1` (where the patch lives) is dispatched by type only, independent of
`ncols_dst` - so it's still exercised by the now-dominant `<12,2,0,0>` and
`<12,4,0,0>` variants, which together account for 41.9% of GPU time, more than the
original batch-1 kernel's 26.4% share. `mul_mat_q<12,64,0>` (23.9%, plus its fixup
kernel) is a genuinely separate code path the patch can never touch, at any batch size.
Conclusion: relevant, but only re-tested against the kernel it was actually validated
on - needed re-deriving against `<12,2,...>`/`<12,4,...>`, per the Next Steps item's own
framing.

### Step 2: correctness and throughput under `-np 4`

| | Pre-patch | Post-patch | Delta |
|---|---|---|---|
| Correctness (4 concurrent slots, fixed seed, byte-diff) | - | **all 4 identical** | pass |
| Aggregate decode tok/s (median of 5) | 265.49 | 267.61 | **+0.80%** |
| Sample range | 265.14 - 266.77 | 266.56 - 268.37 | thin overlap |

All four concurrently-generated outputs are byte-identical pre- vs. post-patch -
correctness holds end-to-end under the actual `ncols_dst=2`/`4` kernels, not just
`test-backend-ops`'s unit-level coverage (which already included those `n` values, just
not in a live concurrent-server setting). Throughput direction reverses from batch-1's
clean -2.84% regression to a small positive delta - though with only 5 samples and a
thin overlap band between the ranges, this reads as "probably a small improvement, not
a regression," not as unambiguous a signal as the batch-1 result was.

### Step 3: DRAM-bytes check under concurrency

| Metric (`mul_mat_vec_q<12,4,0,0>`, n=18 instances) | Pre-patch | Post-patch | Delta |
|---|---|---|---|
| `dram__bytes.sum` (sum) | 318.63 MB | 318.64 MB | +0.0035% (noise) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` (avg) | 33.02% | 33.05% | +0.03pp (noise) |

The hypothesis going in was that heavier concurrent traffic might create enough L1
pressure to make the batch-1 "free" caching absorption less complete, turning the
sector-efficiency gain into a real DRAM saving this time. That's refuted: DRAM bytes
moved are statistically identical pre- vs. post-patch, same as the batch-1 result. L1 is
still absorbing the waste under concurrent load with the now-dominant `ncols_dst=4`
kernel.

## Interpretation

The small throughput improvement isn't explained by reduced memory traffic - ruled out
directly, not just unconfirmed. What's notable is the parallel to the
`GGML_CUDA_GRAPH_OPT` re-test in `next-steps-graphopt-concurrency.md`: in both cases, a clear
batch-1 effect (regression there, no-effect here) changes under real concurrency
(neutral there, small-positive here), and in both cases the DRAM-level check comes back
flat. The common thread across both investigations: whatever fixed per-launch
instruction/scheduling overhead each change adds is more exposed in isolation
(batch-1, nothing else running to hide it behind) and gets absorbed once the GPU has
other concurrent work in flight - not because memory bandwidth usage actually changes.
Neither investigation measured instruction count/occupancy under concurrency directly
(only at batch-1, in Phase 2's regression diagnosis), so this is an inference from the
pattern repeating across two independent experiments, not something directly confirmed
here - a reasonable next check if pursued further, in the same spirit as the earlier
`phase-3.md` gap-closing exercise.

**Artifacts:**
- `run_nsys_concurrency.sh` - concurrent-load `nsys` capture (uses `nsys shutdown
  --session=... --kill=sigterm`, not `nsys stop`).
- `run_patch_concurrency_test.sh` - launches server, runs correctness capture then
  throughput benchmark, tears down.
- `concurrent_bench.py` - extended with `--correctness-capture` mode (saves each
  slot's generated text instead of benchmarking).
- `run_dram_diagnosis_concurrency.sh` - `ncu` DRAM-bytes check under `-np 4`.
- `prepatch.*` / `postpatch.*` - correctness + throughput logs, both states.
- `prepatch_dram_conc.csv`/`.ncu-rep`, `postpatch_dram_conc.csv`/`.ncu-rep` - DRAM
  check raw data, both states.
- `correctness_prepatch_slot{0-3}.txt`, `correctness_postpatch_slot{0-3}.txt` -
  generated text per slot, both states.

## Status

Patch was applied and reverted twice during this investigation (once for the
throughput/correctness test, once for the DRAM check) - `vecdotq.cuh` and the rest of
the `llama.cpp` working tree are back at clean `HEAD`. `int2_shuffle_attempt.patch`
itself (in `phase-2/`) is unchanged - same patch used throughout this project.
