# Re-test: `GGML_CUDA_GRAPH_OPT=1` under real concurrency

This closes out the first "Next Steps" follow-up item from the main README: re-testing
the Q/K/V stream-concurrency flag (`GGML_CUDA_GRAPH_OPT`) under real concurrent load,
rather than the batch-1 decode setting every other measurement in this project used.

## Purpose

Phase 3 measured `GGML_CUDA_GRAPH_OPT=1` at batch-1 and found a real, repeatable
regression (decode -2.98%, prefill -5.93%), traced - by elimination, not direct
measurement - to host-side stream/event overhead rather than anything visible at the
single-kernel level (`ncu` showed the kernel itself completely unaffected: DRAM bytes,
throughput, and duration all flat between flag-off and flag-on).

That explanation ("no bandwidth to share, streams just add overhead against a
bottleneck with nothing to parallelize") is specific to batch-1, where there is exactly
one decode stream and nothing else running on the GPU to hide any added dispatch
overhead behind. Under real concurrent load - multiple simultaneous requests, each with
independent work in flight - the same overhead could plausibly get amortized away
(hidden behind other slots' work) or get worse (more streams now competing for the same
resources). The batch-1 result doesn't tell you which. This test measures it directly
instead of continuing to reason about it.

## Setup

`llama-server` launched with multiple parallel slots and context sized up accordingly,
matching the "Core batching test" setup described in the Next Steps section:

```bash
GGML_CUDA_GRAPH_OPT=<0|1> ./build/bin/llama-server \
  -m ~/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 16384 -np 4
```

`-np 4` (4 slots) with `-c 16384` (4x the single-slot `-c 4096` used everywhere else in
this project, so each slot still gets its original 4096-token budget).

No existing tool in the repo fit a controlled, self-contained A/B test here - the
in-tree server benchmarks (`tools/server/bench/`) pull an external HuggingFace dataset
and require `matplotlib`/`k6`, more than needed for a fixed-prompt, fixed-length
concurrent throughput comparison. Wrote a small script instead
(`concurrent_bench.py`, ~80 lines, stdlib + `requests` only), matching this project's
convention of minimal purpose-built scripts over adding external dependencies:

- Fires N `/completion` requests concurrently (via a thread pool, not sequentially) at
  `temperature=0`, `n_predict=128` each - same decode length as every `tg128` number in
  this project.
- Same prompt used throughout the `ncu` diagnostics: "Write a Python function that
  finds two numbers in a list that sum to a target, then explain the time complexity in
  detail."
- Aggregate tok/s per rep = total tokens generated across all N requests, divided by
  one wall-clock window (start of the first request to completion of the last) - not
  the sum of each request's own reported tok/s, which would double-count shared GPU
  time.
- 5 reps, median reported, same convention as every `llama-bench` comparison in this
  project.

Ran with `-n-requests 4` (matching `-np 4`), once per flag state, server fully
restarted between runs (the flag is read once, at first graph-optimize call, so it
can't be toggled on a live server).

## Expectation

Stated before running, not after: if the batch-1 regression really is a roughly fixed
per-decode-step host-side cost (stream/event setup), it shouldn't disappear just
because other slots are concurrently active - each slot still pays its own per-step
overhead. Predicted the flag would still show a similar-magnitude relative regression
in aggregate throughput under `-np 4` (~3-6%, in line with the batch-1 numbers).

If aggregate throughput instead came back flat or improved, that would contradict the
host-side-overhead explanation and suggest the batch-1 regression actually was about
GPU-side contention after all - just invisible at batch-1 because there was nothing
else in flight to contend *with*.

## Results

| Metric (median of 5 reps, 4 concurrent requests, n_predict=128 each) | `GRAPH_OPT=0` | `GRAPH_OPT=1` | Delta |
|---|---|---|---|
| Aggregate decode tok/s | 264.19 | 266.56 | **+0.90%** |
| Sample range | 261.11 - 265.11 | 261.98 - 268.98 | **overlapping** |

Unlike every other flag-off/on comparison in this project, the sample ranges overlap.
This is not a clean signal in either direction - it reads as flat, within noise. That
is a qualitative difference from the batch-1 result (133.9-134.1 vs. 129.8-130.2,
cleanly separated, real).

Secondary data point, independent of the flag: aggregate throughput at `-np 4`
(~264-267 tok/s either way) is only about **49%** of the naive 4x linear-scaling
projection from the single-stream baseline (4 x 134.05 = 536.20 tok/s). Real
contention exists at the concurrency level - just not something this flag measurably
changes in either direction.

## Interpretation

The stated expectation was wrong, and it's informative that it was wrong. The batch-1
regression's cause doesn't appear to be a fixed, per-request cost that persists
regardless of what else is running - it appears to specifically require an *idle* GPU
to show up. Under `-np 4`, the GPU already has substantial independent work in flight
from the other slots' kernels regardless of the flag; that seems to be enough to absorb
whatever dispatch/stream-management overhead the flag adds, the same way any other
small serialization cost gets hidden once there's enough concurrent work to schedule
around it.

This reframes, but doesn't reverse, Phase 3's conclusion. The flag is not "bad" in
general - it's bad specifically in the regime this whole project has been measuring
(single-stream, batch-1 decode), and roughly neutral under realistic concurrent load.
It doesn't rise to "worth adopting" either: neutral-with-overlapping-noise is not a
measured win, and nothing here suggests it would ever clearly outperform flag-off at
this concurrency level. The practical takeaway is narrower than "the flag helps under
load" - it's "the flag's batch-1 regression is a batch-1-specific artifact, not a
general cost," which matters for interpreting Phase 3 correctly but doesn't change the
project's standing recommendation (continuous batching as the real lever, not this
flag).

Artifacts: `concurrent_bench.py` (the concurrent-request benchmark script),
`run_concurrency_bench.sh` (launches the server for a given flag value, waits for
health, runs the benchmark, tears the server down.), `concurrency_off.bench.log` /
`concurrency_off.server.log`, `concurrency_on.bench.log` / `concurrency_on.server.log`.

## Status

No code changes - `GGML_CUDA_GRAPH_OPT` remains an existing upstream toggle, not
something modified in this project. `vecdotq.cuh` and the rest of the `llama.cpp`
working tree remain at clean `HEAD`.
