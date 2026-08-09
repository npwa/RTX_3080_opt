# Take-Home Follow-Up: CUDA Kernel Optimization (RTX 3080)

## Phase 0: Baseline & Profile

1. Confirm that llama.cpp build environment on my desktop is in a clean, ready
state for modifying and recompiling CUDA source:

```bash
cd ~/work/ollama/llama.cpp
git pull
git tag --points-at HEAD
# Tag is "b10326", Commit: 3653e6d6d

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
# benign error because optional web UI component failed to build
# otherwise built successfully
```

2. Pull the model: `Qwen2.5-7B-Instruct-GGUF`, `q4_k_m`, verify it's
small enough (~4.7GB weights + KV cache) to be fully GPU-resident:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama run qwen2.5:7b-instruct "Hello"
ollama ps
```

Output:
> NAME                   ID              SIZE      PROCESSOR    CONTEXT    UNTIL
> qwen2.5:7b-instruct    845dbda0ea48    4.7 GB    100% GPU     4096       4 minutes from now

This confirms the model fits 100% on GPU. Also verified from log entries:
`Aug 07 21:41:01 npalmass-desk1 ollama[1452130]: slot print_timing: id 0 | task 60 | eval time = 193.17 ms / 23 tokens ( 8.40 ms per token, 119.07 tokens per second)`

if there was a CPU offload, tok/sec would be much slower, around 20 tok/sec. The model
pulled by ollama was only used for this check, the following benchmarks were done with
the model pulled in two shards from huggingface.


3. Run baseline and report medians

> By default it'll report `mean` but we are asked to report `median` (need to calculate from raw json output)

```bash
./build/bin/llama-bench -m ~/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
   -p 512 -n 128 -r 5 \
   -o json > baseline_raw.json
```

**Phase 0 baseline numbers**
| Test | Median throughput
|------|-
| Prefill (pp512) | **5229.88 tok/s**
| Decode (tg128)  | **134.05 tok/s**

4. Kernel profile of steady-state decode

First launch the server with explicit full GPU residency and 4096 context size:

```bash
./build/bin/llama-server -m ~/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 \
  -ngl 99 -c 4096
```

then check and confirm:
```bash
npalmass@npalmass-desk1:~/work/inf_opt/models$ nvidia-smi
Fri Aug  7 22:51:39 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        Off |   00000000:01:00.0 Off |                  N/A |
|  0%   43C    P8             75W /  380W |    4805MiB /  10240MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3036      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A         3580441      C   ./build/bin/llama-server               4782MiB |
+-----------------------------------------------------------------------------------------+
```

This confirms 100% GPU residency before profiling. If any layers were CPU-offloaded,
the kernel profile in step 5 would be measuring a CPU/GPU hybrid, not clean GPU-only
decode.

Reading the numbers: 4805 MiB used, model is 4.36 GiB on disk — the difference is KV
cache for `-c 4096`, in line with expectations. Only two processes hold GPU memory:
`llama-server` and `Xorg`, nothing else is competing for VRAM or GPU time.  `GPU-Util
0%`  is expected: this snapshot was taken right after model load, before any
request. It confirms the model is resident and idle.

Same server invocation (same flags: `-ngl 99 -c 4096`) is what gets wrapped by
`nsys`/`ncu` in step 5 — this check and the profiling use identical parameters.

5. Identify top-3 kernels, compute achieved DRAM bandwidth for kernel #1

`run_nsys_baseline.sh` wraps the server in `nsys profile`, sends one real completion
request, and stops the server cleanly (killed via signal, not Ctrl-C — an earlier
attempt with a manual Ctrl-C produced a corrupted `.qdstrm` file missing its
time-conversion data, so the script now handles shutdown itself).

```bash
nsys stats --force-export=true baseline_decode_nsys.nsys-rep
```

**Top 3 kernels by GPU time** (61% of total kernel time combined):

| Kernel | Time % | Instances |
|---|---|---|
| `mul_mat_vec_q<12,1,1,0>` | 26.4% | 280 |
| `mul_mat_q<12,32,0>` | 21.6% | 166 |
| `mul_mat_vec_q<12,2,0,0>` | 13.1% | 166 |

All three are variants of the same op (quantized matmul, `ggml_type=12` =
`Q4_K`). `mul_mat_vec_q` — the matrix-**vector** variant, i.e. one token at a time — is
the single largest kernel, consistent with batch-1 decode.

Next, measure achieved DRAM bandwidth for that #1 kernel specifically, with `ncu`. One
gotcha worth noting: model loading itself launches kernels matching the same name
(weight repacking), so a plain `--launch-count N` with no skip profiles the *load*, not
real decode — caught by checking kernel launch timestamps against the server's own
`model loaded` log line, then fixed with `--launch-skip`.

```bash
sudo -E ./run_targeted.sh
```

```bash
ncu --import baseline_bandwidth_ncu.ncu-rep --csv --page raw > baseline_bandwidth.csv
```

**Result, averaged over 10 real decode-phase instances:**

| Metric | Value |
|---|---|
| Achieved DRAM bandwidth | **532.4 GB/s** |
| DRAM throughput, % of peak (measured by `ncu`) | **72.1%** |
| SM (compute) throughput, % of peak | **45.7%** |

RTX 3080 published peak is 760 GB/s — 532.4 / 760 = 70.1%, matching `ncu`'s
hardware-counter measurement (72.1%) within a couple points, which is a good sanity
check on the number itself.

**Bottom line: decode is memory-bound, not compute-bound.** The kernel is using ~72% of
available memory bandwidth but only ~46% of available compute. This confirms the target
for Phase 1 should be about moving less data or moving it more efficiently, not raw
arithmetic throughput.

Artifacts: `baseline_decode_nsys.nsys-rep` (nsys capture), `run_nsys_baseline.sh`,
`baseline_bandwidth.csv` (ncu per-kernel metrics), `run_targeted.sh`,
`run_targeted.log`, `baseline_bandwidth_ncu.ncu-rep`.

----

## Phase 1: Target & Hypothesis

**Target:** `mul_mat_vec_q<(ggml_type)12, (int)1, (bool)1, (bool)0>` — the Q4_K batch-1
decode matvec kernel. Dominant kernel from Phase 0: 26.4% of total GPU time, 280
instances per generation.

**Why it's suboptimal — traced to the actual source, then confirmed with `ncu`:**

In `vec_dot_q4_K_q8_1` (`vecdotq.cuh`), each thread reads its weight values as `v[0] =
q4[0]; v[1] = q4[4];` — two 4-byte reads, 16 bytes apart. In the calling loop
(`mmvq.cu`), `iqs` is assigned so that only groups of 4 adjacent threads land in the same
16-byte burst; consecutive groups jump 32 bytes apart instead of staying contiguous.

Ran `ncu` with `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` targeted
at this exact kernel, 5 real decode-phase instances:

| Metric | Result |
|---|---|
| Sector efficiency (bytes used / bytes fetched) | **34.0% average** |
| L1 bank conflicts (sum) | 245,470 average per instance |

Only about a third of every fetched byte is actually used; this explains why Phase 0
measured 72% of *peak bandwidth* while still being memory-bound: a real share of that
bandwidth is spent moving data the kernel immediately discards.

The same 32-38% efficiency also showed up on a second kernel variant (`<14,1,1,0>`, a
different quant type) captured in the same run; this looks like a shared pattern in
`vecdotq.cuh`'s read layout, not a Q4_K-only issue.

**Hypothesis:** reorganizing the warp's memory access, so adjacent threads' reads land
in the same coalesced transaction instead of scattered 32-byte-apart bursts. This should cut
wasted bytes fetched per launch, without changing the arithmetic. Fewer wasted bytes
moved per useful token should show up directly as higher decode tok/s.

**Correctness gate before this counts as done:** `test-backend-ops` must pass, and
generation output must match baseline byte-for-byte at a fixed seed. A faster kernel that
changes output doesn't count.

Artifacts: `run_mem_efficiency_check.sh`, `mem_efficiency_check.ncu-rep`,
`mem_efficiency.csv`.

----

## Phase 2: the int2 + shuffle attempt (and why I reverted it)

**Target:** the `v[0]`/`v[1]` weight load in `vec_dot_q4_K_q8_1`
(`vecdotq.cuh`), the load step of `mul_mat_vec_q<12,...>` — my Phase 1
target, confirmed at 32.87% sector efficiency.

```cpp
const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
v[0] = q4[0];
v[1] = q4[4];
```

Root cause: `v[0]` and `v[1]` are two separate scalar loads, 16 bytes apart. Each one
is its own instruction that only touches half of the 32-byte sector it triggers — the
two together span exactly one sector, but because they're issued separately, each gets
charged for a full sector fetch while using half of it.

**First attempt (int4 broadcast) — didn't move the needle at all.** Had all 4 lanes in
a group load the same 16 bytes, pick their own 4 via a switch. Sector efficiency:
32.87% → 32.87%, no change. Makes sense in hindsight — I changed how the bytes get
handed to each lane, not how many separate memory transactions get issued. Didn't touch
the actual problem.

**Second attempt — int2 load + shuffle.** One 8-byte load per lane, tiling the full
32-byte window across the 4-lane group in a single instruction, then redistributing to
the right lane with `__shfl_sync`:

```cpp
const int lane_in_group = (iqs/2) % 4;
const int q4_group_byte_offset = 16 * bq8_offset;

const int2 chunk = *(const int2 *)(bq4_K->qs + q4_group_byte_offset + 8 * lane_in_group);
const unsigned int mask = __activemask();

const int v0_from_x = __shfl_sync(mask, chunk.x, lane_in_group / 2,     4);
const int v0_from_y = __shfl_sync(mask, chunk.y, lane_in_group / 2,     4);
const int v1_from_x = __shfl_sync(mask, chunk.x, lane_in_group / 2 + 2, 4);
const int v1_from_y = __shfl_sync(mask, chunk.y, lane_in_group / 2 + 2, 4);

v[0] = (lane_in_group % 2 == 0) ? v0_from_x : v0_from_y;
v[1] = (lane_in_group % 2 == 0) ? v1_from_x : v1_from_y;
```

Used `__activemask()` instead of the usual hardcoded `0xffffffff` (the
convention elsewhere in this codebase) — this call site sits inside a
loop in `mmvq.cu` that provably diverges to a partial warp on its last
iteration whenever `blocks_per_row_x` is odd. A full mask there risks an
actual hang on some model shapes, not just wrong output.

**Checked correctness before anything else**, in order:
1. Hand-derived the lane math before writing a line of code.
2. Debug printf comparing shuffle output against the original scalar formula, real
   runtime data: 29,148 matches, 0 mismatches. Reverted the printf once confirmed.
3. `test-backend-ops` on CUDA0: 13,054/13,054 passed.
4. Fixed-seed generation (`--seed 42 --temp 0 -st`): byte-identical to
   baseline.

**Initial data looked like an improvement:**

| Metric | Baseline | int4 broadcast | int2 + shuffle |
|---|---|---|---|
| Sector efficiency (avg) | 32.87% | 32.87% | **36.47%** |
| Sectors loaded (avg) | 4,675,580 | 4,688,624 | **4,095,178** (-12.4%) |
| Bank conflicts (avg) | 178,481.5 | 177,948.2 | 177,511.8 (noise) |

Re-measured baseline for this diagnostic with a fresh sample rather than reusing Phase
1's number (34.0%, 5 instances) — both are real measurements of the same kernel; the
difference is normal run-to-run variance between separate `ncu` sessions.

Fewer sectors fetched, better efficiency, zero correctness cost, this looked done.

**Then I ran `llama-bench` and it wasn't:**

| | Baseline | int2 + shuffle |
|---|---|---|
| Prefill (pp512) | 5229.88 tok/s | 5206.80 tok/s (-0.44%) |
| Decode (tg128) | 134.05 tok/s | **130.24 tok/s (-2.84%)** |

Sample spreads don't overlap (baseline 133.9-134.1, fixed 129.9-130.3) —
this is real, not noise.

**Why:** two reasons.

First, this load site is only part of the kernel's memory traffic. A few lines down,
the activation load (`u[]`, reading `bq8_1->qs`) has the exact same scattered-two-loads
shape and I never touched it. The 12.4% fewer sectors is 12.4% of *this one load*, not
the kernel's total DRAM traffic — the real bandwidth win is smaller than the
sector-efficiency number makes it look.

Second, I wanted a real number here instead of a guess — I profiled instruction count
and occupancy on both builds:

| Metric (avg, 4 instances) | Baseline | int2 + shuffle | Delta |
|---|---|---|---|
| `smsp__inst_executed.sum` | 8,548,736 | 9,698,688 | **+13.45%** |
| Warps active, % of peak | 62.09% | 63.79% | +1.70pp |

Occupancy didn't drop. I had two theories going in (more instructions, or fewer warps
in flight from shuffle register pressure) and this rules the second one out; occupancy
is flat, if anything slightly better. It's just executing 13.45% more instructions per
launch. `__activemask()` + 4 shuffles + 2 selects costs more than the 12.4% memory
saving buys back.  Net: traded a bit of memory traffic for more instruction overhead,
and lost on the wall-clock number that actually matters.

This is exactly the trap the assignment's Phase 3 warns about: a per-kernel proxy
metric got better while the real end-to-end metric got worse. Sector efficiency alone
wasn't enough evidence.

**Closed the loop: did the L1-level "waste" ever cost real DRAM traffic?** Ran one more
`ncu` pass, `dram__bytes.sum` and `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
same 4 instances, both builds:

| Metric (sum/avg, 4 instances) | Baseline | int2 + shuffle | Delta |
|---|---|---|---|
| `dram__bytes.sum` | 93.65 MB | 93.69 MB | **+0.048%** (noise) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 66.70% | 59.45% | **-7.25pp** |

Total bytes moved from DRAM are identical between the two builds. The 32.87%→36.47%
sector-efficiency gain was real at the L1 request level but never turned into fewer
bytes crossing HBM — L1 was already absorbing that "waste" before the fix, so there was
never a DRAM-level saving to capture. The DRAM-throughput% drop is just the mechanical
consequence of the same finding from the instruction-count check: same bytes, but the
shuffle kernel takes longer to move them (the +13.45% extra instructions), so the
average %-of-peak achieved over that longer window is lower.

Full chain, cause to effect: L1 sector efficiency improves -> DRAM bytes moved:
unchanged -> instructions executed: +13.45% -> DRAM throughput%: -7.25% (same bytes,
longer kernel) -> decode tok/s: -2.84%. The fix solved a problem that didn't exist at
the level that actually costs cycles, and the attempt to solve it cost cycles of its
own. This is the fact-checked version of "the fix isn't free" above, not just an
instruction-count inference.

**Why I'm not porting this to `u[]`:** checked first whether the same trick is even
safe there. It isn't, cleanly. `block_q4_K` is 144 bytes — a multiple of 16 — so `qs`
lands on a clean 32-byte boundary for every block, which is what made the `int2` tiling
safe above. `block_q8_1` is 36 bytes, not a multiple of 8 — walking the array, `qs`
alternates between 4-byte-aligned and 8-byte-aligned. `QR4_K=2` means the loop always
hits one of each, so a direct port would issue a misaligned 8-byte load every other
iteration — real undefined behavior, not a style nitpick. `test-backend-ops` passing on
this GPU wouldn't be reliable proof it's actually safe. Given the `v[]`-only version
already regressed, and `u[]` needs more complexity just to be safe, this is where I'm
stopping rather than pushing further.

**Where the code stands:** reverted `vecdotq.cuh` back to baseline. The int2+shuffle
change passes every correctness gate and is a real, repeatable 2.84% regression — kept
as a patch/diff alongside this write-up, not as live code.

Artifacts: `run_regression_diagnosis_baseline.sh`, `int2_shuffle_attempt.patch`,
`run_regression_diagnosis_shuffle.sh`, `baseline_regression_diagnosis.csv`,
`shuffle_regression_diagnosis.csv`, `.ncu-rep` files for both.
`run_dram_diagnosis.sh`, `baseline_dram_diagnosis.csv`/`.ncu-rep`,
`shuffle_dram_diagnosis.csv`/`.ncu-rep`.
