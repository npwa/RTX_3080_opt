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

This confirms the model fits 100% on GPU. Also verfied from log entries:  
`Aug 07 21:41:01 npalmass-desk1 ollama[1452130]: slot print_timing: id 0 | task 60 | eval time = 193.17 ms / 23 tokens ( 8.40 ms per token, 119.07 tokens per second)`  
if there was a CPU offload, tok/sec would be much slower, around 20 tok/sec.

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
ncu --import baseline_bandwidth_ncu4.ncu-rep --csv --page raw > baseline_bandwidth.csv
```

**Result, averaged over 10 real decode-phase instances:**

| Metric | Value |
|---|---|
| Achieved DRAM bandwidth | **532.4 GB/s** |
| DRAM throughput, % of peak (measured by `ncu`) | **72.1%** |
| SM (compute) throughput, % of peak | **45.7%** |

RTX 3080 published peak is 760 GB/s — 532.4 / 760 = 70.1%, matching `ncu`'s own
hardware-counter measurement (72.1%) within a couple points, which is a good sanity
check on the number itself.

**Bottom line: decode is memory-bound, not compute-bound.** The kernel is using ~72% of
available memory bandwidth but only ~46% of available compute — confirms the target for
Phase 1 should be about moving less data or moving it more efficiently, not raw
arithmetic throughput.

Artifacts: `baseline_decode_nsys.nsys-rep` (nsys capture), `run_nsys_baseline.sh`,
`baseline_bandwidth.csv` (ncu per-kernel metrics), `run_targeted.sh`, `run_targeted.log`.
