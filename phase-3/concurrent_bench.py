#!/usr/bin/env python3
"""Minimal concurrent-request throughput benchmark for llama-server.

Fires N completion requests concurrently (not sequentially), measures
aggregate decode tok/s over one wall-clock window per rep, repeats for
`reps` and reports the median (matching this project's llama-bench
convention of reporting median-of-N rather than mean).
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def fire_one(url, prompt, n_predict, seed):
    t0 = time.time()
    resp = requests.post(
        f"{url}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "seed": seed,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "latency_s": time.time() - t0,
        "tokens_predicted": data["tokens_predicted"],
        "predicted_per_second": data["timings"]["predicted_per_second"],
    }


def run_rep(url, n_requests, prompt, n_predict):
    with ThreadPoolExecutor(max_workers=n_requests) as pool:
        t0 = time.time()
        futures = [
            pool.submit(fire_one, url, prompt, n_predict, seed=100 + i)
            for i in range(n_requests)
        ]
        results = [f.result() for f in futures]
        wall_s = time.time() - t0
    total_tokens = sum(r["tokens_predicted"] for r in results)
    aggregate_tok_s = total_tokens / wall_s
    return aggregate_tok_s, wall_s, total_tokens, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--prompt",
        default="Write a Python function that finds two numbers in a list that sum to a target, then explain the time complexity in detail.",
    )
    args = ap.parse_args()

    samples = []
    for rep in range(args.reps):
        agg, wall_s, total_tokens, results = run_rep(
            args.url, args.n_requests, args.prompt, args.n_predict
        )
        per_req = ", ".join(f"{r['predicted_per_second']:.2f}" for r in results)
        print(
            f"rep {rep}: aggregate={agg:.2f} tok/s  wall={wall_s:.3f}s  "
            f"total_tokens={total_tokens}  per-request_tok/s=[{per_req}]"
        )
        samples.append(agg)

    print()
    print(f"samples: {samples}")
    print(f"median aggregate tok/s: {statistics.median(samples):.2f}")


if __name__ == "__main__":
    main()
