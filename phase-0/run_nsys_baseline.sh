#!/bin/bash
set -e

nsys profile -o baseline_decode_nsys --stats=true --trace=cuda,osrt --force-overwrite=true \
  ./build/bin/llama-server -m ~/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 4096 &
NSYS_PID=$!

echo "Waiting for server to load..."
sleep 10

echo "Sending completion request..."
curl -s http://127.0.0.1:8093/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that finds two numbers in a list that sum to a target, then explain the time complexity in detail.", "n_predict": 300, "stream": false}' \
  | python3 -m json.tool

echo "Stopping profiler/server..."
kill -INT "$NSYS_PID" 2>/dev/null || true
sleep 3
kill -9 "$NSYS_PID" 2>/dev/null || true
wait "$NSYS_PID" 2>/dev/null || true

echo "Done. Checking for output file:"
ls -la baseline_decode_nsys.*


nsys stats --force-export=true baseline_decode_nsys.nsys-rep
