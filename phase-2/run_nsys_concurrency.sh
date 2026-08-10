#!/bin/bash
set -e

SESSION_NAME="concurrent_decode_session"

nsys profile -o concurrent_decode_nsys --stats=true --trace=cuda,osrt --force-overwrite=true \
  --session-new="$SESSION_NAME" \
  ./build/bin/llama-server -m ~/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 16384 -np 4 &
NSYS_PID=$!

echo "Waiting for server to load..."
sleep 10
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200; do
  sleep 5
done

echo "Sending 4 concurrent completion requests..."
CURL_PIDS=()
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8093/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a Python function that finds two numbers in a list that sum to a target, then explain the time complexity in detail.", "n_predict": 300, "stream": false}' \
    -o "/tmp/concurrent_nsys_resp_$i.json" &
  CURL_PIDS+=($!)
done
wait "${CURL_PIDS[@]}"

echo "Shutting down profiler session and target cleanly..."
nsys shutdown --session="$SESSION_NAME" --kill=sigterm
sleep 3
wait "$NSYS_PID" 2>/dev/null || true

echo "Done. Checking for output file:"
ls -la concurrent_decode_nsys.*

nsys stats --force-export=true concurrent_decode_nsys.nsys-rep
