#!/bin/bash
LABEL="$1"

./build/bin/llama-server \
  -m /home/npalmass/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 16384 -np 4 > "${LABEL}.server.log" 2>&1 &
SERVER_PID=$!

sleep 5
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200; do
  sleep 5
done

echo "=== correctness capture: $LABEL ==="
python3 concurrent_bench.py --url http://127.0.0.1:8093 --n-requests 4 --n-predict 128 \
  --correctness-capture "correctness_${LABEL}" > "${LABEL}.correctness.log" 2>&1
cat "${LABEL}.correctness.log"

echo "=== throughput benchmark: $LABEL ==="
python3 concurrent_bench.py --url http://127.0.0.1:8093 --n-requests 4 --n-predict 128 --reps 5 \
  > "${LABEL}.bench.log" 2>&1
cat "${LABEL}.bench.log"

kill -INT "$SERVER_PID" 2>/dev/null || true
sleep 2
kill -9 "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
