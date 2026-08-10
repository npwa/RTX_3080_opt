#!/bin/bash
OUTNAME="$1"

sudo -E ncu --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:mul_mat_vec_q \
  --launch-skip 500 --launch-count 20 \
  -o "$OUTNAME" --force-overwrite \
  ./build/bin/llama-server -m /home/npalmass/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 16384 -np 4 &
NCU_PID=$!

sleep 30
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200; do
  sleep 10
done

echo "Sending 4 concurrent completion requests..."
CURL_PIDS=()
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8093/completion \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a Python function that finds two numbers in a list that sum to a target, then explain the time complexity in detail.", "n_predict": 300, "stream": false}' \
    -o "/tmp/dram_conc_resp_$i.json" &
  CURL_PIDS+=($!)
done
wait "${CURL_PIDS[@]}"

sleep 20

sudo kill -INT "$NCU_PID" 2>/dev/null || true
sleep 3
sudo kill -9 "$NCU_PID" 2>/dev/null || true
wait "$NCU_PID" 2>/dev/null || true

ls -la "$OUTNAME".ncu-rep*
