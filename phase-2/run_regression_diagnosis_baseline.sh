#!/bin/bash

sudo -E ncu --metrics smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name regex:mul_mat_vec_q \
  --launch-skip 500 --launch-count 5 \
  -o baseline_regression_diagnosis --force-overwrite \
  ./build/bin/llama-server -m /home/npalmass/work/inf_opt/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8093 --host 127.0.0.1 -ngl 99 -c 4096 &
NCU_PID=$!

sleep 30
until curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200; do
  sleep 10
done

curl -s http://127.0.0.1:8093/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function that finds two numbers in a list that sum to a target, then explain the time complexity in detail.", "n_predict": 300, "stream": false}' \
  | python3 -m json.tool

sleep 20

sudo kill -INT "$NCU_PID" 2>/dev/null || true
sleep 3
sudo kill -9 "$NCU_PID" 2>/dev/null || true
wait "$NCU_PID" 2>/dev/null || true

ls -la baseline_regression_diagnosis.ncu-rep*
