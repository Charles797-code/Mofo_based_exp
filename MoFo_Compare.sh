#!/bin/bash

# MoFo_Compare.sh - Compare original MoFo with MoFo_Circulant

echo "=============================================="
echo "Running Original MoFo on ETTh1..."
echo "=============================================="
python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" \
  --strategy-args '{"horizon": 96}' \
  --model-name "time_series_library.MoFo" \
  --model-hyper-params '{"batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01, "norm": true, "seq_len": 336, "patience": 10, "periodic": 24, "bias": 1, "cias": 1}' \
  --adapter "MoFo_adapter" \
  --gpus 0 --num-workers 1 --timeout 60000 \
  --save-path "ETTh1/MoFo"

echo ""
echo "=============================================="
echo "Running MoFo_Circulant on ETTh1..."
echo "=============================================="
python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "ETTh1.csv" \
  --strategy-args '{"horizon": 96}' \
  --model-name "time_series_library.MoFo_Circulant" \
  --model-hyper-params '{"batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01, "norm": true, "seq_len": 336, "patience": 10, "periodic": 24, "bias": 1, "cias": 1, "lambda_init": 0.1, "use_causal_mask": false}' \
  --adapter "MoFo_Circulant_adapter" \
  --gpus 0 --num-workers 1 --timeout 60000 \
  --save-path "ETTh1/MoFo_Circulant"

echo ""
echo "=============================================="
echo "Comparison complete. Check results in:"
echo "  ETTh1/MoFo/       (Original MoFo)"
echo "  ETTh1/MoFo_Circulant/  (MoFo with Circulant Regularization)"
echo "=============================================="
