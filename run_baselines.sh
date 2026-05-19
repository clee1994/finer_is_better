#!/bin/bash
set -e

# Avoid git credential prompts
export SKIP_GIT=true

MODELS=("$@")
for model_id in "${MODELS[@]}"; do
  echo "================================================================================"
  echo "Starting evaluations for model: $model_id"
  echo "================================================================================"
  
  # Config 1: int8 | bf16 | 512
  echo "--- Config 1: int8 | bf16 | 512 ---"
  python3 torch_eval.py "$model_id" 512 true false bf16 false none "" false int8 none none
  
  # Config 2: int8 | bf16 | 512 | hadamard 256
  echo "--- Config 2: int8 | bf16 | 512 | hadamard 256 ---"
  python3 torch_eval.py "$model_id" 512 true false bf16 false none "" false int8 256 42
  
  # Config 3: int4 | bf16 | 512
  echo "--- Config 3: int4 | bf16 | 512 ---"
  python3 torch_eval.py "$model_id" 512 true false bf16 false none "" false int4 none none
  
  # Config 4: int4 | bf16 | 512 | hadamard 256
  echo "--- Config 4: int4 | bf16 | 512 | hadamard 256 ---"
  python3 torch_eval.py "$model_id" 512 true false bf16 false none "" false int4 256 42
  
  # Config 5: nvfp4 | e4m3 | 16
  echo "--- Config 5: nvfp4 | e4m3 | 16 ---"
  python3 torch_eval.py "$model_id" 16 true false e4m3 false none "" false e2m1 none none
  
  # Config 6: mxfp4 | e8m0 | 32
  echo "--- Config 6: mxfp4 | e8m0 | 32 ---"
  python3 torch_eval.py "$model_id" 32 true false e8m0 false none "" false e2m1 none none
  
  # Config 7: mxfp4 (e1m2) | e8m0 | 32
  echo "--- Config 7: mxfp4 (e1m2) | e8m0 | 32 ---"
  python3 torch_eval.py "$model_id" 32 true false e8m0 false none "" false e1m2 none none

  echo "Finished evaluations for model: $model_id"
done
