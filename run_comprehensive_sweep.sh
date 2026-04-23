#!/bin/bash
source ~/quant_eval_venv/bin/activate
cd ~/finer_is_better

echo "Running Option 1: nvfp4 (no pz)"
python3 torch_eval.py meta-llama/Llama-3.1-8B 4,32 false false false

echo "Running Option 2: nvfp4_pz"
python3 torch_eval.py meta-llama/Llama-3.1-8B 4,32 true false false

echo "Running Option 3: 4over6 (no pz)"
python3 torch_eval.py meta-llama/Llama-3.1-8B 4,32 false true false

echo "Running Option 4: 4over6_pz"
python3 torch_eval.py meta-llama/Llama-3.1-8B 4,32 true true false

echo "Running Option 5: ue5m3_pz"
python3 torch_eval.py meta-llama/Llama-3.1-8B 4,32 true false true

echo "All sweeps completed!"
