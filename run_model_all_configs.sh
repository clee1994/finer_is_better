#!/bin/bash
set -e

echo "Pulling latest changes..."
git pull

echo "Activating environment..."
source ~/quant_eval_venv/bin/activate

MODEL_ID=$1
# Run full dataset (no step limit) by omitting 6th argument!
BLOCK_SIZES="4,8,16,32,64,128,256"

echo "Running evaluation for $MODEL_ID across all configs..."

# FAST CONFIGS FIRST
echo "1. e4m3 (no pz)"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false false

echo "2. e4m3 + PZ"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false false

echo "3. 4over6 (no pz)"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false true false

echo "4. 4over6 + PZ"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true false

# SLOW CONFIGS LAST (ue5m3)
echo "5. ue5m3 (no pz)"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false true

echo "6. ue5m3 + PZ"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false true

echo "7. 4over6 + PZ + ue5m3"
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true true

DISP_NAME="unknown"
if [[ "$MODEL_ID" == *"llama"* ]]; then
    DISP_NAME="llama"
elif [[ "$MODEL_ID" == *"granite"* ]]; then
    DISP_NAME="granite"
elif [[ "$MODEL_ID" == *"qwen"* ]]; then
    DISP_NAME="qwen"
elif [[ "$MODEL_ID" == *"deepseek"* ]]; then
    DISP_NAME="deepseek"
fi

# Polling for all results
EXPECTED_FILES=("results_llama.csv" "results_granite.csv" "results_qwen.csv" "results_deepseek.csv")

all_present=false
while [ "$all_present" = false ]; do
    echo "Syncing repo to check for other results..."
    git pull
    
    all_present=true
    for f in "${EXPECTED_FILES[@]}"; do
        if [ ! -f "$f" ]; then
            all_present=false
            echo "Missing $f"
            break
        fi
    done
    
    if [ "$all_present" = false ]; then
        echo "Waiting 5 minutes for all results to be pushed..."
        sleep 300
    fi
done

echo "All results present! Running final aggregation..."
python3 aggregate_results.py

echo "Master script execution completed for $DISP_NAME!"
