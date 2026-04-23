#!/bin/bash
set -e

echo "Pulling latest changes..."
git pull

echo "Activating environment..."
source ~/quant_eval_venv/bin/activate

MODEL_ID=$1
BLOCK_SIZES="4,8,16,32,64,128,256"

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

echo "Running evaluation for $MODEL_ID across all configs..."

# 1. e4m3 (no pz)
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false false

# 2. e4m3 + PZ
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false false

# 3. ue5m3 (no pz)
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false true

# 4. ue5m3 + PZ
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false true

# 5. 4over6 (no pz)
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false true false

# 6. 4over6 + PZ
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true false

# 7. 4over6 + PZ + ue5m3
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true true

echo "Committing and pushing results for $DISP_NAME..."
git add results_${DISP_NAME}.csv
git commit -m "Add final results for $DISP_NAME"
git push

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

echo "Master script execution completed!"
