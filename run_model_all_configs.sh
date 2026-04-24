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


# FAST CONFIGS (No 4over6)
# 9. e4m3 + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false false true
# 10. e4m3 + PZ + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false false true
# 13. ue5m3 + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false false true true
# 14. ue5m3 + PZ + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true false true true

# SLOW CONFIGS (With 4over6)
# 11. e4m3 + 4o6 + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false true false true
# 12. e4m3 + 4o6 + PZ + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true false true
# 15. ue5m3 + 4o6 + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES false true true true
# 16. ue5m3 + 4o6 + PZ + H
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES true true true true

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
python3 cleanup_and_plot.py

echo "Master script execution completed!"
