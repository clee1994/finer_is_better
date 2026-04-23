#!/bin/bash
set -e

echo "Pulling latest changes..."
git pull

echo "Activating environment..."
source ~/quant_eval_venv/bin/activate

MODEL_ID=$1
BLOCK_SIZES=$2
PREVENT_ZERO=$3
FOUR_OVER_SIX=$4
USE_UE5M3=$5

OPTION="e4m3"
if [ "$FOUR_OVER_SIX" = "true" ]; then
    OPTION="4over6"
fi
if [ "$USE_UE5M3" = "true" ]; then
    OPTION="ue5m3"
fi

if [ "$PREVENT_ZERO" = "true" ]; then
    OPTION="${OPTION}_pz"
else
    OPTION="${OPTION}_no_pz"
fi

echo "Running evaluation for $MODEL_ID with option $OPTION..."
python3 torch_eval.py $MODEL_ID $BLOCK_SIZES $PREVENT_ZERO $FOUR_OVER_SIX $USE_UE5M3

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

echo "Committing and pushing results for $DISP_NAME..."
git add results_${DISP_NAME}.csv
git commit -m "Add results for $DISP_NAME ($OPTION)"
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
        echo "Waiting 1 minute for all results to be pushed..."
        sleep 60
    fi
done

echo "All results present! Running final aggregation..."
python3 aggregate_results.py

echo "Master script execution completed!"
