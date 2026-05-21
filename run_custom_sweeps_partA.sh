#!/bin/bash
REPO_DIR="/home/cjsschaefer_google_com/finer_is_better_repo"
PYTHON_ENV="/home/cjsschaefer_google_com/quant_eval_venv/bin/python3"

cd $REPO_DIR
export SKIP_GIT=true

echo "Starting GCE VM1 Custom Rotation Sweeps PART A (Granite & Llama) at $(date)"

MODELS=(
  "ibm-granite/granite-3.3-8b-base"
  "meta-llama/Llama-3.1-8B"
)

MODEL_NAMES=(
  "Granite"
  "Llama"
)

FP4_ROTATIONS=(
  "fp4_tilted_2s"
  "fp4_tilted_3s"
  "fp4_tilted_5s"
  "fp4_rot_2s"
  "fp4_rot_3s"
  "fp4_rot_5s"
)

FP4_ROT_LABELS=(
  "Tilted_2s"
  "Tilted_3s"
  "Tilted_5s"
  "Rot_2s"
  "Rot_3s"
  "Rot_5s"
)

INT4_ROTATIONS=(
  "int4_tilted_2s"
  "int4_tilted_3s"
  "int4_tilted_5s"
  "int4_rot_2s"
  "int4_rot_3s"
  "int4_rot_5s"
  "int4_ks_320p"
  "int4_rot_80p"
)

INT4_ROT_LABELS=(
  "Tilted_2s"
  "Tilted_3s"
  "Tilted_5s"
  "Rot_2s"
  "Rot_3s"
  "Rot_5s"
  "KS_320p"
  "Rot_80p"
)

# Run FP4 (e2m1)
for i in "${!MODELS[@]}"; do
  model_id="${MODELS[$i]}"
  model_name="${MODEL_NAMES[$i]}"
  for j in "${!FP4_ROTATIONS[@]}"; do
    rot="${FP4_ROTATIONS[$j]}"
    label="${FP4_ROT_LABELS[$j]}"
    log_file="${model_name}_mxfp4_e8m0_scale,_BS_32___${label}.log"
    echo "Running ${model_name} FP4 rotation: ${rot} -> ${log_file} at $(date)"
    $PYTHON_ENV torch_eval.py "$model_id" 32 true false false false none "" true e2m1 none none none none "$rot" > "$log_file" 2>&1
  done
done

# Run INT4 (e1m2)
for i in "${!MODELS[@]}"; do
  model_id="${MODELS[$i]}"
  model_name="${MODEL_NAMES[$i]}"
  for j in "${!INT4_ROTATIONS[@]}"; do
    rot="${INT4_ROTATIONS[$j]}"
    label="${INT4_ROT_LABELS[$j]}"
    log_file="${model_name}_mxfp4_e1m2_e8m0_scale,_BS_32___${label}.log"
    echo "Running ${model_name} INT4 rotation: ${rot} -> ${log_file} at $(date)"
    $PYTHON_ENV torch_eval.py "$model_id" 32 true false false false none "" true e1m2 none none none none "$rot" > "$log_file" 2>&1
  done
done

echo "Finished GCE VM1 Custom Rotation Sweeps PART A at $(date)"
