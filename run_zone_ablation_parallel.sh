#!/bin/bash
# Launches the integrated absolute zone ablation heatmap sweeps in parallel across multiple compute instances!

echo "Deploying the updated integrated torch_eval.py driver to paper-eval-vm1 and paper-eval-vm6 over SUP relay..."

# Deploy to VM1
cat torch_eval.py | gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="cat > ~/torch_eval.py" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

# Deploy to VM6
cat torch_eval.py | gcloud compute ssh paper-eval-vm6 --project supercomputer-testing --zone us-central1-c --command="cat > ~/torch_eval.py" -- -o Hostname=nic0.paper-eval-vm6.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

echo "Deployments complete. Launching integrated concurrent heatmap points..."

# Threshold combinations (B < A)
t_pairs=(
  "0.0 0.003" "0.0 0.01" "0.0 0.05" "0.0 10000.0"
  "0.003 0.01" "0.003 0.05" "0.003 10000.0"
  "0.01 0.05" "0.01 10000.0"
  "0.05 10000.0"
)

# Dispatch Granite on VM1 sequentially in a background worker loop
(
  for pair in "${t_pairs[@]}"; do
    read b a <<< "$pair"
    echo "VM1: Dispatching Granite ablation for zone [$b, $a]..."
    gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c \
      --command="/home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/torch_eval.py ibm-granite/granite-3.3-8b-base ablation $b $a" \
      -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com
  done
  echo "VM1: Granite integrated ablation heatmap matrix populated successfully!"
) > ablation_sweep_granite.log 2>&1 &

# Dispatch Qwen on VM6 sequentially in a background worker loop
(
  for pair in "${t_pairs[@]}"; do
    read b a <<< "$pair"
    echo "VM6: Dispatching Qwen ablation for zone [$b, $a]..."
    gcloud compute ssh paper-eval-vm6 --project supercomputer-testing --zone us-central1-c \
      --command="/home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/torch_eval.py Qwen/Qwen2.5-14B ablation $b $a" \
      -- -o Hostname=nic0.paper-eval-vm6.us-central1-c.c.supercomputer-testing.internal.gcpnode.com
  done
  echo "VM6: Qwen integrated ablation heatmap matrix populated successfully!"
) > ablation_sweep_qwen.log 2>&1 &

echo "Parallel execution loop dispatched concurrently to both compute instances! Monitor progress in ablation_sweep_granite.log and ablation_sweep_qwen.log."
