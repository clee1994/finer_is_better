#!/bin/bash
# Launches the absolute zone ablation heatmap sweeps in parallel across multiple compute instances!

echo "Deploying the finalized zone ablation script to paper-eval-vm1 and paper-eval-vm6 over SUP relay..."

# Deploy to VM1
cat zone_ablation_eval.py | gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="cat > ~/zone_ablation_eval.py" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

# Deploy to VM6
cat zone_ablation_eval.py | gcloud compute ssh paper-eval-vm6 --project supercomputer-testing --zone us-central1-c --command="cat > ~/zone_ablation_eval.py" -- -o Hostname=nic0.paper-eval-vm6.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

echo "Deployments complete. Launching parallel heatmap sweeps concurrently..."

# Launch Granite sweep on VM1
gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="nohup /home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/zone_ablation_eval.py sweep None Granite-3.3-8B > ~/ablation_sweep_granite.log 2>&1 &" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com &

# Launch Qwen sweep on VM6
gcloud compute ssh paper-eval-vm6 --project supercomputer-testing --zone us-central1-c --command="nohup /home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/zone_ablation_eval.py sweep None Qwen2.5-14B > ~/ablation_sweep_qwen.log 2>&1 &" -- -o Hostname=nic0.paper-eval-vm6.us-central1-c.c.supercomputer-testing.internal.gcpnode.com &

wait
echo "Parallel sweeps successfully dispatched to both compute instances! Monitor progress via their respective log files."
