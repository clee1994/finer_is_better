#!/bin/bash
# Queues the ultra-dense 19x19 overnight absolute magnitude zone ablation sweeps!

echo "Queuing the 4.75-hour overnight ultra-dense ablation sweeps. Waiting completely hands-off for active 1-hour runs to exit..."

# Loop silently checking for active ablation processes on VM1
while gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="ps aux | grep '[t]orch_eval.py'" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com | grep "ablation" > /dev/null; do
  sleep 60
done

echo "Active sweeps have officially exited! Launching the 190 concurrent background points..."

# Deploy latest dense scripts to VM1
cat torch_eval.py | gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="cat > ~/torch_eval.py" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com
cat plot_ablation_heatmaps.py | gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="cat > ~/results/plot_ablation_heatmaps.py" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

# Clear out intermediate old matrices on VM1 to enforce fresh initialization
gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="rm -f ~/results/ablation_heatmap_*.csv" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

# Launch the 190 combinations dynamically in remote background worker loops
gcloud compute ssh paper-eval-vm1 --project supercomputer-testing --zone us-central1-c --command="nohup bash -c '
t_dense=(0.0 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.0015 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.01 0.012 0.016 0.024 0.035)

# Granite worker
(
  echo \"Starting Granite overnight 190-point sweep...\"
  for ((i=0; i<\${#t_dense[@]}-1; i++)); do
    for ((j=i+1; j<\${#t_dense[@]}; j++)); do
      b=\${t_dense[i]}
      a=\${t_dense[j]}
      echo \"Granite: Evaluating dense zone [\$b, \$a]...\"
      /home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/torch_eval.py ibm-granite/granite-3.3-8b-base ablation \$b \$a
    done
  done
  echo \"Granite dense sweep completed!\"
) > ~/overnight_sweep_granite.log 2>&1 &

# Qwen worker concurrently
(
  echo \"Starting Qwen overnight 190-point sweep...\"
  for ((i=0; i<\${#t_dense[@]}-1; i++)); do
    for ((j=i+1; j<\${#t_dense[@]}; j++)); do
      b=\${t_dense[i]}
      a=\${t_dense[j]}
      echo \"Qwen: Evaluating dense zone [\$b, \$a]...\"
      /home/cjsschaefer_google_com/quant_eval_venv/bin/python3 ~/torch_eval.py Qwen/Qwen2.5-14B ablation \$b \$a
    done
  done
  echo \"Qwen dense sweep completed! Drawing heatmap...\"
  cd ~/results && /home/cjsschaefer_google_com/quant_eval_venv/bin/python3 plot_ablation_heatmaps.py
) > ~/overnight_sweep_qwen.log 2>&1 &

' > ~/overnight_dispatch.log 2>&1 &" -- -o Hostname=nic0.paper-eval-vm1.us-central1-c.c.supercomputer-testing.internal.gcpnode.com

echo "Overnight queuing daemon successfully detached! It will wake up the moment the current runs drop and dispatch the full 190-point combinations."
