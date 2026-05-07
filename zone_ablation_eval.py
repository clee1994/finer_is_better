import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys

class ZoneAblationLinear(nn.Module):
    """
    Custom linear layer that zeroes out weights and inputs whose absolute magnitudes
    fall within the specified absolute band [b_thresh, a_thresh], leaving all other
    values untouched in full precision.
    """
    def __init__(self, in_features, out_features, bias=True, b_thresh=0.0, a_thresh=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.b_thresh = b_thresh
        self.a_thresh = a_thresh
        
    def forward(self, input):
        # Create masks for values whose absolute magnitude falls inside [B, A]
        w_abs = torch.abs(self.weight)
        w_mask = (w_abs >= self.b_thresh) & (w_abs <= self.a_thresh)
        masked_weight = torch.where(w_mask, torch.zeros_like(self.weight), self.weight)
        
        i_abs = torch.abs(input)
        i_mask = (i_abs >= self.b_thresh) & (i_abs <= self.a_thresh)
        masked_input = torch.where(i_mask, torch.zeros_like(input), input)
        
        return F.linear(masked_input, masked_weight, self.bias)

def run_zone_ablation(model_id, b_thresh, a_thresh, num_steps=None):
    print(f"\nRunning Ablation on {model_id} | Wiping out absolute zone: [{b_thresh}, {a_thresh}]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    
    # Identify output head to skip it
    head_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            head_name = name
    print(f"Identified head layer (skipping): {head_name}")
    
    # Replace standard Linear layers (skipping attention and head)
    replaced_count = 0
    for name, module in model.named_modules():
        if name == head_name:
            continue
        if "attn" in name:
            continue
        if isinstance(module, nn.Linear):
            idx = name.rfind(".")
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            father_module = model
            if father_name:
                for part in father_name.split("."):
                    father_module = getattr(father_module, part)
            
            idx = idx + 1 if idx != 0 else idx
            new_m = ZoneAblationLinear(module.in_features, module.out_features, module.bias is not None, b_thresh=b_thresh, a_thresh=a_thresh)
            new_m.weight.data = module.weight.data
            if module.bias is not None:
                new_m.bias.data = module.bias.data
                
            setattr(father_module, name[idx:], new_m)
            replaced_count += 1
            
    print(f"Successfully replaced {replaced_count} layers with ZoneAblationLinear.")
    
    # Prepare WikiText-2 test dataset
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    seq_len = 2048
    stride = 2048
    nlls = []
    
    loop = range(0, encodings.input_ids.size(1), stride)
    if num_steps is not None:
        loop = zip(loop, range(num_steps))
        
    for item in tqdm(loop, desc="Evaluating Perplexity"):
        if num_steps is not None:
            i, _ = item
        else:
            i = item
            
        begin_loc = i
        end_loc = min(i + seq_len, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)
            
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"Ablation Perplexity: {ppl:.4f}")
    
    del model
    torch.cuda.empty_cache()
    return ppl

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def apply_style():
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['figure.titlesize'] = 22
    mpl.rcParams['figure.figsize'] = (16.0, 8.0)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300

def plot_heatmap(matrix_df, model_name, ax):
    data = matrix_df.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Perplexity Gap', fontweight='bold')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontweight="bold", size=14)
                
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, fontweight='bold')
    ax.set_yticklabels(matrix_df.index, fontweight='bold')
    
    ax.set_title(f"Zone Ablation Sensitivity: {model_name}", fontweight='bold', pad=15)
    ax.set_ylabel("Lower Threshold (B)", fontweight='bold')
    ax.set_xlabel("Upper Threshold (A)", fontweight='bold')

def run_ablation_heatmap_sweep(num_steps=None, target_model=None):
    apply_style()
    
    models = {
        "Granite-3.3-8B": "ibm-granite/granite-3.3-8b-base",
        "Qwen2.5-14B": "Qwen/Qwen2.5-14B"
    }
    
    if target_model is not None and target_model in models:
        models = {target_model: models[target_model]}
        print(f"Filtering sweep strictly for target model: {target_model}")
        
    t_abs = [0.0, 0.003, 0.01, 0.05, 10000.0]
    labels = ["0.0", "0.003", "0.01", "0.05", "inf"]
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Adjust subplot layout dynamically depending on number of models being evaluated
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 8))
    if num_models == 1:
        axes = [axes]
        
    all_records = []
    
    for idx, (name, model_id) in enumerate(models.items()):
        print(f"\n{'='*60}\nStarting Heatmap Sweep for {name}\n{'='*60}")
        
        print("Computing baseline full-precision perplexity...")
        base_ppl = run_zone_ablation(model_id, 0.0, 0.0, num_steps=num_steps)
        print(f"Baseline Perplexity established: {base_ppl:.4f}")
        
        matrix = pd.DataFrame(np.nan, index=labels[:-1], columns=labels[1:])
        
        for i in range(len(t_abs) - 1):
            for j in range(i + 1, len(t_abs)):
                b = t_abs[i]
                a = t_abs[j]
                b_lbl = labels[i]
                a_lbl = labels[j]
                
                ppl = run_zone_ablation(model_id, b, a, num_steps=num_steps)
                gap = ppl - base_ppl
                matrix.loc[b_lbl, a_lbl] = gap
                
                all_records.append({
                    "Model": name,
                    "B_Thresh": b_lbl,
                    "A_Thresh": a_lbl,
                    "Ablation_PPL": ppl,
                    "Perplexity_Gap": gap
                })
                
        matrix.to_csv(f"results/ablation_heatmap_{name}.csv")
        plot_heatmap(matrix, name, axes[idx])
        
    plt.tight_layout()
    out_img = f"plots/zone_ablation_heatmap_{target_model}.png" if target_model is not None else "plots/zone_ablation_heatmap.png"
    plt.savefig(out_img, bbox_inches='tight')
    print(f"\nSuccessfully generated heatmap plot at {out_img}")
    
    df = pd.DataFrame(all_records)
    out_csv = f"results/zone_ablation_records_{target_model}.csv" if target_model is not None else "results/zone_ablation_records.csv"
    df.to_csv(out_csv, index=False)

def test_ablation_module():
    print("Running local test on ZoneAblationLinear module...")
    layer = ZoneAblationLinear(4, 4, b_thresh=0.01, a_thresh=0.05)
    layer.weight.data = torch.tensor([
        [0.001, 0.02, -0.03, 0.1],
        [-0.005, -0.04, 0.05, -0.5],
        [0.0, 0.01, 0.05, 0.0],
        [1.0, 0.03, -0.025, 0.0001]
    ], dtype=torch.float32)
    layer.bias.data = torch.zeros(4)
    
    x = torch.tensor([[0.002, 0.03, -0.1, 0.04]], dtype=torch.float32)
    out = layer(x)
    print("Execution successful! Forward pass complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_ablation_module()
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        steps = None
        t_model = None
        if len(sys.argv) > 2:
            try:
                steps = int(sys.argv[2])
            except ValueError:
                t_model = sys.argv[2]
        if len(sys.argv) > 3:
            t_model = sys.argv[3]
            
        run_ablation_heatmap_sweep(num_steps=steps, target_model=t_model)
    else:
        print("Usage:\n  python3 zone_ablation_eval.py test\n  python3 zone_ablation_eval.py sweep [num_steps] [target_model]")
