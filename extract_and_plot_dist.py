import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys

def apply_style():
  """Configures matplotlib for publication-quality plots (NeurIPS style)."""
  mpl.rcParams.update(mpl.rcParamsDefault)
  mpl.rcParams['font.family'] = 'serif'
  mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
  mpl.rcParams['font.size'] = 16
  mpl.rcParams['axes.labelsize'] = 16
  mpl.rcParams['axes.titlesize'] = 16
  mpl.rcParams['xtick.labelsize'] = 14
  mpl.rcParams['ytick.labelsize'] = 14
  mpl.rcParams['legend.fontsize'] = 14
  mpl.rcParams['figure.titlesize'] = 18
  mpl.rcParams['figure.figsize'] = (8.0, 6.0)
  mpl.rcParams['figure.dpi'] = 300
  mpl.rcParams['savefig.dpi'] = 300
  mpl.rcParams['figure.constrained_layout.use'] = True
  mpl.rcParams['axes.spines.top'] = False
  mpl.rcParams['axes.spines.right'] = False
  mpl.rcParams['axes.linewidth'] = 1.0
  mpl.rcParams['axes.edgecolor'] = 'black'
  mpl.rcParams['xtick.major.width'] = 1.0
  mpl.rcParams['ytick.major.width'] = 1.0
  mpl.rcParams['axes.grid'] = True
  mpl.rcParams['grid.alpha'] = 0.3
  mpl.rcParams['grid.linestyle'] = '--'
  mpl.rcParams['grid.linewidth'] = 0.8
  mpl.rcParams['legend.frameon'] = False
  mpl.rcParams['legend.loc'] = 'best'
  mpl.rcParams['savefig.bbox'] = 'tight'
  mpl.rcParams['savefig.pad_inches'] = 0.05

def plot_ridge_contrast(qwen_data, granite_data, layers, title, output_path):
    """Plots histograms with log-scale X-axis and outliers detected from original data."""
    apply_style()
    
    fig, ax = plt.subplots()
    
    num_layers = len(layers)
    overlap = 0.3
    
    c_qwen = '#4C72B0'
    c_granite = '#DD8452'
    
    for i, layer in enumerate(reversed(layers)):
        y_offset = i * overlap
        
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        
        # Plot Qwen
        if layer_name in qwen_data:
            data = qwen_data[layer_name]
            flat_orig = data.flatten()
            
            # Detect outliers BEFORE taking absolute value!
            # 4 most positive and 4 most negative
            pos_outliers = np.partition(flat_orig, -4)[-4:]
            neg_outliers = np.partition(flat_orig, 4)[:4]
            
            flat_data = np.abs(flat_orig)
            flat_data = flat_data[flat_data > 0]
            
            if len(flat_data) > 0:
                log_data = np.log10(flat_data)
                counts, bins = np.histogram(log_data, bins=100, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                norm_counts = counts / np.max(counts)
                
                lin_centers = 10**bin_centers
                
                ax.fill_between(lin_centers, y_offset, norm_counts + y_offset, color=c_qwen, alpha=0.3, label="Qwen" if i == 0 else "")
                ax.plot(lin_centers, norm_counts + y_offset, color=c_qwen, linewidth=1.5, alpha=1.0)
                
                # Plot outliers (using magnitude for X-axis, but identified from original data)
                # 'x' for positive outliers, 'o' for negative outliers
                ax.plot(np.abs(pos_outliers), [y_offset] * 4, 'x', color=c_qwen, markersize=8, markeredgewidth=1.5)
                ax.plot(np.abs(neg_outliers), [y_offset] * 4, 'o', color=c_qwen, markersize=6, markerfacecolor='none', markeredgecolor=c_qwen)
                
        # Plot Granite
        if layer_name in granite_data:
            data = granite_data[layer_name]
            flat_orig = data.flatten()
            
            # Detect outliers BEFORE taking absolute value!
            pos_outliers = np.partition(flat_orig, -4)[-4:]
            neg_outliers = np.partition(flat_orig, 4)[:4]
            
            flat_data = np.abs(flat_orig)
            flat_data = flat_data[flat_data > 0]
            
            if len(flat_data) > 0:
                log_data = np.log10(flat_data)
                counts, bins = np.histogram(log_data, bins=100, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                norm_counts = counts / np.max(counts)
                
                lin_centers = 10**bin_centers
                
                ax.fill_between(lin_centers, y_offset, norm_counts + y_offset, color=c_granite, alpha=0.3, label="Granite" if i == 0 else "")
                ax.plot(lin_centers, norm_counts + y_offset, color=c_granite, linewidth=1.5, alpha=1.0)
                
                # Plot outliers
                # 'x' for positive outliers, 'o' for negative outliers
                ax.plot(np.abs(pos_outliers), [y_offset] * 4, 'x', color=c_granite, markersize=8, markeredgewidth=1.5)
                ax.plot(np.abs(neg_outliers), [y_offset] * 4, 'o', color=c_granite, markersize=6, markerfacecolor='none', markeredgecolor=c_granite)
        
    ax.set_yticks(np.arange(num_layers) * overlap)
    ax.set_yticklabels([str(l) for l in reversed(layers)])
    
    ax.set_xscale('log')
    
    ax.set_xlabel("Value Magnitude")
    ax.set_ylabel("Layer")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.savefig(output_path)
    plt.close()
    print(f"Contrast plot saved to {output_path}")

def extract_model_data(model_id, target_layers):
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().float().numpy()
        return hook
        
    hooks = []
    for name in target_layers:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        hooks.append(module.register_forward_hook(get_activation(name)))
        
    print("Running inference for 1 step to capture activations...")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :2048].to("cuda")
    
    with torch.no_grad():
        model(input_ids)
        
    for h in hooks:
        h.remove()
        
    weights = {}
    for name in target_layers:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        weights[name] = module.weight.detach().cpu().float().numpy()
        
    del model
    torch.cuda.empty_cache()
    
    return weights, activations

def main():
    layers = [2, 4, 6, 20, 30]
    target_layers = [f"model.layers.{l}.mlp.down_proj" for l in layers]
    
    qwen_w, qwen_a = extract_model_data("Qwen/Qwen2.5-14B", target_layers)
    granite_w, granite_a = extract_model_data("ibm-granite/granite-3.3-8b-base", target_layers)
    
    plots_dir = "/home/cjsschaefer_google_com/finer_is_better/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    print("Plotting weights contrast...")
    plot_ridge_contrast(qwen_w, granite_w, layers, "Weights Magnitude Contrast: Qwen vs Granite", f"{plots_dir}/qwen_vs_granite_weights.png")
    
    print("Plotting activations contrast...")
    plot_ridge_contrast(qwen_a, granite_a, layers, "Activations Magnitude Contrast: Qwen vs Granite", f"{plots_dir}/qwen_vs_granite_acts.png")

if __name__ == "__main__":
    main()
