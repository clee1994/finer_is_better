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
  mpl.rcParams['font.weight'] = 'bold'
  mpl.rcParams['axes.labelsize'] = 16
  mpl.rcParams['axes.labelweight'] = 'bold'
  mpl.rcParams['axes.titlesize'] = 16 # Slightly smaller for multi-line titles
  mpl.rcParams['axes.titleweight'] = 'bold'
  mpl.rcParams['xtick.labelsize'] = 14
  mpl.rcParams['ytick.labelsize'] = 14
  mpl.rcParams['legend.fontsize'] = 14
  mpl.rcParams['figure.titlesize'] = 22
  mpl.rcParams['figure.figsize'] = (16.0, 12.0)
  mpl.rcParams['figure.dpi'] = 300
  mpl.rcParams['savefig.dpi'] = 300
  mpl.rcParams['figure.constrained_layout.use'] = True
  mpl.rcParams['axes.spines.top'] = False
  mpl.rcParams['axes.spines.right'] = False
  mpl.rcParams['axes.linewidth'] = 2.0
  mpl.rcParams['axes.edgecolor'] = 'black'
  mpl.rcParams['xtick.major.width'] = 2.0
  mpl.rcParams['ytick.major.width'] = 2.0
  mpl.rcParams['xtick.minor.width'] = 1.5
  mpl.rcParams['ytick.minor.width'] = 1.5
  mpl.rcParams['xtick.direction'] = 'out'
  mpl.rcParams['ytick.direction'] = 'out'
  mpl.rcParams['axes.grid'] = True
  mpl.rcParams['grid.alpha'] = 0.3
  mpl.rcParams['grid.linestyle'] = '--'
  mpl.rcParams['grid.linewidth'] = 1.0
  mpl.rcParams['legend.frameon'] = False
  mpl.rcParams['legend.loc'] = 'best'
  mpl.rcParams['savefig.bbox'] = 'tight'
  mpl.rcParams['savefig.pad_inches'] = 0.05

def plot_model_dist_on_ax(ax, model_name, weights, acts, layers, overlap, c_wgt, c_act, global_min, global_max, is_first_col=True, is_last_row=True):
    num_layers = len(layers)
    bins = np.linspace(global_min, global_max, 100)
    
    handles = []
    labels = []
    
    all_w_data = []
    all_a_data = []

    for i, layer in enumerate(reversed(layers)):
        y_offset = i * overlap
        layer_name = f"model.layers.{layer}.mlp.down_proj"

        # Plot Weights
        if layer_name in weights:
            data = weights[layer_name].flatten()
            abs_data = np.abs(data)
            abs_data = abs_data[abs_data > 0]
            
            if len(abs_data) > 0:
                log_data = np.log2(abs_data)
                all_w_data.append(log_data)
                
                counts, _ = np.histogram(log_data, bins=bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                if np.max(counts) > 0:
                    norm_counts = counts / np.max(counts)
                    
                    mask = (bin_centers >= np.min(log_data)) & (bin_centers <= np.max(log_data))
                    
                    h = ax.fill_between(bin_centers[mask], y_offset, norm_counts[mask] + y_offset, color=c_wgt, alpha=0.3)
                    l, = ax.plot(bin_centers[mask], norm_counts[mask] + y_offset, color=c_wgt, linewidth=2, alpha=1.0)
                    
                    if i == 0:
                        handles.append((h, l))
                        labels.append("Weights")
                    
                    outliers = np.partition(log_data, -4)[-4:]
                    ax.plot(outliers, [y_offset] * 4, 'x', color=c_wgt, markersize=8, markeredgewidth=2)

        # Plot Activations
        if layer_name in acts:
            data = acts[layer_name].flatten()
            abs_data = np.abs(data)
            abs_data = abs_data[abs_data > 0]
            
            if len(abs_data) > 0:
                log_data = np.log2(abs_data)
                all_a_data.append(log_data)
                
                counts, _ = np.histogram(log_data, bins=bins, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                if np.max(counts) > 0:
                    norm_counts = counts / np.max(counts)
                    
                    mask = (bin_centers >= np.min(log_data)) & (bin_centers <= np.max(log_data))
                    
                    h = ax.fill_between(bin_centers[mask], y_offset, norm_counts[mask] + y_offset, color=c_act, alpha=0.3)
                    l, = ax.plot(bin_centers[mask], norm_counts[mask] + y_offset, color=c_act, linewidth=2, alpha=1.0)
                    
                    if i == 0:
                        handles.append((h, l))
                        labels.append("Activations")
                    
                    outliers = np.partition(log_data, -4)[-4:]
                    ax.plot(outliers, [y_offset] * 4, 'o', color=c_act, markersize=6, markerfacecolor='none', markeredgecolor=c_act, markeredgewidth=2)

    ax.set_title(model_name, y=1.02, fontweight='bold')

    ax.set_yticks(np.arange(num_layers) * overlap)
    ax.set_yticklabels([str(l) for l in reversed(layers)], fontweight='bold')
    
    if is_first_col:
        ax.set_ylabel("Layer", fontweight='bold')
        
    if is_last_row:
        ax.set_xlabel("log2(|Value|)", fontweight='bold')
    
    ax.set_xlim(global_min, global_max)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        
    return handles, labels

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
    apply_style()
    
    layers = [2, 4, 6, 20, 25]
    target_layers = [f"model.layers.{l}.mlp.down_proj" for l in layers]

    models = {
        "Granite-3.3-8B": "ibm-granite/granite-3.3-8b-base",
        "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
        "DeepSeek-LLM-7B-Base": "deepseek-ai/deepseek-llm-7b-base",
        "Qwen2.5-14B": "Qwen/Qwen2.5-14B"
    }

    plots_dir = "/home/cjsschaefer_google_com/finer_is_better/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # First pass to extract data and find GLOBAL limits across all models
    all_extracted_data = {}
    global_min = 0
    global_max = 0
    
    for name, model_id in models.items():
        print(f"Extracting data for {name}...")
        try:
            w, a = extract_model_data(model_id, target_layers)
            all_extracted_data[name] = (w, a)
            
            for layer in layers:
                layer_name = f"model.layers.{layer}.mlp.down_proj"
                if layer_name in w:
                    w_data = np.abs(w[layer_name].flatten())
                    w_data = w_data[w_data > 0]
                    if len(w_data) > 0:
                        log_w = np.log2(w_data)
                        global_min = min(global_min, np.min(log_w))
                        global_max = max(global_max, np.max(log_w))
                if layer_name in a:
                    a_data = np.abs(a[layer_name].flatten())
                    a_data = a_data[a_data > 0]
                    if len(a_data) > 0:
                        log_a = np.log2(a_data)
                        global_min = min(global_min, np.min(log_a))
                        global_max = max(global_max, np.max(log_a))
        except Exception as e:
            print(f"Failed to extract data for {name}: {e}")

    # Force global_min to be at least -18 to ensure we see the zone!
    global_min = min(global_min, -18.0)

    # Now plot with shared global limits!
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    
    c_wgt = '#4C72B0'
    c_act = '#DD8452'
    overlap = 0.3
    
    model_list = list(models.keys())
    
    shared_handles = []
    shared_labels = []

    for i, name in enumerate(model_list):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if name in all_extracted_data:
            w, a = all_extracted_data[name]
            handles, labels = plot_model_dist_on_ax(ax, name, w, a, layers, overlap, c_wgt, c_act, global_min, global_max, is_first_col=(col==0), is_last_row=(row==1))
            if not shared_handles:
                shared_handles = handles
                shared_labels = labels
        else:
            ax.set_title(f"{name} (Failed)", y=1.02, fontweight='bold')

    # Shared legend
    if shared_handles:
        fig.legend([h for h in shared_handles], shared_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    output_path = f"{plots_dir}/dist_all_models.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Consolidated distribution plot saved to {output_path}")

if __name__ == "__main__":
    main()
