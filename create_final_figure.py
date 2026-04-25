import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import torch
import numpy as np
from torch_eval import get_ue5m3_grid
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['figure.titlesize'] = 22
    mpl.rcParams['figure.figsize'] = (18.0, 10.0)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.linewidth'] = 1.0
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

def plot_perplexity(ax, df, short_name, proper_name, label_letter):
    columns = [c for c in df.columns if c.startswith("BS=")]
    block_sizes = [int(c.split("=")[1]) for c in columns]
    rows = [r for r in df.index if r.startswith(short_name)]
    
    if not rows:
        ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', va='center', fontweight='bold')
        return
        
    mapping = {
        "e4m3_no_pz": "e4m3",
        "e4m3_pz": "e4m3 + PZ",
        "4over6_no_pz": "e4m3 + 4o6",
        "4over6_pz": "e4m3 + 4o6 + PZ",
        "ue5m3_no_pz": "ue5m3",
        "ue5m3_pz": "ue5m3 + PZ",
        "4over6_ue5m3": "ue5m3 + 4o6",
        "4over6_pz_ue5m3": "ue5m3 + 4o6 + PZ"
    }
    
    # Redundant rows to drop (keeping e4m3+PZ variants to show the insight!)
    redundant = ["e4m3 + 4o6 + PZ", "e4m3 + 4o6 + PZ + H",
                 "ue5m3 + PZ", "ue5m3 + PZ + H", "ue5m3 + 4o6 + PZ", "ue5m3 + 4o6 + PZ + H"]
    
    styles = {
        "e4m3": ("-", "#8C8C8C"),          # Gray
        "e4m3 + H": ("--", "#8C8C8C"),
        "e4m3 + PZ": ("-", "#C44E52"),      # Red
        "e4m3 + PZ + H": ("--", "#C44E52"),
        "e4m3 + 4o6": ("-", "#8172B3"),     # Purple
        "e4m3 + 4o6 + H": ("--", "#8172B3"),
        "ue5m3": ("-", "#DD8452"),          # Orange
        "ue5m3 + H": ("--", "#DD8452"),
        "ue5m3 + 4o6": ("-", "#64B5CD"),     # Cyan
        "ue5m3 + 4o6 + H": ("--", "#64B5CD")
    }
    
    handles = []
    labels = []
    
    for r in rows:
        option = r.replace(short_name + "_", "")
        lbl = mapping.get(option, option)
        
        if lbl in redundant:
            continue
            
        gaps = []
        for c in columns:
            val = df.loc[r, c]
            if pd.isna(val) or val == "TBD":
                gaps.append(None)
            else:
                gaps.append(float(val))
                
        valid_indices = [i for i, x in enumerate(gaps) if x is not None]
        plot_x = [block_sizes[i] for i in valid_indices]
        plot_y = [gaps[i] for i in valid_indices]
        
        if not plot_x:
            continue
            
        style = styles.get(lbl, ("-", "black"))
        line, = ax.plot(plot_x, plot_y, marker="o", linestyle=style[0], color=style[1], linewidth=2, markersize=6, label=lbl)
        
        handles.append(line)
        labels.append(lbl)
        
    ax.set_xscale("log", base=2)
    ax.set_xticks(block_sizes)
    ax.set_xticklabels(block_sizes, fontweight='bold')
    ax.set_xlabel("Block Size", fontweight='bold')
    ax.set_ylabel("Perplexity Gap", fontweight='bold')
    ax.set_title(f"{label_letter} {proper_name}", loc='left', fontweight='bold')
    
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

def plot_model_dist_on_ax(ax, model_name, weights, acts, layers, overlap, c_wgt, c_act, global_min, global_max, label_letter):
    num_layers = len(layers)
    bins = np.linspace(global_min, global_max, 100)
    
    handles = []
    labels = []
    
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

    ax.set_title(f"{label_letter} {model_name}", loc='left', fontweight='bold')
    ax.set_yticks(np.arange(num_layers) * overlap)
    ax.set_yticklabels([str(l) for l in reversed(layers)], fontweight='bold')
    ax.set_ylabel("Layer", fontweight='bold')
    ax.set_xlabel("log2(|Value|)", fontweight='bold')
    ax.set_xlim(global_min, global_max)
    
    return handles, labels

def main():
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results/results.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, index_col=0)
    apply_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Perplexity plots
    h1, l1 = plot_perplexity(axes[0, 0], df, "Granite", "Granite-3.3-8B", "(a)")
    h2, l2 = plot_perplexity(axes[0, 1], df, "Llama", "Llama-3.1-8B", "(b)")
    h3, l3 = plot_perplexity(axes[1, 0], df, "DeepSeek", "DeepSeek-LLM-7B-Base", "(d)")
    h4, l4 = plot_perplexity(axes[1, 1], df, "Qwen", "Qwen2.5-14B", "(e)")
    
    # Distribution plots
    layers = [2, 4, 6, 20, 25]
    target_layers = [f"model.layers.{l}.mlp.down_proj" for l in layers]
    
    print("Extracting data for Granite...")
    w_granite, a_granite = extract_model_data("ibm-granite/granite-3.3-8b-base", target_layers)
    
    print("Extracting data for Qwen...")
    w_qwen, a_qwen = extract_model_data("Qwen/Qwen2.5-14B", target_layers)
    
    # Find global limits for these two models
    global_min = -18.0 # Force min to see the zone
    global_max = 0.0
    
    for w in [w_granite, w_qwen]:
        for name in target_layers:
            if name in w:
                data = np.abs(w[name].flatten())
                data = data[data > 0]
                if len(data) > 0:
                    global_max = max(global_max, np.max(np.log2(data)))
                    global_min = min(global_min, np.min(np.log2(data)))
                    
    for a in [a_granite, a_qwen]:
        for name in target_layers:
            if name in a:
                data = np.abs(a[name].flatten())
                data = data[data > 0]
                if len(data) > 0:
                    global_max = max(global_max, np.max(np.log2(data)))
                    global_min = min(global_min, np.min(np.log2(data)))
                    
    overlap = 0.3
    c_wgt = '#4C72B0'
    c_act = '#DD8452'
    
    print("Plotting distributions...")
    plot_model_dist_on_ax(axes[0, 2], "Granite-3.3-8B", w_granite, a_granite, layers, overlap, c_wgt, c_act, global_min, global_max, "(c)")
    plot_model_dist_on_ax(axes[1, 2], "Qwen2.5-14B", w_qwen, a_qwen, layers, overlap, c_wgt, c_act, global_min, global_max, "(f)")
    
    # Combine handles and labels for legend
    all_handles = h1 + h2 + h3 + h4
    all_labels = l1 + l2 + l3 + l4
    by_label = dict(zip(all_labels, all_handles))
    
    order = ["e4m3", "e4m3 + PZ", "e4m3 + H", "e4m3 + PZ + H", "e4m3 + 4o6", "e4m3 + 4o6 + H", "ue5m3", "ue5m3 + H", "ue5m3 + 4o6", "ue5m3 + 4o6 + H"]
    
    sorted_handles = [by_label[l] for l in order if l in by_label]
    sorted_labels = [l for l in order if l in by_label]
    
    fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    
    plt.tight_layout()
    output_path = "final_fig.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Final figure saved to {output_path}")

if __name__ == "__main__":
    main()
