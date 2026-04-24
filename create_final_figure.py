import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import torch
from torch_eval import get_ue5m3_grid
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def plot_perplexity(ax, df, model_name, label_letter):
    columns = [c for c in df.columns if c.startswith("BS=")]
    block_sizes = [int(c.split("=")[1]) for c in columns]
    rows = [r for r in df.index if r.startswith(model_name)]
    
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
    
    # Redundant rows to drop
    redundant = ["ue5m3 + PZ", "ue5m3 + 4o6 + PZ", "ue5m3 + PZ + H", "ue5m3 + 4o6 + PZ + H"]
    
    styles = {
        "e4m3": ("-", "gray"),
        "e4m3 + H": ("--", "gray"),
        "e4m3 + PZ": ("-", "blue"),
        "e4m3 + PZ + H": ("--", "blue"),
        "e4m3 + 4o6": ("-", "green"),
        "e4m3 + 4o6 + H": ("--", "green"),
        "e4m3 + 4o6 + PZ": ("-", "red"),
        "e4m3 + 4o6 + PZ + H": ("--", "red"),
        "ue5m3": ("-", "orange"),
        "ue5m3 + H": ("--", "orange"),
        "ue5m3 + 4o6": ("-", "cyan"),
        "ue5m3 + 4o6 + H": ("--", "cyan")
    }
    
    handles = []
    labels = []
    
    for r in rows:
        option = r.replace(model_name + "_", "")
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
    ax.set_title(f"{label_letter} {model_name}", loc='left', fontweight='bold')
    
    return handles, labels

def plot_distribution(ax, model_id, label_letter):
    # Placeholder for distribution plotting
    # We need to load the model and extract weights/activations
    ax.text(0.5, 0.5, f"Dist for {model_id}\n(Requires model load)", transform=ax.transAxes, ha='center', va='center', fontweight='bold')
    ax.set_title(f"{label_letter} {model_id} Distribution", loc='left', fontweight='bold')

def main():
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, index_col=0)
    apply_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Perplexity plots
    h1, l1 = plot_perplexity(axes[0, 0], df, "Granite", "(a)")
    h2, l2 = plot_perplexity(axes[0, 1], df, "Llama", "(b)")
    h3, l3 = plot_perplexity(axes[1, 0], df, "DeepSeek", "(d)")
    h4, l4 = plot_perplexity(axes[1, 1], df, "Qwen", "(e)")
    
    # Distribution plots (Placeholders for now)
    plot_distribution(axes[0, 2], "ibm-granite/granite-3.3-8b-base", "(c)")
    plot_distribution(axes[1, 2], "Qwen/Qwen2.5-14B", "(f)")
    
    # Combine handles and labels for legend
    all_handles = h1 + h2 + h3 + h4
    all_labels = l1 + l2 + l3 + l4
    by_label = dict(zip(all_labels, all_handles))
    
    order = ["e4m3", "e4m3 + H", "e4m3 + PZ", "e4m3 + PZ + H", "e4m3 + 4o6", "e4m3 + 4o6 + H", "e4m3 + 4o6 + PZ", "e4m3 + 4o6 + PZ + H", "ue5m3", "ue5m3 + H", "ue5m3 + 4o6", "ue5m3 + 4o6 + H"]
    
    sorted_handles = [by_label[l] for l in order if l in by_label]
    sorted_labels = [l for l in order if l in by_label]
    
    fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout()
    output_path = "/home/cjsschaefer_google_com/finer_is_better/plots/final_figure.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Final figure saved to {output_path}")

if __name__ == "__main__":
    main()
