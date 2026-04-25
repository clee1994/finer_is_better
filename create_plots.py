import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

def apply_style():
  """Configures matplotlib for publication-quality plots (NeurIPS style)."""
  mpl.rcParams.update(mpl.rcParamsDefault)
  mpl.rcParams['font.family'] = 'serif'
  mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
  mpl.rcParams['font.size'] = 20
  mpl.rcParams['font.weight'] = 'bold'
  mpl.rcParams['axes.labelsize'] = 20
  mpl.rcParams['axes.labelweight'] = 'bold'
  mpl.rcParams['axes.titlesize'] = 22
  mpl.rcParams['axes.titleweight'] = 'bold'
  mpl.rcParams['xtick.labelsize'] = 18
  mpl.rcParams['ytick.labelsize'] = 18
  mpl.rcParams['legend.fontsize'] = 18
  mpl.rcParams['figure.titlesize'] = 26
  mpl.rcParams['figure.figsize'] = (16.0, 12.0) # Increased figure size for larger fonts!
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
  cycle = mpl.cycler(color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'])
  mpl.rcParams['axes.prop_cycle'] = cycle
  mpl.rcParams['legend.frameon'] = False
  mpl.rcParams['legend.loc'] = 'best'
  mpl.rcParams['savefig.bbox'] = 'tight'
  mpl.rcParams['savefig.pad_inches'] = 0.05

def plot_all_models(df, output_path):
    apply_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=False)
    axes = axes.flatten()
    
    models = ["Granite", "Llama", "DeepSeek", "Qwen"]
    
    model_titles = {
        "Granite": "Granite-3.3-8B",
        "Llama": "Llama-3.1-8B",
        "DeepSeek": "DeepSeek-LLM-7B-Base",
        "Qwen": "Qwen2.5-14B"
    }
    
    mapping = {
        "e4m3_no_pz": "e4m3",
        "e4m3_pz": "e4m3 + PZ",
        "4over6_no_pz": "e4m3 + 4o6",
        "4over6_pz": "e4m3 + 4o6 + PZ",
        "ue5m3_no_pz": "ue5m3",
        "ue5m3_pz": "ue5m3 + PZ",
        "4over6_pz_ue5m3": "ue5m3 + 4o6 + PZ"
    }
    
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
        "ue5m3 + PZ": ("-", "purple"),
        "ue5m3 + PZ + H": ("--", "purple"),
        "ue5m3 + 4o6": ("-", "cyan"),
        "ue5m3 + 4o6 + H": ("--", "cyan"),
        "ue5m3 + 4o6 + PZ": ("-", "brown"),
        "ue5m3 + 4o6 + PZ + H": ("--", "brown")
    }

    order = ["e4m3", "e4m3 + PZ", "e4m3 + 4o6", "e4m3 + 4o6 + PZ", "ue5m3", "ue5m3 + PZ", "ue5m3 + 4o6", "ue5m3 + 4o6 + PZ", "e4m3 + H", "e4m3 + PZ + H", "e4m3 + 4o6 + H", "e4m3 + 4o6 + PZ + H", "ue5m3 + H", "ue5m3 + PZ + H", "ue5m3 + 4o6 + H", "ue5m3 + 4o6 + PZ + H"]

    handles_list = []
    labels_list = []

    for i, model_name in enumerate(models):
        ax = axes[i]
        # Move title up a little bit (y=0.95 instead of 0.85)
        ax.set_title(model_titles.get(model_name, model_name), y=0.95, fontweight='bold')
        
        columns = [c for c in df.columns if c.startswith("BS=")]
        block_sizes = [int(c.split("=")[1]) for c in columns]
        rows = [r for r in df.index if r.startswith(model_name)]
        
        if not rows:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', va='center', fontweight='bold')
            continue

        mapped_rows = []
        for r in rows:
            option = r.replace(model_name + "_", "")
            mapped_rows.append((r, mapping.get(option, option)))

        sorted_rows = sorted(mapped_rows, key=lambda x: order.index(x[1]) if x[1] in order else 999)

        for row, label in sorted_rows:
            gaps = []
            for c in columns:
                val = df.loc[row, c]
                if pd.isna(val) or val == "TBD":
                    gaps.append(None)
                else:
                    gaps.append(float(val))

            valid_indices = [i for i, x in enumerate(gaps) if x is not None]
            plot_x = [block_sizes[i] for i in valid_indices]
            plot_y = [gaps[i] for i in valid_indices]

            if not plot_x:
                continue

            style = styles.get(label, ("-", "black"))
            line, = ax.plot(plot_x, plot_y, marker="o", linestyle=style[0], color=style[1], linewidth=3, markersize=8, label=label)
            
            if not handles_list and label in order:
                 handles_list.append(line)
                 labels_list.append(label)
            elif label in order and label not in labels_list:
                 handles_list.append(line)
                 labels_list.append(label)

        ax.set_xscale("log", base=2)
        ax.set_xticks(block_sizes)
        ax.set_xticklabels(block_sizes, fontweight='bold')
        
        ax.set_xlabel("Block Size", fontweight='bold')
        ax.set_ylabel("Perplexity Gap", fontweight='bold')
        
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    sorted_handles_labels = sorted(zip(handles_list, labels_list), key=lambda x: order.index(x[1]))
    sorted_handles = [h for h, l in sorted_handles_labels]
    sorted_labels = [l for h, l in sorted_handles_labels]

    # Legend lower
    fig.legend(sorted_handles, sorted_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Consolidated plot saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Skipping plotting.")
        exit(1)

    df = pd.read_csv(csv_path, index_col=0)
    plots_dir = "/home/cjsschaefer_google_com/finer_is_better/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plot_all_models(df, f"{plots_dir}/gap_all_models.png")
