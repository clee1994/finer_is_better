import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def clean_matrices_and_plot():
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

    fig, axes = plt.subplots(1, 2, figsize=(36, 16))

    t_rows = ["0.0", "0.0001", "0.0002", "0.0003", "0.0005", "0.0007", "0.001", "0.0015", "0.002", "0.003", "0.004", "0.005", "0.006", "0.007", "0.008", "0.01", "0.012", "0.016", "0.024"]
    t_cols = ["0.0001", "0.0002", "0.0003", "0.0005", "0.0007", "0.001", "0.0015", "0.002", "0.003", "0.004", "0.005", "0.006", "0.007", "0.008", "0.01", "0.012", "0.016", "0.024", "0.035"]

    for idx, name in enumerate(["Granite", "Qwen"]):
        csv_file = f"ablation_heatmap_{name}.csv"
        if not os.path.exists(csv_file):
            print(f"Missing file {csv_file}")
            continue
        
        df = pd.read_csv(csv_file, index_col=0)
        df.index = df.index.astype(str)
        
        clean_idx = []
        for k in df.index:
            val_str = k.strip()
            for target in t_rows:
                if val_str.startswith(target):
                    val_str = target
                    break
            clean_idx.append(val_str)
        df.index = clean_idx
        
        clean_cols = []
        for c in df.columns:
            val_str = str(c).strip()
            for target in t_cols:
                if val_str.startswith(target):
                    val_str = target
                    break
            clean_cols.append(val_str)
        df.columns = clean_cols
        
        clean_df = df.groupby(df.index).first()
        clean_df = clean_df.reindex(index=t_rows, columns=t_cols)
        
        clean_df.to_csv(csv_file)
        print(f"Successfully cleaned and deduplicated {csv_file}")
        
        ax = axes[idx]
        data = clean_df.values.astype(float)
        
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Perplexity Gap', fontweight='bold')
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    lbl = f"{val:.1e}" if val > 1000 else (f"{val:.1f}" if val > 10 else f"{val:.2f}")
                    ax.text(j, i, lbl, ha="center", va="center", color="black", fontweight="bold", size=10)
                    
        ax.set_xticks(np.arange(len(t_cols)))
        ax.set_yticks(np.arange(len(t_rows)))
        ax.set_xticklabels(t_cols, fontweight='bold')
        ax.set_yticklabels(t_rows, fontweight='bold')
        ax.set_title(f"Zone Ablation Sensitivity: {name}", fontweight='bold', pad=15)
        ax.set_ylabel("Lower Threshold (B)", fontweight='bold')
        ax.set_xlabel("Upper Threshold (A)", fontweight='bold')

    plt.tight_layout()
    out_path = "zone_ablation_heatmap.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Successfully generated crisp triangular heatmap plot at {out_path}")

if __name__ == "__main__":
    clean_matrices_and_plot()
