import pandas as pd
import matplotlib.pyplot as plt
import os

# Global font setting to serif
plt.rcParams["font.family"] = "serif"

def plot_model(model_name, df, output_path):
    plt.figure(figsize=(8, 6))
    
    columns = [c for c in df.columns if c.startswith("BS=")]
    block_sizes = [int(c.split("=")[1]) for c in columns]
    
    gaps = []
    for c in columns:
        val = df.loc[model_name, c]
        if pd.isna(val) or val == "TBD":
            gaps.append(None)
        else:
            gaps.append(float(val))
            
    valid_indices = [i for i, x in enumerate(gaps) if x is not None]
    plot_x = [block_sizes[i] for i in valid_indices]
    plot_y = [gaps[i] for i in valid_indices]
    
    if not plot_x:
        print(f"No valid data for {model_name}. Skipping plot.")
        plt.close()
        return
        
    # Blue line with circles
    plt.plot(plot_x, plot_y, marker="o", linestyle="-", linewidth=3, color="#1f77b4", label="NVFP4")
    
    plt.xlabel("Block Size", fontsize=16)
    plt.ylabel("Perplexity Gap", fontsize=16)
    
    # Log scale on X axis (base 2)
    plt.xscale("log", base=2)
    
    # X-ticks to match specific block sizes
    plt.xticks(block_sizes, block_sizes, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.grid(True, linestyle="--", alpha=0.7, color="#d3d3d3")
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    
    # No legend frame
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    if not os.path.exists(csv_path):
        print("CSV file not found. Skipping plotting.")
        exit(1)
        
    df = pd.read_csv(csv_path, index_col=0)
    
    # Create plots/ folder if it doesn't exist
    plots_dir = "/home/cjsschaefer_google_com/finer_is_better/plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created folder: {plots_dir}")
        
    models = df.index.tolist()
    for m in models:
        file_suffix = m.lower().replace(" ", "_")
        plot_model(m, df, f"{plots_dir}/gap_{file_suffix}.png")
