import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_model(model_name, df, output_path):
    plt.figure(figsize=(8, 6))
    
    # Extract block sizes from columns
    columns = [c for c in df.columns if c.startswith("BS=")]
    block_sizes = [int(c.split("=")[1]) for c in columns]
    
    # Extract gaps for the specific model
    gaps = []
    for c in columns:
        val = df.loc[model_name, c]
        if pd.isna(val) or val == "TBD":
            gaps.append(None)
        else:
            gaps.append(float(val))
            
    # Filter out None values for plotting
    valid_indices = [i for i, x in enumerate(gaps) if x is not None]
    plot_x = [block_sizes[i] for i in valid_indices]
    plot_y = [gaps[i] for i in valid_indices]
    
    if not plot_x:
        print(f"No valid data for {model_name}. Skipping plot.")
        plt.close()
        return
        
    # Match style as exactly as possible
    # Blue line with circles
    plt.plot(plot_x, plot_y, marker="o", linestyle="-", linewidth=3, color="#1f77b4", label="NVFP4")
    
    # Customizing plot to match style
    plt.xlabel("Block Size", fontsize=16, fontname="serif")
    plt.ylabel("Perplexity Gap", fontsize=16, fontname="serif")
    plt.title(f"Quantization Gap - {model_name}", fontsize=18, fontname="serif")
    
    # X-ticks to match specific block sizes
    plt.xticks(block_sizes, block_sizes, fontsize=14)
    plt.yticks(fontsize=14)
    
    # Light gray dashed grid
    plt.grid(True, linestyle="--", alpha=0.7, color="#d3d3d3")
    
    # Bold black axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    
    plt.legend(loc="lower right", fontsize=14)
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
    
    models = df.index.tolist()
    for m in models:
        file_suffix = m.lower().replace(" ", "_")
        plot_model(m, df, f"/home/cjsschaefer_google_com/finer_is_better/gap_{file_suffix}.png")
