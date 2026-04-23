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
  mpl.rcParams['font.size'] = 16
  mpl.rcParams['axes.labelsize'] = 16
  mpl.rcParams['axes.titlesize'] = 16
  mpl.rcParams['xtick.labelsize'] = 14
  mpl.rcParams['ytick.labelsize'] = 14
  mpl.rcParams['legend.fontsize'] = 14
  mpl.rcParams['figure.titlesize'] = 18
  mpl.rcParams['figure.figsize'] = (8.0, 5.0)
  mpl.rcParams['figure.dpi'] = 300
  mpl.rcParams['savefig.dpi'] = 300
  mpl.rcParams['figure.constrained_layout.use'] = True
  mpl.rcParams['axes.spines.top'] = False
  mpl.rcParams['axes.spines.right'] = False
  mpl.rcParams['axes.linewidth'] = 1.0
  mpl.rcParams['axes.edgecolor'] = 'black'
  mpl.rcParams['xtick.major.width'] = 1.0
  mpl.rcParams['ytick.major.width'] = 1.0
  mpl.rcParams['xtick.minor.width'] = 0.8
  mpl.rcParams['ytick.minor.width'] = 0.8
  mpl.rcParams['xtick.direction'] = 'out'
  mpl.rcParams['ytick.direction'] = 'out'
  mpl.rcParams['axes.grid'] = True
  mpl.rcParams['grid.alpha'] = 0.3
  mpl.rcParams['grid.linestyle'] = '--'
  mpl.rcParams['grid.linewidth'] = 0.8
  cycle = mpl.cycler(color=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'])
  mpl.rcParams['axes.prop_cycle'] = cycle
  mpl.rcParams['legend.frameon'] = False
  mpl.rcParams['legend.loc'] = 'best'
  mpl.rcParams['savefig.bbox'] = 'tight'
  mpl.rcParams['savefig.pad_inches'] = 0.05

def plot_model(model_name, df, output_path):
    plt.figure()
    columns = [c for c in df.columns if c.startswith("BS=")]
    block_sizes = [int(c.split("=")[1]) for c in columns]
    rows = [r for r in df.index if r.startswith(model_name)]
    if not rows:
        print(f"No data for {model_name}. Skipping plot.")
        plt.close()
        return

    mapping = {
        "e4m3_no_pz": "e4m3",
        "e4m3_pz": "e4m3 + PZ",
        "4over6_no_pz": "e4m3 + 4o6",
        "4over6_pz": "e4m3 + 4o6 + PZ",
        "ue5m3_no_pz": "ue5m3",
        "ue5m3_pz": "ue5m3 + 4o6 + PZ",
    }

    order = ["e4m3", "e4m3 + PZ", "e4m3 + 4o6", "e4m3 + 4o6 + PZ", "ue5m3", "ue5m3 + PZ", "ue5m3 + 4o6", "ue5m3 + 4o6 + PZ"]

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

        plt.plot(plot_x, plot_y, marker="o", linestyle="-", linewidth=2, label=label)

    plt.xlabel("Block Size")
    plt.ylabel("Perplexity Gap")
    plt.xscale("log", base=2)
    plt.xticks(block_sizes, block_sizes)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    apply_style()
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
        print(f"Created folder: {plots_dir}")

    models = list(set([r.split("_")[0] for r in df.index]))
    for m in models:
        file_suffix = m.lower().replace(" ", "_")
        plot_model(m, df, f"{plots_dir}/gap_{file_suffix}.png")
