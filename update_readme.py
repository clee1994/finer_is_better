import pandas as pd
import os

def update_readme():
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    readme_path = "/home/cjsschaefer_google_com/finer_is_better/README.md"

    if not os.path.exists(csv_path):
        print("CSV file not found. Skipping README update.")
        return

    df = pd.read_csv(csv_path, index_col=0)

    # Format cells to 2 decimal places with always two digits!
    def format_cell(x):
        if pd.isna(x) or x == "TBD":
            return x
        try:
            val = float(str(x).replace("+", ""))
            return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
        except ValueError:
            return x

    # Extract model names (prefix before _)
    all_models = list(set([r.split("_")[0] for r in df.index]))
    requested_order = ["Granite", "Llama", "DeepSeek", "Qwen"]
    # Sort by requested order, then alphabetical for anything remaining
    models = [m for m in requested_order if m in all_models] + sorted([m for m in all_models if m not in requested_order])
    
    full_content = ""
    
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
    
    order = ["e4m3", "e4m3 + PZ", "e4m3 + 4o6", "e4m3 + 4o6 + PZ", "ue5m3", "ue5m3 + PZ", "ue5m3 + 4o6", "ue5m3 + 4o6 + PZ"]
    
    for m in models:
        full_content += f"### {m}\n\n"
        # Filter rows for this model
        model_df = df[df.index.str.startswith(m)]
        
        # Rename rows using mapping
        new_index = []
        for idx in model_df.index:
            parts = idx.split("_")
            if len(parts) < 2:
                new_index.append(idx)
                continue
            option = "_".join(parts[1:])
            new_index.append(mapping.get(option, option))
            
        model_df.index = new_index
        
        # Sort according to requested order
        sorted_index = [x for x in order if x in model_df.index]
        # Add any remaining that were not in the order list
        sorted_index += [x for x in model_df.index if x not in order]
        
        model_df = model_df.reindex(sorted_index)
        
        model_df_formatted = model_df.map(format_cell).astype(str)
        full_content += model_df_formatted.to_markdown() + "\n\n"

    # Add consolidated graph at the end
    # Add consolidated graph at the end
    full_content += "### Perplexity Gap All Models\n"
    full_content += "![Perplexity Gap All Models](plots/gap_all_models.png)\n\n"
    

    if not os.path.exists(readme_path):
        print("README.md not found. Creating a skeleton one.")
        with open(readme_path, "w") as f:
            f.write("# Finer is Better\n\n## Results\n\n<!-- RESULTS_START -->\n\n<!-- RESULTS_END -->\n")

    with open(readme_path, "r") as f:
        readme_content = f.read()

    marker_start = "<!-- RESULTS_START -->"
    marker_end = "<!-- RESULTS_END -->"

    idx_start = readme_content.find(marker_start)
    idx_end = readme_content.find(marker_end)

    if idx_start == -1 or idx_end == -1:
        print("Markers not found in README.md. Appending results at the end.")
        with open(readme_path, "a") as f:
            f.write(f"\n## Results\n{marker_start}\n{full_content}\n{marker_end}\n")
    else:
        # Replace content between markers
        new_content = readme_content[:idx_start + len(marker_start)] + "\n\n" + full_content + "\n\n" + readme_content[idx_end:]
        with open(readme_path, "w") as f:
            f.write(new_content)

    print("README.md updated with results table and graphs.")

if __name__ == "__main__":
    update_readme()
