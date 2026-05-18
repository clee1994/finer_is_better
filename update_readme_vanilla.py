import csv
import os

def format_cell(x):
    if not x or x.strip() == "" or x == "TBD" or x == "nan":
        return ""
    try:
        val = float(str(x).replace("+", ""))
        return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"
    except ValueError:
        return x

def update_readme():
    csv_path = "/google/src/cloud/cjsschaefer/finer_is_better/results.csv"
    readme_path = "/google/src/cloud/cjsschaefer/finer_is_better/README.md"

    if not os.path.exists(csv_path):
        print("CSV file not found.")
        return

    # Read CSV rows
    headers = []
    rows = {}
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row: continue
            rows[row[0]] = row[1:]

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

    order = [
        "e4m3", "e4m3 + PZ", "e4m3 + 4o6", "e4m3 + 4o6 + PZ",
        "ue5m3", "ue5m3 + PZ", "ue5m3 + 4o6", "ue5m3 + 4o6 + PZ",
        "e4m3 + H", "e4m3 + PZ + H", "e4m3 + 4o6 + H", "e4m3 + 4o6 + PZ + H",
        "ue5m3 + H", "ue5m3 + PZ + H", "ue5m3 + 4o6 + H", "ue5m3 + 4o6 + PZ + H"
    ]

    requested_order = ["Granite", "Llama", "DeepSeek", "Qwen"]
    all_models = list(set([r.split("_")[0] for r in rows.keys()]))
    models = [m for m in requested_order if m in all_models] + sorted([m for m in all_models if m not in requested_order])

    full_content = "### Consolidated Experimental Figures\n\n"
    full_content += "#### Publication Final 2x3 Consolidated Figure (Perplexity & Distributions)\n"
    full_content += "![Publication Final Figure](plots/final_fig.png)\n\n"
    full_content += "### Detailed Evaluation Perplexity Matrices\n\n"

    for m in models:
        full_content += f"#### {m}\n\n"
        # Gather rows for this model
        m_rows = {}
        for idx, vals in rows.items():
            if idx.startswith(m):
                parts = idx.split("_")
                if len(parts) < 2:
                    option = idx
                else:
                    option = "_".join(parts[1:])
                lbl = mapping.get(option, option)
                m_rows[lbl] = vals

        # Sort rows according to order
        sorted_keys = [x for x in order if x in m_rows]
        sorted_keys += [x for x in m_rows if x not in order]

        # Build Markdown Table
        table_headers = ["Model_Config"] + headers[1:]
        full_content += "| " + " | ".join(table_headers) + " |\n"
        full_content += "| " + " | ".join(["---"] * len(table_headers)) + " |\n"
        
        for k in sorted_keys:
            formatted_vals = [format_cell(v) for v in m_rows[k]]
            full_content += f"| {k} | " + " | ".join(formatted_vals) + " |\n"
        full_content += "\n"

    if not os.path.exists(readme_path):
        print("README.md not found.")
        return

    with open(readme_path, "r") as f:
        readme_content = f.read()

    marker_start = "<!-- RESULTS_START -->"
    marker_end = "<!-- RESULTS_END -->"

    idx_start = readme_content.find(marker_start)
    idx_end = readme_content.find(marker_end)

    if idx_start == -1 or idx_end == -1:
        print("Markers not found.")
    else:
        new_content = readme_content[:idx_start + len(marker_start)] + "\n\n" + full_content + "\n\n" + readme_content[idx_end:]
        with open(readme_path, "w") as f:
            f.write(new_content)
    print("README.md successfully updated locally.")

if __name__ == "__main__":
    update_readme()
