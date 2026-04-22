import pandas as pd
import os

def update_readme():
    csv_path = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    readme_path = "/home/cjsschaefer_google_com/finer_is_better/README.md"
    
    if not os.path.exists(csv_path):
        print("CSV file not found. Skipping README update.")
        return
        
    df = pd.read_csv(csv_path, index_col=0)
    
    # Generate Markdown table
    md_table = df.to_markdown()
    
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
            f.write(f"\n## Results\n{marker_start}\n{md_table}\n{marker_end}\n")
    else:
        # Replace content between markers
        new_content = readme_content[:idx_start + len(marker_start)] + "\n\n" + md_table + "\n\n" + readme_content[idx_end:]
        with open(readme_path, "w") as f:
            f.write(new_content)
            
    print("README.md updated with results table.")

if __name__ == "__main__":
    update_readme()
