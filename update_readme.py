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
    
    # Append image links side-by-side in an HTML table with captions below
    image_links = "\n### Graphs\n"
    image_links += "<table>\n"
    models = df.index.tolist()
    for i in range(0, len(models), 2):
        m1 = models[i]
        suffix1 = m1.lower().replace(" ", "_")
        img1 = f"<img src=\"plots/gap_{suffix1}.png\" alt=\"{m1} Gap\" width=\"400\">"
        cap1 = f"<p align=\"center\">{m1} Gap</p>"
        
        if i + 1 < len(models):
            m2 = models[i+1]
            suffix2 = m2.lower().replace(" ", "_")
            img2 = f"<img src=\"plots/gap_{suffix2}.png\" alt=\"{m2} Gap\" width=\"400\">"
            cap2 = f"<p align=\"center\">{m2} Gap</p>"
            
            image_links += f"  <tr>\n    <td>{img1}\n{cap1}</td>\n    <td>{img2}\n{cap2}</td>\n  </tr>\n"
        else:
            image_links += f"  <tr>\n    <td>{img1}\n{cap1}</td>\n    <td></td>\n  </tr>\n"
            
    image_links += "</table>\n"
    
    full_content = md_table + "\n" + image_links
    
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
