import pandas as pd
import os
import subprocess
import sys
import glob

def cleanup_and_plot():
    path_csv = "/home/cjsschaefer_google_com/finer_is_better/results.csv"
    if not os.path.exists(path_csv):
        print("results.csv not found.")
        return
        
    df = pd.read_csv(path_csv, index_col=0)
    
    # Rename nvfp4 to e4m3 to merge old and new results
    # And rename model names to be consistent!
    new_index = []
    for idx in df.index:
        new_idx = idx.replace("nvfp4", "e4m3")
        
        if "Llama 3.1 8B" in new_idx:
            new_idx = new_idx.replace("Llama 3.1 8B", "Llama")
        elif "Granite 3.3 8B" in new_idx:
            new_idx = new_idx.replace("Granite 3.3 8B", "Granite")
        elif "Qwen 2.5 14B" in new_idx:
            new_idx = new_idx.replace("Qwen 2.5 14B", "Qwen")
        elif "DeepSeek 7B" in new_idx:
            new_idx = new_idx.replace("DeepSeek 7B", "DeepSeek")
            
        new_index.append(new_idx)
    df.index = new_index
    
    # Drop rows that don't have an underscore (like just "Llama")
    # They are likely broken or old entries
    df = df[df.index.str.contains("_")]
    
    # Drop duplicate rows, keeping the last one (most recent)
    df = df[~df.index.duplicated(keep="last")]
    
    # Extract model names (prefix before _)
    models = list(set([r.split("_")[0] for r in df.index]))
    
    # Fill Baseline column for all rows of the same model
    for m in models:
        model_rows = df.index.str.startswith(m)
        base_val = df.loc[model_rows, "Baseline"].dropna()
        if not base_val.empty:
            val = base_val.values[0]
            df.loc[model_rows, "Baseline"] = val
            
    # Drop rows that have NO data points at all (e.g., pure baseline rows)
    bs_columns = [c for c in df.columns if c.startswith("BS=")]
    df = df.dropna(subset=bs_columns, how="all")
            
    # Save back
    df.to_csv(path_csv)
    print("Cleaned results.csv")
    
    # Run update_readme.py
    subprocess.run(["python3", "/home/cjsschaefer_google_com/finer_is_better/update_readme.py"])
    
    # Run create_plots.py with the combined file
    subprocess.run(["python3", "/home/cjsschaefer_google_com/finer_is_better/create_plots.py", path_csv])
    
    # Remove old broken plots from disk and git
    old_plots = glob.glob("/home/cjsschaefer_google_com/finer_is_better/plots/gap_*.png")
    valid_plots = ["gap_llama.png", "gap_granite.png", "gap_qwen.png", "gap_deepseek.png"]
    
    for p in old_plots:
        base = os.path.basename(p)
        if base not in valid_plots:
            print(f"Deleting old broken plot: {p}")
            try:
                os.remove(p)
                subprocess.run(["git", "rm", p])
            except Exception as e:
                print(f"Error deleting {p}: {e}")
    
    # Commit and push the final results!
    subprocess.run(["git", "add", "results.csv", "README.md", "plots/*"])
    subprocess.run(["git", "commit", "-m", "Remove phantom baseline rows and clean up tables"])
    subprocess.run(["git", "push"])
    print("Pushed cleaned results and plots to GitHub.")

if __name__ == "__main__":
    cleanup_and_plot()
