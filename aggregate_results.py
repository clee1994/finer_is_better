import pandas as pd
import os
import glob
import subprocess

def aggregate_results():
    path_pattern = "/home/cjsschaefer_google_com/finer_is_better/results_*.csv"
    csv_files = glob.glob(path_pattern)
    
    if not csv_files:
        print("No result files found.")
        return
        
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        dfs.append(df)
        
    combined_df = pd.concat(dfs)
    
    # Drop duplicate rows (keep the last one, which should be the most recent)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    
    # Save combined results to results.csv to let update_readme use it!
    combined_df.to_csv("/home/cjsschaefer_google_com/finer_is_better/results.csv")
    print("Saved combined results to results.csv")
    
    # Run update_readme.py
    subprocess.run(["python3", "/home/cjsschaefer_google_com/finer_is_better/update_readme.py"])
    
    # Run create_plots.py with the combined file
    subprocess.run(["python3", "/home/cjsschaefer_google_com/finer_is_better/create_plots.py", "/home/cjsschaefer_google_com/finer_is_better/results.csv"])
    
    # Commit and push the final results!
    subprocess.run(["git", "add", "results.csv", "README.md", "plots/*"])
    subprocess.run(["git", "commit", "-m", "Update README and plots with final results"])
    subprocess.run(["git", "push"])
    print("Pushed final results to GitHub.")

if __name__ == "__main__":
    aggregate_results()
