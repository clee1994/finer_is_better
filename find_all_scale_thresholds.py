import jax.numpy as jnp
import numpy as np
import torch

def find_all_scale_thresholds():
    x = jnp.linspace(-448, 448, 1000000)
    q_x = jnp.float8_e4m3fn(x)
    unique_vals = np.unique(np.array(q_x))
    unique_vals = unique_vals[~np.isnan(unique_vals)]
    
    print(f"Found {len(unique_vals)} unique values.")
    
    thresholds = []
    for i in range(len(unique_vals) - 1):
        v1 = unique_vals[i]
        v2 = unique_vals[i+1]
        
        x_search = jnp.linspace(v1, v2, 1000)
        q_search = jnp.float8_e4m3fn(x_search)
        
        idx = np.where(np.array(q_search) == v2)[0]
        if len(idx) > 0:
            thresholds.append(float(x_search[idx[0]]))
        else:
            print(f"Failed to find threshold between {v1} and {v2}!")
            
    print(f"Found {len(thresholds)} thresholds.")
    
    torch.save(torch.tensor(thresholds, dtype=torch.float32), "/home/cjsschaefer_google_com/finer_is_better/fp8_thresholds.pt")
    print("Saved FP8 thresholds.")

if __name__ == "__main__":
    find_all_scale_thresholds()
