import torch
import time
import sys
import numpy as np

_UE5M3_GRID = None
_UE5M3_TH = None

def get_ue5m3_grid():
    global _UE5M3_GRID, _UE5M3_TH
    if _UE5M3_GRID is None:
        grid = []
        for m in range(8):
            val = (m / 8.0) * (2**-14)
            grid.append(val)
        for e in range(1, 31):
            for m in range(8):
                val = (1.0 + m / 8.0) * (2**(e - 15))
                grid.append(val)
        _UE5M3_GRID = torch.tensor(sorted(list(set(grid))), dtype=torch.float32).cuda()
        _UE5M3_TH = (_UE5M3_GRID[:-1] + _UE5M3_GRID[1:]) / 2
    return _UE5M3_GRID, _UE5M3_TH

def quantize_fp8_simulate_old(val):
    grid, _ = get_ue5m3_grid()
    dist = torch.abs(val.unsqueeze(-1) - grid)
    idx = torch.argmin(dist, dim=-1)
    quant = grid[idx]
    return quant

def quantize_fp8_simulate_new(val):
    grid, th = get_ue5m3_grid()
    idx = torch.bucketize(val, th)
    quant = grid[idx]
    return quant

def verify():
    print("Generating wide-range test data...")
    # Test values from 1e-16 to 1e5!
    exponents = torch.linspace(-16, 5, 1000000).cuda()
    val = 10**exponents
    
    print("Running verification...")
    q_old = quantize_fp8_simulate_old(val)
    q_new = quantize_fp8_simulate_new(val)
    
    # Check exact equality!
    is_equal = torch.equal(q_old, q_new)
    print(f"Are results EXACTLY equal? {is_equal}")
    
    if not is_equal:
        diff = torch.abs(q_old - q_new)
        max_diff = torch.max(diff)
        print(f"Max difference: {max_diff}")
        
        # Find where they differ
        idx = torch.where(diff > 0)[0]
        print(f"Number of differences: {len(idx)}")
        print(f"First mismatch at idx {idx[0]}: val={val[idx[0]]}, old={q_old[idx[0]]}, new={q_new[idx[0]]}")

if __name__ == "__main__":
    verify()
