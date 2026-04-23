import torch
import jax.numpy as jnp
import numpy as np
from torch_eval import quantize_fp8_simulate

def test_mismatch():
    import jax
    
    print("--- Testing Scale Quantization ---")
    x_np = np.random.rand(10000).astype(np.float32) * 10.0
    x_jax = jnp.array(x_np)
    x_torch = torch.tensor(x_np).cuda()
    
    q_jax = jnp.float8_e4m3fn(x_jax)
    q_jax_np = np.array(q_jax)
    
    q_torch = quantize_fp8_simulate(x_torch.unsqueeze(-1)).squeeze(-1)
    q_torch_np = q_torch.cpu().numpy()
    
    diff = np.abs(q_jax_np.astype(np.float32) - q_torch_np.astype(np.float32))
    
    mismatch_indices = np.where(diff > 0.01)[0]
    print(f"Number of mismatches: {len(mismatch_indices)}")
    
    print("Top 10 mismatches for scales:")
    for idx in mismatch_indices[:10]:
        print(f"Val: {float(x_np[idx]):.6f}, JAX: {float(q_jax_np[idx]):.6f}, Torch: {float(q_torch_np[idx]):.6f}")
        
    print("\n--- Testing Element Quantization ---")
    x_np = (np.random.rand(10000).astype(np.float32) - 0.5) * 12.0
    x_jax = jnp.array(x_np)
    x_torch = torch.tensor(x_np).cuda()
    
    q_jax = jnp.float4_e2m1fn(x_jax)
    q_jax_np = np.array(q_jax)
    
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32).cuda()
    sign = torch.sign(x_torch)
    abs_x = torch.abs(x_torch)
    dist = torch.abs(abs_x.unsqueeze(-1) - grid)
    idx_argmin = torch.argmin(dist, dim=-1)
    
    next_idx = torch.clamp(idx_argmin + 1, max=len(grid) - 1)
    dist_current = torch.gather(dist, -1, idx_argmin.unsqueeze(-1))
    dist_next = torch.gather(dist, -1, next_idx.unsqueeze(-1))
    is_tie = torch.isclose(dist_current, dist_next)
    idx_is_even = (idx_argmin % 2) == 0
    new_idx = torch.where(is_tie.squeeze(-1) & ~idx_is_even, next_idx, idx_argmin)
    
    q_torch = grid[new_idx] * sign
    q_torch_np = q_torch.cpu().numpy()
    
    diff = np.abs(q_jax_np.astype(np.float32) - q_torch_np.astype(np.float32))
    
    mismatch_indices = np.where(diff > 0.01)[0]
    print(f"Number of mismatches: {len(mismatch_indices)}")
    
    print("Top 10 mismatches for elements:")
    for idx in mismatch_indices[:10]:
        print(f"Val: {float(x_np[idx]):.6f}, JAX: {float(q_jax_np[idx]):.6f}, Torch: {float(q_torch_np[idx]):.6f}")

if __name__ == "__main__":
    test_mismatch()
