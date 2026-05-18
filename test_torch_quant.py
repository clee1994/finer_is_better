import torch
from torch_eval import (
    FP4_quant_torch,
    get_hadamard_matrix,
    apply_hadamard_torch,
    had_mod_torch,
    TorchMXLinear,
    quantize_mx_torch
)

def test_torch_quant():
    x = (torch.rand(1024, 1024, dtype=torch.float32).cuda() - 0.5) * 20.0
    print(f"Original x (Case 1) shape: {x.shape}")

    block_size = 32
    with torch.no_grad():
        qx, quant, scale, _ = FP4_quant_torch(x, block_size)
        print(f"Quantized x (Case 1) shape: {quant.shape}")
        print(f"Scale (Case 1) shape: {scale.shape}")

        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32).cuda()

        quant_flat = quant.flatten()
        dist = torch.abs(torch.abs(quant_flat).unsqueeze(-1) - grid)
        min_dist = torch.min(dist, dim=-1).values

        is_on_grid = torch.allclose(min_dist, torch.zeros_like(min_dist), atol=1e-5)
        print(f"Are all values on grid? {is_on_grid}")
        assert is_on_grid, "Element quantization failed! Not all values mapped to grid!"

        unique_elements = torch.unique(quant)
        print(f"Number of unique elements: {len(unique_elements)}")
        assert len(unique_elements) <= 15, "Too many unique values in FP4 elements!"

        min_elem = torch.min(quant)
        max_elem = torch.max(quant)
        print(f"Min element: {min_elem}, Max element: {max_elem}")
        assert min_elem >= -6.0 and max_elem <= 6.0, "Elements out of range [-6, 6]!"

        unique_scales = torch.unique(scale)
        print(f"Number of unique scales: {len(unique_scales)}")
        assert len(unique_scales) <= 256, "Too many unique values in FP8 scales!"

        min_scale = torch.min(scale)
        max_scale = torch.max(scale)
        print(f"Min scale: {min_scale}, Max scale: {max_scale}")
        assert min_scale >= 2**-9 and max_scale <= 240.0, f"Scales out of range [2^-9, 240]! Got min={min_scale}, max={max_scale}"

        print("Case 1 passed!")

    print("All comprehensive tests passed!")

def test_against_jax():
    print("\n--- Test Case 5: Compare against JAX gold standard ---")
    import jax
    import jax.numpy as jnp
    import numpy as np

    def FP4_quant_jax(x):
        max_val = jnp.max(jnp.abs(x), axis=0, keepdims=True)
        raw_scale = max_val / 6.0
        scaling_factor = jnp.float32(jnp.float8_e4m3fn(raw_scale))
        scaled = jnp.where(scaling_factor != 0, x / scaling_factor, 0.0)
        clipped = jnp.clip(scaled, -6.0, 6.0)
        quant = np.float32(jnp.float4_e2m1fn(clipped))
        return quant, scaling_factor

    x_np = (np.random.rand(1024, 1024).astype(np.float32) - 0.5) * 20.0
    x_np = np.round(x_np / 0.002) * 0.002 + 0.0005

    block_size = 32
    x_jax = jnp.array(x_np.reshape(-1, block_size).T)

    quant_jax, scale_jax = FP4_quant_jax(x_jax)
    dequant_jax = quant_jax * scale_jax
    dequant_jax_orig = dequant_jax.T.reshape(1024, 1024)

    x_torch = torch.tensor(x_np).cuda()
    qx_torch, _, _, _ = FP4_quant_torch(x_torch, block_size, prevent_zero=False)

    dequant_jax_torch = torch.tensor(np.array(dequant_jax_orig)).cuda()

    def snr_jax(qout, grnd):
        qout = qout.astype(jnp.float32)
        grnd = grnd.astype(jnp.float32)
        error = grnd - qout
        return jnp.log(1 + jnp.sum(qout**2) / jnp.sum(error**2))

    grnd_jax = x_np.reshape(-1, block_size).T
    snr_j = snr_jax(dequant_jax, grnd_jax)
    print(f"JAX SNR: {snr_j}")

    def snr_torch(qout, grnd):
        error = grnd - qout
        return torch.log(1 + torch.sum(qout**2) / torch.sum(error**2))

    snr_t = snr_torch(qx_torch, x_torch)
    print(f"Torch SNR: {snr_t}")

    is_snr_close = torch.allclose(snr_t, torch.tensor(float(snr_j)).cuda(), atol=1e-1)
    print(f"Is SNR close? {is_snr_close}")

    max_diff = torch.max(torch.abs(qx_torch - dequant_jax_torch))
    print(f"Max difference against JAX: {max_diff}")

    print("Checking if Torch SNR >= 4.540...")
    assert snr_t >= 4.540, f"Torch SNR degraded! Got {snr_t}, expected >= 4.540"
    print("SNR check passed!")

    assert is_snr_close, "SNR comparison failed! Torch simulation not close to JAX gold standard!"
    print("Case 5 passed!")

def test_four_over_six():
    print("\n--- Test Case 6: Verify Four Over Six block selection ---")

    x_a = torch.tensor([[4.0, 0.0]], dtype=torch.float32).cuda()
    _, _, _, use_4_a = FP4_quant_torch(x_a, block_size=2, four_over_six=True)
    print(f"Use 4 (Part A): {use_4_a}")
    assert use_4_a.item() == True, "Failed to pick scale 4 when it was better!"
    print("Part A passed!")

    x_b = torch.tensor([[3.0, 1.0]], dtype=torch.float32).cuda()
    _, _, _, use_4_b = FP4_quant_torch(x_b, block_size=2, four_over_six=True)
    print(f"Use 4 (Part B): {use_4_b}")
    assert use_4_b.item() == False, "Failed to pick scale 6 when it was better!"
    print("Part B passed!")

def test_snr_improvement():
    print("\n--- Test Case 7: Verify SNR improvement with Four Over Six ---")
    x = (torch.rand(1024, 1024, dtype=torch.float32).cuda() - 0.5) * 20.0

    def snr_torch(qout, grnd):
        error = grnd - qout
        return torch.log(1 + torch.sum(qout**2) / torch.sum(error**2))

    qx_6, _, _, _ = FP4_quant_torch(x, block_size=32, four_over_six=False)
    snr_6 = snr_torch(qx_6, x)
    print(f"SNR without 4/6: {snr_6}")

    qx_46, _, _, _ = FP4_quant_torch(x, block_size=32, four_over_six=True)
    snr_46 = snr_torch(qx_46, x)
    print(f"SNR with 4/6: {snr_46}")

    print(f"SNR improvement: {snr_46 - snr_6}")

    assert snr_46 > snr_6, "SNR did not improve with Four Over Six!"
    print("Case 7 passed!")

def test_heterodoxy():
    print("\n--- Test Case 8: Verify heterodoxy in picks with bigger data ---")
    x = (torch.rand(1024, 1024, dtype=torch.float32).cuda() - 0.5) * 20.0

    _, _, _, use_4 = FP4_quant_torch(x, block_size=32, four_over_six=True)

    num_4 = torch.sum(use_4).item()
    num_6 = len(use_4) - num_4

    print(f"Number of blocks picking scale 4: {num_4}")
    print(f"Number of blocks picking scale 6: {num_6}")

    assert num_4 > 0 and num_6 > 0, "Heterodoxy check failed! Not both scales were picked!"
    print("Case 8 passed!")

def test_ue5m3():
    print("\n--- Test Case 9: Verify UE5M3 scale format ---")
    from torch_eval import get_ue5m3_grid

    grid, _ = get_ue5m3_grid()
    print(f"UE5M3 grid size: {len(grid)}")

    assert len(grid) == 248, f"Unexpected UE5M3 grid size! Got {len(grid)}, expected 248"

    min_val = torch.min(grid)
    max_val = torch.max(grid)
    print(f"Min value: {min_val}, Max value: {max_val}")

    assert min_val == 0.0, "Min value should be 0.0!"
    assert torch.isclose(max_val, torch.tensor(61440.0).cuda()), f"Max value should be 61440.0! Got {max_val}"
    print("Case 9 passed!")

def test_hierarchical_scaling_snr():
    print("\n--- Test Case 10: Verify SNR Improvement with Hierarchical Scaling ---")
    
    # Row 0: small values in [-0.1, 0.1]
    # Row 1: large values in [-10.0, 10.0]
    x = torch.zeros(2, 32, dtype=torch.float32).cuda()
    x[0] = (torch.rand(32).cuda() - 0.5) * 0.2
    x[1] = (torch.rand(32).cuda() - 0.5) * 20.0
    
    def snr_torch(qout, grnd):
        error = grnd - qout
        return torch.log(1 + torch.sum(qout**2) / torch.sum(error**2))
        
    # Without hierarchical scaling
    qx_std, _, _, _ = FP4_quant_torch(x, block_size=32, use_hierarchical=False)
    snr_std = snr_torch(qx_std, x)
    print(f"SNR Standard: {snr_std.item()}")
    
    # With hierarchical scaling
    qx_hier, _, _, _ = FP4_quant_torch(x, block_size=32, use_hierarchical=True)
    snr_hier = snr_torch(qx_hier, x)
    print(f"SNR Hierarchical: {snr_hier.item()}")
    
    print(f"SNR improvement: {snr_hier - snr_std}")
    
    # We expect at least similar or slightly better SNR, not worse!
    assert snr_hier >= snr_std - 0.1, "Hierarchical scaling degraded SNR!"
    print("Case 10 passed!")

def test_hadamard_orthonormality():
    print("\n--- Test Case 11: Verify Hadamard Orthonormality ---")
    
    for n in [32, 64, 256]:
        H = get_hadamard_matrix(n, device="cuda", dtype=torch.float32)
        assert H.shape == (n, n), f"Expected shape ({n}, {n}), got {H.shape}"
        
        # Check H @ H.T / n == I
        I_approx = (H @ H.t()) / n
        I_true = torch.eye(n, device="cuda", dtype=torch.float32)
        
        diff = torch.abs(I_approx - I_true).max()
        print(f"Size {n} orthonormality max diff: {diff.item()}")
        assert diff < 1e-5, f"Hadamard matrix of size {n} is not orthonormal! Max diff: {diff.item()}"
        
    print("Case 11 passed!")

def test_hadamard_inner_product_preservation():
    print("\n--- Test Case 12: Verify Hadamard Inner-Product Preservation ---")
    
    # x: (M, K), w: (N, K)
    # Let's choose K = 256, M = 4, N = 8
    # hadamard size must divide K. Here hadamard size = 256.
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    w = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    
    x_had, w_had = had_mod_torch(x, w, had_size=256, seed=123)
    
    out_orig = x @ w.t()
    out_had = x_had @ w_had.t()
    
    diff = torch.abs(out_orig - out_had).max()
    print(f"Inner-product preservation max diff: {diff.item()}")
    assert diff < 1e-4, f"Hadamard transform failed to preserve inner products! Max diff: {diff.item()}"
    
    print("Case 12 passed!")

def test_hadamard_seeding_consistency():
    print("\n--- Test Case 13: Verify Hadamard Seeding and Sign Randomization ---")
    
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    w = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    
    # Same seed -> same outputs
    x_had_a, w_had_a = had_mod_torch(x, w, had_size=256, seed=42)
    x_had_b, w_had_b = had_mod_torch(x, w, had_size=256, seed=42)
    
    diff_same_x = torch.abs(x_had_a - x_had_b).max().item()
    diff_same_w = torch.abs(w_had_a - w_had_b).max().item()
    print(f"Same seed diff - X: {diff_same_x}, W: {diff_same_w}")
    assert diff_same_x == 0.0 and diff_same_w == 0.0, "Different outputs with same seed!"
    
    # Different seed -> different outputs
    x_had_c, w_had_c = had_mod_torch(x, w, had_size=256, seed=43)
    diff_diff_x = torch.abs(x_had_a - x_had_c).max().item()
    diff_diff_w = torch.abs(w_had_a - w_had_c).max().item()
    print(f"Different seed diff - X: {diff_diff_x}, W: {diff_diff_w}")
    assert diff_diff_x > 1e-3 and diff_diff_w > 1e-3, "Same outputs with different seeds!"
    
    print("Case 13 passed!")

def test_mxfp4_grid_snapping():
    print("\n--- Test Case 14: Verify MXFP4 Grid Snapping ---")
    
    x = (torch.rand(4, 128, device="cuda", dtype=torch.float32) - 0.5) * 50.0
    
    # Test e2m1
    qx_e2m1, _, _, _ = quantize_mx_torch(x, block_size=32, elem_format="e2m1", scale_format="e8m0")
    grid_e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    
    # Reconstruct E8M0 scale per block
    x_b = x.reshape(-1, 32)
    max_abs = torch.max(torch.abs(x_b), dim=-1, keepdim=True).values
    max_abs = torch.clamp(max_abs, min=1e-7)
    log2_scale = torch.ceil(torch.log2(max_abs / 6.0))
    scale_e2m1 = (2.0 ** log2_scale).reshape(-1, 1)
    
    qx_b = qx_e2m1.reshape(-1, 32)
    scaled_qx = torch.abs(qx_b / scale_e2m1)
    
    dists = torch.abs(scaled_qx.unsqueeze(-1) - grid_e2m1)
    min_dists = torch.min(dists, dim=-1).values
    assert torch.allclose(min_dists, torch.zeros_like(min_dists), atol=1e-5), "e2m1 snapped values not on grid!"
    
    # Test e1m2
    qx_e1m2, _, _, _ = quantize_mx_torch(x, block_size=32, elem_format="e1m2", scale_format="e8m0")
    grid_e1m2 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda")
    
    log2_scale_e1m2 = torch.ceil(torch.log2(max_abs / 7.0))
    scale_e1m2 = (2.0 ** log2_scale_e1m2).reshape(-1, 1)
    
    qx_b_e1m2 = qx_e1m2.reshape(-1, 32)
    scaled_qx_e1m2 = torch.abs(qx_b_e1m2 / scale_e1m2)
    
    dists_e1m2 = torch.abs(scaled_qx_e1m2.unsqueeze(-1) - grid_e1m2)
    min_dists_e1m2 = torch.min(dists_e1m2, dim=-1).values
    assert torch.allclose(min_dists_e1m2, torch.zeros_like(min_dists_e1m2), atol=1e-5), "e1m2 snapped values not on grid!"
    
    print("Case 14 passed!")
 
def test_torch_mx_linear_pipeline():
    print("\n--- Test Case 15: Verify TorchMXLinear Pipeline Stability ---")
    
    # 1. Standard MXFP4 forward
    layer = TorchMXLinear(
        in_features=256, 
        out_features=128, 
        bias=True, 
        block_size=32, 
        elem_format="e2m1",
        scale_format="e8m0", 
        hadamard_size=256, 
        hadamard_seed=42
    ).cuda().bfloat16()
    
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    out = layer(x)
    
    assert out.shape == (8, 128), f"Expected shape (8, 128), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs!"
    assert not torch.isinf(out).any(), "Output contains Infs!"
    
    # 2. Standard MXFP4 forward with e1m2 format
    layer_e1m2 = TorchMXLinear(
        in_features=256, 
        out_features=128, 
        bias=True, 
        block_size=32, 
        elem_format="e1m2",
        scale_format="e8m0", 
        hadamard_size=256, 
        hadamard_seed=99
    ).cuda().bfloat16()
    
    out_e1m2 = layer_e1m2(x)
    assert out_e1m2.shape == (8, 128), f"Expected shape (8, 128), got {out_e1m2.shape}"
    assert not torch.isnan(out_e1m2).any(), "Output contains NaNs!"
    
    print("Case 15 passed!")

def test_generate_float_grid():
    print("\n--- Test Case 16: Verify generate_float_grid with Known Formats ---")
    from torch_eval import get_element_format_grid

    # Test e2m1 (Standard OCP MXFP4 element grid)
    grid_e2m1, _, max_e2m1 = get_element_format_grid("e2m1")
    expected_e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=grid_e2m1.device)
    assert torch.allclose(grid_e2m1, expected_e2m1), f"e2m1 grid mismatch! Got {grid_e2m1}"
    assert max_e2m1 == 6.0, f"e2m1 max mismatch! Got {max_e2m1}"
    print("e2m1 grid verified!")

    # Test e1m2 (Alternative FP4 format element grid)
    grid_e1m2, _, max_e1m2 = get_element_format_grid("e1m2")
    expected_e1m2 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device=grid_e1m2.device)
    assert torch.allclose(grid_e1m2, expected_e1m2), f"e1m2 grid mismatch! Got {grid_e1m2}"
    assert max_e1m2 == 7.0, f"e1m2 max mismatch! Got {max_e1m2}"
    print("e1m2 grid verified!")

    # Test e4m3 (Standard OCP FP8 scale format)
    _, _, max_e4m3 = get_element_format_grid("e4m3")
    assert max_e4m3 == 240.0, f"e4m3 max mismatch! Got {max_e4m3}, expected 240.0"
    print("e4m3 max value verified!")

    # Test ue5m3 (Standard UE5M3 scale format)
    _, _, max_ue5m3 = get_element_format_grid("ue5m3")
    assert max_ue5m3 == 61440.0, f"ue5m3 max mismatch! Got {max_ue5m3}, expected 61440.0"
    print("ue5m3 max value verified!")

    print("Case 16 passed!")

if __name__ == "__main__":
    test_torch_quant()
    test_against_jax()
    test_four_over_six()
    test_snr_improvement()
    test_heterodoxy()
    test_ue5m3()
    test_hierarchical_scaling_snr()
    
    # New tests
    test_hadamard_orthonormality()
    test_hadamard_inner_product_preservation()
    test_hadamard_seeding_consistency()
    test_mxfp4_grid_snapping()
    test_torch_mx_linear_pipeline()
    test_generate_float_grid()
