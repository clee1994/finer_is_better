import torch
from torch_eval import FP4_quant_torch

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

def test_hierarchical_scaling_overflow():
    print("\n--- Test Case 10: Verify Hierarchical Scaling Overflow Protection ---")
    
    x = torch.zeros(1, 32, dtype=torch.float32).cuda()
    # Values up to 5000, will cause block scale to be ~833 > 448
    x[0] = torch.linspace(1000.0, 5000.0, 32).cuda()
    
    # Without hierarchical scaling
    qx_std, _, scale_std, _ = FP4_quant_torch(x, block_size=32, use_hierarchical=False)
    print(f"Standard scale: {scale_std.item()}")
    # It should be clipped to 448.0!
    assert torch.isclose(scale_std, torch.tensor(448.0).cuda()), f"Expected scale to be clipped to 448.0, got {scale_std.item()}"
    
    # With hierarchical scaling
    qx_hier, _, scale_hier, _ = FP4_quant_torch(x, block_size=32, use_hierarchical=True, channel_dim=0)
    print(f"Hierarchical block scale: {scale_hier.item()}")
    # It should be much smaller!
    assert scale_hier.item() < 100.0, f"Expected smaller block scale, got {scale_hier.item()}"
    
    # Compare MSE
    mse_std = torch.mean((x - qx_std)**2)
    mse_hier = torch.mean((x - qx_hier)**2)
    print(f"MSE Standard: {mse_std.item()}")
    print(f"MSE Hierarchical: {mse_hier.item()}")
    
    assert mse_hier < mse_std / 10.0, "Hierarchical scaling did not significantly reduce MSE for large values!"
    print("Case 10 passed!")

if __name__ == "__main__":
    test_torch_quant()
    test_against_jax()
    test_four_over_six()
    test_snr_improvement()
    test_heterodoxy()
    test_ue5m3()
    test_hierarchical_scaling_overflow()
