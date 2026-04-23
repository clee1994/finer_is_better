import torch
from torch_eval import FP4_quant_torch

def test_torch_quant():
    # Use random values in [-10, 10] to test negative values too!
    x = (torch.rand(1024, 1024, dtype=torch.float32).cuda() - 0.5) * 20.0
    print(f"Original x (Case 1) shape: {x.shape}")
    
    block_size = 32
    with torch.no_grad():
        qx, quant, scale = FP4_quant_torch(x, block_size)
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

if __name__ == "__main__":
    test_torch_quant()
