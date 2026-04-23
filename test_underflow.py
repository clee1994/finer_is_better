import torch
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs

def test_underflow():
    min_scale = 2**-9
    print(f"Target min scale: {min_scale}")
    
    x = torch.tensor([2**-15, 2**-15 * 2], dtype=torch.float32).cuda()
    print(f"Original x: {x}")
    
    specs_no_sm = MxSpecs(
        scale_bits=8,
        a_elem_format="fp4_e2m1",
        block_size=2,
        custom_cuda=True,
    )
    
    specs_sm = MxSpecs(
        scale_bits=8,
        a_elem_format="fp4_e2m1",
        block_size=2,
        custom_cuda=True,
        scale_min=min_scale,
    )
    
    with torch.no_grad():
        qx_no_sm = quantize_mx_op(
            x.unsqueeze(0),
            specs_no_sm,
            elem_format="fp4_e2m1",
            axes=[-1],
            round=specs_no_sm["round_mx_output"],
        )
        print(f"Quantized without scale_min: {qx_no_sm}")
        
        qx_sm = quantize_mx_op(
            x.unsqueeze(0),
            specs_sm,
            elem_format="fp4_e2m1",
            axes=[-1],
            round=specs_sm["round_mx_output"],
        )
        print(f"Quantized with scale_min: {qx_sm}")
        
        is_zero_no_sm = torch.allclose(qx_no_sm, torch.zeros_like(qx_no_sm))
        is_zero_sm = torch.allclose(qx_sm, torch.zeros_like(qx_sm))
        
        print(f"Is zero without scale_min? {is_zero_no_sm}")
        print(f"Is zero with scale_min? {is_zero_sm}")
        
        assert is_zero_no_sm, "Expected underflow to zero without scale_min failed even at 2**-15!"
        print("Underflow test passed!")

if __name__ == "__main__":
    test_underflow()
