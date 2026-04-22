import torch
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs
from mx.formats import _get_format_params

def test_scale_min():
    ebits, mbits, emax, max_norm, min_norm = _get_format_params("fp8_e4m3")
    print(f"Min norm for FP8 scale: {min_norm}")
    
    x = torch.tensor([min_norm * 0.01, min_norm * 0.05], dtype=torch.float32).cuda()
    print(f"Original x: {x}")
    
    specs_no_sm = MxSpecs(
        scale_bits=8,
        a_elem_format="fp4_e2m1",
        block_size=2,
        custom_cuda=True,
        bfloat_subnorms=False,
    )
    
    specs_sm = MxSpecs(
        scale_bits=8,
        a_elem_format="fp4_e2m1",
        block_size=2,
        custom_cuda=True,
        bfloat_subnorms=False,
        scale_min=min_norm,
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
        
        assert not is_zero_sm, "Prevent Zero failed! Scale min had no effect!"
        print("Scale min test passed!")

if __name__ == "__main__":
    test_scale_min()
